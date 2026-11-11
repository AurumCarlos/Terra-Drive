//! Fenetre, boucle de jeu et API que le jeu utilise.

use crate::camera::Camera;
use crate::error::Error;
use crate::input::Input;
use crate::render::{Renderer, UiFrame};
use crate::sky::Sky;
use crate::ui::{self, HudData, MenuAction, MenuState};
use crate::world::generator::{dir_to_lat_lon, lat_lon_to_dir, Generator, PlanetParams};
use crate::world::streaming::{StreamConfig, Streamer};
use glam::DVec3;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

#[derive(Clone)]
pub struct EngineConfig {
    pub title: String,
    pub planet: PlanetParams,
    pub width: u32,
    pub height: u32,
    pub vsync: bool,
    pub stream: StreamConfig,
    /// Capture automatique : attend que le monde soit charge, enregistre l'image et quitte.
    pub auto_screenshot: Option<PathBuf>,
    /// Ouvre le menu des le depart.
    pub start_with_menu: bool,
    /// Son actif au demarrage (M le coupe et le remet).
    pub sound: bool,
    /// Ou le jeu posera la camera sur une nouvelle planete (direction, cap), selon le ciel et l'altitude
    /// de la camera au moment de la demande ; `None` si elle reste ou elle est. Calcule en arriere-plan,
    /// avec le paysage autour, pendant qu'on joue encore sur l'ancienne planete.
    pub landing: Option<Landing>,
}

/// Choix du point d'arrivee sur une planete : voir [`EngineConfig::landing`].
pub type Landing = fn(&Generator, &Sky, f64) -> Option<(DVec3, f32)>;

/// Hauteur des yeux au-dessus du sol a l'arrivee.
pub const EYE_HEIGHT: f64 = 1.75;

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            title: "Terra".to_string(),
            planet: PlanetParams::default(),
            width: 1600,
            height: 900,
            vsync: true,
            stream: StreamConfig::default(),
            auto_screenshot: None,
            start_with_menu: false,
            sound: true,
            landing: None,
        }
    }
}

/// Ce qui vient de se passer hors du jeu et qui appelle une reaction de sa part.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GameEvent {
    /// La planete a ete remplacee.
    PlanetChanged,
    GoToSpace,
    GoToSurface,
    /// Aller survoler ce point de la planete, choisi sur la carte.
    GoTo(DVec3),
    /// Arriver en vol au-dessus d'une ville, face a son centre.
    GoToCity(DVec3),
}

/// Ce que le moteur expose au jeu a chaque image.
pub struct Context<'a> {
    pub input: &'a Input,
    pub camera: &'a mut Camera,
    pub sky: &'a mut Sky,
    pub world: &'a Arc<Generator>,
    /// Etat du monde charge autour de la camera (objets, collisions).
    pub streamer: &'a Streamer,
    pub events: &'a [GameEvent],
    /// Le jeu y note la vitesse et le mode, pour l'affichage.
    pub hud: &'a mut HudData,
    /// Remplace la planete a la fin de l'image.
    pub regenerate: &'a mut Option<PlanetParams>,
    /// Demande une capture d'ecran vers ce fichier.
    pub screenshot: &'a mut Option<PathBuf>,
    /// Passe a vrai pour quitter.
    pub exit: &'a mut bool,
    /// La voiture a dessiner a cette image, s'il y en a une.
    pub vehicle: &'a mut Option<crate::vehicle::VehicleFrame>,
    /// Le son : le jeu y met la voiture conduite et les chocs ; le moteur calcule l'ambiance.
    pub sound: &'a mut crate::audio::SoundParams,
}

/// A implementer par le jeu.
pub trait Game: 'static {
    /// Appele une fois, avant la premiere image : place la camera, regle le ciel.
    fn start(&mut self, ctx: &mut Context);
    /// Appele a chaque image, avant le rendu : `dt` est le temps ecoule en secondes, borne a 0,1 s.
    fn update(&mut self, ctx: &mut Context, dt: f32);
}

struct State {
    window: Arc<Window>,
    renderer: Renderer,
    streamer: Streamer,
    generator: Arc<Generator>,
    input: Input,
    camera: Camera,
    sky: Sky,
    ui_ctx: egui::Context,
    ui_winit: egui_winit::State,
    menu: MenuState,
    hud: HudData,
    events: Vec<GameEvent>,
    regenerate: Option<PlanetParams>,
    /// Planete en cours de simulation : l'ancienne reste affichee en attendant.
    pending: Option<(PlanetParams, Instant, Receiver<Prepared>)>,
    last_frame: Instant,
    started: Instant,
    frames: u64,
    fps_timer: f32,
    fps_frames: u32,
    idle_frames: u32,
    screenshot: Option<PathBuf>,
    exit: bool,
    vehicle: Option<crate::vehicle::VehicleFrame>,
    /// Temps passe dans chaque phase de l'image (TERRA_PROFILE=1).
    profile: Option<Profile>,
    audio: Option<crate::audio::Output>,
    sound: crate::audio::SoundParams,
    muted: bool,
    /// L'ambiance se recalcule quelques fois par seconde : temps ecoule, position precedente.
    ambience_timer: f32,
    last_camera: DVec3,
    camera_speed: f32,
    /// Voir [`EngineConfig::landing`].
    config_landing: Option<Landing>,
}

/// Une planete simulee en arriere-plan, avec le paysage du point d'arrivee s'il y en a un.
struct Prepared {
    generator: Arc<Generator>,
    warm: Option<(Streamer, crate::world::streaming::StreamUpdate)>,
    weather: Vec<Vec<u8>>,
    maps: [egui::ColorImage; 2],
}

/// Temps moyen des phases d'une image, affiche toutes les 5 s.
#[derive(Default)]
struct Profile {
    sums: [f64; 6],
    frames: u32,
    since: f64,
}

impl Profile {
    const NAMES: [&'static str; 6] = ["jeu", "streaming", "envoi chunks", "hud", "interface", "rendu"];

    fn add(&mut self, times: [f64; 6], dt: f64) {
        for (s, t) in self.sums.iter_mut().zip(times) {
            *s += t;
        }
        self.frames += 1;
        self.since += dt;
        if self.since >= 5.0 {
            let n = self.frames as f64;
            let parts: Vec<String> =
                Self::NAMES.iter().zip(self.sums).map(|(name, s)| format!("{name} {:.2}", s / n * 1000.0)).collect();
            println!("profil (ms par image, {} images) : {}", self.frames, parts.join(", "));
            *self = Self::default();
        }
    }
}

struct App<G: Game> {
    config: EngineConfig,
    game: G,
    state: Option<State>,
    /// L'erreur qui a arrete la boucle, rendue par `run`.
    error: Option<Error>,
}

impl<G: Game> App<G> {
    /// Arrete la boucle sur une erreur fatale, que `run` rendra a l'appelant.
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: Error) {
        self.error.get_or_insert(error);
        event_loop.exit();
    }
}

/// Lance le moteur ; ne rend la main qu'a la fermeture de la fenetre.
///
/// Les reglages de la planete sont ramenes dans leurs bornes (voir [`PlanetParams::sanitized`]).
///
/// # Errors
///
/// Si la fenetre ou le rendu ne peuvent pas demarrer, ou si le rendu echoue sans reprise possible.
pub fn run<G: Game>(mut config: EngineConfig, game: G) -> crate::Result<()> {
    config.planet = config.planet.sanitized();
    config.width = config.width.clamp(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE);
    config.height = config.height.clamp(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE);
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { config, game, state: None, error: None };
    event_loop.run_app(&mut app)?;
    app.error.map_or(Ok(()), Err)
}

/// Plus petite fenetre demandee a l'ouverture, en pixels : une surface de taille nulle est invalide.
pub const MIN_WINDOW_SIZE: u32 = 64;
/// Plus grande fenetre demandee a l'ouverture : au-dela, les textures depassent ce que les cartes acceptent.
pub const MAX_WINDOW_SIZE: u32 = 8192;

fn set_capture(window: &Window, input: &mut Input, capture: bool) {
    if capture {
        let locked = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        input.captured = locked.is_ok();
    } else {
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        input.captured = false;
    }
    window.set_cursor_visible(!input.captured);
}

impl<G: Game> ApplicationHandler for App<G> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(&self.config.title)
            .with_inner_size(PhysicalSize::new(self.config.width, self.config.height));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => return self.fail(event_loop, e.into()),
        };
        let mut renderer = match pollster::block_on(Renderer::new(window.clone(), self.config.vsync)) {
            Ok(r) => r,
            Err(e) => return self.fail(event_loop, e),
        };
        println!("GPU : {}", renderer.adapter_name);

        crate::world::generator::set_planet_radius(self.config.planet.radius());
        let generator = Arc::new(Generator::new(self.config.planet));
        renderer.set_weather(|dir| crate::weather::encode_climate(&generator, dir));
        let streamer = Streamer::new(generator.clone(), self.config.stream);
        let ui_ctx = egui::Context::default();
        let ui_winit = egui_winit::State::new(
            ui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let mut menu = MenuState::new(self.config.planet);
        menu.set_map(ui::planet_maps(&generator));
        menu.open = self.config.start_with_menu;
        menu.render_distance_km = self.config.stream.render_distance / 1000.0;

        let mut state = State {
            window,
            renderer,
            streamer,
            generator,
            input: Input::default(),
            camera: Camera::new(DVec3::Y * crate::world::generator::planet_radius(), 0.0, 0.0),
            sky: Sky::default(),
            ui_ctx,
            ui_winit,
            menu,
            hud: HudData::default(),
            events: Vec::new(),
            regenerate: None,
            pending: None,
            last_frame: Instant::now(),
            started: Instant::now(),
            frames: 0,
            fps_timer: 0.0,
            fps_frames: 0,
            idle_frames: 0,
            screenshot: None,
            exit: false,
            vehicle: None,
            profile: std::env::var("TERRA_PROFILE").is_ok().then(Profile::default),
            audio: crate::audio::Output::open(),
            sound: crate::audio::SoundParams::default(),
            muted: !self.config.sound,
            ambience_timer: 1.0,
            last_camera: DVec3::ZERO,
            camera_speed: 0.0,
            config_landing: self.config.landing,
        };
        if state.audio.is_none() {
            println!("son : aucune sortie audio, le jeu reste muet");
        }
        let mut ctx = state.context();
        self.game.start(&mut ctx);
        self.state = Some(state);
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let (Some(s), DeviceEvent::MouseMotion { delta }) = (&mut self.state, event) {
            s.input.mouse_delta.0 += delta.0;
            s.input.mouse_delta.1 += delta.1;
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(s) = &mut self.state else { return };
        // Souris capturee : l'interface ne doit pas voir les mouvements du jeu.
        let consumed = if s.input.captured { false } else { s.ui_winit.on_window_event(&s.window, &event).consumed };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => s.renderer.resize(size.width, size.height),
            WindowEvent::Focused(false) => {
                set_capture(&s.window, &mut s.input, false);
                s.input.release_all();
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                if !s.menu.open && !consumed {
                    set_capture(&s.window, &mut s.input, true);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !s.menu.open {
                    s.input.scroll += match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                    };
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let down = event.state == ElementState::Pressed;
                    if code == KeyCode::Escape {
                        if down && !event.repeat {
                            s.set_menu(!s.menu.open);
                        }
                    } else if !s.menu.open || !down {
                        s.input.set_key(code, down, event.repeat);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = s.frame(&mut self.game, &self.config, event_loop) {
                    self.fail(event_loop, e);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

impl State {
    fn context(&mut self) -> Context<'_> {
        Context {
            input: &self.input,
            camera: &mut self.camera,
            sky: &mut self.sky,
            world: &self.generator,
            streamer: &self.streamer,
            events: &self.events,
            hud: &mut self.hud,
            regenerate: &mut self.regenerate,
            screenshot: &mut self.screenshot,
            exit: &mut self.exit,
            vehicle: &mut self.vehicle,
            sound: &mut self.sound,
        }
    }

    /// L'ambiance sonore du lieu : vent, eau, arbres, oiseaux ou grillons selon l'heure, ville.
    fn ambience(&mut self, dt: f32) {
        let moved = if self.last_camera == DVec3::ZERO {
            0.0
        } else {
            (self.camera.position - self.last_camera).length() as f32
        };
        self.last_camera = self.camera.position;
        if dt > 0.0 {
            let k = 1.0 - (-dt / 0.3).exp();
            self.camera_speed += (moved / dt - self.camera_speed) * k;
        }
        self.sound.rush = if self.sound.car.is_some() { 0.0 } else { self.camera_speed };
        self.ambience_timer += dt;
        if self.ambience_timer < 0.25 {
            return;
        }
        self.ambience_timer = 0.0;
        let gen = &self.generator;
        let r = crate::world::generator::planet_radius();
        let up = self.camera.up();
        let here = gen.surface(up);
        let altitude = self.camera.position.length() - r;
        let above = (altitude - (here.height.max(here.water_level)) as f64).max(0.0) as f32;
        // Les sons du sol s'effacent en prenant de la hauteur.
        let near = (1.0 - above / 250.0).clamp(0.0, 1.0);
        let s = &mut self.sound;
        s.wind = if altitude > 60_000.0 { 0.0 } else { 0.25 + 0.6 * (above / 800.0).min(1.0) };

        // L'eau autour : mer (vagues) ou riviere et lac (eau qui coule).
        let east = DVec3::Y.cross(up).normalize_or(DVec3::X);
        let north = up.cross(east);
        let (mut sea, mut fresh, mut land) = (0, 0, 0);
        for ring in [60.0, 220.0] {
            for k in 0..8 {
                let a = k as f64 * std::f64::consts::TAU / 8.0;
                let d = (up + (east * a.cos() + north * a.sin()) * (ring / r)).normalize();
                let sf = gen.surface(d);
                if sf.water_level > sf.height + 0.3 {
                    if sf.fresh {
                        fresh += 1;
                    } else {
                        sea += 1;
                    }
                } else {
                    land += 1;
                }
            }
        }
        let coast = if land > 0 { 1.0 } else { 0.5 };
        s.waves = (sea as f32 / 5.0).min(1.0) * coast * near;
        s.stream = (fresh as f32 / 3.0).min(1.0) * near;

        // Arbres a moins de 60 m.
        let mut trees = 0usize;
        let cam = self.camera.position;
        for cell in self.streamer.trees.values() {
            if (cell.center - cam).length() > 600.0 {
                continue;
            }
            for list in cell.objects.iter().take(6) {
                trees += list
                    .iter()
                    .filter(|it| {
                        (cell.center + glam::Vec3::from_array(it.pos).as_dvec3() - cam).length_squared() < 3600.0
                    })
                    .count();
            }
        }
        let wooded = (trees as f32 / 25.0).min(1.0);
        let climate = gen.climate(up, here.height);
        let t = self.sky.local_time(up);
        let day = ((t - 0.25) / 0.04).clamp(0.0, 1.0) * ((0.77 - t) / 0.04).clamp(0.0, 1.0);
        let dry = here.water_level > here.height;
        s.leaves = wooded * near;
        s.birds = if dry || climate.mean < -5.0 { 0.0 } else { (0.15 + 0.85 * wooded) * day * near };
        s.insects = if dry || climate.mean < 8.0 { 0.0 } else { (0.4 + 0.6 * wooded) * (1.0 - day) * near };
        s.city = gen.cities.nearest(up).map_or(0.0, |(_, d)| (1.0 - d as f32 / 1500.0).clamp(0.0, 1.0))
            * (1.0 - above / 600.0).clamp(0.0, 1.0);
    }

    fn set_menu(&mut self, open: bool) {
        self.menu.open = open;
        set_capture(&self.window, &mut self.input, !open);
        self.input.release_all();
    }

    /// La simulation prend quelques secondes : elle tourne a part, le jeu continue pendant ce temps.
    /// Une nouvelle demande remplace la precedente, dont le resultat sera perdu.
    fn request_planet(&mut self, params: PlanetParams) {
        let params = params.sanitized();
        let (send, receive) = channel();
        let (sky, landing, stream) = (self.sky.probe(), self.config_landing, self.streamer.config);
        let altitude = self.camera.position.length() - crate::world::generator::planet_radius();
        std::thread::spawn(move || {
            let radius = params.radius();
            crate::world::generator::worker_thread(radius);
            let generator = Arc::new(Generator::new(params));
            let weather = Renderer::weather_texels(|dir| crate::weather::encode_climate(&generator, dir));
            let maps = ui::planet_maps(&generator);
            // Le point d'arrivee et le paysage autour se preparent ici : a l'echange, tout est pret.
            let warm = landing.and_then(|f| f(&generator, &sky, altitude)).map(|(dir, yaw)| {
                let _ = generator.landing.set((dir, yaw));
                let eye = dir * (radius + generator.height(dir).max(0.0) as f64 + EYE_HEIGHT);
                let mut streamer = Streamer::new(generator.clone(), stream);
                let chunks = streamer.prewarm(eye, std::time::Duration::from_secs(8));
                (streamer, chunks)
            });
            let _ = send.send(Prepared { generator, warm, weather, maps });
        });
        self.pending = Some((params, Instant::now(), receive));
        self.menu.sync(params);
    }

    /// Vrai si la planete vient d'etre remplacee.
    fn receive_planet(&mut self) -> bool {
        let Some((params, started, receiver)) = &self.pending else { return false };
        let prepared = match receiver.try_recv() {
            Ok(prepared) => prepared,
            Err(TryRecvError::Empty) => return false,
            // Le thread de simulation s'est arrete sans rien rendre (panique) : on garde la planete
            // actuelle, et le menu redevient disponible au lieu d'attendre indefiniment.
            Err(TryRecvError::Disconnected) => {
                eprintln!("la creation de la planete (graine {}) a echoue : planete actuelle conservee", params.seed);
                self.pending = None;
                self.menu.sync(self.generator.params);
                return false;
            }
        };
        let (params, seconds) = (*params, started.elapsed().as_secs_f32());
        self.pending = None;
        // L'ancienne generation s'arrete ici : ses threads finissent leur tache puis sortent.
        // La distance de rendu choisie dans le menu se conserve d'une planete a l'autre.
        let stream = self.streamer.config;
        let Prepared { generator, warm, weather, maps } = prepared;
        crate::world::generator::set_planet_radius(generator.params.radius());
        self.generator = generator;
        self.renderer.clear_world();
        match warm {
            // Le paysage du point d'arrivee est deja calcule : il part au GPU d'un coup.
            Some((streamer, chunks)) => {
                self.streamer = streamer;
                self.renderer.sync_chunks(chunks);
            }
            None => self.streamer = Streamer::new(self.generator.clone(), stream),
        }
        self.renderer.set_weather_texels(&weather);
        self.menu.set_map(maps);
        self.events.push(GameEvent::PlanetChanged);
        self.idle_frames = 0;
        println!("nouvelle planete : graine {} en {seconds:.2} s", params.seed);
        true
    }

    /// Interface de l'image : decrit le menu et le tessele pour le GPU.
    fn run_ui(&mut self) -> (UiFrame, Vec<MenuAction>) {
        let input = self.ui_winit.take_egui_input(&self.window);
        let mut actions = Vec::new();
        let (menu, hud) = (&mut self.menu, &self.hud);
        let output = self.ui_ctx.run(input, |ctx| actions = ui::draw(ctx, menu, hud));
        if self.menu.open {
            self.ui_winit.handle_platform_output(&self.window, output.platform_output);
        }
        let primitives = self.ui_ctx.tessellate(output.shapes, output.pixels_per_point);
        (UiFrame { primitives, textures: output.textures_delta, pixels_per_point: output.pixels_per_point }, actions)
    }

    /// Une image : jeu, streaming, rendu. Rend l'erreur de rendu quand elle est sans reprise possible.
    fn frame<G: Game>(
        &mut self,
        game: &mut G,
        config: &EngineConfig,
        event_loop: &ActiveEventLoop,
    ) -> crate::Result<()> {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        // Une grosse saccade ne doit pas faire traverser le decor au joueur.
        let step = dt.min(0.1);

        let mut times = [0.0f64; 6];
        let mut mark = Instant::now();
        let mut lap = |k: usize, times: &mut [f64; 6]| {
            let t = Instant::now();
            times[k] += (t - mark).as_secs_f64();
            mark = t;
        };
        self.sky.advance(step);
        let generator = self.generator.clone();
        self.sky.weather.update(&generator, self.camera.position, step);
        let (up, alt) = (self.camera.up(), self.camera.position.length() - crate::world::generator::planet_radius());
        self.sky.weather.sheltered = generator
            .cities
            .highways
            .tunnel(&generator, up)
            .is_some_and(|t| alt < t.road + crate::world::highway::VAULT);
        self.sound.car = None;
        {
            let mut ctx = self.context();
            game.update(&mut ctx, step);
        }
        self.events.clear();
        if self.input.just_pressed(KeyCode::KeyM) {
            self.muted = !self.muted;
        }
        self.input.end_frame();
        if self.exit {
            event_loop.exit();
            return Ok(());
        }

        lap(0, &mut times);
        // A l'echange, la camera est encore a l'ancienne place : le jeu la deplace a l'image suivante.
        let swapped = self.receive_planet();
        self.hud.generating = self.pending.is_some();
        let update = if swapped { Default::default() } else { self.streamer.update(self.camera.position) };
        lap(1, &mut times);
        self.renderer.sync_chunks(update);
        lap(2, &mut times);

        // Laisse quelques images apres le chargement pour que l'exposition soit stable.
        let mut capture = self.screenshot.take();
        let mut finish_after = false;
        if let Some(path) = &config.auto_screenshot {
            self.idle_frames = if self.streamer.is_idle() && self.pending.is_none() && self.regenerate.is_none() {
                self.idle_frames + 1
            } else {
                0
            };
            if self.idle_frames == 30 {
                capture = Some(path.clone());
                finish_after = true;
                println!(
                    "monde charge en {:.2} s ({} chunks en cache, {} images)",
                    self.started.elapsed().as_secs_f32(),
                    self.streamer.loaded_chunks(),
                    self.frames
                );
            }
        }

        let (lat, lon) = dir_to_lat_lon(self.camera.up());
        self.hud.altitude = self.camera.position.length() - crate::world::generator::planet_radius();
        self.hud.latitude = lat;
        self.hud.longitude = lon;
        self.hud.local_time = self.sky.local_time(self.camera.up());
        self.hud.weather = self.sky.weather.describe();
        self.hud.seed = self.generator.params.seed;
        self.hud.render_distance = self.streamer.config.render_distance;
        let up = self.camera.up();
        self.hud.circuit = self.generator.cities.circuits.nearest(up).and_then(|c| {
            let dist = self
                .generator
                .cities
                .circuits
                .clearance(up)
                .unwrap_or_else(|| {
                    ((c.center - up).length() * crate::world::generator::planet_radius() - c.radius).max(0.0)
                })
                .max(0.0);
            (dist < 6000.0).then(|| (c.name(), c.location(), dist))
        });
        self.hud.mountain = self.generator.cities.mountains.nearest(up).and_then(|(road, far)| {
            let dist = self.generator.cities.mountains.clearance(up).unwrap_or(far).max(0.0);
            (dist < 4000.0).then(|| (road.name.clone(), road.hairpins.len(), dist))
        });
        self.hud.city = self.generator.cities.nearest(up).map(|(i, dist)| {
            let c = &self.generator.cities.list[i];
            let east = DVec3::Y.cross(up).normalize_or(DVec3::X);
            let north = up.cross(east);
            let to = c.dir - up;
            let bearing = to.dot(east).atan2(to.dot(north)).to_degrees();
            (c.name.clone(), c.style.label(), dist, bearing)
        });
        self.hud.loading = !self.streamer.is_idle();

        self.ambience(step);
        self.sound.volume = if self.muted { 0.0 } else { 0.8 };
        if let Some(audio) = &self.audio {
            audio.set(self.sound);
        }
        lap(3, &mut times);
        let (ui_frame, actions) = self.run_ui();
        lap(4, &mut times);

        let stats = self.renderer.render(
            &self.camera,
            &self.sky,
            &self.streamer,
            ui_frame,
            capture.as_deref(),
            self.vehicle.as_ref(),
        );
        lap(5, &mut times);
        if let Some(p) = &mut self.profile {
            if p.frames == 0 {
                self.streamer.timings = [0.0; 6];
                self.renderer.timings = [0.0; 4];
            }
            if p.since + dt as f64 >= 5.0 {
                let n = (p.frames + 1) as f64;
                let t: Vec<String> = self.streamer.timings.iter().map(|t| format!("{:.2}", t / n * 1000.0)).collect();
                println!("  streaming (resultats, sol, quadtree, cellules, eviction, file) : {} ; {} chunks choisis, {} charges, {} generes", t.join(" "), self.streamer.selected.len(), self.streamer.loaded_chunks(), self.streamer.generated);
                let t: Vec<String> = self.renderer.timings.iter().map(|t| format!("{:.2}", t / n * 1000.0)).collect();
                println!("  rendu (attente image, preparation, enregistrement, envoi) : {}", t.join(" "));
                if let Some(gpu) = self.renderer.gpu_timer.as_mut().and_then(|t| t.take()) {
                    let parts: Vec<String> =
                        crate::render::GPU_SECTIONS.iter().zip(gpu).map(|(n, ms)| format!("{n} {ms:.2}")).collect();
                    println!("  GPU (ms) : {}, total {:.2}", parts.join(", "), gpu.iter().sum::<f64>());
                }
            }
            p.add(times, dt as f64);
        }
        let stats = match stats {
            Ok(stats) => stats,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let size = self.window.inner_size();
                self.renderer.resize(size.width, size.height);
                return Ok(());
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        if finish_after {
            println!(
                "{} chunks dessines, {:.2} M triangles, {} objets, derniere image {:.2} ms",
                stats.chunks_drawn,
                stats.triangles as f32 / 1.0e6,
                stats.objects,
                dt * 1000.0
            );
            event_loop.exit();
            return Ok(());
        }

        self.hud.chunks = stats.chunks_drawn;
        self.hud.triangles = stats.triangles;
        self.hud.objects = stats.objects;
        self.frames += 1;
        self.fps_frames += 1;
        self.fps_timer += dt;
        if self.fps_timer >= 0.5 {
            self.hud.fps = self.fps_frames as f32 / self.fps_timer;
            self.fps_timer = 0.0;
            self.fps_frames = 0;
        }

        for action in actions {
            match action {
                MenuAction::Resume => self.set_menu(false),
                MenuAction::NewPlanet(params) => self.request_planet(params),
                MenuAction::SetRenderDistance(metres) => self.streamer.config.set_render_distance(metres),
                MenuAction::SetSeason(season) => {
                    self.sky.weather.season = season;
                    self.sky.declination = self.sky.weather.declination();
                }
                MenuAction::GoToSpace => {
                    self.events.push(GameEvent::GoToSpace);
                    self.set_menu(false);
                }
                MenuAction::GoToSurface => {
                    self.events.push(GameEvent::GoToSurface);
                    self.set_menu(false);
                }
                MenuAction::GoToCity => {
                    if let Some((i, _)) = self.generator.cities.nearest(self.camera.up()) {
                        self.events.push(GameEvent::GoToCity(self.generator.cities.list[i].dir));
                    }
                    self.set_menu(false);
                }
                MenuAction::GoTo(lat, lon) => {
                    self.events.push(GameEvent::GoTo(lat_lon_to_dir(lat, lon)));
                    self.set_menu(false);
                }
                MenuAction::Quit => event_loop.exit(),
            }
        }
        if let Some(params) = self.regenerate.take() {
            self.request_planet(params);
        }
        Ok(())
    }
}
