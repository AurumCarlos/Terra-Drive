//! Demo : explorer une planete entiere, a pied, en vol, et jusque dans l'espace.

use glam::DVec3;
use std::path::PathBuf;
use terra::app::{Context, EngineConfig, Game, GameEvent, MAX_WINDOW_SIZE, MIN_WINDOW_SIZE};
use terra::player::{world_lock, PlayerController};
use terra::vehicle::{parking_spot, Car, View};
use terra::sky::Sky;
use terra::world::generator::{lat_lon_to_dir, planet_radius, Generator, PlanetParams};
use terra::Camera;
use winit::keyboard::KeyCode;

#[derive(Default)]
struct Options {
    /// Latitude et longitude de depart, en degres.
    position: Option<(f64, f64)>,
    altitude: Option<f64>,
    yaw: Option<f32>,
    pitch: Option<f32>,
    time: Option<f32>,
    /// Duree d'une journee en secondes ; 0 fige le soleil.
    day_length: f32,
    /// Vol en ligne droite a cette vitesse (m/s), pour mesurer le streaming, puis sortie.
    autopilot: Option<f64>,
    /// Remplace la planete apres quelques images, pour verifier la regeneration.
    regenerate_seed: Option<u64>,
    /// Demarre au volant d'une voiture.
    drive: bool,
    /// Vue du conducteur au demarrage.
    cockpit: bool,
    /// Demarre sur la grille du circuit le plus proche.
    circuit: bool,
    /// Demarre au pied de la route de montagne la plus proche.
    mountain: bool,
}

struct Explorer {
    options: Options,
    player: PlayerController,
    /// La voiture, garee ou conduite.
    car: Option<Car>,
    driving: bool,
    /// Place de la premiere voiture : la pole position d'un circuit.
    grid: Option<(DVec3, DVec3)>,
    shots: u32,
    flight_time: f32,
    frames: u32,
    frame_times: Vec<f32>,
}

const AUTOPILOT_SECONDS: f32 = 30.0;
/// Distance d'ou la planete entiere tient dans l'image, depuis son centre.
fn space_distance() -> f64 {
    planet_radius() * 2.6
}
/// Au-dela, une nouvelle planete ne deplace pas la camera.
const KEEP_ALTITUDE: f64 = 50_000.0;
/// Hauteur du soleil au-dessus de l'horizon a l'arrivee, en degres : l'heure doree.
const EVENING_SUN: f32 = 5.0;
/// Le regard s'ecarte un peu du soleil : la composition respire mieux qu'avec l'astre au centre.
const FRAMING: f32 = 0.2;

impl Explorer {
    /// Le monde a eu quelques images pour se charger.
    fn frames_alive(&mut self) -> bool {
        self.frames += 1;
        self.frames > 3
    }

    fn autopilot(&mut self, ctx: &mut Context, speed: f64, dt: f32) {
        // Les premieres images servent au chargement initial, pas a la mesure.
        self.frames += 1;
        if self.frames < 60 {
            return;
        }
        self.frame_times.push(dt);
        self.flight_time += dt;
        if dt > 0.012 && std::env::var("TERRA_PROFILE").is_ok() {
            println!("image lente : {:.1} ms a {:.2} s de vol", dt * 1000.0, self.flight_time);
        }
        let f = ctx.camera.forward_d();
        ctx.camera.move_by(f * speed * dt as f64, world_lock(ctx.streamer.camera_altitude));
        let up = ctx.camera.up();
        let floor = planet_radius() + ctx.world.height(up).max(0.0) as f64 + 30.0;
        if ctx.camera.position.length() < floor {
            ctx.camera.position = up * floor;
        }

        if self.flight_time >= AUTOPILOT_SECONDS {
            self.frame_times.sort_by(f32::total_cmp);
            let n = self.frame_times.len();
            let at = |q: f32| self.frame_times[((n - 1) as f32 * q) as usize] * 1000.0;
            println!(
                "vol : {:.1} km parcourus, altitude finale {:.1} km, {n} images, mediane {:.2} ms, 99e centile {:.2} ms, pire {:.2} ms",
                speed * self.flight_time as f64 / 1000.0,
                (ctx.camera.position.length() - planet_radius()) / 1000.0,
                at(0.5),
                at(0.99),
                at(1.0)
            );
            *ctx.exit = true;
        }
    }

    /// Le point de vue d'arrivee : la rive d'une grande etendue d'eau, face au soleil couchant.
    fn vista_on(world: &Generator, sky: &Sky) -> (DVec3, f32) {
        world.find_vista(|d| sky.evening_sun(d, EVENING_SUN)).unwrap_or_else(|| {
            let dir = world.find_spawn();
            (dir, world.yaw_towards_water(dir))
        })
    }

    /// Pour le moteur : sur une nouvelle planete, la camera se pose seulement si elle n'est pas dans
    /// l'espace (voir `PlanetChanged`). Le paysage de ce point se prepare alors en arriere-plan.
    fn landing(world: &Generator, sky: &Sky, altitude: f64) -> Option<(DVec3, f32)> {
        (altitude < KEEP_ALTITUDE).then(|| Self::vista_on(world, sky))
    }

    fn vista(ctx: &mut Context) -> (DVec3, f32) {
        match ctx.world.landing.get() {
            Some(spot) => *spot,
            None => Self::vista_on(ctx.world, ctx.sky),
        }
    }

    fn go_to_surface(&mut self, ctx: &mut Context) {
        let (dir, yaw) = Self::vista(ctx);
        let r = planet_radius() + ctx.world.height(dir).max(0.0) as f64 + terra::app::EYE_HEIGHT;
        *ctx.camera = Camera::new(dir * r, yaw - FRAMING, 0.03);
        let time = ctx.sky.evening_time(dir, EVENING_SUN);
        ctx.sky.set_local_time(dir, time);
        self.player.fly = false;
        self.player.fly_speed = 60.0;
        self.player.stop();
    }

    /// Survol d'un point choisi sur la carte, assez haut pour voir la region.
    fn fly_over(&mut self, ctx: &mut Context, dir: DVec3) {
        let ground = ctx.world.height(dir).max(0.0) as f64;
        *ctx.camera = Camera::new(dir * (planet_radius() + ground + 4000.0), 0.0, -0.3);
        self.player.fly = true;
        self.player.fly_speed = 600.0;
        self.player.stop();
    }

    /// Arrivee sur une ville : a 2 km au sud de son centre, a 350 m, face aux tours.
    fn arrive_at_city(&mut self, ctx: &mut Context, center: DVec3) {
        let east = DVec3::Y.cross(center).normalize_or(DVec3::X);
        let north = center.cross(east);
        let dir = (center - north * (2000.0 / planet_radius())).normalize();
        let ground = ctx.world.height(dir).max(0.0) as f64;
        *ctx.camera = Camera::new(dir * (planet_radius() + ground + 350.0), 0.0, -0.12);
        self.player.fly = true;
        self.player.fly_speed = 60.0;
        self.player.stop();
    }

    /// Monter dans la voiture (la plus proche, ou une nouvelle garee dans la rue la plus proche), ou en descendre.
    fn toggle_car(&mut self, ctx: &mut Context) {
        if self.driving {
            if let Some(car) = &self.car {
                if car.speed.abs() > 3.0 {
                    return;
                }
                let door = car.door().normalize();
                let ground = ctx.world.height(door).max(0.0) as f64;
                let heading = car.forward;
                *ctx.camera = Camera::new(door * (planet_radius() + ground + 1.75), 0.0, 0.0);
                ctx.camera.heading = (heading - door * heading.dot(door)).normalize_or(ctx.camera.heading);
            }
            self.driving = false;
            self.player.fly = false;
            self.player.stop();
            return;
        }
        let near = self.car.as_ref().is_some_and(|c| (c.position - ctx.camera.position).length() < 12.0);
        if !near {
            let (at, forward) =
                self.grid.take().unwrap_or_else(|| parking_spot(ctx.world, ctx.camera.up(), ctx.camera.heading));
            let mut car = Car::new(ctx.world, at, forward);
            car.palette = (ctx.world.params.seed % 8) as u8;
            if self.options.cockpit {
                car.view = View::Cockpit;
            }
            self.car = Some(car);
        }
        self.driving = true;
        self.player.fly = false;
    }

    fn go_to_space(&mut self, ctx: &mut Context) {
        let up = ctx.camera.up();
        let heading = ctx.camera.heading;
        ctx.camera.position = up * space_distance();
        ctx.camera.heading = heading;
        ctx.camera.pitch = -1.55;
        self.player.fly = true;
        self.player.fly_speed = 1_000_000.0;
        self.player.stop();
    }
}

impl Game for Explorer {
    fn start(&mut self, ctx: &mut Context) {
        let (mut dir, vista_yaw) = match self.options.position {
            Some((lat, lon)) => {
                let dir = lat_lon_to_dir(lat, lon);
                (dir, ctx.world.yaw_towards_water(dir))
            }
            None => {
                let (dir, yaw) = Self::vista(ctx);
                (dir, yaw - FRAMING)
            }
        };
        let mut altitude = self.options.altitude.unwrap_or(1.75);
        // Sur la grille du circuit le plus proche : pole position, a 16 m de la ligne.
        if self.options.circuit {
            if let Some(c) = ctx.world.cities.circuits.nearest(dir) {
                let track = &c.paths[0];
                let n = track.points.len();
                let at = track.points[n - 2];
                let fwd = track.tangent(n - 2);
                let left = at.cross(fwd).normalize();
                let pole = (at + left * (3.2 / planet_radius())).normalize();
                self.grid = Some((pole, fwd));
                let east = DVec3::Y.cross(pole).normalize_or(DVec3::X);
                let north = pole.cross(east);
                self.options.yaw = Some(fwd.dot(east).atan2(fwd.dot(north)) as f32);
                dir = pole;
                altitude = altitude.min(1.75);
                println!("circuit : {} ({}), {:.0} m", c.name(), c.location(), track.length());
            }
        }
        // Au pied de la route de montagne la plus proche, tourne vers la montee.
        if self.options.mountain {
            if let Some((road, _)) = ctx.world.cities.mountains.nearest(dir) {
                // TERRA_MOUNTAIN_AT=0.6 : plus haut sur la route (verification).
                let (at, fwd) = match std::env::var("TERRA_MOUNTAIN_AT").ok().and_then(|v| v.parse::<f64>().ok()) {
                    Some(f) => {
                        let path = &road.paths[0];
                        let k = ((path.points.len() - 2) as f64 * f.clamp(0.0, 1.0)) as usize;
                        (path.points[k], path.tangent(k))
                    }
                    None => road.start(),
                };
                let left = at.cross(fwd).normalize();
                let lane = (at - left * (1.7 / planet_radius())).normalize();
                self.grid = Some((lane, fwd));
                let east = DVec3::Y.cross(lane).normalize_or(DVec3::X);
                let north = lane.cross(east);
                self.options.yaw = Some(fwd.dot(east).atan2(fwd.dot(north)) as f32);
                dir = lane;
                altitude = altitude.min(1.75);
                println!(
                    "route de montagne : {}, {:.1} km, {:.0} m de denivele, {} epingles",
                    road.name,
                    road.paths[0].length() / 1000.0,
                    road.rise(ctx.world),
                    road.hairpins.len()
                );
                if std::env::var("TERRA_DEBUG_MOUNTAIN").is_ok() {
                    for h in &road.hairpins {
                        let (lat, lon) = terra::world::generator::dir_to_lat_lon(*h);
                        println!("  epingle {lat:.5},{lon:.5}");
                    }
                }
            }
        }
        // A pied en ville : sur la chaussee la plus proche, pas dans un mur.
        if altitude <= 10.0 && self.grid.is_none() {
            if let Some(street) = ctx.world.cities.street_near(dir, ctx.world) {
                dir = street;
            }
        }
        let mut ground = ctx.world.height(dir).max(0.0) as f64;
        // Dans un tunnel : sur la chaussee, pas sur la montagne.
        if altitude <= 10.0 {
            if let Some(t) = ctx.world.cities.highways.tunnel(ctx.world, dir) {
                ground = t.road + 0.1;
            }
        }
        let yaw = self.options.yaw.unwrap_or(vista_yaw);
        let pitch = self.options.pitch.unwrap_or(0.03);
        *ctx.camera = Camera::new(dir * (planet_radius() + ground + altitude), yaw, pitch);

        // Le soleil reste fige a l'heure doree, sauf si on demande une journee qui passe.
        let time = self.options.time.unwrap_or_else(|| ctx.sky.evening_time(dir, EVENING_SUN));
        ctx.sky.set_local_time(dir, time);
        ctx.sky.day_length = self.options.day_length;
        // Hors du sol au depart : on commence en vol, pour ne pas tomber du ciel.
        self.player.fly = altitude > 10.0;
        if altitude > 100_000.0 {
            self.player.fly_speed = 1_000_000.0;
        }
    }

    fn update(&mut self, ctx: &mut Context, dt: f32) {
        for event in ctx.events.iter().copied() {
            match event {
                GameEvent::GoToSpace => self.go_to_space(ctx),
                GameEvent::GoToSurface => self.go_to_surface(ctx),
                GameEvent::GoTo(dir) => self.fly_over(ctx, dir),
                GameEvent::GoToCity(dir) => self.arrive_at_city(ctx, dir),
                GameEvent::PlanetChanged => {
                    self.car = None;
                    self.driving = false;
                    if ctx.camera.position.length() - planet_radius() < KEEP_ALTITUDE {
                        self.go_to_surface(ctx);
                    }
                }
            }
            if !matches!(event, GameEvent::PlanetChanged) {
                // Teleporte : on laisse la voiture ou elle est.
                self.driving = false;
            }
        }

        if let Some(speed) = self.options.autopilot {
            self.autopilot(ctx, speed, dt);
            return;
        }
        if let Some(seed) = self.options.regenerate_seed {
            self.frames += 1;
            if self.frames == 5 {
                let mut params: PlanetParams = ctx.world.params;
                params.seed = seed;
                *ctx.regenerate = Some(params);
            }
        }

        if self.options.drive && self.car.is_none() && self.frames_alive() {
            self.toggle_car(ctx);
        }
        if ctx.input.just_pressed(KeyCode::KeyE) && (self.driving || !self.player.fly) {
            self.toggle_car(ctx);
        }
        let world = ctx.world.clone();
        if let Some(car) = &mut self.car {
            car.weather_grip = ctx.sky.weather.grip() as f64;
            car.update(ctx.input, &world, ctx.streamer, dt, self.driving);
            if self.driving {
                ctx.sound.car = Some(car.sound(car.view == View::Cockpit));
            }
            if car.impact > 2.5 {
                ctx.sound.impact = (ctx.sound.impact.0.wrapping_add(1), car.impact as f32);
            }
        }
        ctx.hud.driving = None;
        if let (true, Some(car)) = (self.driving, &mut self.car) {
            car.place_camera(ctx.camera, ctx.input, &world, dt);
            ctx.hud.speed = car.speed.abs();
            ctx.hud.flying = false;
            ctx.hud.driving = Some((car.speed, car.gear, car.rpm, car.headlights));
        } else {
            self.player.update(ctx.input, ctx.camera, ctx.streamer, dt);
            ctx.hud.speed = self.player.speed(ctx.input);
            ctx.hud.flying = self.player.fly;
        }
        *ctx.vehicle = self.car.as_ref().map(|c| c.frame(self.driving && c.view == View::Cockpit));

        if ctx.input.held(KeyCode::KeyT) {
            ctx.sky.shift_time(dt * 0.04);
        }
        if ctx.input.held(KeyCode::KeyG) {
            ctx.sky.shift_time(-dt * 0.04);
        }
        if ctx.input.just_pressed(KeyCode::F12) {
            self.shots += 1;
            *ctx.screenshot = Some(PathBuf::from(format!("terra_{:03}.png", self.shots)));
        }
    }
}

/// Deux nombres finis separes par une virgule : `48.85,2.35`.
fn pair(s: &str) -> Option<(f64, f64)> {
    let (a, b) = s.split_once(',')?;
    let (a, b): (f64, f64) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
    (a.is_finite() && b.is_finite()).then_some((a, b))
}

/// Un nombre fini compris entre `min` et `max`. Tout le reste (texte, `nan`, `inf`, hors bornes)
/// est refuse : une valeur absurde ne doit pas etre ignoree en silence ni atteindre la simulation.
fn number<T>(s: &str, min: T, max: T) -> Option<T>
where
    T: std::str::FromStr + PartialOrd,
{
    // Les comparaisons sont fausses pour NaN, qui est donc refuse aussi.
    s.trim().parse::<T>().ok().filter(|v| *v >= min && *v <= max)
}

fn main() {
    let mut config = EngineConfig { title: "Terra".to_string(), landing: Some(Explorer::landing), ..Default::default() };
    let mut options = Options::default();

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage(&format!("valeur manquante pour {flag}")));
        // La valeur de l'option, entre ses bornes ; sinon l'aide, avec le rappel de ce qui est attendu.
        let mut real = |min: f64, max: f64, help: &str| number(&value(), min, max).unwrap_or_else(|| usage(help));
        match flag.as_str() {
            "--seed" => config.planet.seed = number(&value(), 0, u64::MAX).unwrap_or_else(|| usage("--seed N")),
            "--planet-size" => config.planet.size = real(0.25, 2.0, "--planet-size 0.25..2") as f32,
            "--cities" => {
                config.planet.cities =
                    number(&value(), 0, PlanetParams::MAX_CITIES).unwrap_or_else(|| usage("--cities 0..600"))
            }
            "--no-rivers" => config.planet.rivers = false,
            "--ocean" => config.planet.ocean = real(0.0, 1.0, "--ocean 0..1") as f32,
            "--relief" => config.planet.relief = real(0.2, 2.0, "--relief 0.2..2") as f32,
            "--temperature" => config.planet.temperature = real(-0.5, 0.5, "--temperature -0.5..0.5") as f32,
            "--pos" => {
                let (lat, lon) = pair(&value()).unwrap_or_else(|| usage("--pos lat,lon"));
                if !(-90.0..=90.0).contains(&lat) || !(-360.0..=360.0).contains(&lon) {
                    usage("--pos lat,lon : latitude de -90 a 90, longitude de -360 a 360");
                }
                options.position = Some((lat, lon));
            }
            "--altitude" => options.altitude = Some(real(0.0, 1.0e9, "--altitude m, de 0 a 1e9")),
            "--yaw" => options.yaw = Some((real(-360.0, 360.0, "--yaw deg, de -360 a 360") as f32).to_radians()),
            "--pitch" => options.pitch = Some((real(-90.0, 90.0, "--pitch deg, de -90 a 90") as f32).to_radians()),
            "--time" => options.time = Some(real(0.0, 1.0, "--time 0..1") as f32),
            "--day-length" => options.day_length = real(0.0, 1.0e6, "--day-length s, 0 ou plus") as f32,
            "--size" => {
                let (min, max) = (f64::from(MIN_WINDOW_SIZE), f64::from(MAX_WINDOW_SIZE));
                let (w, h) = pair(&value())
                    .filter(|(w, h)| (min..=max).contains(w) && (min..=max).contains(h))
                    .unwrap_or_else(|| usage("--size largeur,hauteur : de 64 a 8192 pixels"));
                (config.width, config.height) = (w as u32, h as u32);
            }
            "--render-distance" => {
                config.stream.set_render_distance(real(5.0, 1000.0, "--render-distance km, de 5 a 1000") * 1000.0);
            }
            "--no-vsync" => config.vsync = false,
            "--autopilot" => options.autopilot = Some(real(0.0, 3.0e10, "--autopilot m/s, 0 ou plus")),
            "--regenerate" => {
                options.regenerate_seed =
                    Some(number(&value(), 0, u64::MAX).unwrap_or_else(|| usage("--regenerate N")));
            }
            "--menu" => config.start_with_menu = true,
            "--mute" => config.sound = false,
            "--drive" => options.drive = true,
            "--cockpit" => {
                options.drive = true;
                options.cockpit = true;
            }
            "--circuit" => {
                options.drive = true;
                options.circuit = true;
            }
            "--montagne" => {
                options.drive = true;
                options.mountain = true;
            }
            "--screenshot" => config.auto_screenshot = Some(PathBuf::from(value())),
            "--help" | "-h" => usage(""),
            other => usage(&format!("option inconnue : {other}")),
        }
    }

    let game = Explorer {
        options,
        player: PlayerController::default(),
        car: None,
        driving: false,
        grid: None,
        shots: 0,
        flight_time: 0.0,
        frames: 0,
        frame_times: Vec::new(),
    };
    if let Err(e) = terra::run(config, game) {
        eprintln!("erreur : {e}");
        std::process::exit(1);
    }
}

fn usage(message: &str) -> ! {
    if !message.is_empty() {
        eprintln!("{message}\n");
    }
    eprintln!(
        "terra-demo [options]\n\
         \n  --seed N            graine de la planete\
         \n  --ocean 0..1        part d'ocean (0.68 par defaut)\
         \n  --planet-size x     rayon, relatif a la Terre (1 par defaut)\
         \n  --cities n          nombre de villes (140 par defaut)\
         \n  --no-rivers         planete sans rivieres\
         \n  --relief 0.2..2     hauteur des montagnes\
         \n  --temperature x     climat, de -0.5 a 0.5\
         \n  --pos lat,lon       position de depart, en degres\
         \n  --altitude m        hauteur au-dessus du sol (> 10 : demarre en vol)\
         \n  --yaw deg           orientation, a droite du nord\
         \n  --pitch deg         inclinaison du regard (-89 : vers la planete)\
         \n  --time 0..1         heure locale (0.5 = midi, heure doree par defaut)\
         \n  --day-length s      duree d'une journee (0 par defaut : soleil fige)\
         \n  --size l,h          taille de la fenetre\
         \n  --render-distance km  distance de rendu, de 5 a 1000 (80 par defaut)\
         \n  --no-vsync          sans synchronisation verticale\
         \n  --menu              ouvre le menu au demarrage\
         \n  --mute              sans son (M le coupe et le remet en jeu)\
         \n  --drive             demarre au volant d'une voiture (E : monter ou descendre)\
         \n  --cockpit           idem, vue du conducteur\
         \n  --circuit           au volant, sur la grille du circuit le plus proche\
         \n  --montagne          au volant, au pied de la route de montagne la plus proche\
         \n  --regenerate N      remplace la planete au demarrage (verification)\
         \n  --autopilot m/s     vol de mesure de 30 s, affiche les temps d'image\
         \n  --screenshot f.png  attend le chargement, enregistre l'image et quitte"
    );
    std::process::exit(if message.is_empty() { 0 } else { 2 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_accepts_only_finite_values_in_range() {
        assert_eq!(number(" 0.5 ", 0.0, 1.0), Some(0.5));
        assert_eq!(number("1", 0.0, 1.0), Some(1.0));
        assert_eq!(number("1.01", 0.0, 1.0), None);
        assert_eq!(number("nan", 0.0, 1.0), None);
        assert_eq!(number("inf", 0.0, f64::MAX), None);
        assert_eq!(number("abc", 0.0, 1.0), None);
        assert_eq!(number("600", 0u32, 600), Some(600));
        assert_eq!(number("601", 0u32, 600), None);
        assert_eq!(number("-1", 0u32, 600), None);
    }

    #[test]
    fn pair_needs_two_finite_numbers() {
        assert_eq!(pair("48.85, 2.35"), Some((48.85, 2.35)));
        assert_eq!(pair("48.85"), None);
        assert_eq!(pair("nan,2"), None);
        assert_eq!(pair("1,inf"), None);
        assert_eq!(pair("a,b"), None);
    }
}
