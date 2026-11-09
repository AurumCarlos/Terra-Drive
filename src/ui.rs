//! Menu (Echap) et affichage tete haute, avec egui. La description de l'interface est
//! separee de la fenetre pour pouvoir la tester sans GPU.

use crate::world::generator::{Generator, PlanetParams};
use crate::world::noise::Rng;
use crate::world::streaming::{DEFAULT_RENDER_DISTANCE, MAX_RENDER_DISTANCE, MIN_RENDER_DISTANCE};

/// Ce que l'utilisateur a demande dans le menu.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MenuAction {
    Resume,
    NewPlanet(PlanetParams),
    /// Nouvelle distance de rendu, en metres.
    SetRenderDistance(f64),
    GoToSpace,
    GoToSurface,
    /// Survoler ce point de la carte, en latitude et longitude.
    GoTo(f64, f64),
    /// Arriver au-dessus de la ville la plus proche.
    GoToCity,
    Quit,
}

const MAP_SIZE: [usize; 2] = [760, 380];

/// Les deux cartes d'une planete : sans et avec les frontieres de plaques.
pub fn planet_maps(gen: &Generator) -> [egui::ColorImage; 2] {
    [false, true].map(|plates| egui::ColorImage::from_rgb(MAP_SIZE, &gen.map_rgb(MAP_SIZE[0], MAP_SIZE[1], plates)))
}

/// Donnees affichees en permanence.
#[derive(Clone, Debug, Default)]
pub struct HudData {
    pub altitude: f64,
    pub speed: f64,
    pub flying: bool,
    pub latitude: f64,
    pub longitude: f64,
    pub local_time: f32,
    pub seed: u64,
    pub render_distance: f64,
    pub fps: f32,
    pub chunks: usize,
    pub triangles: u64,
    pub objects: usize,
    pub loading: bool,
    /// Une nouvelle planete est en cours de simulation.
    pub generating: bool,
    /// Ville la plus proche : nom, plan, distance au centre (m), direction (degres depuis le nord).
    pub city: Option<(String, &'static str, f64, f64)>,
    /// Au volant : vitesse (m/s), rapport (-1 marche arriere), regime (tr/min), phares.
    pub driving: Option<(f64, i32, f64, bool)>,
    /// Circuit proche : nom, lieu du vrai circuit, distance (m).
    pub circuit: Option<(&'static str, &'static str, f64)>,
}

pub struct MenuState {
    pub open: bool,
    pub params: PlanetParams,
    pub render_distance_km: f64,
    seed_text: String,
    /// Graine de tirage des planetes aleatoires.
    rng: Rng,
    /// Emplacement du bouton "Nouvelle planete" (pour les tests).
    pub new_planet_rect: Option<egui::Rect>,
    /// Images de carte en attente d'envoi au GPU.
    map_images: Option<[egui::ColorImage; 2]>,
    map_textures: Option<[egui::TextureHandle; 2]>,
    show_plates: bool,
}

impl MenuState {
    /// Menu ferme, regle sur les parametres de la planete affichee.
    pub fn new(params: PlanetParams) -> Self {
        let clock = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(params.seed);
        Self {
            open: false,
            params,
            render_distance_km: DEFAULT_RENDER_DISTANCE / 1000.0,
            seed_text: params.seed.to_string(),
            rng: Rng(clock),
            new_planet_rect: None,
            map_images: None,
            map_textures: None,
            show_plates: false,
        }
    }

    /// Note les parametres de la planete en cours (generee autrement que par le menu).
    pub fn sync(&mut self, params: PlanetParams) {
        self.params = params;
        self.seed_text = params.seed.to_string();
    }

    /// Remplace les cartes de la planete : sans et avec les plaques tectoniques.
    pub fn set_map(&mut self, images: [egui::ColorImage; 2]) {
        self.map_images = Some(images);
    }
}

/// Distance lisible : en metres, puis en kilometres (au dixieme jusqu'a 1000 km).
pub fn format_distance(m: f64) -> String {
    let a = m.abs();
    if a < 1000.0 {
        format!("{m:.0} m")
    } else if a < 1.0e6 {
        format!("{:.1} km", m / 1000.0)
    } else {
        format!("{:.0} km", m / 1000.0)
    }
}

/// Vitesse lisible, a partir de metres par seconde : km/h, puis km/s au-dela de 1000 m/s.
pub fn format_speed(v: f64) -> String {
    if v < 1000.0 {
        format!("{:.0} km/h", v * 3.6)
    } else {
        format!("{:.1} km/s", v / 1000.0)
    }
}

fn hud(ctx: &egui::Context, d: &HudData) {
    egui::Area::new(egui::Id::new("hud"))
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::NONE.fill(egui::Color32::from_black_alpha(130)).corner_radius(4.0).inner_margin(8.0).show(
                ui,
                |ui| {
                    ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                    let color = egui::Color32::from_gray(225);
                    let line = |ui: &mut egui::Ui, text: String| {
                        ui.label(egui::RichText::new(text).color(color));
                    };
                    let mode = if d.driving.is_some() {
                        "voiture"
                    } else if d.flying {
                        "vol"
                    } else {
                        "marche"
                    };
                    line(ui, format!("Altitude  {}", format_distance(d.altitude)));
                    line(ui, format!("Vitesse   {} ({mode})", format_speed(d.speed)));
                    let (ns, ew) =
                        (if d.latitude >= 0.0 { 'N' } else { 'S' }, if d.longitude >= 0.0 { 'E' } else { 'O' });
                    line(ui, format!("Position  {:.2} {ns}  {:.2} {ew}", d.latitude.abs(), d.longitude.abs()));
                    let minutes = (d.local_time * 1440.0) as u32;
                    line(ui, format!("Heure     {:02}h{:02}", minutes / 60, minutes % 60));
                    line(ui, format!("Planete   {}", d.seed));
                    if let Some((name, plan, dist, bearing)) = &d.city {
                        if *dist < 1500.0 {
                            line(ui, format!("Ville     {name} ({plan})"));
                        } else {
                            const POINTS: [&str; 8] = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"];
                            let point = POINTS[(((bearing + 22.5).rem_euclid(360.0)) / 45.0) as usize % 8];
                            line(ui, format!("Ville     {name} a {} {point}", format_distance(*dist)));
                        }
                    }
                    if let Some((name, place, dist)) = &d.circuit {
                        if *dist < 300.0 {
                            line(ui, format!("Circuit   {name} ({place})"));
                        } else {
                            line(ui, format!("Circuit   {name} a {}", format_distance(*dist)));
                        }
                    }
                    line(ui, format!("Rendu     {}", format_distance(d.render_distance)));
                    let status = if d.loading { "  chargement" } else { "" };
                    line(
                        ui,
                        format!(
                            "{:.0} fps  {} chunks  {:.1} M tri  {} objets{status}",
                            d.fps,
                            d.chunks,
                            d.triangles as f32 / 1.0e6,
                            d.objects
                        ),
                    );
                    if d.generating {
                        ui.label(
                            egui::RichText::new("Simulation d'une nouvelle planete...")
                                .color(egui::Color32::from_rgb(255, 210, 120)),
                        );
                    }
                    if !ui.ctx().is_using_pointer() {
                        ui.add_space(2.0);
                        line(ui, "Echap : menu".to_string());
                        if d.driving.is_some() {
                            line(
                                ui,
                                "E : descendre   C : vue   L : phares   Espace : frein a main   R : remettre sur roues"
                                    .to_string(),
                            );
                        } else if !d.flying {
                            line(ui, "E : prendre une voiture".to_string());
                        }
                    }
                },
            );
        });
    if let Some((speed, gear, rpm, head)) = d.driving {
        dashboard(ctx, speed, gear, rpm, head);
    }
}

/// Compteur de vitesse et compte-tours, en bas a droite.
fn dashboard(ctx: &egui::Context, speed: f64, gear: i32, rpm: f64, head: bool) {
    egui::Area::new(egui::Id::new("compteur"))
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-24.0, -24.0))
        .interactable(false)
        .show(ctx, |ui| {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(230.0, 230.0), egui::Sense::hover());
            let p = ui.painter();
            let c = rect.center();
            let r = 105.0;
            p.circle_filled(c, r + 6.0, egui::Color32::from_black_alpha(150));
            // Graduations de 0 a 300 km/h sur 270 degres.
            let angle = |k: f32| std::f32::consts::PI * (0.75 + 1.5 * k);
            let at = |a: f32, d: f32| c + egui::vec2(a.cos(), a.sin()) * d;
            for i in 0..=30 {
                let k = i as f32 / 30.0;
                let a = angle(k);
                let long = i % 3 == 0;
                p.line_segment(
                    [at(a, r - if long { 14.0 } else { 7.0 }), at(a, r - 2.0)],
                    egui::Stroke::new(if long { 2.0_f32 } else { 1.0 }, egui::Color32::from_gray(220)),
                );
                if long {
                    p.text(
                        at(a, r - 28.0),
                        egui::Align2::CENTER_CENTER,
                        format!("{}", i * 10),
                        egui::FontId::proportional(11.0),
                        egui::Color32::from_gray(200),
                    );
                }
            }
            // Zone rouge du compte-tours, en arc interieur.
            let rev = (rpm / 7200.0).clamp(0.0, 1.0) as f32;
            for i in 0..60 {
                let k = i as f32 / 60.0;
                if k > rev {
                    break;
                }
                let color =
                    if k > 0.85 { egui::Color32::from_rgb(230, 50, 40) } else { egui::Color32::from_rgb(90, 170, 255) };
                p.line_segment(
                    [at(angle(k), r - 44.0), at(angle(k + 1.0 / 60.0), r - 44.0)],
                    egui::Stroke::new(4.0_f32, color),
                );
            }
            let kmh = (speed.abs() * 3.6) as f32;
            let a = angle((kmh / 300.0).clamp(0.0, 1.0));
            p.line_segment(
                [at(a + std::f32::consts::PI, 12.0), at(a, r - 10.0)],
                egui::Stroke::new(3.0_f32, egui::Color32::from_rgb(255, 80, 40)),
            );
            p.circle_filled(c, 7.0, egui::Color32::from_gray(40));
            p.text(
                c + egui::vec2(0.0, 36.0),
                egui::Align2::CENTER_CENTER,
                format!("{kmh:.0}"),
                egui::FontId::monospace(28.0),
                egui::Color32::WHITE,
            );
            p.text(
                c + egui::vec2(0.0, 60.0),
                egui::Align2::CENTER_CENTER,
                "km/h",
                egui::FontId::proportional(11.0),
                egui::Color32::from_gray(180),
            );
            let g = if gear < 0 { "R".to_string() } else { gear.to_string() };
            p.text(
                c + egui::vec2(0.0, 80.0),
                egui::Align2::CENTER_CENTER,
                g,
                egui::FontId::monospace(18.0),
                egui::Color32::from_rgb(120, 220, 120),
            );
            if head {
                p.text(
                    c + egui::vec2(-58.0, 70.0),
                    egui::Align2::CENTER_CENTER,
                    "phares",
                    egui::FontId::proportional(10.0),
                    egui::Color32::from_rgb(110, 200, 255),
                );
            }
        });
}

/// La carte de la planete : un clic y envoie le joueur.
fn map_panel(ctx: &egui::Context, state: &mut MenuState, actions: &mut Vec<MenuAction>) {
    if let Some([plain, plates]) = state.map_images.take() {
        state.map_textures = Some([
            ctx.load_texture("carte", plain, egui::TextureOptions::LINEAR),
            ctx.load_texture("carte-plaques", plates, egui::TextureOptions::LINEAR),
        ]);
    }
    let Some(textures) = &state.map_textures else { return };
    let texture = &textures[state.show_plates as usize];
    egui::Window::new("Carte")
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-12.0, 12.0))
        .resizable(false)
        .collapsible(true)
        .default_open(true)
        .show(ctx, |ui| {
            let size = egui::vec2(MAP_SIZE[0] as f32, MAP_SIZE[1] as f32) * 0.72;
            let image = egui::Image::new((texture.id(), size)).sense(egui::Sense::click());
            let response = ui.add(image);
            if response.clicked() {
                if let Some(at) = response.interact_pointer_pos() {
                    let (u, v) = ((at - response.rect.min) / response.rect.size()).into();
                    actions.push(MenuAction::GoTo((0.5 - v as f64) * 180.0, (u as f64 - 0.5) * 360.0));
                }
            }
            ui.horizontal(|ui| {
                ui.checkbox(&mut state.show_plates, "Plaques tectoniques");
                ui.label("Cliquer pour y aller");
            });
            if state.show_plates {
                ui.label("Rouge : collision.  Orange : subduction.  Jaune : dorsale.");
            }
        });
}

fn menu(ctx: &egui::Context, state: &mut MenuState, actions: &mut Vec<MenuAction>) {
    egui::Window::new("Terra")
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .resizable(false)
        .collapsible(false)
        .default_width(340.0)
        .show(ctx, |ui| {
            ui.spacing_mut().button_padding = egui::vec2(10.0, 6.0);
            ui.vertical_centered_justified(|ui| {
                if ui.button("Reprendre").clicked() {
                    actions.push(MenuAction::Resume);
                }
                if ui.button("Aller dans l'espace").clicked() {
                    actions.push(MenuAction::GoToSpace);
                }
                if ui.button("Retourner a la surface").clicked() {
                    actions.push(MenuAction::GoToSurface);
                }
                if ui.button("Aller a la ville la plus proche").clicked() {
                    actions.push(MenuAction::GoToCity);
                }
            });

            ui.separator();
            ui.heading("Affichage");
            ui.horizontal(|ui| {
                ui.label("Distance de rendu");
                let range = MIN_RENDER_DISTANCE / 1000.0..=MAX_RENDER_DISTANCE / 1000.0;
                let slider = egui::Slider::new(&mut state.render_distance_km, range)
                    .logarithmic(true)
                    .suffix(" km")
                    .fixed_decimals(0);
                if ui.add(slider).changed() {
                    actions.push(MenuAction::SetRenderDistance(state.render_distance_km * 1000.0));
                }
            });

            ui.separator();
            ui.heading("Planete");
            egui::Grid::new("planet").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                ui.label("Graine");
                ui.add(egui::TextEdit::singleline(&mut state.seed_text).desired_width(170.0));
                ui.end_row();
                ui.label("Oceans");
                ui.add(egui::Slider::new(&mut state.params.ocean, 0.1..=0.95).show_value(true));
                ui.end_row();
                ui.label("Relief");
                ui.add(egui::Slider::new(&mut state.params.relief, 0.2..=2.0));
                ui.end_row();
                ui.label("Temperature");
                ui.add(egui::Slider::new(&mut state.params.temperature, -0.5..=0.5));
                ui.end_row();
                ui.label("Taille");
                ui.add(
                    egui::Slider::new(&mut state.params.size, 0.25..=2.0)
                        .custom_formatter(|v, _| format!("{:.0} km", v * 6371.0)),
                );
                ui.end_row();
                ui.label("Villes");
                ui.add(egui::Slider::new(&mut state.params.cities, 0..=600));
                ui.end_row();
                ui.label("Rivieres");
                ui.checkbox(&mut state.params.rivers, "");
                ui.end_row();
            });

            ui.add_space(4.0);
            ui.vertical_centered_justified(|ui| {
                let random = ui.button("Nouvelle planete aleatoire");
                state.new_planet_rect = Some(random.rect);
                if random.clicked() {
                    state.params.seed = state.rng.next_u64() >> 16;
                    state.seed_text = state.params.seed.to_string();
                    actions.push(MenuAction::NewPlanet(state.params));
                }
                if ui.button("Generer avec cette graine").clicked() {
                    // Une graine en texte libre donne aussi une planete, toujours la meme.
                    let text = state.seed_text.trim();
                    state.params.seed = text.parse().unwrap_or_else(|_| {
                        text.bytes()
                            .fold(0xCBF2_9CE4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3))
                            >> 16
                    });
                    actions.push(MenuAction::NewPlanet(state.params));
                }
            });

            ui.separator();
            egui::CollapsingHeader::new("Commandes").show(ui, |ui| {
                ui.label(
                    "Z Q S D : se deplacer\n\
                     Souris : regarder\n\
                     F : marche / vol\n\
                     Molette, Page haut / bas : vitesse de vol\n\
                     Maj : vitesse x8 en vol\n\
                     Espace / Ctrl : monter / descendre en vol\n\
                     T / G : avancer / reculer l'heure\n\
                     F12 : capture d'ecran",
                );
            });

            ui.separator();
            ui.vertical_centered_justified(|ui| {
                if ui.button("Quitter").clicked() {
                    actions.push(MenuAction::Quit);
                }
            });
        });
}

/// Decrit l'interface d'une image et retourne les actions demandees.
pub fn draw(ctx: &egui::Context, state: &mut MenuState, data: &HudData) -> Vec<MenuAction> {
    let mut actions = Vec::new();
    hud(ctx, data);
    if state.open {
        menu(ctx, state, &mut actions);
        map_panel(ctx, state, &mut actions);
    }
    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(ctx: &egui::Context, state: &mut MenuState, events: Vec<egui::Event>) -> Vec<MenuAction> {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 720.0))),
            events,
            ..Default::default()
        };
        let mut actions = Vec::new();
        let _ = ctx.run(input, |c| actions = draw(c, state, &HudData::default()));
        actions
    }

    fn click(at: egui::Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        }
    }

    #[test]
    fn clicking_new_planet_gives_a_different_seed_each_time() {
        let ctx = egui::Context::default();
        let mut state = MenuState::new(PlanetParams::default());
        state.open = true;
        // egui place les fenetres ancrees apres quelques images.
        for _ in 0..4 {
            frame(&ctx, &mut state, vec![]);
        }

        let mut seeds = Vec::new();
        for _ in 0..2 {
            let at = state.new_planet_rect.expect("bouton dessine").center();
            frame(&ctx, &mut state, vec![egui::Event::PointerMoved(at)]);
            frame(&ctx, &mut state, vec![click(at, true)]);
            let actions = frame(&ctx, &mut state, vec![click(at, false)]);
            match actions.as_slice() {
                [MenuAction::NewPlanet(p)] => seeds.push(p.seed),
                other => panic!("attendu NewPlanet, obtenu {other:?}"),
            }
            frame(&ctx, &mut state, vec![]);
        }
        assert_ne!(seeds[0], seeds[1]);
        assert_ne!(seeds[0], PlanetParams::default().seed);
    }

    #[test]
    fn closed_menu_draws_no_button() {
        let ctx = egui::Context::default();
        let mut state = MenuState::new(PlanetParams::default());
        frame(&ctx, &mut state, vec![]);
        assert!(state.new_planet_rect.is_none());
    }

    #[test]
    fn units_scale_with_the_values() {
        assert_eq!(format_distance(350.0), "350 m");
        assert_eq!(format_distance(12_300.0), "12.3 km");
        assert_eq!(format_distance(10_000_000.0), "10000 km");
        assert_eq!(format_speed(10.0), "36 km/h");
        assert_eq!(format_speed(2000.0), "2.0 km/s");
    }
}
