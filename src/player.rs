//! Controleur a la premiere personne : marche sur la sphere, nage, et vol libre jusqu'a l'espace.

use crate::camera::Camera;
use crate::input::Input;
use crate::world::generator::{planet_radius, smoothstep};
use crate::world::streaming::Streamer;
use glam::DVec3;
use winit::keyboard::KeyCode;

const EYE_HEIGHT: f64 = 1.75;
const GRAVITY: f64 = 9.81 * 2.2;
const BODY_RADIUS: f64 = 0.4;
pub const MIN_FLY_SPEED: f64 = 2.0;
/// 10 % de la vitesse de la lumiere : un tour de planete en moins d'une seconde.
pub const MAX_FLY_SPEED: f64 = 30_000_000.0;
const SPEED_STEP: f64 = 1.5;

pub struct PlayerController {
    pub fly: bool,
    pub walk_speed: f64,
    pub fly_speed: f64,
    pub sensitivity: f32,
    velocity: DVec3,
    grounded: bool,
}

impl Default for PlayerController {
    fn default() -> Self {
        Self {
            fly: false,
            walk_speed: 6.0,
            fly_speed: 60.0,
            sensitivity: 0.0022,
            velocity: DVec3::ZERO,
            grounded: false,
        }
    }
}

/// Part de "trajectoire droite" selon l'altitude : le vol suit la courbure pres du sol,
/// puis file tout droit dans l'espace.
pub fn world_lock(altitude: f64) -> f32 {
    smoothstep(30_000.0, 300_000.0, altitude as f32)
}

impl PlayerController {
    /// Vitesse reelle du joueur, pour l'affichage.
    pub fn speed(&self, input: &Input) -> f64 {
        if self.fly {
            self.fly_speed * if input.held(KeyCode::ShiftLeft) { 8.0 } else { 1.0 }
        } else {
            self.velocity.length()
        }
    }

    /// Annule la vitesse acquise : a appeler apres une teleportation.
    pub fn stop(&mut self) {
        self.velocity = DVec3::ZERO;
        self.grounded = false;
    }

    /// Une image de deplacement : regard a la souris, marche ou vol au clavier, gravite et collisions
    /// avec le sol et les objets charges par `streamer`.
    pub fn update(&mut self, input: &Input, cam: &mut Camera, streamer: &Streamer, dt: f32) {
        let world = &*streamer.generator;
        if input.captured {
            cam.turn(input.mouse_delta.0 as f32 * self.sensitivity, -input.mouse_delta.1 as f32 * self.sensitivity);
        }
        if input.just_pressed(KeyCode::KeyF) {
            self.fly = !self.fly;
            self.velocity = DVec3::ZERO;
        }
        if input.scroll != 0.0 {
            self.fly_speed *= SPEED_STEP.powf(input.scroll as f64);
        }
        let hold = input.held(KeyCode::PageUp) as i32 as f64 - input.held(KeyCode::PageDown) as i32 as f64;
        self.fly_speed *= (hold * 1.6 * dt as f64).exp();
        self.fly_speed = self.fly_speed.clamp(MIN_FLY_SPEED, MAX_FLY_SPEED);

        let dt = dt as f64;
        let axis = |pos: KeyCode, neg: KeyCode| input.held(pos) as i32 as f64 - input.held(neg) as i32 as f64;
        let (ahead, side) = (axis(KeyCode::KeyW, KeyCode::KeyS), axis(KeyCode::KeyD, KeyCode::KeyA));
        let sprint = input.held(KeyCode::ShiftLeft);

        if self.fly {
            let up = cam.up();
            let climb = axis(KeyCode::Space, KeyCode::ControlLeft);
            let dir = (cam.forward_d() * ahead + cam.right() * side + up * climb).normalize_or_zero();
            let speed = self.fly_speed * if sprint { 8.0 } else { 1.0 };
            cam.move_by(dir * speed * dt, world_lock(streamer.camera_altitude));
            self.keep_above_ground(cam, world);
            let ground = world.height(cam.up()) as f64;
            world.cities.push_out(&mut cam.position, BODY_RADIUS + 1.0, ground);
            return;
        }

        let up = cam.up();
        let radius = cam.position.length();
        // Une riviere ou un lac porte comme la mer, chacun a son niveau.
        let eye_water = radius - (planet_radius() + world.surface(up).water_level as f64 - 0.3);
        let swimming = eye_water < 0.0;
        let speed = self.walk_speed * if sprint { 2.6 } else { 1.0 } * if swimming { 0.55 } else { 1.0 };
        let wish = (cam.heading * ahead + cam.right() * side).normalize_or_zero() * speed;

        // Reponse rapide au sol, plus d'inertie en l'air.
        let response = if self.grounded || swimming { 12.0 } else { 2.0 };
        let blend = 1.0 - (-response * dt).exp();
        let mut vertical = self.velocity.dot(up);
        let horizontal = self.velocity - up * vertical;
        let horizontal = horizontal + (wish - horizontal) * blend;

        if swimming {
            // Remonte a la surface et y reste, la tete hors de l'eau.
            vertical += ((-eye_water).min(2.0) * 3.0 - vertical) * blend;
        } else {
            vertical -= GRAVITY * dt;
            if self.grounded && input.held(KeyCode::Space) {
                vertical = 8.0;
                self.grounded = false;
            }
        }
        self.velocity = horizontal + up * vertical;

        cam.move_by(self.velocity * dt, 0.0);
        streamer.push_out_of_objects(&mut cam.position, BODY_RADIUS);
        let ground = world.height(cam.up()) as f64;
        world.cities.push_out(&mut cam.position, BODY_RADIUS, ground);

        let up = cam.up();
        let radius = cam.position.length();
        let mut floor = world.height(up) as f64;
        // Sur une autoroute ou un pont : le tablier porte, sauf a passer dessous.
        if let Some(deck) = world.cities.highways.deck(world, up) {
            if radius - planet_radius() - EYE_HEIGHT > deck - 2.0 {
                floor = floor.max(deck);
            }
        }
        // Dans un tunnel, la chaussee porte : le relief est au-dessus de la voute.
        if let Some(t) = world.cities.highways.tunnel(world, up) {
            if radius - planet_radius() < t.road + crate::world::highway::VAULT {
                floor = t.road + 0.1;
                if let Some((depth, side)) = t.wall(BODY_RADIUS) {
                    cam.position += t.left * (side * depth);
                }
            }
        }
        let ground = planet_radius() + floor + EYE_HEIGHT;
        if radius <= ground {
            cam.position = up * ground;
            let v = self.velocity.dot(up);
            if v < 0.0 {
                self.velocity -= up * v;
            }
            self.grounded = true;
        } else if radius > ground + 0.35 {
            self.grounded = false;
        } else if self.grounded && self.velocity.dot(up) <= 0.0 {
            // Colle au sol dans les descentes au lieu de rebondir.
            cam.position = up * ground;
        }
    }

    fn keep_above_ground(&self, cam: &mut Camera, world: &crate::world::generator::Generator) {
        let up = cam.up();
        let surface = world.surface(up);
        let floor = planet_radius() + surface.height.max(surface.water_level) as f64 + 0.5;
        if cam.position.length() < floor {
            cam.position = up * floor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::generator::{Generator, PlanetParams};
    use crate::world::streaming::StreamConfig;
    use std::sync::Arc;

    const DT: f32 = 1.0 / 60.0;

    fn setup(seed: u64) -> (Streamer, Camera, PlayerController, Input) {
        let gen = Arc::new(Generator::new(PlanetParams { seed, ..Default::default() }));
        let dir = gen.find_spawn();
        let r = planet_radius() + gen.height(dir) as f64 + EYE_HEIGHT;
        let cam = Camera::new(dir * r, 0.0, 0.0);
        // Sans objet charge : ces tests portent sur le deplacement, pas sur les troncs.
        let streamer = Streamer::new(gen, StreamConfig { workers: 1, ..Default::default() });
        (streamer, cam, PlayerController::default(), Input::default())
    }

    fn steps(p: &mut PlayerController, i: &Input, c: &mut Camera, s: &mut Streamer, n: usize) {
        for _ in 0..n {
            s.camera_altitude = c.position.length() - planet_radius();
            p.update(i, c, s, DT);
        }
    }

    fn altitude(c: &Camera) -> f64 {
        c.position.length() - planet_radius()
    }

    #[test]
    fn walking_follows_the_ground_at_the_expected_speed() {
        let (mut s, mut cam, mut p, mut input) = setup(1);
        steps(&mut p, &input, &mut cam, &mut s, 5);
        let start = cam.position;
        let heading = cam.heading;
        input.set_key(KeyCode::KeyW, true, false);
        steps(&mut p, &input, &mut cam, &mut s, 120);

        assert!((cam.position - start).dot(heading) > 8.0);
        let ground = s.generator.height(cam.up()) as f64 + EYE_HEIGHT;
        assert!((altitude(&cam) - ground).abs() < 0.5);
    }

    #[test]
    fn jump_goes_up_then_lands() {
        let (mut s, mut cam, mut p, mut input) = setup(1);
        steps(&mut p, &input, &mut cam, &mut s, 5);
        let start = altitude(&cam);
        input.set_key(KeyCode::Space, true, false);
        steps(&mut p, &input, &mut cam, &mut s, 1);
        input.set_key(KeyCode::Space, false, false);
        steps(&mut p, &input, &mut cam, &mut s, 15);
        assert!(altitude(&cam) > start + 0.8, "le saut doit decoller");
        steps(&mut p, &input, &mut cam, &mut s, 120);
        assert!((altitude(&cam) - start).abs() < 0.01, "le joueur doit retomber");
    }

    #[test]
    fn scroll_raises_flight_speed_up_to_the_limit() {
        let (mut s, mut cam, mut p, mut input) = setup(1);
        let before = p.fly_speed;
        input.scroll = 3.0;
        steps(&mut p, &input, &mut cam, &mut s, 1);
        assert!(p.fly_speed > before * 3.0);
        input.scroll = 200.0;
        steps(&mut p, &input, &mut cam, &mut s, 1);
        assert_eq!(p.fly_speed, MAX_FLY_SPEED);
    }

    #[test]
    fn can_fly_from_the_ground_to_space_and_look_back() {
        let (mut s, mut cam, mut p, mut input) = setup(1);
        input.set_key(KeyCode::KeyF, true, false);
        steps(&mut p, &input, &mut cam, &mut s, 1);
        input.end_frame();
        assert!(p.fly);

        // Vers le haut, tres vite, pendant dix secondes.
        p.fly_speed = 2_000_000.0;
        input.set_key(KeyCode::Space, true, false);
        steps(&mut p, &input, &mut cam, &mut s, 600);
        assert!(altitude(&cam) > 15_000_000.0, "altitude {}", altitude(&cam));

        // Regarder vers le bas revient a regarder la planete.
        input.set_key(KeyCode::Space, false, false);
        cam.pitch = -1.55;
        assert!(cam.forward_d().dot(-cam.up()) > 0.99);
    }

    #[test]
    fn flight_never_enters_the_ground_even_at_full_speed() {
        let (mut s, mut cam, mut p, mut input) = setup(1);
        p.fly = true;
        p.fly_speed = MAX_FLY_SPEED;
        cam.position = cam.up() * (planet_radius() + 50_000.0);
        input.set_key(KeyCode::ControlLeft, true, false);
        steps(&mut p, &input, &mut cam, &mut s, 60);
        let floor = s.generator.height(cam.up()).max(0.0) as f64;
        assert!(altitude(&cam) >= floor);
        assert!(cam.position.length().is_finite());
    }

    #[test]
    fn water_keeps_the_head_at_the_surface() {
        let (mut s, mut cam, mut p, input) = setup(1);
        // Cherche un point de mer profonde sur la planete.
        let g = s.generator.clone();
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let dir = (0..4000)
            .map(|i| {
                let y = 1.0 - (i as f64 + 0.5) / 2000.0;
                let r = (1.0 - y * y).sqrt();
                let a = golden * i as f64;
                DVec3::new(a.cos() * r, y, a.sin() * r)
            })
            .find(|d| g.height(*d) < -30.0)
            .expect("de la mer sur la planete");
        cam = Camera::new(dir * (planet_radius() - 5.0), cam.pitch, 0.0);
        steps(&mut p, &input, &mut cam, &mut s, 600);
        let y = altitude(&cam);
        assert!((-1.0..1.0).contains(&y), "y = {y}");
    }
}
