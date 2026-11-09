//! Voiture : un modele detaille (voir `model`) et sa conduite sur la planete.
//!
//! La dynamique (voir `dynamics`) calcule la caisse en 3D sur ses quatre suspensions, les pneus, le moteur,
//! la boite et les freins. Ici, on lui donne le sol sous chaque roue (relief, autoroutes, ponts, viaducs et
//! circuits compris), on gere les chocs contre les batiments et les arbres, et les bosses de la carrosserie.

pub mod dynamics;
pub mod model;

use crate::camera::Camera;
use crate::input::Input;
use crate::world::generator::{planet_radius, Generator};
use crate::world::streaming::Streamer;
use glam::DVec3;
use model::DRIVER_EYE;
pub use model::{lights, CarModel, Part};
use winit::keyboard::KeyCode;

use dynamics::{Chassis, Contact, Controls, Plane};

/// Ce que le rendu doit savoir de la voiture a chaque image.
#[derive(Clone, Copy, Debug)]
pub struct VehicleFrame {
    /// Origine du modele : au sol, entre les essieux.
    pub position: DVec3,
    pub right: DVec3,
    pub up: DVec3,
    pub forward: DVec3,
    /// Braquage des roues avant (positif a droite).
    pub steer: f32,
    /// Rotation de chaque roue.
    pub spin: [f32; 4],
    /// Debattement vertical de chaque roue.
    pub travel: [f32; 4],
    pub lights: u8,
    pub speed_kmh: f32,
    /// Teinte de la carrosserie (0 a 15).
    pub palette: u8,
    /// Bosses : point du modele et profondeur (0 : pas de bosse).
    pub dents: [[f32; 4]; 8],
}

/// Pose le modele : sommets relatifs a `frame.position`, dans le repere de la planete.
pub fn pose(model: &CarModel, frame: &VehicleFrame, out: &mut Vec<crate::world::city_mesh::CityVertex>) {
    use glam::{Mat3, Vec3};
    out.clear();
    out.extend_from_slice(&model.vertices);
    let (r, u, f) = (frame.right.as_vec3(), frame.up.as_vec3(), frame.forward.as_vec3());
    let world = Mat3::from_cols(r, u, f);
    let steer = Mat3::from_rotation_y(frame.steer);
    for (part, range) in &model.parts {
        let (m, t) = match *part {
            Part::Body => (Mat3::IDENTITY, Vec3::ZERO),
            Part::Wheel(k) => {
                let spin = Mat3::from_rotation_x(frame.spin[k]);
                let m = if k < 2 { steer * spin } else { spin };
                (m, model::WHEELS[k] + Vec3::Y * frame.travel[k])
            }
            Part::Hub(k) => (if k < 2 { steer } else { Mat3::IDENTITY }, model::WHEELS[k] + Vec3::Y * frame.travel[k]),
            Part::Steering => {
                (Mat3::from_axis_angle(model::steering_axis(), frame.steer * 12.0), model::STEERING_CENTER)
            }
        };
        let (wm, wt) = (world * m, world * t);
        let body = *part == Part::Body;
        for v in &mut out[range.clone()] {
            let mut p = Vec3::from_array(v.pos);
            if body {
                // Bosses : la tole recule vers l'interieur autour du point de choc.
                for d in &frame.dents {
                    if d[3] <= 0.0 {
                        continue;
                    }
                    let c = Vec3::new(d[0], d[1], d[2]);
                    let k = 1.0 - (p - c).length() / 0.55;
                    if k > 0.0 {
                        let inward = (Vec3::new(0.0, 0.75, 0.0) - c).normalize_or_zero();
                        p += inward * d[3] * k * k;
                    }
                }
            }
            v.pos = (wm * p + wt).to_array();
            v.normal = (wm * Vec3::from_array(v.normal)).to_array();
            match v.mat[0] {
                model::mat::CAR => {
                    v.mat[1] = (v.mat[1] & 0xF0) | (frame.palette & 15);
                    v.mat[2] = frame.lights;
                }
                model::mat::CABIN => v.mat[3] = frame.speed_kmh.clamp(0.0, 255.0) as u8,
                _ => {}
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum View {
    Chase,
    Cockpit,
}

pub struct Car {
    /// Origine du modele : au sol entre les essieux quand la voiture est au repos.
    pub position: DVec3,
    /// Cap, tangent a la sphere.
    pub forward: DVec3,
    /// Vitesse le long du cap (negative en marche arriere).
    pub speed: f64,
    /// Caisse, suspensions, pneus, moteur, boite, freins.
    pub chassis: Chassis,
    pub steer: f64,
    spin: [f64; 4],
    lights: u8,
    pub headlights: bool,
    time: f64,
    pub gear: i32,
    pub rpm: f64,
    pub view: View,
    look: (f64, f64),
    look_idle: f64,
    chase: DVec3,
    /// La voiture vient de heurter quelque chose.
    pub bumped: bool,
    /// Vitesse du choc le plus fort de cette image (m/s), 0 sans choc.
    pub impact: f64,
    pub palette: u8,
    /// Bosses de la carrosserie : point (repere du modele) et profondeur.
    dents: Vec<[f32; 4]>,
    /// Temps passe immobile, garee : au-dela d'une seconde, la physique s'endort.
    rest: f64,
}

/// Le sol sous un point pour une roue a l'altitude `alt` : relief, tabliers d'autoroutes et de ponts, eau.
fn floor(world: &Generator, dir: DVec3, alt: f64) -> (f64, f64) {
    let s = world.surface(dir);
    let mut h = s.height as f64;
    if let Some(d) = world.cities.highways.deck(world, dir) {
        if alt > d - 2.5 {
            h = h.max(d);
        }
    }
    if let Some(d) = world.cities.deck(dir, world) {
        if alt > d - 2.5 {
            h = h.max(d);
        }
    }
    (h + 0.03, s.water_level as f64)
}

/// Le plan du sol sous un point (centre de roue ou de caisse) : trois sondes, revetement, eau.
fn plane_under(world: &Generator, p: DVec3, heading: DVec3) -> (Plane, f64) {
    let r = planet_radius();
    let dir = p.normalize();
    let alt = p.length() - r;
    let e1 = (heading - dir * heading.dot(dir)).normalize_or(dir.any_orthonormal_vector());
    let e2 = e1.cross(dir);
    // Dans l'eau profonde, la voiture s'enfonce jusqu'a 55 cm sous la surface, pas plus.
    let height = |d: DVec3| {
        let (h, w) = floor(world, d, alt + 0.5);
        (h.max(w - 0.55), w - h)
    };
    let (h0, depth) = height(dir);
    let q0 = dir * (r + h0);
    let s = 0.7 / r;
    let (d1, d2) = ((dir + e1 * s).normalize(), (dir + e2 * s).normalize());
    let q1 = d1 * (r + height(d1).0);
    let q2 = d2 * (r + height(d2).0);
    let mut n = (q1 - q0).cross(q2 - q0).normalize_or(dir);
    if n.dot(dir) < 0.0 {
        n = -n;
    }
    // Asphalte : rues, autoroutes, circuits, routes de montagne ; sinon terre et herbe.
    let paved = world.cities.ground(dir, world) == crate::world::city::Ground::Road
        || world.cities.highways.clearance(dir).is_some_and(|c| c < 0.5)
        || world.cities.circuits.on_track(dir)
        || world.cities.mountains.on_road(dir);
    let grip = if depth > 0.15 {
        0.55
    } else if paved {
        1.0
    } else {
        0.8
    };
    (Plane { point: q0, normal: n, grip, rolling: if paved { 0.012 } else { 0.05 } }, depth)
}

/// Une place pour garer une voiture pres d'un point : la voie de droite de la rue ou de
/// l'autoroute la plus proche, dans le sens le plus proche de `heading`.
pub fn parking_spot(world: &Generator, near: DVec3, heading: DVec3) -> (DVec3, DVec3) {
    let r = planet_radius();
    let up = near.normalize();
    let heading = (heading - up * heading.dot(up)).normalize_or(up.any_orthonormal_vector());
    let mut best: Option<(DVec3, DVec3, f64)> = world.cities.road_near(up, world, heading, 300.0);
    let hws = &world.cities.highways;
    let mut hit: Option<crate::world::highway::Hit> = None;
    hws.near(up, 400.0, |h| {
        if hit.is_none_or(|b| h.dist < b.dist) {
            hit = Some(h);
        }
    });
    if let Some(h) = hit {
        if best.is_none_or(|b| h.dist < b.2) {
            let hw = &hws.list[h.hw];
            let axis = hw.dir_at(h.s);
            let mut t = hw.tangent_at(h.s);
            if t.dot(heading) < 0.0 {
                t = -t;
            }
            let right = t.cross(axis);
            let offset = hw.median() + 0.9 + hw.lanes as f64 * 3.5 - 1.75;
            best = Some(((axis + right * (offset / r)).normalize(), t, h.dist));
        }
    }
    best.map_or((up, heading), |b| (b.0, b.1))
}

impl Car {
    /// Une voiture posee sur le sol en `dir`, tournee vers `forward`.
    pub fn new(world: &Generator, dir: DVec3, forward: DVec3) -> Self {
        let up = dir.normalize();
        let forward = (forward - up * forward.dot(up)).normalize_or(up.any_orthonormal_vector());
        let (ground, water) = floor(world, up, f64::MAX);
        let base = ground.max(water - 0.55);
        let position = up * (planet_radius() + base);
        Self {
            position,
            forward,
            speed: 0.0,
            chassis: Chassis::new(position, up, forward),
            steer: 0.0,
            spin: [0.0; 4],
            lights: 0,
            headlights: false,
            time: 0.0,
            gear: 1,
            rpm: 900.0,
            view: View::Chase,
            look: (0.0, 0.0),
            look_idle: 0.0,
            chase: forward,
            bumped: false,
            impact: 0.0,
            palette: 0,
            dents: Vec::new(),
            rest: 0.0,
        }
    }

    /// La verticale du lieu.
    pub fn up(&self) -> DVec3 {
        self.position.normalize()
    }

    #[cfg(test)]
    fn altitude(&self) -> f64 {
        self.position.length() - planet_radius()
    }

    /// Le repere de la caisse : droite, haut, avant.
    pub fn axes(&self) -> (DVec3, DVec3, DVec3) {
        (self.chassis.right(), self.chassis.up(), self.chassis.forward())
    }

    /// Un point du repere de la voiture, dans le repere de la planete.
    pub fn to_world(&self, local: DVec3) -> DVec3 {
        self.chassis.to_world(local)
    }

    /// Un point du monde dans le repere du modele.
    fn to_model(&self, p: DVec3) -> DVec3 {
        let (r, u, f) = self.axes();
        let d = p - self.chassis.origin();
        DVec3::new(d.dot(r), d.dot(u), d.dot(f))
    }

    /// Enfonce la carrosserie autour d'un point du modele, selon la vitesse du choc.
    fn dent(&mut self, at: DVec3, speed: f64) {
        let depth = ((speed - 2.5) * 0.025).clamp(0.0, 0.12) as f32;
        if depth <= 0.0 {
            return;
        }
        let at = DVec3::new(at.x.clamp(-0.92, 0.92), at.y.clamp(0.3, 1.4), at.z.clamp(-2.31, 2.31)).as_vec3();
        if let Some(d) = self.dents.iter_mut().find(|d| (glam::Vec3::new(d[0], d[1], d[2]) - at).length() < 0.4) {
            d[3] = (d[3] + depth).min(0.25);
            return;
        }
        if self.dents.len() >= 8 {
            self.dents.remove(0);
        }
        self.dents.push([at.x, at.y, at.z, depth]);
    }

    /// Une image de conduite. Sans `controls`, la voiture garee reste au frein a main.
    pub fn update(&mut self, input: &Input, world: &Generator, streamer: &Streamer, dt: f32, controls: bool) {
        let dt = (dt as f64).min(0.05);
        self.time += dt;
        let held = |k: KeyCode| controls && input.held(k);
        if controls && input.just_pressed(KeyCode::KeyL) {
            self.headlights = !self.headlights;
        }

        // Remise sur roues (R) : sur place, a l'arret ou presque.
        if controls && input.just_pressed(KeyCode::KeyR) && self.chassis.speed() < 4.0 {
            let up = self.chassis.pos.normalize();
            let fwd = self.chassis.forward();
            let heading = (fwd - up * fwd.dot(up)).normalize_or(self.forward);
            let alt = self.chassis.pos.length() - planet_radius();
            let (ground, water) = floor(world, up, alt + 1.0);
            let at = up * (planet_radius() + ground.max(water - 0.55) + 0.25);
            let gear = self.chassis.gear.max(1);
            self.chassis = Chassis::new(at, up, heading);
            self.chassis.gear = gear;
        }

        // Commandes : Z accelere ; S freine, puis recule une fois arrete ; Espace serre le frein a main.
        let (gas, back, hand) = (held(KeyCode::KeyW), held(KeyCode::KeyS), held(KeyCode::Space));
        let steer = (held(KeyCode::KeyD) as i32 - held(KeyCode::KeyA) as i32) as f64;
        let vx = self.chassis.vx();
        let reversing = self.chassis.gear == -1;
        let mut c = Controls { steer, handbrake: hand as i32 as f64, ..Default::default() };
        if reversing {
            c.reverse = back || (vx < -0.5 && !gas);
            if back {
                c.throttle = 1.0;
            }
            if gas {
                c.brake = 1.0;
                c.reverse = vx < -0.5;
            }
        } else {
            if gas {
                c.throttle = 1.0;
            }
            if back {
                if vx > 0.8 {
                    c.brake = 1.0;
                } else if !gas {
                    c.reverse = true;
                }
            }
        }
        let braking = c.brake > 0.0;
        // A l'arret sans commande, le pied reste sur le frein ; garee, la voiture est au frein a main.
        if !gas && !back && vx.abs() < 0.4 {
            c.brake = 1.0;
        }
        if !controls {
            c = Controls { brake: 1.0, handbrake: 1.0, ..Default::default() };
        }

        // Garee et immobile : la physique s'endort.
        let still = self.chassis.speed() < 0.03 && self.chassis.ang.length() < 0.03;
        self.rest = if !controls && still { self.rest + dt } else { 0.0 };
        if self.rest < 1.0 {
            self.simulate(&c, world, streamer, dt);
        }

        let ch = &self.chassis;
        self.position = ch.origin();
        let up = self.up();
        let heading = ch.forward() - up * ch.forward().dot(up);
        if heading.length() > 0.2 {
            self.forward = heading.normalize();
        }
        self.steer = ch.steer;
        self.speed = ch.vx();
        for k in 0..4 {
            self.spin[k] = (self.spin[k] + ch.omega[k] * dt) % std::f64::consts::TAU;
        }
        self.gear = ch.gear;
        self.rpm = ch.rpm;
        let v = ch.speed();
        let mut l = 0;
        if self.headlights {
            l |= lights::HEAD;
        }
        if braking || hand || (!gas && v < 0.3 && controls) || (self.gear == -1 && gas) {
            l |= lights::BRAKE;
        }
        if self.speed < -0.2 {
            l |= lights::REVERSE;
        }
        // Clignotants : quand on braque fort a basse vitesse.
        let blink = (self.time * 2.6).fract() < 0.5;
        if blink && v < 14.0 && self.steer.abs() > 0.25 {
            l |= if self.steer < 0.0 { lights::LEFT } else { lights::RIGHT };
        }
        self.lights = l;
    }

    /// La physique d'une image : le sol sous les roues, la dynamique, les chocs.
    fn simulate(&mut self, c: &Controls, world: &Generator, streamer: &Streamer, dt: f64) {
        let heading = self.chassis.forward();
        let wheels: [Plane; 4] =
            std::array::from_fn(|k| plane_under(world, self.chassis.to_world(model::WHEELS[k].as_dvec3()), heading).0);
        let (body, depth) = plane_under(world, self.chassis.to_world(DVec3::new(0.0, 0.5, 0.0)), heading);
        let contact = Contact { up: self.chassis.pos.normalize(), wheels, body };
        self.chassis.update(c, &contact, dt);

        // Dans l'eau : la voiture peine a avancer.
        if depth > 0.25 {
            let up = contact.up;
            let k = (-1.6 * dt).exp();
            let vertical = up * self.chassis.vel.dot(up);
            self.chassis.vel = vertical + (self.chassis.vel - vertical) * k;
        }

        // Chocs : trois cercles le long de la voiture, contre les batiments et les troncs.
        self.bumped = false;
        self.impact = 0.0;
        let mut push = DVec3::ZERO;
        let mut probe_at = DVec3::ZERO;
        for z in [-1.45, 0.0, 1.45] {
            let probe0 = self.chassis.to_world(DVec3::new(0.0, 0.6, z));
            let mut probe = probe0;
            let ground = world.height(probe.normalize()) as f64;
            world.cities.push_out(&mut probe, 0.95, ground);
            streamer.push_out_of_objects(&mut probe, 0.95);
            let d = probe - probe0;
            if d.length() > push.length() {
                push = d;
                probe_at = probe0;
            }
        }
        if push.length() > 1e-3 {
            let n = push.normalize();
            self.chassis.pos += push;
            let hit = probe_at + push - n * 0.95;
            let impact = self.chassis.impulse(hit, n, 0.2);
            for w in &mut self.chassis.omega {
                *w = w.clamp(-90.0, 90.0);
            }
            self.bumped = impact > 3.0;
            self.impact = impact;
            let at = self.to_model(hit);
            self.dent(at, impact);
        }
        for (at, speed) in std::mem::take(&mut self.chassis.hits) {
            self.impact = self.impact.max(speed);
            self.dent(at, speed);
        }
    }

    /// Ce que le son doit jouer : moteur, pneus, roulement.
    pub fn sound(&self, cockpit: bool) -> crate::audio::CarSound {
        let ch = &self.chassis;
        // Glissement des roues au sol : un pneu crisse au-dela du pic d'adherence.
        let mut slip: f64 = 0.0;
        let mut rough = 0.0;
        let mut grounded = 0.0;
        for k in 0..4 {
            if ch.touching[k] {
                slip = slip.max(((ch.slide[k] - 0.1) / 0.25).clamp(0.0, 1.0) * (ch.load[k] / 3000.0).min(1.0));
                rough += (ch.grip[k] < 0.9) as i32 as f64;
                grounded += 1.0;
            }
        }
        crate::audio::CarSound {
            rpm: ch.rpm as f32,
            load: ch.throttle() as f32,
            gear: ch.gear,
            speed: ch.speed() as f32,
            slip: slip as f32,
            rough: if grounded > 0.0 { (rough / grounded) as f32 } else { 0.0 },
            cockpit,
        }
    }

    /// Ce que le rendu dessine.
    pub fn frame(&self, cockpit_camera: bool) -> VehicleFrame {
        let (r, u, f) = self.axes();
        let mut dents = [[0.0f32; 4]; 8];
        for (d, s) in dents.iter_mut().zip(&self.dents) {
            *d = *s;
        }
        VehicleFrame {
            position: self.position,
            right: r,
            up: u,
            forward: f,
            steer: self.steer as f32,
            spin: self.spin.map(|s| s as f32),
            travel: self.chassis.travel().map(|t| t.clamp(-0.13, 0.09) as f32),
            lights: self.lights | if cockpit_camera { lights::COCKPIT } else { 0 },
            speed_kmh: (self.speed.abs() * 3.6) as f32,
            palette: self.palette,
            dents,
        }
    }

    /// Place la camera : derriere la voiture, ou a la place du conducteur. La souris tourne le regard.
    pub fn place_camera(&mut self, cam: &mut Camera, input: &Input, world: &Generator, dt: f32) {
        let dt = dt as f64;
        if input.just_pressed(KeyCode::KeyC) {
            self.view = if self.view == View::Chase { View::Cockpit } else { View::Chase };
            self.look = (0.0, 0.0);
        }
        if input.captured {
            let (dx, dy) = input.mouse_delta;
            self.look.0 = (self.look.0 + dx * 0.0025).clamp(-2.6, 2.6);
            self.look.1 = (self.look.1 - dy * 0.0025).clamp(-0.6, 0.7);
            if dx.abs() + dy.abs() > 0.5 {
                self.look_idle = 0.0;
            }
        }
        self.look_idle += dt;
        if self.view == View::Chase && self.look_idle > 1.5 {
            // Le regard revient doucement dans l'axe.
            let k = 1.0 - (-2.0 * dt).exp();
            self.look.0 -= self.look.0 * k;
            self.look.1 -= (self.look.1 - 0.0) * k;
        }
        let (r, u, f) = self.axes();
        let up = self.up();
        let (sy, cy) = self.look.0.sin_cos();
        let (eye, look) = match self.view {
            View::Cockpit => {
                let eye = self.to_world(DVec3::new(DRIVER_EYE.x as f64, DRIVER_EYE.y as f64, DRIVER_EYE.z as f64));
                let dir = (f * cy + r * sy).normalize();
                let (sp, cp) = (self.look.1 - 0.08).sin_cos();
                (eye, (dir * cp + u * sp).normalize())
            }
            View::Chase => {
                let flat = (f - up * f.dot(up)).normalize_or(self.forward);
                let k = 1.0 - (-5.0 * dt).exp();
                self.chase = (self.chase + (flat - self.chase) * k).normalize_or(flat);
                let side = self.chase.cross(up);
                let back = (self.chase * cy + side * sy).normalize();
                let distance = 6.4 + (self.speed.abs() * 0.02).min(1.2);
                let height = 1.9 + self.look.1 * 3.0;
                let mut eye = self.position - back * distance + up * height.max(0.6);
                // Jamais sous le sol.
                let ground = world.height(eye.normalize()) as f64;
                let min = planet_radius() + ground + 0.5;
                if eye.length() < min {
                    eye = eye.normalize() * min;
                }
                let target = self.position + up * 1.1 + back * 1.5;
                (eye, (target - eye).normalize())
            }
        };
        cam.position = eye;
        let eu = eye.normalize();
        cam.heading = (look - eu * look.dot(eu)).normalize_or(cam.heading);
        cam.pitch = look.dot(eu).clamp(-1.0, 1.0).asin() as f32;
    }

    /// Ou poser le pieton qui descend : cote conducteur.
    pub fn door(&self) -> DVec3 {
        let (r, _, _) = self.axes();
        self.position - r * 1.7
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::generator::PlanetParams;
    use crate::world::streaming::StreamConfig;
    use std::sync::Arc;

    fn setup() -> (Arc<Generator>, Streamer, Car) {
        let gen = Arc::new(Generator::new(PlanetParams { seed: 1, cities: 20, ..Default::default() }));
        let city = &gen.cities.list[0];
        let (at, fwd) = parking_spot(&gen, city.dir, city.axes().1);
        let car = Car::new(&gen, at, fwd);
        let streamer = Streamer::new(gen.clone(), StreamConfig { workers: 1, ..Default::default() });
        (gen, streamer, car)
    }

    #[test]
    fn the_car_accelerates_brakes_and_stays_on_the_ground() {
        let (gen, streamer, mut car) = setup();
        let mut input = Input::default();
        input.set_key(KeyCode::KeyW, true, false);
        for _ in 0..90 {
            car.update(&input, &gen, &streamer, 1.0 / 60.0, true);
        }
        assert!(car.speed > 5.0, "vitesse {}", car.speed);
        let ground = floor(&gen, car.up(), car.altitude() + 0.5).0;
        assert!((car.altitude() - ground).abs() < 1.0, "altitude {} sol {ground}", car.altitude());
        input.set_key(KeyCode::KeyW, false, false);
        input.set_key(KeyCode::KeyS, true, false);
        for _ in 0..300 {
            car.update(&input, &gen, &streamer, 1.0 / 60.0, true);
            if car.speed <= 0.0 {
                break;
            }
        }
        assert!(car.speed <= 0.5, "la voiture freine : {}", car.speed);
    }

    #[test]
    fn steering_turns_the_car_the_right_way() {
        let (gen, streamer, mut car) = setup();
        let start = car.forward;
        let right = start.cross(car.up());
        let mut input = Input::default();
        input.set_key(KeyCode::KeyW, true, false);
        input.set_key(KeyCode::KeyD, true, false);
        for _ in 0..90 {
            car.update(&input, &gen, &streamer, 1.0 / 60.0, true);
        }
        assert!(car.forward.dot(right) > 0.1, "la voiture tourne a droite");
    }

    #[test]
    fn the_car_is_parked_on_a_street() {
        let (gen, _, car) = setup();
        assert_eq!(gen.cities.ground(car.up(), &gen), crate::world::city::Ground::Road);
    }
}
