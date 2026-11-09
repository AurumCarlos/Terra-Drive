//! Dynamique de la voiture : une caisse rigide en 3D portee par quatre suspensions, sans le monde.
//!
//! - Caisse : corps rigide a six degres de liberte (position, vitesse, orientation, vitesse angulaire,
//!   tenseur d'inertie). Plongee, cabrage, roulis et sauts ne sont pas animes : ils sortent du calcul.
//! - Suspensions : ressort, amortisseur a deux taux (compression, detente), butee, barre antiroulis par
//!   essieu. La charge de chaque pneu est la force de sa suspension : les transferts de charge en decoulent.
//!   Une roue peut decoller ; la caisse peut se retourner et frotter le sol.
//! - Pneus : formule magique de Pacejka sur le glissement combine, avec longueur de relaxation (le pneu se
//!   deforme avant de glisser). A l'arret, cette deformation tient la voiture en pente, sans artifice.
//! - Moteur atmospherique de 4,0 l : courbe de couple, frein moteur, limiteur ; convertisseur de couple
//!   au demarrage ; boite automatique a 7 rapports avec temps de passage.
//! - Propulsion avec differentiel a glissement limite, freins repartis avec ABS, frein a main sur
//!   l'arriere sans ABS, antipatinage leger, resistance au roulement, trainee et appui aerodynamiques.
//!
//! Repere de la caisse : celui du modele, x a droite, y en haut, z vers l'avant, origine au sol entre les
//! essieux. Le sol est donne a chaque image par des plans tangents sous les roues (voir `Contact`).

use super::model::{FRONT_AXLE, REAR_AXLE, WHEELS, WHEEL_RADIUS};
use glam::{DMat3, DQuat, DVec3};

pub const MASS: f64 = 1550.0;
pub const GRAVITY: f64 = 9.81;
/// Inertie de la caisse (kg m2) : tangage, lacet, roulis.
const INERTIA: DVec3 = DVec3::new(2300.0, 2600.0, 560.0);
/// Hauteur du centre de gravite au-dessus du sol, au repos.
pub const CG_HEIGHT: f64 = 0.52;
/// Inertie d'une roue avec son frein et sa part de transmission.
const WHEEL_INERTIA: f64 = 1.4;
/// 0,5 * densite de l'air * Cx * surface frontale.
const DRAG: f64 = 0.5 * 1.2 * 0.31 * 2.2;
const DOWNFORCE: f64 = 0.5 * 1.2 * 0.12 * 2.2;
pub const IDLE_RPM: f64 = 850.0;
pub const REDLINE: f64 = 7400.0;
/// Rapports de boite ; le pont multiplie encore par `FINAL_DRIVE`.
const RATIOS: [f64; 7] = [4.11, 2.65, 1.86, 1.43, 1.17, 1.0, 0.84];
const REVERSE: f64 = 3.7;
const FINAL_DRIVE: f64 = 3.15;
const DRIVETRAIN: f64 = 0.9;
/// Couple maximal de freinage par roue (N m), avant et arriere.
const BRAKE_TORQUE: [f64; 2] = [2600.0, 1450.0];
const HANDBRAKE_TORQUE: f64 = 2600.0;
const SHIFT_TIME: f64 = 0.18;
/// Adherence des pneus sport sur asphalte sec.
const TIRE_MU: f64 = 1.1;
/// Longueurs de relaxation du pneu (m) : en long, en travers.
const RELAX: [f64; 2] = [0.3, 0.55];

/// Raideur des ressorts ramenee a la roue (N/m), avant et arriere : environ 1,5 Hz, une berline sportive.
const SPRING: [f64; 2] = [36_000.0, 39_000.0];
/// Amortisseurs (N s/m) : compression, detente.
const BUMP: [f64; 2] = [2700.0, 2900.0];
const REBOUND: [f64; 2] = [4300.0, 4700.0];
/// Barres antiroulis (N/m de difference de debattement), avant plus raide : la voiture sous-vire un peu.
const ANTI_ROLL: [f64; 2] = [26_000.0, 11_000.0];
/// Longueur de la suspension au repos (fixation - centre de roue), debattements en compression et detente.
const STATIC_LENGTH: f64 = 0.26;
const BUMP_TRAVEL: f64 = 0.09;
const DROOP_TRAVEL: f64 = 0.13;
/// Butee de fin de course.
const BUMP_STOP: f64 = 260_000.0;

/// Points de la caisse qui peuvent toucher le sol : bas de caisse, boucliers, toit.
const HULL: [[f64; 3]; 14] = [
    [-0.9, 0.3, 2.25],
    [0.9, 0.3, 2.25],
    [-0.9, 0.3, -2.25],
    [0.9, 0.3, -2.25],
    [0.0, 0.32, 2.31],
    [0.0, 0.36, -2.31],
    [-0.92, 0.3, 0.0],
    [0.92, 0.3, 0.0],
    [-0.75, 1.4, 0.7],
    [0.75, 1.4, 0.7],
    [-0.75, 1.4, -1.1],
    [0.75, 1.4, -1.1],
    [-0.92, 0.95, 0.0],
    [0.92, 0.95, 0.0],
];

/// Pas d'integration : les pneus sont raides, il faut un pas court.
const STEP: f64 = 1.0 / 240.0;

/// Couple du moteur a pleine charge (N m) selon le regime.
pub fn torque_curve(rpm: f64) -> f64 {
    const CURVE: [(f64, f64); 8] = [(0.0, 250.0), (1000.0, 320.0), (2000.0, 400.0), (3000.0, 450.0), (4500.0, 470.0), (5800.0, 455.0), (6800.0, 420.0), (7400.0, 380.0)];
    let rpm = rpm.clamp(0.0, 7400.0);
    for w in CURVE.windows(2) {
        if rpm <= w[1].0 {
            let t = (rpm - w[0].0) / (w[1].0 - w[0].0);
            return w[0].1 + (w[1].1 - w[0].1) * t;
        }
    }
    CURVE[7].1
}

/// Formule magique de Pacejka : effort rapporte a l'effort maximal, selon le glissement.
fn magic(slip: f64) -> f64 {
    const B: f64 = 11.0;
    const C: f64 = 1.6;
    const E: f64 = 0.55;
    let x = B * slip;
    (C * (x - E * (x - x.atan())).atan()).sin()
}

/// Commandes, de 0 a 1 (le volant de -1 a 1).
#[derive(Clone, Copy, Default, Debug)]
pub struct Controls {
    pub throttle: f64,
    pub brake: f64,
    pub handbrake: f64,
    pub steer: f64,
    /// Marche arriere demandee.
    pub reverse: bool,
}

/// Le sol sous un point : plan tangent et revetement.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub point: DVec3,
    pub normal: DVec3,
    /// Adherence : 1 sur l'asphalte sec, moins sur l'herbe ou dans l'eau.
    pub grip: f64,
    /// Resistance au roulement, en part de la charge.
    pub rolling: f64,
}

/// Le sol autour de la voiture pour une image : la verticale, un plan sous chaque roue et un sous la caisse.
#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub up: DVec3,
    pub wheels: [Plane; 4],
    pub body: Plane,
}

impl Contact {
    /// Un sol plan (pour les essais) : passe par `point`, de normale `normal`, la pesanteur selon -`up`.
    pub fn plane(up: DVec3, point: DVec3, normal: DVec3) -> Self {
        let p = Plane { point, normal: normal.normalize(), grip: 1.0, rolling: 0.012 };
        Self { up: up.normalize(), wheels: [p; 4], body: p }
    }
}

#[derive(Clone, Debug)]
pub struct Chassis {
    /// Centre de gravite, vitesse (repere du monde).
    pub pos: DVec3,
    pub vel: DVec3,
    /// Orientation : du repere de la caisse (x a gauche, y en haut, z en avant ; repere direct) au monde.
    pub rot: DQuat,
    /// Vitesse angulaire (repere du monde).
    pub ang: DVec3,
    /// Angle des roues avant (rad), positif a droite.
    pub steer: f64,
    /// Vitesse de rotation des roues (rad/s) : AVG, AVD, ARG, ARD.
    pub omega: [f64; 4],
    pub rpm: f64,
    /// Rapport engage : 1 a 7, -1 pour la marche arriere.
    pub gear: i32,
    shifting: f64,
    /// Pas de retrogradage juste apres un passage : la boite ne doit pas hesiter.
    settle: f64,
    throttle: f64,
    /// Longueur de chaque suspension.
    pub length: [f64; 4],
    /// Deformation de chaque pneu (m) : en long, en travers.
    deflect: [[f64; 2]; 4],
    /// Charge, glissement et contact de chaque roue.
    pub load: [f64; 4],
    pub slide: [f64; 4],
    pub touching: [bool; 4],
    /// Adherence du sol sous chaque roue (1 : asphalte sec).
    pub grip: [f64; 4],
    pub abs: bool,
    /// Chocs de la caisse contre le sol depuis la derniere lecture : point (repere du modele), vitesse.
    pub hits: Vec<(DVec3, f64)>,
}

/// Position d'un point du modele par rapport au centre de gravite, dans le repere de la caisse
/// (x a gauche).
fn body_local(model: DVec3) -> DVec3 {
    DVec3::new(-model.x, model.y - CG_HEIGHT, model.z)
}

/// Longueur libre du ressort : il porte sa part du poids a la longueur de repos.
fn free_length(k: usize) -> f64 {
    let wb = (FRONT_AXLE - REAR_AXLE) as f64;
    let share = if k < 2 { -REAR_AXLE as f64 / wb } else { FRONT_AXLE as f64 / wb };
    STATIC_LENGTH + MASS * GRAVITY * share * 0.5 / SPRING[(k >= 2) as usize]
}

fn mount(k: usize) -> DVec3 {
    let w = WHEELS[k];
    body_local(DVec3::new(w.x as f64, w.y as f64 + STATIC_LENGTH, w.z as f64))
}

impl Chassis {
    /// Une voiture au repos, origine du modele en `ground` (au sol), tournee vers `forward`.
    pub fn new(ground: DVec3, up: DVec3, forward: DVec3) -> Self {
        let up = up.normalize();
        let fwd = (forward - up * forward.dot(up)).normalize();
        let left = up.cross(fwd);
        let rot = DQuat::from_mat3(&DMat3::from_cols(left, up, fwd)).normalize();
        Self {
            pos: ground + up * CG_HEIGHT,
            vel: DVec3::ZERO,
            rot,
            ang: DVec3::ZERO,
            steer: 0.0,
            omega: [0.0; 4],
            rpm: IDLE_RPM,
            gear: 1,
            shifting: 0.0,
            settle: 0.0,
            throttle: 0.0,
            length: [STATIC_LENGTH; 4],
            deflect: [[0.0; 2]; 4],
            load: [MASS * GRAVITY / 4.0; 4],
            slide: [0.0; 4],
            touching: [true; 4],
            grip: [1.0; 4],
            abs: false,
            hits: Vec::new(),
        }
    }

    /// Lance la voiture tout droit a `v` m/s.
    pub fn launch(&mut self, v: f64) {
        self.vel = self.forward() * v;
        self.omega = [v / WHEEL_RADIUS as f64; 4];
    }

    pub fn up(&self) -> DVec3 {
        self.rot * DVec3::Y
    }

    pub fn forward(&self) -> DVec3 {
        self.rot * DVec3::Z
    }

    pub fn right(&self) -> DVec3 {
        self.rot * DVec3::NEG_X
    }

    /// Origine du modele (au sol entre les essieux, quand la voiture est au repos).
    pub fn origin(&self) -> DVec3 {
        self.pos - self.up() * CG_HEIGHT
    }

    /// Un point du repere du modele, dans le monde.
    pub fn to_world(&self, model: DVec3) -> DVec3 {
        self.pos + self.rot * body_local(model)
    }

    /// Vitesse le long de la voiture (negative en marche arriere) et vers sa droite.
    pub fn vx(&self) -> f64 {
        self.vel.dot(self.forward())
    }

    pub fn vy(&self) -> f64 {
        self.vel.dot(self.right())
    }

    pub fn speed(&self) -> f64 {
        self.vel.length()
    }

    /// Vitesse de lacet (rad/s) autour de la verticale `up`, positive vers la droite.
    pub fn yaw_rate(&self, up: DVec3) -> f64 {
        -self.ang.dot(up)
    }

    /// Angle de derive de la caisse (rad).
    pub fn drift_angle(&self) -> f64 {
        let vx = self.vx();
        if vx.abs() < 2.0 { 0.0 } else { self.vy().atan2(vx.abs()) }
    }

    /// Position de la pedale d'accelerateur, telle que le moteur la voit (0 a 1).
    pub fn throttle(&self) -> f64 {
        self.throttle
    }

    /// Debattement de chaque roue par rapport au repos (positif : la roue remonte).
    pub fn travel(&self) -> [f64; 4] {
        self.length.map(|l| STATIC_LENGTH - l)
    }

    /// Rapport total moteur / roues du rapport engage.
    fn ratio(&self) -> f64 {
        match self.gear {
            -1 => -REVERSE * FINAL_DRIVE,
            g => RATIOS[(g.clamp(1, 7) - 1) as usize] * FINAL_DRIVE,
        }
    }

    /// Braquage maximal : le volant se demultiplie avec la vitesse, comme une direction a assistance variable.
    pub fn max_steer(&self) -> f64 {
        let v = self.vx().abs();
        (0.6 / (1.0 + (v / 17.0).powi(2)).sqrt()).max(0.07)
    }

    /// Inertie dans le repere du monde, et son inverse.
    fn inertia(&self) -> (DMat3, DMat3) {
        let m = DMat3::from_quat(self.rot);
        let mt = m.transpose();
        (m * DMat3::from_diagonal(INERTIA) * mt, m * DMat3::from_diagonal(INERTIA.recip()) * mt)
    }

    /// Choc ponctuel en `at` contre une surface de normale `n` (vers l'exterieur de l'obstacle) :
    /// impulsion avec rebond et frottement. Renvoie la vitesse d'impact.
    pub fn impulse(&mut self, at: DVec3, n: DVec3, restitution: f64) -> f64 {
        let r = at - self.pos;
        let v = self.vel + self.ang.cross(r);
        let vn = v.dot(n);
        if vn >= 0.0 {
            return 0.0;
        }
        let (_, inv) = self.inertia();
        let k = |d: DVec3| 1.0 / MASS + d.dot((inv * r.cross(d)).cross(r));
        let j = -(1.0 + restitution) * vn / k(n);
        self.vel += n * (j / MASS);
        self.ang += inv * r.cross(n * j);
        // Frottement le long de l'obstacle, sans inverser le glissement.
        let vt = v - n * vn;
        if vt.length() > 1e-3 {
            let t = vt.normalize();
            let jt = (vt.length() / k(t)).min(0.4 * j);
            self.vel -= t * (jt / MASS);
            self.ang -= inv * r.cross(t * jt);
        }
        -vn
    }

    /// Avance de `dt` secondes, par pas courts.
    pub fn update(&mut self, c: &Controls, ground: &Contact, dt: f64) {
        let steps = (dt / STEP).ceil().max(1.0) as usize;
        let h = dt / steps as f64;
        for _ in 0..steps {
            self.step(c, ground, h);
        }
    }

    fn step(&mut self, c: &Controls, ground: &Contact, dt: f64) {
        let (up, fwd, right) = (self.up(), self.forward(), self.right());
        let vx = self.vel.dot(fwd);
        let r = WHEEL_RADIUS as f64;

        // Volant : vitesse de braquage limitee, retour plus rapide.
        let wish = c.steer.clamp(-1.0, 1.0) * self.max_steer();
        let rate = if wish.abs() < self.steer.abs() { 4.0 } else { 2.2 };
        self.steer += (wish - self.steer).clamp(-rate * dt, rate * dt);

        // Pedale d'accelerateur : un pied, pas un interrupteur.
        let pedal = c.throttle.clamp(0.0, 1.0);
        self.throttle += (pedal - self.throttle).clamp(-8.0 * dt, 5.0 * dt);

        // Boite automatique.
        self.shifting = (self.shifting - dt).max(0.0);
        self.settle = (self.settle - dt).max(0.0);
        let wheel_rpm = (self.omega[2] + self.omega[3]) * 0.5 * 60.0 / std::f64::consts::TAU;
        if c.reverse {
            if self.gear != -1 && vx < 1.0 {
                self.gear = -1;
                self.shifting = SHIFT_TIME;
            }
        } else if self.gear == -1 {
            if vx > -1.0 {
                self.gear = 1;
                self.shifting = SHIFT_TIME;
            }
        } else if self.shifting <= 0.0 {
            let rpm = wheel_rpm * self.ratio();
            let up_at = 6900.0 - (1.0 - self.throttle) * 3600.0;
            if self.gear < 7 && rpm > up_at {
                self.gear += 1;
                self.shifting = SHIFT_TIME;
                self.settle = 1.2;
            } else if self.gear > 1 && self.settle <= 0.0 {
                let lower = wheel_rpm * RATIOS[(self.gear - 2) as usize] * FINAL_DRIVE;
                let down_at = 1500.0 + self.throttle * 1700.0;
                if rpm < down_at && lower < 5600.0 {
                    self.gear -= 1;
                    self.shifting = SHIFT_TIME * 0.7;
                }
            }
        }

        // Moteur : le convertisseur laisse le regime monter au demarrage et multiplie le couple.
        let ratio = self.ratio();
        let coupled = (wheel_rpm * ratio).abs();
        let stall = IDLE_RPM + 1900.0 * self.throttle;
        let engine_rpm = coupled.max(stall).min(REDLINE + 200.0);
        let slip = (1.0 - coupled / engine_rpm).clamp(0.0, 1.0);
        let mut torque = if engine_rpm >= REDLINE { 0.0 } else { torque_curve(engine_rpm) * self.throttle };
        torque -= (1.0 - self.throttle) * (25.0 + engine_rpm * 0.012) * (1.0 - slip);
        torque *= 1.0 + 1.0 * slip;
        if self.shifting > 0.0 {
            torque *= 0.15;
        }
        self.rpm += (engine_rpm - self.rpm) * (1.0 - (-25.0 * dt).exp());

        // Antipatinage : coupe les gaz quand l'arriere patine trop (pas au frein a main).
        let rear_spin = ((self.omega[2] + self.omega[3]) * 0.5 * r - vx) / vx.abs().max(4.0);
        if c.handbrake < 0.1 && rear_spin * ratio.signum() > 0.22 {
            torque *= 0.4;
        }
        let axle_torque = torque * ratio * DRIVETRAIN;

        let mut force = -ground.up * (MASS * GRAVITY);
        let mut moment = DVec3::ZERO;
        let pos = self.pos;
        let push = |f: DVec3, at: DVec3, force: &mut DVec3, moment: &mut DVec3| {
            *force += f;
            *moment += (at - pos).cross(f);
        };

        // Air : trainee et appui.
        let speed = self.vel.length();
        if speed > 1e-3 {
            force -= self.vel * (DRAG * speed);
        }
        force -= up * (DOWNFORCE * vx * vx);

        // Suspensions : longueur jusqu'au sol le long de l'axe de la caisse, ressort, amortisseur, butee.
        let mut spring = [0.0; 4];
        let mut contact = [DVec3::ZERO; 4];
        let mut mounts = [DVec3::ZERO; 4];
        for k in 0..4 {
            let axle = (k >= 2) as usize;
            let m = self.pos + self.rot * mount(k);
            mounts[k] = m;
            let plane = &ground.wheels[k];
            let n = plane.normal;
            let un = up.dot(n);
            let (lmin, lmax) = (STATIC_LENGTH - BUMP_TRAVEL, STATIC_LENGTH + DROOP_TRAVEL);
            let reach = if un > 0.25 { ((m - plane.point).dot(n) - r) / un } else { f64::MAX };
            self.touching[k] = reach < lmax;
            if !self.touching[k] {
                // En l'air : la roue pend, le ressort la ramene vite en bas.
                self.length[k] += (lmax - self.length[k]) * (1.0 - (-30.0 * dt).exp());
                continue;
            }
            let l = reach.max(lmin);
            let vm = self.vel + self.ang.cross(m - self.pos);
            let ldot = vm.dot(n) / un;
            let damper = if ldot < 0.0 { BUMP[axle] } else { REBOUND[axle] };
            let mut f = SPRING[axle] * (free_length(k) - l) - damper * ldot;
            if reach < lmin {
                f += BUMP_STOP * (lmin - reach) - 8000.0 * ldot.min(0.0);
            }
            self.length[k] = l;
            spring[k] = f;
            contact[k] = m - up * l - n * r;
        }
        for axle in 0..2 {
            let (i, j) = (axle * 2, axle * 2 + 1);
            let f = ANTI_ROLL[axle] * (self.length[j] - self.length[i]);
            if self.touching[i] {
                spring[i] += f;
            }
            if self.touching[j] {
                spring[j] -= f;
            }
        }

        // Pneus.
        self.abs = false;
        for k in 0..4 {
            if !self.touching[k] {
                self.load[k] = 0.0;
                self.slide[k] = 0.0;
                self.deflect[k] = [0.0; 2];
                // Roue libre en l'air : la transmission et les freins la font tourner ou l'arretent.
                let mut w = self.omega[k];
                if k >= 2 {
                    w += axle_torque * 0.5 / (WHEEL_INERTIA * 4.0) * dt;
                }
                let stop = (c.brake * BRAKE_TORQUE[(k >= 2) as usize] + if k >= 2 { c.handbrake * HANDBRAKE_TORQUE } else { 0.0 }) / WHEEL_INERTIA * dt;
                self.omega[k] = if w > 0.0 { (w - stop).max(0.0) } else { (w + stop).min(0.0) };
                continue;
            }
            let fz = spring[k].max(0.0);
            self.load[k] = fz;
            self.grip[k] = ground.wheels[k].grip;
            push(up * fz, mounts[k], &mut force, &mut moment);
            let plane = &ground.wheels[k];
            let n = plane.normal;
            let cp = contact[k];

            // Repere du pneu dans le plan du sol.
            let delta = if k < 2 { self.steer } else { 0.0 };
            let (sd, cd) = delta.sin_cos();
            let heading = fwd * cd + right * sd;
            let tf = (heading - n * heading.dot(n)).normalize_or(fwd);
            let tr = tf.cross(n);
            let vc = self.vel + self.ang.cross(cp - self.pos);
            let (vl, vs) = (vc.dot(tf), vc.dot(tr));

            // Couples sur la roue : transmission (propulsion), freins.
            let mut drive = 0.0;
            if k >= 2 {
                drive = axle_torque * 0.5;
                // Differentiel a glissement limite : il freine la roue qui tourne plus vite que l'autre.
                let other = self.omega[if k == 2 { 3 } else { 2 }];
                drive -= (self.omega[k] - other) * 160.0;
            }
            let mut brake = c.brake.clamp(0.0, 1.0) * BRAKE_TORQUE[(k >= 2) as usize];
            let kappa = self.deflect[k][0] / RELAX[0];
            // ABS : relache la roue qui se bloque.
            if brake > 0.0 && kappa < -0.14 && vl.abs() > 2.0 {
                brake *= 0.15;
                self.abs = true;
            }
            if k >= 2 {
                brake += c.handbrake.clamp(0.0, 1.0) * HANDBRAKE_TORQUE;
            }

            // Deformation du pneu : elle suit le glissement avec la longueur de relaxation.
            let a = vl.abs();
            let slips = [self.omega[k] * r - vl, vs];
            for (i, s) in slips.iter().enumerate() {
                let d = &mut self.deflect[k][i];
                if a < 1e-3 {
                    *d += s * dt;
                } else {
                    let e = (-a * dt / RELAX[i]).exp();
                    *d = *d * e + s * RELAX[i] / a * (1.0 - e);
                }
            }
            // Pneus arriere plus larges, donc plus raides en derive : la voiture sous-vire un peu et reste
            // stable a haute vitesse, comme une voiture de serie.
            let stiff = if k >= 2 { 1.35 } else { 1.0 };
            let mut sx = self.deflect[k][0] / RELAX[0];
            let mut sy = stiff * self.deflect[k][1] / RELAX[1];
            let mut sigma = (sx * sx + sy * sy).sqrt();
            // Au-dela, le pneu glisse : sa deformation ne grandit plus.
            if sigma > 1.2 {
                let q = 1.2 / sigma;
                self.deflect[k][0] *= q;
                self.deflect[k][1] *= q;
                sx *= q;
                sy *= q;
                sigma = 1.2;
            }
            let peak = plane.grip * TIRE_MU * fz;
            let (mut fl, mut fs) = if sigma > 1e-9 {
                let f = peak * magic(sigma);
                (f * sx / sigma, -f * sy / sigma)
            } else {
                (0.0, 0.0)
            };
            // A tres basse vitesse, un amortissement remplace l'hysteresis du caoutchouc.
            let fade = (1.0 - a / 3.0).clamp(0.0, 1.0);
            let damping = 5.0 * fz * fade;
            fs = (fs - damping * vs).clamp(-peak, peak);
            self.slide[k] = sigma;

            // La roue : moteur et reaction du sol, amortissement (implicite), puis freins et roulement.
            let mut w = self.omega[k] + (drive - fl * r) / WHEEL_INERTIA * dt;
            let damp = damping * r * r * dt / WHEEL_INERTIA;
            w = (w + damping * r * vl * dt / WHEEL_INERTIA) / (1.0 + damp);
            let stop = (brake + plane.rolling * fz * r) / WHEEL_INERTIA * dt;
            w = if w > 0.0 { (w - stop).max(0.0) } else { (w + stop).min(0.0) };
            // Roue bloquee par le frein a main : elle le reste.
            if k >= 2 && c.handbrake > 0.5 && drive.abs() < brake {
                w *= 0.2;
            }
            fl = (fl + damping * (w * r - vl)).clamp(-peak, peak);
            self.omega[k] = w;
            push(tf * fl + tr * fs, cp, &mut force, &mut moment);
        }

        // La caisse contre le sol : ressort raide, amortissement, frottement de la tole.
        for p in HULL {
            let at = self.pos + self.rot * body_local(DVec3::from_array(p));
            let plane = ground
                .wheels
                .iter()
                .chain(std::iter::once(&ground.body))
                .min_by(|a, b| (a.point - at).length_squared().total_cmp(&(b.point - at).length_squared()))
                .unwrap();
            let depth = -(at - plane.point).dot(plane.normal);
            if depth <= 0.0 {
                continue;
            }
            let n = plane.normal;
            let v = self.vel + self.ang.cross(at - self.pos);
            let vn = v.dot(n);
            let fn_ = (400_000.0 * depth - 30_000.0 * vn).max(0.0);
            let vt = v - n * vn;
            let ft = -vt / (vt.length() + 0.3) * (0.55 * fn_);
            push(n * fn_ + ft, at, &mut force, &mut moment);
            if vn < -3.0 && self.hits.len() < 16 {
                self.hits.push((DVec3::from_array(p), -vn));
            }
        }

        // Integration : vitesse puis position (Euler semi-implicite), orientation par quaternion.
        self.vel += force / MASS * dt;
        let (iw, inv) = self.inertia();
        self.ang += inv * (moment - self.ang.cross(iw * self.ang)) * dt;
        self.pos += self.vel * dt;
        self.rot = (DQuat::from_scaled_axis(self.ang * dt) * self.rot).normalize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat() -> Contact {
        Contact::plane(DVec3::Y, DVec3::ZERO, DVec3::Y)
    }

    fn car() -> Chassis {
        Chassis::new(DVec3::ZERO, DVec3::Y, DVec3::Z)
    }

    fn run(ch: &mut Chassis, c: Controls, ground: &Contact, seconds: f64) {
        let mut t = 0.0;
        while t < seconds {
            ch.update(&c, ground, 1.0 / 60.0);
            t += 1.0 / 60.0;
        }
    }

    fn trace() -> bool {
        std::env::var("TRACE").is_ok()
    }

    #[test]
    fn it_settles_on_its_springs_after_a_drop() {
        let mut ch = car();
        ch.pos += DVec3::Y * 0.6;
        let mut lowest = f64::MAX;
        for i in 0..240 {
            ch.update(&Controls { brake: 1.0, ..Default::default() }, &flat(), 1.0 / 60.0);
            lowest = lowest.min(ch.pos.y);
            if trace() && i % 6 == 0 {
                eprintln!("y {:.3} vy {:.3} len {:?}", ch.pos.y, ch.vel.y, ch.length.map(|l| (l * 1000.0) as i32));
            }
        }
        assert!(lowest > 0.2, "la caisse ne traverse pas le sol : {lowest:.2}");
        assert!((ch.pos.y - CG_HEIGHT).abs() < 0.03, "hauteur au repos {:.3}", ch.pos.y);
        assert!(ch.vel.length() < 0.02 && ch.ang.length() < 0.02, "immobile : {} {}", ch.vel, ch.ang);
        assert!(ch.up().y > 0.999, "a plat");
    }

    #[test]
    fn it_reaches_100_kmh_like_a_sports_saloon() {
        let mut ch = car();
        let mut t = 0.0;
        while ch.vx() < 100.0 / 3.6 && t < 20.0 {
            ch.update(&Controls { throttle: 1.0, ..Default::default() }, &flat(), 1.0 / 60.0);
            t += 1.0 / 60.0;
            if trace() && ((t * 60.0) as i32) % 15 == 0 {
                eprintln!("t {t:.2} v {:.1} g {} rpm {:.0} load {:?} pitch {:.3}", ch.vx() * 3.6, ch.gear, ch.rpm, ch.load.map(|l| l as i32), ch.forward().y);
            }
        }
        assert!((3.8..7.0).contains(&t), "0-100 km/h en {t:.2} s");
        assert!(ch.gear >= 2, "la boite a passe la seconde : {}", ch.gear);
        for _ in 0..30 {
            run(&mut ch, Controls { throttle: 1.0, ..Default::default() }, &flat(), 3.0);
        }
        let top = ch.vx() * 3.6;
        assert!((230.0..320.0).contains(&top), "vitesse de pointe {top:.0} km/h");
        assert!(ch.vy().abs() < 0.1 && ch.up().y > 0.99, "stable a fond : {} {}", ch.vy(), ch.up());
    }

    #[test]
    fn it_brakes_from_100_in_about_35_m_and_dives() {
        let mut ch = car();
        run(&mut ch, Controls::default(), &flat(), 0.5);
        ch.launch(100.0 / 3.6);
        ch.gear = 4;
        let start = ch.pos;
        let mut dive: f64 = 0.0;
        while ch.vx() > 0.1 && (ch.pos - start).length() < 200.0 {
            ch.update(&Controls { brake: 1.0, ..Default::default() }, &flat(), 1.0 / 60.0);
            dive = dive.min(ch.forward().y);
        }
        let dist = (ch.pos - start).length();
        assert!((28.0..45.0).contains(&dist), "distance d'arret {dist:.1} m");
        assert!((-0.06..-0.005).contains(&dive), "le nez plonge : {dive:.3}");
    }

    #[test]
    fn cornering_grip_is_close_to_one_g_and_the_body_rolls_out() {
        let mut ch = car();
        ch.launch(20.0);
        ch.gear = 3;
        let mut roll: f64 = 0.0;
        for _ in 0..240 {
            ch.update(&Controls { throttle: 0.35, steer: 0.5, ..Default::default() }, &flat(), 1.0 / 60.0);
            roll = roll.max(ch.right().y);
        }
        let yaw = ch.yaw_rate(DVec3::Y);
        assert!(yaw > 0.0, "a droite");
        let lateral = yaw * ch.vx();
        assert!((5.0..12.0).contains(&lateral), "acceleration laterale {lateral:.1} m/s2");
        // Virage a droite : la caisse penche vers la gauche, le cote droit se leve.
        assert!((0.01..0.12).contains(&roll), "roulis {roll:.3}");
    }

    #[test]
    fn the_handbrake_makes_the_rear_slide() {
        let mut ch = car();
        ch.launch(18.0);
        ch.gear = 3;
        run(&mut ch, Controls { steer: 0.6, ..Default::default() }, &flat(), 0.4);
        run(&mut ch, Controls { steer: 0.6, handbrake: 1.0, ..Default::default() }, &flat(), 0.7);
        assert!(ch.drift_angle().abs() > 0.12, "derive {:.2} rad", ch.drift_angle());
    }

    #[test]
    fn the_car_stays_stable_at_rest_and_at_speed() {
        let mut ch = car();
        run(&mut ch, Controls::default(), &flat(), 3.0);
        assert!(ch.speed() < 0.05 && ch.ang.length() < 1e-2, "{} {}", ch.speed(), ch.ang);
        let mut ch = car();
        ch.launch(40.0);
        ch.gear = 5;
        run(&mut ch, Controls { throttle: 0.4, ..Default::default() }, &flat(), 5.0);
        assert!(ch.vy().abs() < 0.05 && ch.yaw_rate(DVec3::Y).abs() < 0.01, "tout droit : {} {}", ch.vy(), ch.yaw_rate(DVec3::Y));
    }

    #[test]
    fn the_brakes_hold_it_on_a_steep_slope() {
        // Pente de 20 % : sans frein, la voiture descend ; freinee, elle tient.
        let n = DVec3::new(0.0, 1.0, 0.2).normalize();
        let ground = Contact::plane(DVec3::Y, DVec3::ZERO, n);
        let mut ch = Chassis::new(DVec3::ZERO, n, DVec3::Z);
        run(&mut ch, Controls { brake: 1.0, handbrake: 1.0, ..Default::default() }, &ground, 4.0);
        let held = ch.pos;
        run(&mut ch, Controls { brake: 1.0, handbrake: 1.0, ..Default::default() }, &ground, 4.0);
        assert!((ch.pos - held).length() < 0.02, "glisse de {:.3} m", (ch.pos - held).length());
        run(&mut ch, Controls::default(), &ground, 3.0);
        // La normale penche vers +z : le sol descend vers l'avant.
        assert!(ch.vx() > 1.0, "roue libre, elle descend : {:.2}", ch.vx());
    }

    #[test]
    fn a_jump_lands_on_its_wheels() {
        let mut ch = car();
        ch.pos += DVec3::Y * 2.0;
        ch.launch(25.0);
        ch.gear = 4;
        let mut airborne = false;
        for _ in 0..300 {
            ch.update(&Controls { throttle: 0.3, ..Default::default() }, &flat(), 1.0 / 60.0);
            airborne |= ch.touching.iter().all(|t| !t);
        }
        assert!(airborne, "la voiture a vole");
        assert!(ch.up().y > 0.99 && (ch.pos.y - CG_HEIGHT).abs() < 0.05, "retombee : {} {}", ch.up(), ch.pos.y);
    }

    #[test]
    fn a_car_on_its_roof_stays_there() {
        let mut ch = Chassis::new(DVec3::Y * 1.6, -DVec3::Y, DVec3::Z);
        run(&mut ch, Controls::default(), &flat(), 3.0);
        assert!(ch.up().y < -0.9, "sur le toit : {}", ch.up());
        assert!(ch.pos.y > 0.3 && ch.speed() < 0.1, "pose sur le toit : {} {}", ch.pos.y, ch.speed());
    }
}
