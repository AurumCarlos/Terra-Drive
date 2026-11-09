//! Camera a position f64 dans le repere de la planete. Tout le rendu se fait relativement
//! a elle, ce qui evite le tremblement des f32. Le "haut" est toujours la verticale locale.

use glam::{DVec3, Mat4, Vec3};

pub struct Camera {
    pub position: DVec3,
    /// Direction horizontale du regard, tangente a la sphere.
    pub heading: DVec3,
    /// Inclinaison au-dessus de l'horizon local.
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
}

const MAX_PITCH: f32 = 1.55;

impl Camera {
    /// Camera en `position`, `yaw` radians a droite du nord.
    pub fn new(position: DVec3, yaw: f32, pitch: f32) -> Self {
        let up = position.normalize_or(DVec3::Y);
        let axis = if up.y.abs() > 0.99 { DVec3::Z } else { DVec3::Y };
        let north = (axis - up * axis.dot(up)).normalize();
        let east = north.cross(up);
        let (s, c) = (yaw as f64).sin_cos();
        Self {
            position,
            heading: north * c + east * s,
            pitch: pitch.clamp(-MAX_PITCH, MAX_PITCH),
            fov_y: 65f32.to_radians(),
            near: 0.1,
        }
    }

    /// Verticale du lieu : du centre de la planete vers la camera.
    pub fn up(&self) -> DVec3 {
        self.position.normalize_or(DVec3::Y)
    }

    /// Direction du regard, en double precision.
    pub fn forward_d(&self) -> DVec3 {
        let (s, c) = (self.pitch as f64).sin_cos();
        self.heading * c + self.up() * s
    }

    /// Direction du regard.
    pub fn forward(&self) -> Vec3 {
        self.forward_d().as_vec3()
    }

    /// Vers la droite du regard, a l'horizontale.
    pub fn right(&self) -> DVec3 {
        self.heading.cross(self.up())
    }

    /// Tourne le regard : `yaw` vers la droite, `pitch` vers le haut.
    pub fn turn(&mut self, yaw: f32, pitch: f32) {
        let (s, c) = (yaw as f64).sin_cos();
        self.heading = (self.heading * c + self.right() * s).normalize();
        self.pitch = (self.pitch + pitch).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Deplace la camera et recale son repere sur la nouvelle verticale.
    /// `world_lock` : 0 suit la courbure (vol au ras du sol), 1 garde le regard fixe dans
    /// l'espace (trajectoire droite, indispensable pour quitter la planete).
    pub fn move_by(&mut self, delta: DVec3, world_lock: f32) {
        let forward = self.forward_d();
        self.position += delta;
        let up = self.up();

        let carried = (self.heading - up * self.heading.dot(up)).normalize_or(self.heading);
        let kept = forward - up * forward.dot(up);
        let lock = world_lock as f64;
        self.heading = if kept.length_squared() > 1e-4 {
            (carried * (1.0 - lock) + kept.normalize() * lock).normalize()
        } else {
            carried
        };
        let world_pitch = forward.dot(up).clamp(-1.0, 1.0).asin() as f32;
        self.pitch = (self.pitch + (world_pitch - self.pitch) * world_lock).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Matrice vue-projection relative a la camera : Z inverse, plan lointain a l'infini.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_to_rh(Vec3::ZERO, self.forward(), self.up().as_vec3());
        Mat4::perspective_infinite_reverse_rh(self.fov_y, aspect, self.near) * view
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_up_flight_keeps_looking_up() {
        let mut c = Camera::new(DVec3::new(6.4e6, 0.0, 0.0), 0.3, 1.5);
        let start = c.forward_d();
        for _ in 0..100 {
            let f = c.forward_d();
            c.move_by(f * 1.0e5, 1.0);
        }
        assert!(c.forward_d().dot(start) > 0.999);
    }

    #[test]
    fn locked_flight_goes_in_a_straight_line() {
        let mut c = Camera::new(DVec3::new(7.0e6, 0.0, 0.0), 1.0, 0.2);
        let start = c.position;
        let dir = c.forward_d();
        for _ in 0..1000 {
            let f = c.forward_d();
            c.move_by(f * 2.0e3, 1.0);
        }
        let done = (c.position - start).normalize();
        assert!(done.dot(dir) > 0.9999, "{}", done.dot(dir));
    }

    #[test]
    fn low_flight_follows_the_curve() {
        // Autour de toute la planete a pitch nul, sans s'en eloigner.
        let r = 6.4e6;
        let mut c = Camera::new(DVec3::new(r, 0.0, 0.0), 0.7, 0.0);
        for _ in 0..4000 {
            let f = c.forward_d();
            c.move_by(f * 10_000.0, 0.0);
            c.position = c.position.normalize() * r;
        }
        assert!(c.pitch.abs() < 1e-3);
        assert!(c.heading.dot(c.up()).abs() < 1e-6);
        assert!(c.heading.is_normalized());
    }

    #[test]
    fn turning_right_goes_to_the_right() {
        let mut c = Camera::new(DVec3::new(6.4e6, 0.0, 0.0), 0.0, 0.0);
        let right = c.right();
        c.turn(0.1, 0.0);
        assert!(c.forward_d().dot(right) > 0.09);
    }
}
