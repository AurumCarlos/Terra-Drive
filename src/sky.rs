//! Position du soleil. L'axe des poles est Y ; l'heure locale depend donc de la longitude,
//! et une moitie de la planete est toujours dans la nuit.

use glam::{DVec3, Vec3};

pub struct Sky {
    /// Longitude ou il est midi, en radians.
    pub sun_longitude: f32,
    /// Inclinaison du soleil par rapport a l'equateur.
    pub declination: f32,
    /// Duree d'une journee en secondes ; 0 fige le soleil.
    pub day_length: f32,
    /// Temps ecoule, pour le vent, les vagues et les nuages.
    pub elapsed: f32,
}

impl Default for Sky {
    fn default() -> Self {
        Self { sun_longitude: 0.0, declination: 15f32.to_radians(), day_length: 1800.0, elapsed: 0.0 }
    }
}

fn longitude(up: DVec3) -> f32 {
    up.z.atan2(up.x) as f32
}

impl Sky {
    /// Fait passer `dt` secondes : le soleil ne tourne que si `day_length` n'est pas nul.
    pub fn advance(&mut self, dt: f32) {
        self.elapsed += dt;
        if self.day_length > 0.0 {
            self.shift_time(dt / self.day_length);
        }
    }

    /// Avance l'heure partout, en fractions de journee.
    pub fn shift_time(&mut self, days: f32) {
        self.sun_longitude = (self.sun_longitude - days * std::f32::consts::TAU).rem_euclid(std::f32::consts::TAU);
    }

    /// Vecteur unitaire vers le soleil.
    pub fn sun_dir(&self) -> Vec3 {
        let (sd, cd) = self.declination.sin_cos();
        let (sl, cl) = self.sun_longitude.sin_cos();
        Vec3::new(cd * cl, sd, cd * sl)
    }

    /// Heure locale a la verticale `up`, de 0 (minuit) a 1, 0.5 etant midi.
    pub fn local_time(&self, up: DVec3) -> f32 {
        (0.5 + (longitude(up) - self.sun_longitude) / std::f32::consts::TAU).rem_euclid(1.0)
    }

    /// Place le soleil pour qu'il soit `time` (0 a 1, 0,5 = midi) au lieu de verticale `up`.
    pub fn set_local_time(&mut self, up: DVec3, time: f32) {
        self.sun_longitude = (longitude(up) - (time - 0.5) * std::f32::consts::TAU).rem_euclid(std::f32::consts::TAU);
    }

    /// Heure du soir ou le soleil descend a `degrees` au-dessus de l'horizon en ce lieu.
    /// Sous les latitudes ou il ne s'y trouve jamais, rend la fin d'apres-midi.
    pub fn evening_time(&self, up: DVec3, degrees: f32) -> f32 {
        let target = degrees.to_radians().sin() as f64;
        let mut probe = Sky { ..*self };
        (0..250)
            .map(|k| 0.5 + k as f32 * 0.002)
            .find(|&time| {
                probe.set_local_time(up, time);
                probe.sun_dir().as_dvec3().dot(up) <= target
            })
            .filter(|&time| time > 0.5)
            .unwrap_or(0.7)
    }

    /// Direction du soleil a l'heure dite du soir, vue du lieu donne.
    pub fn evening_sun(&self, up: DVec3, degrees: f32) -> DVec3 {
        let mut probe = Sky { ..*self };
        probe.set_local_time(up, self.evening_time(up, degrees));
        probe.sun_dir().as_dvec3()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noon_is_overhead_and_midnight_is_under_the_feet() {
        let up = DVec3::new(0.6, 0.3, -0.74).normalize();
        let mut s = Sky::default();
        s.set_local_time(up, 0.5);
        assert!((s.local_time(up) - 0.5).abs() < 1e-4);
        assert!(s.sun_dir().as_dvec3().dot(up) > 0.7);
        s.set_local_time(up, 0.0);
        assert!(s.sun_dir().as_dvec3().dot(up) < -0.3);
    }

    #[test]
    fn evening_time_puts_the_sun_low_in_the_west() {
        let up = DVec3::new(0.5, 0.6, -0.62).normalize();
        let mut s = Sky::default();
        let time = s.evening_time(up, 6.0);
        assert!(time > 0.5 && time < 1.0);
        s.set_local_time(up, time);
        let elevation = s.sun_dir().as_dvec3().dot(up).asin().to_degrees();
        assert!((elevation - 6.0).abs() < 1.0, "{elevation}");
    }

    #[test]
    fn time_moves_forward_at_a_fixed_place() {
        let up = DVec3::X;
        let mut s = Sky::default();
        s.set_local_time(up, 0.3);
        s.shift_time(0.1);
        assert!((s.local_time(up) - 0.4).abs() < 1e-4);
    }
}
