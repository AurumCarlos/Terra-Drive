//! Saisons et temps qu'il fait la ou se trouve la camera. L'annee fait tourner l'inclinaison du soleil
//! et les temperatures de chaque lieu entre celles de janvier et de juillet que donne la simulation du
//! climat. Des perturbations derivent avec le temps ; il pleut ou il neige sous elles, d'autant plus
//! souvent que le lieu est arrose a cette saison.

use crate::world::generator::{planet_radius, smoothstep, Generator};
use crate::world::noise::{hash_seed, Noise};
use glam::DVec3;

/// Inclinaison de l'axe de la planete, comme celui de la Terre.
pub const OBLIQUITY: f32 = 23.44;
/// Jours de jeu par an, par defaut : un mois de journees de jeu fait une annee.
pub const YEAR_DAYS: f32 = 30.0;
/// Moment de l'annee de l'equinoxe de mars, depuis la mi-janvier.
const SPRING: f32 = 0.175;
/// Taille des perturbations et des averses, en metres.
const FRONT_SCALE: f64 = 320_000.0;
const SHOWER_SCALE: f64 = 40_000.0;
/// Vitesse de derive des perturbations, en metres par seconde de temps du monde.
const FRONT_SPEED: f64 = 12.0;
/// Au-dessus des nuages, on ne voit plus la pluie.
const CLOUD_TOP: f64 = 2_600.0;

pub struct Weather {
    /// Moment de l'annee, de 0 a 1 : 0 a la mi-janvier, 0,5 a la mi-juillet.
    pub season: f32,
    /// Jours de jeu par an ; 0 fige la saison.
    pub year_days: f32,
    /// Precipitations autour de la camera, de 0 a 1.
    pub rain: f32,
    /// Part de neige dans ces precipitations.
    pub snow: f32,
    /// Sol mouille, de 0 a 1 : monte sous la pluie, seche ensuite.
    pub wet: f32,
    /// Temperature de l'air a la camera, en degres.
    pub temperature: f32,
    /// Neige au sol sous la camera, de 0 a 1.
    pub snow_cover: f32,
    /// La camera est a l'abri (dans un tunnel) : il ne pleut pas devant elle.
    pub sheltered: bool,
    /// Temps du monde ecoule, en secondes : les perturbations derivent avec lui.
    world_time: f64,
    noise: Option<(u64, Noise, Vec<f32>)>,
}

/// Pour les captures : TERRA_MONTH (1 a 12) fixe le mois, TERRA_RAIN (0 a 1) la pluie.
fn forced(name: &str) -> Option<f32> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

impl Default for Weather {
    fn default() -> Self {
        Self {
            season: forced("TERRA_MONTH").map_or(0.52, |m| ((m - 1.0) / 12.0).rem_euclid(1.0)),
            year_days: YEAR_DAYS,
            rain: 0.0,
            snow: 0.0,
            wet: 0.0,
            temperature: 15.0,
            snow_cover: 0.0,
            sheltered: false,
            world_time: 0.0,
            noise: None,
        }
    }
}

/// Valeurs du bruit des perturbations, triees : leur rang donne la part du temps ou il pleut.
fn front_quantiles(noise: &Noise) -> Vec<f32> {
    let mut values: Vec<f32> = (0..4096)
        .map(|k| {
            let a = k as f64 * 2.399963;
            let z = 1.0 - 2.0 * (k as f64 + 0.5) / 4096.0;
            let r = (1.0 - z * z).sqrt();
            front_value(noise, DVec3::new(a.cos() * r, z, a.sin() * r), 0.0)
        })
        .collect();
    values.sort_by(f32::total_cmp);
    values
}

fn front_value(noise: &Noise, dir: DVec3, drift: f64) -> f32 {
    let p = dir * planet_radius();
    let shift = DVec3::new(drift, 0.0, drift * 0.4);
    0.7 * noise.fbm((p + shift) / FRONT_SCALE, 4) + 0.3 * noise.fbm((p + shift * 1.4) / SHOWER_SCALE + 50.0, 3)
}

impl Weather {
    /// Inclinaison du soleil par rapport a l'equateur a ce moment de l'annee.
    pub fn declination(&self) -> f32 {
        (OBLIQUITY.to_radians()) * (std::f32::consts::TAU * (self.season - SPRING)).sin()
    }

    /// Ecart a la moyenne de l'annee, de -1 (plein hiver) a 1 (plein ete), dans l'hemisphere de `dir`.
    pub fn summer(&self, dir: DVec3) -> f32 {
        let north = -(std::f32::consts::TAU * self.season).cos();
        if dir.y >= 0.0 {
            north
        } else {
            -north
        }
    }

    /// Fait passer `days` jours de jeu, soit `seconds` secondes du monde.
    pub fn advance(&mut self, days: f32, seconds: f64) {
        if self.year_days > 0.0 {
            self.season = (self.season + days / self.year_days).rem_euclid(1.0);
        }
        self.world_time += seconds;
    }

    /// Met a jour le temps qu'il fait autour de la camera `cam`, apres `dt` secondes reelles.
    pub fn update(&mut self, gen: &Generator, cam: DVec3, dt: f32) {
        let seed = gen.params.seed;
        if self.noise.as_ref().is_none_or(|n| n.0 != seed) {
            let noise = Noise::new(hash_seed(seed, 0x3E7, 1));
            let levels = front_quantiles(&noise);
            self.noise = Some((seed, noise, levels));
        }
        let Some((_, noise, levels)) = &self.noise else { return };
        let dir = cam.normalize();
        let ground = gen.height(dir).max(0.0);
        let altitude = cam.length() - planet_radius();
        let climate = gen.climate(dir, ground);

        // Temperature et pluie du mois en cours.
        let summer = self.summer(dir);
        let half = (climate.hot - climate.cold) * 0.5;
        let air = climate.mean + half * summer - crate::world::climate::LAPSE * (altitude as f32 - ground).max(0.0);
        let month = climate.rain / 12.0 * (1.0 + (2.0 * climate.summer - 1.0) * summer).max(0.0);
        // Part du temps ou il tombe quelque chose : quelques heures par mois au desert, un jour sur trois
        // sous les tropiques humides.
        let chance = 0.5 * month / (month + 120.0);

        let value = front_value(noise, dir, self.world_time * FRONT_SPEED);
        let rank = levels.partition_point(|v| *v < value) as f32 / levels.len() as f32;
        let start = 1.0 - chance;
        let mut target = smoothstep(start, (start + 0.12).min(1.0), rank);
        // Au-dessus des nuages, plus rien ne tombe.
        let above = (altitude - ground as f64) as f32;
        target *= 1.0 - smoothstep(CLOUD_TOP as f32 * 0.6, CLOUD_TOP as f32, above);

        let k = 1.0 - (-dt / 12.0).exp();
        self.rain += (target - self.rain) * k;
        if let Some(rain) = forced("TERRA_RAIN") {
            self.rain = rain.clamp(0.0, 1.0);
            self.wet = self.wet.max(rain.min(1.0));
        }
        self.temperature = air;
        self.snow = smoothstep(2.0, -0.5, air);
        // Le sol se mouille en quelques minutes de pluie, seche en une dizaine.
        let liquid = self.rain * (1.0 - self.snow);
        if liquid > 0.05 {
            self.wet = (self.wet + dt * liquid / 60.0).min(1.0);
        } else {
            let drying = 1.0 / (900.0 - 600.0 * smoothstep(5.0, 30.0, air));
            self.wet = (self.wet - dt * drying).max(0.0);
        }
        // Neige au sol : un hiver assez froid et assez de precipitations, comme dans le shader du terrain.
        let ground_air = climate.mean + half * summer;
        self.snow_cover = smoothstep(0.5, -3.5, ground_air) * smoothstep(45.0, 180.0, climate.rain);
    }

    /// Le temps qu'il fait, en quelques mots : mois, temperature, ciel.
    pub fn describe(&self) -> String {
        const MONTHS: [&str; 12] = [
            "janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre",
            "novembre", "decembre",
        ];
        let month = MONTHS[((self.season * 12.0).round() as usize) % 12];
        let sky = if self.rain > 0.3 && self.snow > 0.5 {
            "neige"
        } else if self.rain > 0.7 {
            "forte pluie"
        } else if self.rain > 0.3 {
            "pluie"
        } else if self.rain > 0.08 {
            "averses"
        } else if self.snow_cover > 0.5 {
            "sec, sol enneige"
        } else if self.wet > 0.3 {
            "sec, sol mouille"
        } else {
            "sec"
        };
        format!("{month}, {:.0} C, {sky}", self.temperature)
    }

    /// Adherence des pneus, de 1 sur sol sec a 0,45 sur la neige.
    pub fn grip(&self) -> f32 {
        let wet = 1.0 - 0.22 * self.wet;
        let snowy = self.snow_cover.max(self.rain * self.snow);
        wet * (1.0 - 0.55 * snowy)
    }
}

/// Le climat d'un lieu tel que le lit le shader : nebulosite, temperature moyenne au niveau de la mer
/// (de -50 a 40 degres), demi-ecart entre janvier et juillet (de -30 a 30, positif si janvier est l'hiver)
/// et pluie de l'annee (jusqu'a 3 000 mm).
pub fn encode_climate(gen: &Generator, dir: DVec3) -> [f32; 4] {
    let c = gen.weather.at(dir, 0.0, 1.0);
    let half = (c.hot - c.cold) * 0.5 * if dir.y >= 0.0 { 1.0 } else { -1.0 };
    [gen.weather.cloud.sample_linear(dir), (c.mean + 50.0) / 90.0, (half + 30.0) / 60.0, c.rain / 3000.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::generator::PlanetParams;

    #[test]
    fn the_sun_climbs_in_summer_and_seasons_are_opposite_across_the_equator() {
        let mut w = Weather { season: 0.5, ..Default::default() };
        assert!(w.declination().to_degrees() > 18.0);
        w.season = 0.0;
        assert!(w.declination().to_degrees() < -18.0);
        assert!(w.summer(DVec3::Y) < -0.99 && w.summer(-DVec3::Y) > 0.99);
        w.advance(YEAR_DAYS * 0.5, 0.0);
        assert!((w.season - 0.5).abs() < 1e-4);
    }

    /// Sur toute une planete, il pleut une part raisonnable du temps, et davantage la ou il pleut beaucoup.
    #[test]
    fn it_rains_more_often_where_the_climate_is_wet() {
        let gen = Generator::new(PlanetParams { seed: 4, ..Default::default() });
        let (mut wet_hits, mut wet_n, mut dry_hits, mut dry_n) = (0, 0, 0, 0);
        for k in 0..600 {
            let a = k as f64 * 2.399963;
            let z = 1.0 - 2.0 * (k as f64 + 0.5) / 600.0;
            let r = (1.0 - z * z).sqrt();
            let dir = DVec3::new(a.cos() * r, z, a.sin() * r);
            let h = gen.height(dir);
            if h <= 0.0 {
                continue;
            }
            let rain = gen.climate(dir, h).rain;
            for t in 0..20 {
                let mut w = Weather::default();
                w.advance(0.0, t as f64 * 40_000.0);
                for _ in 0..40 {
                    w.update(&gen, dir * (planet_radius() + h as f64 + 2.0), 1.0);
                }
                let raining = w.rain > 0.5;
                if rain > 1500.0 {
                    wet_n += 1;
                    wet_hits += raining as u32;
                } else if rain < 300.0 {
                    dry_n += 1;
                    dry_hits += raining as u32;
                }
            }
        }
        let (wet, dry) = (wet_hits as f32 / wet_n.max(1) as f32, dry_hits as f32 / dry_n.max(1) as f32);
        assert!(wet_n > 40 && dry_n > 40, "{wet_n} essais humides, {dry_n} secs");
        assert!(wet > dry * 2.0 && wet > 0.08 && wet < 0.6, "pluie {wet} du temps en pays humide, {dry} en pays sec");
    }
}
