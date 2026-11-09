//! Son : tout est synthetise en direct, sans fichier audio.
//!
//! - Voiture : moteur V8 (harmoniques de l'allumage selon le regime, bruit d'echappement a chaque
//!   explosion, filtre qui s'ouvre avec la charge, limiteur, petarades au leve de pied), passages de
//!   rapport, crissement des pneus, bruit de roulement (asphalte ou gravier), vent, chocs.
//! - Ambiance : vent en rafales, feuillages, oiseaux le jour, grillons la nuit, vagues, rumeur de ville,
//!   souffle de la vitesse en vol.
//!
//! Le jeu remplit `SoundParams` a chaque image ; le thread audio lit la derniere version et lisse les
//! changements. `Synth` ne depend pas de la carte son : on peut le tester et l'ecouter hors ligne.

use std::f32::consts::TAU;
use std::sync::{Arc, Mutex};

/// La voiture conduite.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CarSound {
    /// Regime (tr/min) et charge du moteur (0 a 1).
    pub rpm: f32,
    pub load: f32,
    /// Rapport engage : un changement declenche le bruit de passage.
    pub gear: i32,
    /// Vitesse (m/s).
    pub speed: f32,
    /// Glissement des pneus (0 : adherence, 1 : derapage franc).
    pub slip: f32,
    /// Part des roues hors de l'asphalte (0 a 1).
    pub rough: f32,
    /// Vue du conducteur : sons etouffes par l'habitacle.
    pub cockpit: bool,
}

/// Ce que le jeu veut entendre a cette image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoundParams {
    pub car: Option<CarSound>,
    /// Choc : numero (un nouveau numero declenche le son) et vitesse d'impact (m/s).
    pub impact: (u32, f32),
    /// Vent de fond (0 a 1), et souffle de la vitesse en vol (m/s de la camera).
    pub wind: f32,
    pub rush: f32,
    /// Arbres autour : feuillages et oiseaux (0 a 1).
    pub leaves: f32,
    pub birds: f32,
    /// Insectes de la nuit (0 a 1).
    pub insects: f32,
    /// Vagues d'un rivage et eau qui coule (0 a 1).
    pub waves: f32,
    pub stream: f32,
    /// Rumeur de ville (0 a 1).
    pub city: f32,
    /// Volume general (0 a 1).
    pub volume: f32,
}

impl Default for SoundParams {
    fn default() -> Self {
        Self {
            car: None,
            impact: (0, 0.0),
            wind: 0.2,
            rush: 0.0,
            leaves: 0.0,
            birds: 0.0,
            insects: 0.0,
            waves: 0.0,
            stream: 0.0,
            city: 0.0,
            volume: 0.8,
        }
    }
}

/// Generateur pseudo-aleatoire (xorshift) : rapide et sans dependance.
#[derive(Clone)]
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniforme dans [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Bruit blanc dans [-1, 1).
    fn noise(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }
}

/// Filtre passe-bas a un pole.
#[derive(Clone, Copy, Default)]
struct Lowpass(f32);

impl Lowpass {
    fn run(&mut self, x: f32, cutoff: f32, rate: f32) -> f32 {
        // 1 - exp(-w) sans exponentielle : w / (1 + w), assez juste pour des coupures basses.
        let w = TAU * cutoff / rate;
        let a = w / (1.0 + w);
        self.0 += a * (x - self.0);
        self.0
    }
}

/// Filtre a variable d'etat (Chamberlin) : passe-bas, passe-bande, passe-haut.
#[derive(Clone, Copy, Default)]
struct Svf {
    low: f32,
    band: f32,
}

impl Svf {
    /// `q` : amortissement (petit = resonant). Renvoie (bas, bande, haut).
    fn run(&mut self, x: f32, cutoff: f32, q: f32, rate: f32) -> (f32, f32, f32) {
        // 2 sin(x) par son developpement : pas de sinus par echantillon.
        let w = std::f32::consts::PI * cutoff.min(rate * 0.16) / rate;
        let f = 2.0 * w * (1.0 - w * w / 6.0);
        self.low += f * self.band;
        let high = x - self.low - q * self.band;
        self.band += f * high;
        // Securite : un filtre qui diverge ne doit jamais sortir de bruit infini.
        if !self.band.is_finite() || !self.low.is_finite() {
            *self = Self::default();
        }
        (self.low, self.band, high)
    }
}

/// Approche une cible avec une constante de temps (s).
fn approach(v: &mut f32, target: f32, tau: f32, dt: f32) {
    *v += (target - *v) * (1.0 - (-dt / tau).exp());
}

/// Un oiseau : une phrase de quelques notes glissees, ou un trille.
#[derive(Clone, Copy, Default)]
struct Bird {
    notes: u32,
    t: f32,
    duration: f32,
    gap: f32,
    f0: f32,
    f1: f32,
    trill: f32,
    amp: f32,
    pan: f32,
    phase: f32,
}

/// Un grillon : des stridulations groupees.
#[derive(Clone, Copy, Default)]
struct Cricket {
    freq: f32,
    rate: f32,
    pulses: f32,
    phase: f32,
    clock: f32,
    pan: f32,
}

/// Amplitudes des harmoniques du cycle moteur (deux tours) : l'ordre 8 est l'allumage d'un V8, l'ordre 4
/// le grondement du vilebrequin croise, les autres colorent le timbre.
const ENGINE_HARMONICS: [f32; 24] = [
    0.05, 0.2, 0.08, 0.55, 0.1, 0.25, 0.07, 1.0, 0.12, 0.3, 0.06, 0.45, 0.05, 0.15, 0.04, 0.35, 0.03, 0.1, 0.03, 0.15,
    0.02, 0.06, 0.02, 0.1,
];

/// Dephasage de chaque harmonique (sinus, cosinus de k^2 * 0,7) : un timbre rauque, pas une sirene.
const ENGINE_OFFSETS: [(f32, f32); 24] = {
    let mut t = [(0.0f32, 0.0f32); 24];
    let mut k = 0;
    while k < 24 {
        let k1 = (k + 1) as f64;
        let x = (k1 * k1 * 0.7) % std::f64::consts::TAU;
        t[k] = (sin_const(x) as f32, sin_const(x + std::f64::consts::FRAC_PI_2) as f32);
        k += 1;
    }
    t
};

/// Sinus calculable a la compilation (serie de Taylor apres reduction a [-pi, pi]).
const fn sin_const(x: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let mut x = x % (2.0 * pi);
    if x > pi {
        x -= 2.0 * pi;
    }
    let (mut term, mut sum, mut n) = (x, x, 1);
    while n < 15 {
        term = -term * x * x / ((2 * n) as f64 * (2 * n + 1) as f64);
        sum += term;
        n += 1;
    }
    sum
}

/// Le synthetiseur : un echantillon stereo a la fois.
pub struct Synth {
    rate: f32,
    rng: Rng,
    time: f32,
    /// Parametres lisses.
    p: SoundParams,
    target: SoundParams,
    car_level: f32,
    rpm: f32,
    load: f32,
    speed: f32,
    slip: f32,
    rough: f32,
    muffle: f32,
    engine_phase: f32,
    engine_lp: Svf,
    exhaust_lp: Lowpass,
    intake: Svf,
    limiter: f32,
    pops: f32,
    gear: i32,
    shift: f32,
    road: [Lowpass; 2],
    gravel: Svf,
    squeal: Svf,
    squeal_phase: f32,
    car_wind: [Svf; 2],
    impact_id: u32,
    impact: f32,
    impact_lp: Lowpass,
    thump_phase: f32,
    wind: [Svf; 2],
    gust: f32,
    gust_target: f32,
    leaves: [Svf; 2],
    rush: [Lowpass; 2],
    waves: [Lowpass; 2],
    swell: f32,
    stream: [Svf; 2],
    city: [Lowpass; 2],
    horn: f32,
    horn_phase: f32,
    birds: [Bird; 3],
    crickets: [Cricket; 4],
    cabin: [Lowpass; 2],
}

impl Synth {
    /// Synthetiseur pour une sortie a `rate` echantillons par seconde.
    pub fn new(rate: f32) -> Self {
        let mut rng = Rng(0x9E37_79B9);
        let crickets = std::array::from_fn(|k| Cricket {
            freq: rng.range(3800.0, 4800.0),
            rate: rng.range(1.2, 2.4),
            pulses: (3 + k % 3) as f32,
            phase: 0.0,
            clock: rng.unit(),
            pan: rng.range(-0.8, 0.8),
        });
        Self {
            rate,
            rng,
            time: 0.0,
            p: SoundParams { volume: 0.0, wind: 0.0, ..Default::default() },
            target: SoundParams::default(),
            car_level: 0.0,
            rpm: 850.0,
            load: 0.0,
            speed: 0.0,
            slip: 0.0,
            rough: 0.0,
            muffle: 0.0,
            engine_phase: 0.0,
            engine_lp: Svf::default(),
            exhaust_lp: Lowpass::default(),
            intake: Svf::default(),
            limiter: 0.0,
            pops: 0.0,
            gear: 1,
            shift: 0.0,
            road: Default::default(),
            gravel: Svf::default(),
            squeal: Svf::default(),
            squeal_phase: 0.0,
            car_wind: Default::default(),
            impact_id: 0,
            impact: 0.0,
            impact_lp: Lowpass::default(),
            thump_phase: 0.0,
            wind: Default::default(),
            gust: 0.5,
            gust_target: 0.5,
            leaves: Default::default(),
            rush: Default::default(),
            waves: Default::default(),
            swell: 0.0,
            stream: Default::default(),
            city: Default::default(),
            horn: 0.0,
            horn_phase: 0.0,
            birds: Default::default(),
            crickets,
            cabin: Default::default(),
        }
    }

    /// Recoit les reglages de l'image en cours ; le son les rejoint en douceur, sans clic.
    pub fn set(&mut self, params: SoundParams) {
        if params.impact.0 != self.impact_id {
            self.impact_id = params.impact.0;
            self.impact = (params.impact.1 / 12.0).clamp(0.0, 1.0).max(self.impact);
        }
        if let Some(car) = params.car {
            if car.gear != self.gear {
                self.gear = car.gear;
                self.shift = 1.0;
            }
        }
        self.target = params;
    }

    /// Remplit un tampon entrelace de `channels` voies.
    pub fn fill(&mut self, out: &mut [f32], channels: usize) {
        // Les parametres d'ambiance se lissent une fois par tampon (quelques ms).
        let dt = (out.len() / channels.max(1)) as f32 / self.rate;
        let (p, t) = (&mut self.p, &self.target);
        approach(&mut p.volume, t.volume, 0.2, dt);
        approach(&mut p.wind, t.wind, 1.5, dt);
        approach(&mut p.rush, t.rush, 0.3, dt);
        approach(&mut p.leaves, t.leaves, 1.5, dt);
        approach(&mut p.birds, t.birds, 2.0, dt);
        approach(&mut p.insects, t.insects, 2.0, dt);
        approach(&mut p.waves, t.waves, 1.5, dt);
        approach(&mut p.stream, t.stream, 1.5, dt);
        approach(&mut p.city, t.city, 1.5, dt);
        let car = t.car;
        approach(&mut self.car_level, car.is_some() as i32 as f32, 0.25, dt);
        if let Some(c) = car {
            approach(&mut self.rpm, c.rpm, 0.03, dt);
            approach(&mut self.load, c.load, 0.05, dt);
            approach(&mut self.speed, c.speed, 0.1, dt);
            approach(&mut self.slip, c.slip, 0.06, dt);
            approach(&mut self.rough, c.rough, 0.15, dt);
            approach(&mut self.muffle, c.cockpit as i32 as f32, 0.1, dt);
        }
        for frame in out.chunks_mut(channels.max(1)) {
            let (l, r) = self.sample();
            match frame.len() {
                1 => frame[0] = (l + r) * 0.5,
                _ => {
                    frame[0] = l;
                    frame[1] = r;
                    for x in &mut frame[2..] {
                        *x = 0.0;
                    }
                }
            }
        }
    }

    fn sample(&mut self) -> (f32, f32) {
        let dt = 1.0 / self.rate;
        let rate = self.rate;
        self.time += dt;
        let (mut l, mut r) = (0.0f32, 0.0f32);

        // ------------------------------------------------------------------ voiture
        if self.car_level > 1e-3 {
            let level = self.car_level;
            let (rpm, load) = (self.rpm.max(300.0), self.load);
            // Moteur : cycle a quatre temps, deux tours.
            self.engine_phase = (self.engine_phase + rpm / 120.0 * dt).fract();
            let ph = self.engine_phase * TAU;
            // Harmoniques par recurrence : sin(k ph) et cos(k ph) tournent d'un pas a chaque rang.
            let mut tone = 0.0;
            let top = (rate * 0.4 / (rpm / 120.0)).min(24.0) as usize;
            let (s1, c1) = ph.sin_cos();
            let (mut sk, mut ck) = (s1, c1);
            for k in 0..top {
                let (so, co) = ENGINE_OFFSETS[k];
                tone += ENGINE_HARMONICS[k] * (sk * co + ck * so);
                (sk, ck) = (sk * c1 + ck * s1, ck * c1 - sk * s1);
            }
            // Une explosion par huitieme de cycle, un peu irreguliere : le grondement.
            let fire = (ph * 8.0).cos().max(0.0).powi(6);
            let burble = 1.0 + 0.35 * (ph * 4.0).sin();
            let exhaust = self.exhaust_lp.run(self.rng.noise(), 900.0 + rpm * 0.3, rate) * fire * burble;
            // Limiteur : le moteur coupe l'allumage par a-coups.
            if rpm > 7300.0 {
                self.limiter = (self.limiter + dt * 18.0).fract();
            } else {
                self.limiter = 0.0;
            }
            let cut = if self.limiter > 0.5 { 0.35 } else { 1.0 };
            // Petarades : pied leve a haut regime.
            if load < 0.1 && rpm > 4000.0 && self.rng.unit() < 0.00025 {
                self.pops = 1.0;
            }
            self.pops *= 1.0 - 60.0 * dt;
            let pop = self.pops * self.rng.noise() * 0.8;
            let cutoff = 250.0 + rpm * 0.22 + load * 2200.0;
            let (low, _, _) = self.engine_lp.run(
                tone * 0.18 * (0.35 + 0.65 * load) + exhaust * (0.6 + 1.6 * load) + pop,
                cutoff,
                0.5,
                rate,
            );
            // Admission : un souffle qui siffle avec la charge.
            let (_, band, _) = self.intake.run(self.rng.noise(), 1800.0 + rpm * 0.25, 0.3, rate);
            let engine = (low * (0.55 + 0.45 * load) + band * 0.03 * load * rpm / 7000.0) * cut;
            // Passage de rapport : un claquement sourd et bref.
            self.shift *= 1.0 - 25.0 * dt;
            self.thump_phase = (self.thump_phase + 55.0 * dt).fract();
            let shift = self.shift * (self.thump_phase * TAU).sin() * 0.25;

            // Roulement : grondement qui monte avec la vitesse ; graviers hors de l'asphalte.
            let v = self.speed;
            let roll_amp = (v / 35.0).min(1.0) * (0.12 + 0.25 * self.rough);
            let road_l = self.road[0].run(self.rng.noise(), 60.0 + v * 9.0, rate) * roll_amp;
            let road_r = self.road[1].run(self.rng.noise(), 60.0 + v * 9.0, rate) * roll_amp;
            let crackle =
                if self.rng.unit() < self.rough * (v / 20.0).min(1.0) * 0.02 { self.rng.noise() * 2.0 } else { 0.0 };
            let (_, gravel, _) = self.gravel.run(self.rng.noise() * 0.4 + crackle, 2500.0, 0.8, rate);
            let gravel = gravel * self.rough * (v / 15.0).min(1.0) * 0.4;

            // Pneus qui crissent : sur l'asphalte seulement.
            let squeal_amp = (self.slip * (1.0 - self.rough) * (v / 5.0).min(1.0)).clamp(0.0, 1.0);
            let mut squeal = 0.0;
            if squeal_amp > 1e-3 {
                let f = 950.0 + 120.0 * self.slip + 40.0 * (self.time * 7.0).sin();
                self.squeal_phase = (self.squeal_phase + f * dt).fract();
                let (_, band, _) = self.squeal.run(self.rng.noise(), 1300.0, 0.08, rate);
                squeal = (band * 0.5 + (self.squeal_phase * TAU).sin() * 0.25) * squeal_amp * 0.5;
            }

            // Vent sur la carrosserie.
            let wind_amp = (v / 55.0).powi(2).min(1.5) * 0.25;
            let (_, _, wl) = self.car_wind[0].run(self.rng.noise(), 500.0 + v * 15.0, 0.7, rate);
            let (_, _, wr) = self.car_wind[1].run(self.rng.noise(), 500.0 + v * 15.0, 0.7, rate);

            // Chocs : bruit sourd de tole et coup grave.
            self.impact *= 1.0 - 9.0 * dt;
            let hit = self.impact_lp.run(self.rng.noise(), 700.0, rate) * self.impact * 1.6
                + (self.time * TAU * 48.0).sin() * self.impact * 0.5;

            let shared = engine + shift + gravel + squeal + hit;
            let (mut cl, mut cr) = (shared + road_l + wl * wind_amp, shared + road_r + wr * wind_amp);
            // Dans l'habitacle : les aigus sont etouffes, le moteur reste present.
            if self.muffle > 0.01 {
                let m = self.muffle;
                let fl = self.cabin[0].run(cl, 1400.0, rate);
                let fr = self.cabin[1].run(cr, 1400.0, rate);
                cl = cl * (1.0 - m) + fl * m * 1.1;
                cr = cr * (1.0 - m) + fr * m * 1.1;
            }
            l += cl * level;
            r += cr * level;
        }

        // ------------------------------------------------------------------ ambiance
        let p = self.p;
        // Rafales : une cible au hasard toutes les quelques secondes.
        if self.rng.unit() < dt * 0.4 {
            self.gust_target = self.rng.range(0.15, 1.0);
        }
        approach(&mut self.gust, self.gust_target, 1.2, dt);
        let outside = 1.0 - 0.75 * self.muffle * self.car_level;
        if p.wind > 1e-3 {
            let f = 250.0 + 500.0 * self.gust;
            let (_, a, _) = self.wind[0].run(self.rng.noise(), f, 0.9, rate);
            let (_, b, _) = self.wind[1].run(self.rng.noise(), f * 1.07, 0.9, rate);
            let amp = p.wind * (0.25 + 0.75 * self.gust) * 0.8 * outside;
            l += a * amp;
            r += b * amp;
        }
        if p.leaves > 1e-3 {
            let (_, a, _) = self.leaves[0].run(self.rng.noise(), 3200.0, 1.2, rate);
            let (_, b, _) = self.leaves[1].run(self.rng.noise(), 3600.0, 1.2, rate);
            let amp = p.leaves * (0.2 + self.gust * self.gust) * 0.5 * outside;
            l += a * amp;
            r += b * amp;
        }
        if p.rush > 0.5 {
            let f = 150.0 + p.rush.min(400.0) * 4.0;
            let amp = (p.rush / 120.0).min(1.0) * 0.3;
            l += self.rush[0].run(self.rng.noise(), f, rate) * amp;
            r += self.rush[1].run(self.rng.noise(), f, rate) * amp;
        }
        if p.waves > 1e-3 {
            // Houle : une vague toutes les 6 a 9 s, qui deferle puis se retire.
            self.swell = (self.swell + dt / 7.5).fract();
            let crest = (self.swell * TAU).sin().max(0.0).powi(2) + 0.15;
            let f = 300.0 + 900.0 * crest;
            let amp = p.waves * crest * 1.1 * outside;
            l += self.waves[0].run(self.rng.noise(), f, rate) * amp;
            r += self.waves[1].run(self.rng.noise(), f * 0.9, rate) * amp;
        }
        if p.stream > 1e-3 {
            let (_, a, _) = self.stream[0].run(self.rng.noise(), 1100.0, 0.6, rate);
            let (_, b, _) = self.stream[1].run(self.rng.noise(), 1300.0, 0.6, rate);
            let bubble = 1.0 + 0.4 * (self.time * 5.3).sin() * (self.time * 3.1).sin();
            let amp = p.stream * 0.4 * bubble * outside;
            l += a * amp;
            r += b * amp;
        }
        if p.city > 1e-3 {
            let hum = 0.6 + 0.4 * (self.time * 0.13).sin() * (self.time * 0.07).cos();
            let amp = p.city * hum * 1.0 * outside;
            l += self.city[0].run(self.rng.noise(), 180.0, rate) * amp;
            r += self.city[1].run(self.rng.noise(), 200.0, rate) * amp;
            // Un klaxon au loin, de temps en temps.
            if self.horn <= 0.0 && self.rng.unit() < dt * 0.02 * p.city {
                self.horn = 1.0;
            }
            if self.horn > 0.0 {
                self.horn -= dt * 2.0;
                self.horn_phase = (self.horn_phase + 415.0 * dt).fract();
                let tone = ((self.horn_phase * TAU).sin() + 0.6 * (self.horn_phase * TAU * 1.26).sin())
                    * self.horn.min(0.2)
                    * 5.0;
                l += tone * 0.015 * p.city * outside;
                r += tone * 0.01 * p.city * outside;
            }
        }
        if p.birds > 1e-3 {
            for k in 0..self.birds.len() {
                let (bl, br) = self.bird(k, p.birds, dt);
                l += bl * outside;
                r += br * outside;
            }
        }
        if p.insects > 1e-3 {
            for k in 0..self.crickets.len() {
                let c = &mut self.crickets[k];
                c.clock = (c.clock + dt * c.rate).fract();
                // Une salve de quelques stridulations au debut de chaque periode.
                let burst = c.clock * 12.0;
                let on = if burst < c.pulses { ((burst.fract()) * std::f32::consts::PI).sin().powi(2) } else { 0.0 };
                c.phase = (c.phase + c.freq * dt).fract();
                let s = (c.phase * TAU).sin() * on * p.insects * 0.06 * outside;
                l += s * (1.0 - c.pan) * 0.5;
                r += s * (1.0 + c.pan) * 0.5;
            }
        }

        // Sortie : volume, puis saturation douce plutot qu'un ecretage.
        let v = p.volume;
        ((l * v).tanh(), (r * v).tanh())
    }

    fn bird(&mut self, k: usize, density: f32, dt: f32) -> (f32, f32) {
        let mut b = self.birds[k];
        if b.notes == 0 {
            // Au repos : un oiseau chante de temps en temps, plus souvent dans un bois dense.
            if self.rng.unit() < dt * 0.35 * density {
                let trill = self.rng.unit() < 0.3;
                b = Bird {
                    notes: if trill { 1 } else { 1 + self.rng.next() % 6 },
                    t: 0.0,
                    duration: if trill { self.rng.range(0.3, 0.8) } else { self.rng.range(0.05, 0.16) },
                    gap: self.rng.range(0.04, 0.14),
                    f0: self.rng.range(2200.0, 5200.0),
                    f1: 0.0,
                    trill: if trill { self.rng.range(18.0, 35.0) } else { 0.0 },
                    amp: self.rng.range(0.04, 0.14),
                    pan: self.rng.range(-0.9, 0.9),
                    phase: 0.0,
                };
                b.f1 = b.f0 * self.rng.range(0.6, 1.5);
            }
            self.birds[k] = b;
            return (0.0, 0.0);
        }
        b.t += dt;
        let mut out = 0.0;
        if b.t < b.duration {
            let x = b.t / b.duration;
            let f = b.f0 + (b.f1 - b.f0) * x;
            b.phase = (b.phase + f * dt).fract();
            let env = (x * std::f32::consts::PI).sin();
            let am = if b.trill > 0.0 { 0.5 + 0.5 * (b.t * b.trill * TAU).sin() } else { 1.0 };
            out = (b.phase * TAU).sin() * env * am * b.amp;
        } else if b.t > b.duration + b.gap {
            b.notes -= 1;
            b.t = 0.0;
            // La note suivante reprend un peu plus haut ou plus bas.
            let shift = self.rng.range(0.85, 1.15);
            b.f0 *= shift;
            b.f1 *= shift;
        }
        self.birds[k] = b;
        (out * (1.0 - b.pan) * 0.5, out * (1.0 + b.pan) * 0.5)
    }
}

/// La sortie audio : un flux de la carte son qui joue le synthetiseur.
pub struct Output {
    shared: Arc<Mutex<SoundParams>>,
    _stream: cpal::Stream,
}

impl Output {
    /// Ouvre la sortie par defaut ; `None` sans carte son (le jeu reste muet).
    pub fn open() -> Option<Self> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        let host = cpal::default_host();
        let device = host.default_output_device()?;
        let supported = device.default_output_config().ok()?;
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let shared = Arc::new(Mutex::new(SoundParams::default()));
        let stream = match format {
            cpal::SampleFormat::F32 => Self::build::<f32>(&device, &config, shared.clone()),
            cpal::SampleFormat::I16 => Self::build::<i16>(&device, &config, shared.clone()),
            cpal::SampleFormat::U16 => Self::build::<u16>(&device, &config, shared.clone()),
            cpal::SampleFormat::I32 => Self::build::<i32>(&device, &config, shared.clone()),
            _ => None,
        }?;
        stream.play().ok()?;
        Some(Self { shared, _stream: stream })
    }

    fn build<T>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        shared: Arc<Mutex<SoundParams>>,
    ) -> Option<cpal::Stream>
    where
        T: cpal::SizedSample + cpal::FromSample<f32>,
    {
        use cpal::traits::DeviceTrait;
        let channels = config.channels as usize;
        let mut synth = Synth::new(config.sample_rate as f32);
        let mut buffer: Vec<f32> = Vec::new();
        let reported = std::sync::atomic::AtomicBool::new(false);
        device
            .build_output_stream(
                config,
                move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                    // Jamais d'attente dans le thread audio : si le jeu ecrit, on garde les parametres d'avant.
                    if let Ok(p) = shared.try_lock() {
                        synth.set(*p);
                    }
                    buffer.resize(data.len(), 0.0);
                    synth.fill(&mut buffer, channels);
                    for (d, s) in data.iter_mut().zip(&buffer) {
                        *d = T::from_sample(*s);
                    }
                },
                move |e| {
                    // Un peripherique en defaut rappelle a chaque tampon : la premiere erreur suffit.
                    if !reported.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!("son : {e} (les erreurs suivantes ne sont plus affichees)");
                    }
                },
                None,
            )
            .ok()
    }

    /// Transmet les reglages au thread audio, qui les prendra a son prochain tampon.
    pub fn set(&self, params: SoundParams) {
        *crate::sync::lock(&self.shared) = params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(synth: &mut Synth, seconds: f32) -> Vec<f32> {
        let mut out = vec![0.0; (48_000.0 * seconds) as usize * 2];
        for chunk in out.chunks_mut(960) {
            synth.fill(chunk, 2);
        }
        out
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn silence_when_nothing_plays() {
        let mut s = Synth::new(48_000.0);
        s.set(SoundParams { wind: 0.0, ..Default::default() });
        let out = render(&mut s, 1.0);
        assert!(rms(&out) < 1e-4, "{}", rms(&out));
    }

    #[test]
    fn the_engine_gets_louder_and_brighter_with_load_and_never_clips() {
        let car = CarSound { rpm: 2500.0, load: 0.0, gear: 2, speed: 10.0, ..Default::default() };
        let mut s = Synth::new(48_000.0);
        s.set(SoundParams { car: Some(car), wind: 0.0, ..Default::default() });
        let idle = rms(&render(&mut s, 1.5)[48_000..]);
        s.set(SoundParams { car: Some(CarSound { rpm: 6500.0, load: 1.0, ..car }), wind: 0.0, ..Default::default() });
        let full = render(&mut s, 1.5);
        let loud = rms(&full[48_000..]);
        assert!(idle > 0.01 && loud > idle * 1.5, "ralenti {idle:.3}, pleine charge {loud:.3}");
        assert!(full.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }

    #[test]
    fn ambiences_play_and_stay_bounded() {
        let mut s = Synth::new(48_000.0);
        s.set(SoundParams {
            wind: 1.0,
            rush: 200.0,
            leaves: 1.0,
            birds: 1.0,
            insects: 1.0,
            waves: 1.0,
            stream: 1.0,
            city: 1.0,
            impact: (1, 30.0),
            ..Default::default()
        });
        let out = render(&mut s, 6.0);
        let level = rms(&out[96_000..]);
        assert!(level > 0.02, "{level}");
        assert!(out.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }

    #[test]
    fn the_synth_costs_a_few_percent_of_a_core() {
        let car = CarSound { rpm: 5000.0, load: 1.0, gear: 3, speed: 30.0, slip: 0.5, rough: 0.3, cockpit: true };
        let mut s = Synth::new(48_000.0);
        s.set(SoundParams {
            car: Some(car),
            wind: 1.0,
            rush: 0.0,
            leaves: 1.0,
            birds: 1.0,
            insects: 1.0,
            waves: 1.0,
            stream: 1.0,
            city: 1.0,
            ..Default::default()
        });
        // Le meilleur de trois mesures : les autres tests chargent les coeurs en meme temps.
        let cost = (0..3)
            .map(|_| {
                let start = std::time::Instant::now();
                render(&mut s, 3.0);
                start.elapsed().as_secs_f32() / 3.0
            })
            .fold(f32::MAX, f32::min);
        if std::env::var("TRACE").is_ok() {
            eprintln!("son : {:.2} % d'un coeur", cost * 100.0);
        }
        // Seul, environ 4 % ; la marge couvre une machine chargee par les autres tests ou d'autres programmes.
        assert!(cost < 0.15, "{:.1} % d'un coeur", cost * 100.0);
    }

    /// Ecrit une demonstration a ecouter (TERRA_WAV=chemin) : depart arrete, passages de rapports,
    /// derapage, choc, puis bois de jour, nuit d'ete, rivage et ville.
    #[test]
    fn write_a_listening_demo_on_request() {
        let Ok(path) = std::env::var("TERRA_WAV") else { return };
        let rate = 48_000.0;
        let mut s = Synth::new(rate);
        let mut out: Vec<f32> = Vec::new();
        let quiet = SoundParams { wind: 0.15, ..Default::default() };
        let play = |s: &mut Synth, p: SoundParams, seconds: f32, out: &mut Vec<f32>| {
            s.set(p);
            let mut buf = vec![0.0; (rate * seconds) as usize * 2];
            for chunk in buf.chunks_mut(960) {
                s.fill(chunk, 2);
            }
            out.extend(buf);
        };
        // Voiture : on rejoue une acceleration avec la vraie dynamique.
        let mut ch = crate::vehicle::dynamics::Chassis::new(glam::DVec3::ZERO, glam::DVec3::Y, glam::DVec3::Z);
        let ground = crate::vehicle::dynamics::Contact::plane(glam::DVec3::Y, glam::DVec3::ZERO, glam::DVec3::Y);
        let car = |ch: &crate::vehicle::dynamics::Chassis, slip: f32| CarSound {
            rpm: ch.rpm as f32,
            load: ch.throttle() as f32,
            gear: ch.gear,
            speed: ch.speed() as f32,
            slip,
            rough: 0.0,
            cockpit: false,
        };
        let idle = crate::vehicle::dynamics::Controls::default();
        let full = crate::vehicle::dynamics::Controls { throttle: 1.0, ..Default::default() };
        let brake = crate::vehicle::dynamics::Controls { brake: 1.0, ..Default::default() };
        for (controls, seconds) in [(idle, 1.5), (full, 9.0), (idle, 2.0), (brake, 3.0)] {
            let mut t = 0.0;
            while t < seconds {
                ch.update(&controls, &ground, 1.0 / 60.0);
                play(&mut s, SoundParams { car: Some(car(&ch, 0.0)), ..quiet }, 1.0 / 60.0, &mut out);
                t += 1.0 / 60.0;
            }
        }
        play(
            &mut s,
            SoundParams {
                car: Some(CarSound { rpm: 3500.0, load: 0.6, gear: 2, speed: 15.0, slip: 0.9, ..Default::default() }),
                ..quiet
            },
            2.0,
            &mut out,
        );
        play(
            &mut s,
            SoundParams {
                car: Some(CarSound { rpm: 1500.0, load: 0.0, gear: 2, speed: 8.0, ..Default::default() }),
                impact: (1, 14.0),
                ..quiet
            },
            1.5,
            &mut out,
        );
        play(&mut s, quiet, 1.0, &mut out);
        play(
            &mut s,
            SoundParams { wind: 0.4, leaves: 1.0, birds: 1.0, stream: 0.5, ..Default::default() },
            8.0,
            &mut out,
        );
        play(&mut s, SoundParams { wind: 0.2, insects: 1.0, leaves: 0.3, ..Default::default() }, 6.0, &mut out);
        play(&mut s, SoundParams { wind: 0.6, waves: 1.0, ..Default::default() }, 8.0, &mut out);
        play(&mut s, SoundParams { wind: 0.2, city: 1.0, birds: 0.2, ..Default::default() }, 6.0, &mut out);
        // WAV PCM 16 bits stereo.
        let mut bytes = Vec::with_capacity(44 + out.len() * 2);
        let data = (out.len() * 2) as u32;
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&48_000u32.to_le_bytes());
        bytes.extend_from_slice(&(48_000u32 * 4).to_le_bytes());
        bytes.extend_from_slice(&4u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data.to_le_bytes());
        for v in &out {
            bytes.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        eprintln!("demonstration : {path} ({:.0} s)", out.len() as f32 / 2.0 / rate);
    }

    #[test]
    fn tires_squeal_only_on_tarmac() {
        let car = CarSound { rpm: 3000.0, load: 0.3, gear: 3, speed: 20.0, slip: 0.0, rough: 0.0, cockpit: false };
        // Energie des aigus : difference entre echantillons voisins (passe-haut grossier).
        let treble = |slip: f32, rough: f32| {
            let mut s = Synth::new(48_000.0);
            s.set(SoundParams { car: Some(CarSound { slip, rough, ..car }), wind: 0.0, ..Default::default() });
            let out = render(&mut s, 1.0);
            let left: Vec<f32> = out.iter().step_by(2).copied().collect();
            let hp: Vec<f32> = left.windows(2).map(|w| w[1] - w[0]).collect();
            rms(&hp[24_000..])
        };
        let (grip, slide) = (treble(0.0, 0.0), treble(1.0, 0.0));
        assert!(slide > grip * 1.5, "asphalte : adherence {grip:.4}, derapage {slide:.4}");
        let (grip, slide) = (treble(0.0, 1.0), treble(1.0, 1.0));
        assert!(slide < grip * 1.1, "graviers : adherence {grip:.4}, derapage {slide:.4}");
    }
}
