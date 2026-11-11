//! Climat de la planete, simule une fois a sa creation, en unites reelles (degres Celsius, mm de pluie
//! par an), pour les deux saisons extremes : janvier et juillet.
//!
//! - Temperature : bilan de l'ensoleillement par latitude, amplitude des saisons selon l'eloignement de
//!   la mer sous le vent (continentalite), courants marins (froids sur les facades ouest subtropicales,
//!   chauds sur les facades ouest des hautes latitudes), puis 6 degres de moins par kilometre d'altitude.
//! - Vents : alizes, vents d'ouest et vents polaires autour d'un equateur thermique qui suit le soleil
//!   (plus loin sur les continents), devies par les depressions thermiques des terres chaudes : la
//!   mousson.
//! - Pluie : la vapeur d'eau est transportee par le vent. Elle se charge sur la mer, selon ce que l'air
//!   peut contenir (Clausius-Clapeyron), et retombe la ou l'air monte : zone de convergence equatoriale,
//!   perturbations des latitudes moyennes, versants au vent. Les cellules de Hadley assechent les
//!   tropiques, et les versants sous le vent restent dans l'ombre des chaines.
//!
//! Le climat d'un lieu se lit ensuite comme sur Terre : classification de Koppen-Geiger (seuils de Peel
//! et al. 2007), evapotranspiration potentielle de Holdridge, indice d'aridite de l'UNEP.

use super::cube::{face_of, CubeField, CubeGrid};
use super::generator::{par_map, planet_radius, smoothstep, PlanetParams};
use super::hydrology::par_cells;
use glam::{DVec3, Vec3};
use std::f64::consts::FRAC_PI_2;

/// Resolution de la simulation : environ 60 km par cellule sur une planete de la taille de la Terre.
pub const CLIMATE_N: usize = 160;
/// Baisse de la temperature avec l'altitude, en degres par metre.
pub const LAPSE: f32 = 0.006;
/// Pas de transport de la vapeur d'eau, et pas dont on garde la pluie.
const STEPS: usize = 170;
const KEPT: usize = 60;
/// Vent le plus fort, en m/s : il parcourt un peu moins d'une cellule par pas.
const MAX_WIND: f32 = 11.0;
/// Part de la vapeur qui tombe par pas dans un air sature en pleine ascendance.
const RAIN_RATE: f32 = 0.034;
/// Part de l'ecart a la saturation que la mer comble par pas.
const EVAPORATION: f32 = 0.055;
/// Humidite relative de l'air au-dessus de la mer.
const SEA_HUMIDITY: f32 = 0.85;
/// Part de la pluie que la vegetation rend a l'air.
const RECYCLED: f32 = 0.32;

/// Climat d'un lieu, a son altitude.
#[derive(Clone, Copy, Debug)]
pub struct Climate {
    /// Temperature moyenne de l'annee, du mois le plus froid et du mois le plus chaud, en degres Celsius.
    pub mean: f32,
    pub cold: f32,
    pub hot: f32,
    /// Pluie (et neige), en mm par an.
    pub rain: f32,
    /// Part de cette pluie qui tombe pendant le semestre d'ete.
    pub summer: f32,
    /// Pluie du mois le plus sec, en mm.
    pub driest: f32,
    /// Chaleur de 0 (glace) a 1 (equateur) : 45 degres d'un bout a l'autre.
    pub temperature: f32,
    /// Humidite de 0 (desert) a 1 (foret pluviale), d'apres l'indice d'aridite.
    pub moisture: f32,
}

/// Les grands types de climat de Koppen-Geiger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Koppen {
    /// A : tropical humide (Af, Am).
    Rainforest,
    /// Aw : tropical a saison seche.
    Savanna,
    /// BW : desert.
    Desert,
    /// BS : steppe.
    Steppe,
    /// Cs : mediterraneen.
    Mediterranean,
    /// Cf, Cw : tempere humide.
    Temperate,
    /// Dfa, Dfb, Dw : continental a ete chaud ou tiede.
    Continental,
    /// Dfc, Dfd : subarctique.
    Subarctic,
    /// ET : toundra.
    Tundra,
    /// EF : glace.
    Ice,
}

impl Koppen {
    pub const ALL: [Koppen; 10] = [
        Koppen::Rainforest,
        Koppen::Savanna,
        Koppen::Desert,
        Koppen::Steppe,
        Koppen::Mediterranean,
        Koppen::Temperate,
        Koppen::Continental,
        Koppen::Subarctic,
        Koppen::Tundra,
        Koppen::Ice,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Koppen::Rainforest => "tropical humide",
            Koppen::Savanna => "tropical a saison seche",
            Koppen::Desert => "desertique",
            Koppen::Steppe => "semi-aride",
            Koppen::Mediterranean => "mediterraneen",
            Koppen::Temperate => "tempere",
            Koppen::Continental => "continental",
            Koppen::Subarctic => "subarctique",
            Koppen::Tundra => "polaire",
            Koppen::Ice => "glaciaire",
        }
    }
}

impl Climate {
    /// Le climat a partir de ses mesures : temperatures de janvier et de juillet (dans l'hemisphere nord :
    /// l'hiver et l'ete), pluie de chaque semestre en mm par an.
    pub fn new(winter: f32, summer: f32, winter_rain: f32, summer_rain: f32) -> Self {
        let (cold, hot) = (winter.min(summer), winter.max(summer));
        let mean = (cold + hot) * 0.5;
        // Le semestre d'ete est celui du mois le plus chaud.
        let (wet_cold, wet_hot) =
            if summer >= winter { (winter_rain, summer_rain) } else { (summer_rain, winter_rain) };
        let rain = ((wet_cold + wet_hot) * 0.5).max(0.0);
        let share = if rain > 1e-3 { (wet_hot / (wet_cold + wet_hot)).clamp(0.0, 1.0) } else { 0.5 };
        // La pluie de l'annee suit une sinusoide dont chaque semestre a la moyenne mesuree.
        let swing = ((wet_hot - wet_cold).abs() * 0.5 * std::f32::consts::FRAC_PI_2).min(rain);
        let driest = (rain - swing).max(0.0) / 12.0;
        let mut c = Self { mean, cold, hot, rain, summer: share, driest, temperature: 0.0, moisture: 0.0 };
        c.temperature = ((mean + 18.0) / 45.0).clamp(0.0, 1.0);
        c.moisture = (1.0 - (-c.aridity() * 1.1).exp()).clamp(0.0, 1.0);
        c
    }

    /// Le meme climat, mieux arrose : `k` vaut 1 au bord d'une riviere, qui apporte l'eau des montagnes
    /// jusque dans le desert.
    pub fn wetter(mut self, k: f32) -> Self {
        if k > 0.0 {
            let rain = self.rain * (1.0 + k) + 450.0 * k;
            self.driest = self.driest * (1.0 + k) + 20.0 * k;
            self.rain = rain;
            self.moisture = (1.0 - (-self.aridity() * 1.1).exp()).clamp(0.0, 1.0);
        }
        self
    }

    /// Temperature du mois `m`, de 0 (le plus froid) a 1 (le plus chaud) en passant par une demi-annee.
    fn month(&self, t: f32) -> f32 {
        self.mean - (self.hot - self.cold) * 0.5 * (t * std::f32::consts::PI).cos()
    }

    /// Nombre de mois plus chauds que `limit` degres.
    pub fn months_above(&self, limit: f32) -> f32 {
        let half = (self.hot - self.cold) * 0.5;
        if half < 1e-3 {
            return if self.mean > limit { 12.0 } else { 0.0 };
        }
        // mean - half cos(x) > limit, pour x de 0 a pi.
        let k = ((self.mean - limit) / half).clamp(-1.0, 1.0);
        12.0 * (1.0 - (k.acos() / std::f32::consts::PI))
    }

    /// Evapotranspiration potentielle de Holdridge, en mm par an : 58,93 fois la biotemperature, moyenne
    /// des temperatures de l'annee ramenees entre 0 et 30 degres.
    pub fn evaporation(&self) -> f32 {
        let mut sum = 0.0;
        for k in 0..6 {
            sum += self.month((k as f32 + 0.5) / 6.0).clamp(0.0, 30.0);
        }
        58.93 * sum / 6.0
    }

    /// Indice d'aridite de l'UNEP : pluie sur evapotranspiration potentielle. Moins de 0,05 : hyperaride ;
    /// 0,2 : aride ; 0,5 : semi-aride ; 0,65 : subhumide sec ; au-dela, humide.
    pub fn aridity(&self) -> f32 {
        self.rain / self.evaporation().max(60.0)
    }

    /// Seuil de secheresse de Koppen, en mm par an : 20 fois la temperature moyenne, plus 280 si la pluie
    /// tombe en ete (quand l'evaporation est forte), 0 si elle tombe en hiver.
    pub fn dry_limit(&self) -> f32 {
        (20.0 * self.mean + 280.0 * smoothstep(0.3, 0.7, self.summer)).max(0.0)
    }

    /// Le type de climat de Koppen-Geiger.
    pub fn koppen(&self) -> Koppen {
        if self.hot < 0.0 {
            return Koppen::Ice;
        }
        if self.hot < 10.0 {
            return Koppen::Tundra;
        }
        let limit = self.dry_limit();
        if self.rain < limit * 0.5 {
            return Koppen::Desert;
        }
        if self.rain < limit {
            return Koppen::Steppe;
        }
        if self.cold >= 18.0 {
            // Af et Am : pas de vraie saison seche, ou une mousson qui la compense.
            return if self.driest >= 60.0 || self.driest >= 100.0 - self.rain / 25.0 {
                Koppen::Rainforest
            } else {
                Koppen::Savanna
            };
        }
        if self.cold > 0.0 {
            // Cs : un ete sec, trois fois moins arrose que l'hiver.
            return if self.summer < 0.27 && self.driest < 40.0 { Koppen::Mediterranean } else { Koppen::Temperate };
        }
        if self.months_above(10.0) >= 4.0 {
            Koppen::Continental
        } else {
            Koppen::Subarctic
        }
    }
}

/// Le climat de toute la planete : temperatures au niveau de la mer et pluies des deux saisons.
pub struct ClimateMap {
    /// Temperature au niveau de la mer en janvier puis en juillet, pluie des memes saisons.
    fields: [CubeField; 4],
    /// Nebulosite moyenne, de 0 a 1.
    pub cloud: CubeField,
    /// Vent moyen de l'annee, en m/s, par composante.
    wind: [CubeField; 3],
}

fn frame(dir: DVec3) -> (DVec3, DVec3) {
    let east = DVec3::Y.cross(dir).normalize_or(DVec3::X);
    (east, dir.cross(east))
}

fn bump(x: f32, center: f32, width: f32) -> f32 {
    let t = (x - center) / width;
    (-t * t).exp()
}

/// Cellules voisines et poids de l'interpolation bilineaire au point `dir`, sur sa face.
fn stencil(grid: CubeGrid, dir: DVec3) -> ([u32; 4], [f32; 4]) {
    let (face, u, v) = face_of(dir);
    let n = grid.n;
    let x = ((u + 1.0) * 0.5 * n as f64 - 0.5).clamp(0.0, n as f64 - 1.0);
    let y = ((v + 1.0) * 0.5 * n as f64 - 0.5).clamp(0.0, n as f64 - 1.0);
    let (ix, iy) = ((x as usize).min(n - 2), (y as usize).min(n - 2));
    let (fx, fy) = ((x - ix as f64) as f32, (y - iy as f64) as f32);
    let at = |a: usize, b: usize| ((face * n + b) * n + a) as u32;
    (
        [at(ix, iy), at(ix + 1, iy), at(ix, iy + 1), at(ix + 1, iy + 1)],
        [(1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy],
    )
}

/// Ce que l'air peut porter de vapeur d'eau, en mm, au-dessus d'un sol a `t` degres et `h` metres : il
/// double tous les 11 degres (Clausius-Clapeyron), et l'air s'amincit avec l'altitude.
fn capacity(t: f32, h: f32) -> f32 {
    13.0 * (0.064 * t.clamp(-45.0, 38.0)).exp() * (-h / 2600.0).exp()
}

/// Temperature moyenne de l'annee au niveau de la mer, selon le sinus de la latitude : 27 degres a
/// l'equateur, 21 a 30 degres de latitude, 10 a 45, -20 au pole.
fn annual(sin_lat: f32) -> f32 {
    27.0 - 47.0 * sin_lat.abs().powi(3)
}

/// Les vents de la circulation generale a `x` degres de l'equateur thermique : (vers l'est, vers
/// l'equateur) en m/s, et la force des ascendances (zone de convergence, perturbations).
fn belts(x: f32) -> (f32, f32, f32) {
    let a = x.abs();
    let wave = |from: f32| ((a - from) / 30.0 * std::f32::consts::PI).sin();
    let (east, equator) = if a < 30.0 {
        (-6.5 * (a / 60.0 * std::f32::consts::PI).cos(), 3.2 * wave(0.0))
    } else if a < 60.0 {
        (7.5 * wave(30.0), -2.6 * wave(30.0))
    } else {
        (-3.5 * wave(60.0), 1.6 * wave(60.0))
    };
    // Sous les tropiques, l'air des cellules de Hadley redescend et s'asseche : c'est la ceinture des deserts.
    let sinking = 1.0 - 0.9 * bump(a, 23.5, 11.0);
    let lift = (0.07 + 1.0 * bump(a, 0.0, 7.5) + 0.9 * bump(a, 52.0, 15.0)) * sinking;
    (east, equator, lift)
}

impl ClimateMap {
    /// Simule le climat d'une planete dont `height` donne le relief en metres (negatif sous la mer).
    pub fn build(params: &PlanetParams, height: &(dyn Fn(DVec3) -> f32 + Sync)) -> Self {
        let grid = CubeGrid { n: CLIMATE_N };
        let cells = grid.cells();
        let radius = planet_radius();
        let spacing = radius * FRAC_PI_2 / grid.n as f64;
        let dirs: Vec<DVec3> = (0..cells).map(|c| grid.center(c)).collect();
        let dt = 0.9 * spacing / MAX_WIND as f64;
        let per_year = (365.25 * 86_400.0 / dt) as f32;

        // Relief moyen de chaque cellule.
        let ground = par_cells(grid, &|c| {
            let (e, n) = frame(dirs[c]);
            let k = 0.25 * spacing / radius;
            let mut sum = height(dirs[c]);
            for (a, b) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                sum += height((dirs[c] + (e * a + n * b) * k).normalize());
            }
            sum / 5.0
        });
        let land: Vec<bool> = ground.iter().map(|h| *h > 0.0).collect();
        let high: Vec<f32> = ground.iter().map(|h| h.max(0.0)).collect();

        let around: Vec<([u32; 8], usize)> = (0..cells)
            .map(|c| {
                let mut out = [0u32; 8];
                let count = grid.neighbours(c, &mut out);
                (out, count)
            })
            .collect();
        let blur = |values: &[f32], passes: usize| -> Vec<f32> {
            let mut v = values.to_vec();
            for _ in 0..passes {
                v = (0..cells)
                    .map(|c| {
                        let (n, count) = &around[c];
                        let sum: f32 = n[..*count].iter().map(|k| v[*k as usize]).sum();
                        (v[c] * 2.0 + sum) / (2.0 + *count as f32)
                    })
                    .collect();
            }
            v
        };
        let smooth = blur(&high, 2);
        let land_share: Vec<f32> = land.iter().map(|l| *l as u8 as f32).collect();
        // Terres a quelques centaines de km, puis a l'echelle du continent.
        let near_land = blur(&land_share, 6);
        let wide_land = blur(&near_land, 28);

        // Courants marins : distance de la premiere terre vers l'est et vers l'ouest, le long du parallele.
        let scan = |c: usize, sign: f64| -> f32 {
            let p = dirs[c];
            let cos_lat = (1.0 - p.y * p.y).sqrt().max(0.2);
            let step = 0.8 * spacing / radius / cos_lat * sign;
            for k in 1..=56 {
                let (s, co) = (step * k as f64).sin_cos();
                let q = DVec3::new(p.x * co + p.z * s, p.y, -p.x * s + p.z * co);
                if land[grid.cell_of(q)] {
                    return (k as f64 * 0.8 * spacing) as f32;
                }
            }
            f32::INFINITY
        };
        let current = par_cells(grid, &|c| {
            if land[c] {
                return 0.0;
            }
            let lat = (dirs[c].y.abs().asin().to_degrees()) as f32;
            let near = |d: f32| (-d / 9.0e5).exp();
            // Terre a l'est : bord est de l'ocean, donc facade ouest d'un continent.
            let side = near(scan(c, 1.0)) - near(scan(c, -1.0));
            side * (-5.5 * bump(lat, 24.0, 13.0) + 6.5 * bump(lat, 57.0, 14.0))
        });
        let current = blur(&current, 2);

        // Les deux saisons, en parallele.
        let season = |s: usize| -> Season {
            let sun = if s == 0 { -1.0f32 } else { 1.0 };
            // Coordonnee de latitude autour de l'equateur thermique, qui suit le soleil.
            let band: Vec<f32> = (0..cells)
                .map(|c| {
                    let lat = (dirs[c].y.asin().to_degrees()) as f32;
                    // Il s'avance loin sur les continents en ete ; les ceintures des latitudes moyennes, elles,
                    // ne se deplacent que de quelques degres.
                    let itcz = sun * (5.0 + 15.0 * wide_land[c]);
                    let rel = lat - itcz;
                    let edge = (30.0 + 4.0 * sun * rel.signum()) - itcz * rel.signum();
                    if rel.abs() < edge {
                        rel / edge * 30.0
                    } else {
                        rel.signum() * (30.0 + (rel.abs() - edge) / (90.0 - itcz * rel.signum() - edge) * 60.0)
                    }
                })
                .collect();
            let general: Vec<Vec3> = (0..cells)
                .map(|c| {
                    let (e, n) = frame(dirs[c]);
                    let (east, equator, _) = belts(band[c]);
                    (e * east as f64 - n * (band[c].signum() * equator) as f64).as_vec3()
                })
                .collect();
            let upwind = |wind: &[Vec3]| -> Vec<([u32; 4], [f32; 4])> {
                par_map(cells, |c| stencil(grid, (dirs[c] - wind[c].as_dvec3() * (dt / radius)).normalize()))
            };
            let pull = |up: &([u32; 4], [f32; 4]), v: &[f32]| -> f32 {
                (0..4).map(|k| v[up.0[k] as usize] * up.1[k]).sum()
            };

            // Influence de la mer portee par le vent : elle s'efface en avancant dans les terres, plus vite
            // derriere un relief. Elle transporte aussi la chaleur ou le froid des courants.
            let first = upwind(&general);
            let fade = (-(spacing * 0.8) as f32 / 1.6e6).exp();
            let mut sea: Vec<f32> = land.iter().map(|l| if *l { 0.3 } else { 1.0 }).collect();
            let mut carried = current.clone();
            for _ in 0..70 {
                let (mut next_sea, mut next_carried) = (sea.clone(), carried.clone());
                for c in 0..cells {
                    if !land[c] {
                        continue;
                    }
                    let up = &first[c];
                    let climb = (high[c] - pull(up, &high)).max(0.0);
                    let keep = fade * (-climb / 1800.0).exp();
                    next_sea[c] = pull(up, &sea) * keep;
                    next_carried[c] = pull(up, &carried) * keep;
                }
                (sea, carried) = (next_sea, next_carried);
            }
            let sea = blur(&sea, 2);
            let carried = blur(&carried, 2);

            // Temperature au niveau de la mer.
            let temp: Vec<f32> = (0..cells)
                .map(|c| {
                    let y = dirs[c].y as f32;
                    let inland = (1.0 - (0.7 * sea[c] + 0.3 * (1.0 - near_land[c]))).clamp(0.0, 1.0);
                    let inland = if land[c] { inland } else { inland * 0.25 };
                    let swing = (0.8 + 4.5 * y.abs()) * (1.0 - inland) + (1.5 + 25.0 * y.abs().powf(1.15)) * inland;
                    // Loin de la mer, les hautes latitudes sont aussi plus froides en moyenne.
                    let chill = 6.0 * inland * smoothstep(0.45, 0.9, y.abs());
                    // La mer libre, elle, garde la chaleur que les courants apportent des tropiques.
                    let mild = 5.0 * (1.0 - inland) * smoothstep(0.6, 0.85, y.abs());
                    annual(y) + params.temperature * 30.0 + carried[c] - chill + mild
                        + sun * smoothstep(0.0, 0.1, y.abs()) * y.signum() * swing
                })
                .collect();

            // Depressions thermiques : l'air converge vers les terres plus chaudes que leur latitude, devie
            // par la rotation de la planete. C'est la mousson d'ete, et l'anticyclone d'hiver des continents.
            let anomaly: Vec<f32> = (0..cells)
                .map(|c| temp[c] - annual(dirs[c].y as f32) - params.temperature * 30.0)
                .collect();
            let anomaly = blur(&anomaly, 8);
            let wind: Vec<Vec3> = (0..cells)
                .map(|c| {
                    let (n, count) = &around[c];
                    let mut grad = DVec3::ZERO;
                    for k in &n[..*count] {
                        let d = dirs[*k as usize] - dirs[c];
                        let d = d - dirs[c] * d.dot(dirs[c]);
                        let len = d.length() * radius;
                        grad += d.normalize_or_zero() * ((anomaly[*k as usize] - anomaly[c]) as f64 / len);
                    }
                    grad *= 2.0 / *count as f64;
                    // Vers le chaud (la basse pression), tourne de 50 degres : a droite au nord, a gauche au sud.
                    let turn = 50f64.to_radians() * smoothstep(0.05, 0.35, dirs[c].y.abs() as f32) as f64;
                    let side = dirs[c].cross(grad) * -dirs[c].y.signum();
                    let thermal = (grad * turn.cos() + side * turn.sin()) * 3.6e5;
                    let v = general[c] + thermal.as_vec3().clamp_length_max(6.0);
                    v.clamp_length_max(MAX_WIND)
                })
                .collect();

            // Transport de la vapeur d'eau jusqu'a l'equilibre.
            let up = upwind(&wind);
            let surface: Vec<f32> = (0..cells).map(|c| temp[c] - LAPSE * high[c]).collect();
            let hold: Vec<f32> = (0..cells).map(|c| capacity(surface[c], high[c])).collect();
            // La montee de l'air se mesure sur un relief lisse : c'est le massif qui compte, pas ses cretes.
            let climb: Vec<f32> = (0..cells).map(|c| smooth[c] - pull(&up[c], &smooth)).collect();
            let lift: Vec<f32> = (0..cells)
                .map(|c| {
                    let (_, _, belt) = belts(band[c]);
                    // Orages des terres chaudes (sauf sous l'air qui redescend), pluies de versant, foehn.
                    let sinking = 1.0 - 0.9 * bump(band[c].abs(), 23.5, 11.0);
                    let storms = if land[c] { 0.22 * smoothstep(14.0, 27.0, surface[c]) * sinking } else { 0.0 };
                    let slope = (climb[c] / 500.0).clamp(0.0, 2.2);
                    let lee = 1.0 - (-climb[c] / 450.0).clamp(0.0, 0.88);
                    (belt * (1.0 + 0.45 * slope) + storms + 0.32 * slope) * lee
                })
                .collect();
            let mut vapour: Vec<f32> = hold.iter().map(|w| w * 0.5).collect();
            let mut rain = vec![0.0f32; cells];
            let mut damp = vec![0.0f32; cells];
            let mut next = vec![0.0f32; cells];
            for step in 0..STEPS {
                for c in 0..cells {
                    let (n, count) = &around[c];
                    let mixed: f32 = n[..*count].iter().map(|k| vapour[*k as usize]).sum::<f32>() / *count as f32;
                    let mut q = pull(&up[c], &vapour) * 0.86 + mixed * 0.14;
                    if !land[c] && surface[c] > -2.0 {
                        q += (SEA_HUMIDITY * hold[c] - q).max(0.0) * EVAPORATION;
                    }
                    let wet = (q / hold[c]).min(1.3);
                    let fallen = ((q - hold[c]).max(0.0) * 0.4 + RAIN_RATE * q * wet * wet * lift[c]).min(q * 0.9);
                    q -= fallen;
                    if land[c] {
                        q += fallen * RECYCLED * smoothstep(-2.0, 12.0, surface[c]);
                    }
                    next[c] = q;
                    if step >= STEPS - KEPT {
                        rain[c] += fallen;
                        damp[c] += wet.min(1.0) * (0.25 + 0.75 * (lift[c] / 1.2).min(1.0));
                    }
                }
                std::mem::swap(&mut vapour, &mut next);
            }
            // Au-dela de quelques metres par an, la pluie plafonne : les records de la Terre sont vers 11 m.
            let rain: Vec<f32> = rain
                .iter()
                .map(|r| {
                    let mm = r / KEPT as f32 * per_year;
                    6000.0 * (1.0 - (-mm / 6000.0).exp())
                })
                .collect();
            let damp: Vec<f32> = damp.iter().map(|r| r / KEPT as f32).collect();
            Season { temp, rain: blur(&rain, 1), cloud: blur(&damp, 1), wind }
        };
        let (winter, summer) = std::thread::scope(|s| {
            let r = planet_radius();
            let a = s.spawn(move || {
                super::generator::worker_thread(r);
                season(0)
            });
            let b = season(1);
            (crate::sync::join(a), b)
        });

        let cloud: Vec<f32> = (0..cells).map(|c| (winter.cloud[c] + summer.cloud[c]) * 0.5).collect();
        let wind = |k: usize| -> CubeField {
            let v: Vec<f32> = (0..cells).map(|c| (winter.wind[c][k] + summer.wind[c][k]) * 0.5).collect();
            CubeField::from_cells(grid, &v)
        };
        Self {
            fields: [
                CubeField::from_cells(grid, &winter.temp),
                CubeField::from_cells(grid, &summer.temp),
                CubeField::from_cells(grid, &winter.rain),
                CubeField::from_cells(grid, &summer.rain),
            ],
            cloud: CubeField::from_cells(grid, &cloud),
            wind: [wind(0), wind(1), wind(2)],
        }
    }

    /// Le climat du lieu `dir` a l'altitude `height`. `rain` multiplie la pluie (variations locales).
    #[inline]
    pub fn at(&self, dir: DVec3, height: f32, rain: f32) -> Climate {
        let [january, july, wet_january, wet_july] =
            CubeField::sample_linear_many(std::array::from_fn(|i| &self.fields[i]), dir);
        let drop = LAPSE * height.max(0.0);
        Climate::new(january - drop, july - drop, wet_january.max(0.0) * rain, wet_july.max(0.0) * rain)
    }

    /// Vent dominant au lieu `dir`, en m/s, tangent a la planete.
    pub fn wind(&self, dir: DVec3) -> Vec3 {
        Vec3::new(self.wind[0].sample_linear(dir), self.wind[1].sample_linear(dir), self.wind[2].sample_linear(dir))
    }
}

struct Season {
    temp: Vec<f32>,
    rain: Vec<f32>,
    cloud: Vec<f32>,
    wind: Vec<Vec3>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn koppen_classes_match_reference_cities() {
        // (hiver, ete, pluie du semestre d'hiver, du semestre d'ete en mm/an, type attendu)
        let cases = [
            (26.0, 27.0, 2400.0, 2200.0, Koppen::Rainforest),   // Singapour
            (21.0, 27.0, 150.0, 2300.0, Koppen::Savanna),       // Bamako
            (13.0, 35.0, 10.0, 30.0, Koppen::Desert),           // Sahara
            (-2.0, 24.0, 250.0, 500.0, Koppen::Steppe),         // Denver
            (9.0, 25.0, 900.0, 150.0, Koppen::Mediterranean),   // Lisbonne
            (4.0, 19.0, 640.0, 640.0, Koppen::Temperate),       // Paris
            (-9.0, 19.0, 500.0, 900.0, Koppen::Continental),    // Moscou
            (-38.0, 17.0, 150.0, 350.0, Koppen::Subarctic),     // Iakoutsk
            (-27.0, 5.0, 100.0, 200.0, Koppen::Tundra),         // Barrow
            (-60.0, -28.0, 20.0, 20.0, Koppen::Ice),            // Vostok
        ];
        for (winter, summer, wet_winter, wet_summer, want) in cases {
            let c = Climate::new(winter, summer, wet_winter, wet_summer);
            assert_eq!(c.koppen(), want, "{c:?}");
        }
    }

    #[test]
    fn potential_evaporation_follows_holdridge() {
        // Biotemperature de 24 degres : 1 414 mm par an.
        let c = Climate::new(24.0, 24.0, 1000.0, 1000.0);
        assert!((c.evaporation() - 1414.0).abs() < 2.0);
        // Les mois de gel ne comptent pas.
        let c = Climate::new(-20.0, 20.0, 300.0, 300.0);
        assert!((250.0..450.0).contains(&c.evaporation()), "{}", c.evaporation());
        assert!((c.months_above(0.0) - 6.0).abs() < 0.01);
    }
}
