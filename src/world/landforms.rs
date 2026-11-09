//! Les formes du terrain : ravines d'erosion sans simulation, plateaux en gradins, tours rocheuses,
//! dunes et volcans. Tout se calcule point par point, sans voisins, donc en parallele et par morceaux.

use super::cube::{CubeField, CubeGrid};
use super::generator::{planet_radius, smoothstep, Climate};
use super::hydrology::par_cells;
use super::noise::{hash_seed, Noise, Rng};
use super::tectonics::{Boundary, Tectonics};
use glam::{DVec3, Vec3};
use std::f32::consts::TAU;

/// Pente de chaque octave de ravines : une amplitude proportionnelle a la periode.
const GULLY_SLOPE: f32 = 0.27;
/// Chaque octave est moins pentue que la precedente : des montagnes, pas des aiguilles.
const GULLY_DECAY: f32 = 0.87;
/// Plus il est grand, plus les petites ravines remontent sur les versants.
const GULLY_DETAIL: f32 = 1.6;

/// Ce que le filtre d'erosion rend : l'ecart de hauteur, et la carte des cretes et des creux.
#[derive(Clone, Copy, Debug, Default)]
pub struct Eroded {
    pub offset: f32,
    /// De -1 (fond de ravine) a 1 (crete).
    pub ridge: f32,
}

/// sin et cos de `t` tours (t * 2 pi), a la precision d'un f32 (erreur < 1e-6) : plusieurs fois
/// plus rapide que la libm, et appele huit fois par octave de ravines.
#[inline(always)]
pub fn turn_sin_cos(t: f32) -> (f32, f32) {
    // Ramene a [-1/2, 1/2] tour, puis a [-1/4, 1/4] par symetrie : x dans [-pi/2, pi/2].
    let r = t - t.round();
    let (r, flip) = if r > 0.25 {
        (0.5 - r, -1.0)
    } else if r < -0.25 {
        (-0.5 - r, -1.0)
    } else {
        (r, 1.0)
    };
    let x = r * TAU;
    let x2 = x * x;
    // Series de Taylor jusqu'a x^11 et x^12 : erreur sous 1e-7 sur [-pi/2, pi/2].
    let sin = x
        * (1.0
            + x2 * (-1.0 / 6.0
                + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362_880.0 + x2 * (-1.0 / 39_916_800.0))))));
    let cos = 1.0
        + x2 * (-0.5
            + x2 * (1.0 / 24.0
                + x2 * (-1.0 / 720.0
                    + x2 * (1.0 / 40_320.0 + x2 * (-1.0 / 3_628_800.0 + x2 * (1.0 / 479_001_600.0))))));
    (sin, cos * flip)
}

fn ease_out(x: f32) -> f32 {
    let t = 1.0 - x.clamp(0.0, 1.0);
    1.0 - t * t
}

/// Generateur de motifs aleatoires par maille, sans table : il couvre toute la planete.
pub struct Cells {
    seed: u64,
}

impl Cells {
    /// Cellules de la graine `seed`.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    #[inline]
    #[cfg(test)]
    fn hash(&self, x: i64, y: i64, z: i64) -> u64 {
        let mut h = self.seed
            ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ (z as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
        h = (h ^ (h >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h = (h ^ (h >> 32)).wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^ (h >> 29)
    }

    /// Les hachages des huit coins d'une maille, dans l'ordre k = x + 2y + 4z : memes valeurs que
    /// `hash`, avec deux produits par axe au lieu de huit.
    #[inline(always)]
    fn corner_hashes(&self, ix: i64, iy: i64, iz: i64) -> [u64; 8] {
        let x =
            [(ix as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15), ((ix + 1) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)];
        let y =
            [(iy as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F), ((iy + 1) as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)];
        let z = [
            self.seed ^ (iz as u64).wrapping_mul(0x1656_67B1_9E37_79F9),
            self.seed ^ ((iz + 1) as u64).wrapping_mul(0x1656_67B1_9E37_79F9),
        ];
        std::array::from_fn(|k| {
            let mut h = z[k >> 2] ^ x[k & 1] ^ y[(k >> 1) & 1];
            h = (h ^ (h >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h = (h ^ (h >> 32)).wrapping_mul(0x94D0_49BB_1331_11EB);
            h ^ (h >> 29)
        })
    }

    /// Bandes paralleles orientees selon `wave`, inspirees de Phacelle : chaque maille a son pivot,
    /// et les huit mailles voisines se melangent. Rend le cosinus et le sinus de la phase.
    /// `wave` est de longueur 1 ; une periode de bande vaut une maille.
    #[inline]
    pub fn stripes(&self, q: DVec3, wave: Vec3) -> (f32, f32) {
        let cell = q.floor();
        let f = (q - cell).as_vec3();
        let hashes = self.corner_hashes(cell.x as i64, cell.y as i64, cell.z as i64);
        let fade = |t: f32| t * t * (3.0 - 2.0 * t);
        let (u, v, w) = (fade(f.x), fade(f.y), fade(f.z));
        let (mut sum_c, mut sum_s) = (0.0f32, 0.0f32);
        for k in 0..8u32 {
            let (a, b, c) = ((k & 1) as f32, ((k >> 1) & 1) as f32, ((k >> 2) & 1) as f32);
            let h = hashes[k as usize];
            let jitter = Vec3::new(
                (h & 0x1F_FFFF) as f32 / 2_097_152.0 - 0.5,
                ((h >> 21) & 0x1F_FFFF) as f32 / 2_097_152.0 - 0.5,
                ((h >> 42) & 0x1F_FFFF) as f32 / 2_097_152.0 - 0.5,
            );
            let weight = (if a > 0.5 { u } else { 1.0 - u })
                * (if b > 0.5 { v } else { 1.0 - v })
                * (if c > 0.5 { w } else { 1.0 - w });
            let d = f - Vec3::new(a, b, c) - jitter;
            let (s, co) = turn_sin_cos(d.dot(wave));
            sum_c += weight * co;
            sum_s += weight * s;
        }
        // Normaliser completement ferait des pointes la ou les ondes s'annulent.
        let len = (sum_c * sum_c + sum_s * sum_s).sqrt().max(1e-5);
        let scale = (2.0 * len).min(1.0) / len;
        (sum_c * scale, sum_s * scale)
    }

    /// Ravines et cretes emboitees, alignees sur la pente. `slope` : gradient local du relief, en m/m,
    /// dans le plan tangent. `fade` : ce que devient le terrain plat, de -1 (creux) a 1 (sommet).
    /// Chaque octave suit le gradient cumule, donc les petites ravines descendent le flanc des grandes.
    pub fn erosion(
        &self,
        p: DVec3,
        up: Vec3,
        slope: Vec3,
        period: f64,
        octaves: u32,
        fade: f32,
        strength: f32,
    ) -> Eroded {
        let mut mask = ease_out(slope.length() * 9.0);
        let mut target = fade;
        let mut gradient = slope;
        let mut offset = 0.0;
        let mut size = period;
        let mut steep = GULLY_SLOPE * strength;
        for i in 0..octaves {
            let amp = size as f32 * steep / TAU;
            if mask > 1e-5 && gradient.length_squared() > 1e-12 {
                let wave = up.cross(gradient).normalize();
                let shift = DVec3::new(i as f64 * 17.3, i as f64 * -5.9, i as f64 * 41.7);
                let (c, s) = self.stripes(p / size + shift, wave);
                let faded = target + (c - target) * mask;
                offset += faded * amp;
                target = faded;
                // Le signe seul : les ravines plus petites branchent franchement au lieu de longer les grandes.
                gradient += wave * (-s.signum()) * steep * mask;
                mask = (1.0 - (1.0 - mask).powf(GULLY_DETAIL)) * smoothstep(0.0, 0.8, s.abs());
            } else {
                offset += target * amp;
            }
            size *= 0.5;
            steep *= GULLY_DECAY;
        }
        Eroded { offset, ridge: target }
    }

    /// Reliefs isoles dans une grille de mailles : la hauteur du plus haut en ce point. `chance` : part des
    /// mailles occupees. Le rayon reste sous 0.75 maille : jamais une tour ne se coupe en changeant de maille.
    /// Le profil monte de 0 a 1 sur la part `soft` du rayon ; la hauteur est tiree entre `low` et `low + range`.
    #[inline]
    pub fn towers(&self, q: DVec3, chance: f32, radius: f32, soft: f32, low: f32, range: f32) -> f32 {
        let cell = q.floor();
        let f = (q - cell).as_vec3();
        let hashes = self.corner_hashes(cell.x as i64, cell.y as i64, cell.z as i64);
        let mut best = 0.0f32;
        for k in 0..8u32 {
            let h = hashes[k as usize];
            if ((h >> 48) as f32 / 65_536.0) > chance {
                continue;
            }
            let jitter = Vec3::new(
                (h & 0xFFFF) as f32 / 65_536.0 * 0.5 - 0.25,
                ((h >> 16) & 0xFFFF) as f32 / 65_536.0 * 0.5 - 0.25,
                ((h >> 32) & 0xFFFF) as f32 / 65_536.0 * 0.5 - 0.25,
            );
            let d = (f - Vec3::new((k & 1) as f32, ((k >> 1) & 1) as f32, ((k >> 2) & 1) as f32) - jitter).length();
            let t = 1.0 - d / radius;
            if t > 0.0 {
                // Les tours qui se touchent se fondent : on garde la plus haute, sans saut.
                let tall = (h.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32 / 16_777_216.0;
                best = best.max(smoothstep(0.0, soft, t) * (low + range * tall));
            }
        }
        best
    }

    /// Le pivot qui couvre ce point, s'il y en a un : son centre (en mailles), sa proximite de 0 a 1,
    /// et un tirage propre a ce pivot. Sous un rayon de 0.25 maille, deux pivots ne se touchent jamais.
    pub fn pivot(&self, q: DVec3, radius: f32) -> Option<(DVec3, f32, f32)> {
        let cell = q.floor();
        let f = (q - cell).as_vec3();
        let hashes = self.corner_hashes(cell.x as i64, cell.y as i64, cell.z as i64);
        for k in 0..8u32 {
            let h = hashes[k as usize];
            let corner = Vec3::new((k & 1) as f32, ((k >> 1) & 1) as f32, ((k >> 2) & 1) as f32);
            let jitter = Vec3::new(
                (h & 0xFFFF) as f32 / 65_536.0 * 0.5 - 0.25,
                ((h >> 16) & 0xFFFF) as f32 / 65_536.0 * 0.5 - 0.25,
                ((h >> 32) & 0xFFFF) as f32 / 65_536.0 * 0.5 - 0.25,
            );
            let t = 1.0 - (f - corner - jitter).length() / radius;
            if t > 0.0 {
                return Some((cell + (corner + jitter).as_dvec3(), t, (h >> 48) as f32 / 65_536.0));
            }
        }
        None
    }

    /// Dunes : un versant long au vent, une crete vive, un versant court et raide sous le vent. Dans [0, 1].
    pub fn dunes(&self, q: DVec3, wave: Vec3) -> f32 {
        let (c, s) = self.stripes(q, wave);
        let strength = (c * c + s * s).sqrt().min(1.0);
        let phase = s.atan2(c) / TAU + 0.5;
        let profile = if phase < 0.72 { (phase / 0.72).powf(1.35) } else { 1.0 - smoothstep(0.72, 1.0, phase) };
        profile * strength
    }
}

/// Gradins : de longs replats, puis une falaise. `sharp` : part de la hauteur d'un gradin prise par la falaise.
pub fn terrace(h: f32, step: f32, sharp: f32) -> f32 {
    let sharp = sharp.min(0.92);
    let x = h / step;
    let i = x.floor();
    let t = x - i;
    let bench = t * 0.14 / (1.0 - sharp);
    let v = if t < 1.0 - sharp { bench } else { 0.14 + 0.86 * smoothstep(1.0 - sharp, 1.0, t) };
    (i + v) * step
}

/// Provinces : la nature du paysage. Leur somme vaut 1 en tout point de terre.
#[derive(Clone, Copy, Debug, Default)]
pub struct Province {
    /// Collines ravinees.
    pub hills: f32,
    /// Plateaux en gradins et canyons : pays secs, ou tepuis des pays tres humides.
    pub mesa: f32,
    /// Terres ravinees et bariolees.
    pub badlands: f32,
    /// Mer de dunes.
    pub dunes: f32,
    /// Tours calcaires.
    pub karst: f32,
    /// Landes rocheuses semees de lacs.
    pub moor: f32,
    /// Plaines d'ou surgissent des monolithes.
    pub inselberg: f32,
    /// Vallees glaciaires et fjords.
    pub glacial: f32,
}

pub struct Provinces {
    fields: [CubeField; 8],
    /// Teinte regionale, de 0 a 1.
    pub palette: CubeField,
    /// Automne des forets, de 0 a 1.
    pub autumn: CubeField,
}

impl Provinces {
    /// Calcule les provinces de relief d'apres l'altitude de base, l'orogenese et le climat.
    pub fn build(
        seed: u64,
        elevation: &CubeField,
        orogeny: &CubeField,
        climate: &(dyn Fn(DVec3, f32) -> Climate + Sync),
    ) -> Self {
        let grid = CubeGrid { n: 256 };
        let layer = |i: i64| Noise::new(hash_seed(seed, 0x9207, i));
        let noises: [Noise; 8] = std::array::from_fn(|i| layer(i as i64));
        let warp = (layer(20), layer(21), layer(22));
        // Des regions de quelques centaines de km : on en traverse plusieurs en un vol.
        let score_of = |c: usize| -> [f32; 8] {
            let dir = grid.center(c);
            let w = DVec3::new(
                warp.0.fbm(dir * 3.0, 3) as f64,
                warp.1.fbm(dir * 3.0, 3) as f64,
                warp.2.fbm(dir * 3.0, 3) as f64,
            );
            let at = dir + w * 0.28;
            let n = |i: usize| (noises[i].fbm(at * 9.0, 3) * 1.9 + 0.5).clamp(0.02, 1.2);
            let e = elevation.sample(dir).max(0.0);
            let o = orogeny.sample_linear(dir).clamp(0.0, 1.4);
            let cl = climate(dir, 0.0);
            let (t, m) = (cl.temperature, cl.moisture);
            let arid = 1.0 - smoothstep(0.26, 0.5, m);
            let tepui = smoothstep(0.62, 0.8, m) * smoothstep(0.55, 0.75, t);
            let inland = smoothstep(30.0, 160.0, e);
            let warm = smoothstep(0.28, 0.5, t);
            [
                0.16 + n(0) * 0.38,
                n(1) * (0.2 + 1.15 * arid.max(tepui * 0.85)) * inland * smoothstep(0.1, 0.3, t),
                n(2) * smoothstep(0.1, 0.24, m) * (1.0 - smoothstep(0.36, 0.5, m)) * warm * 1.1,
                n(3) * (1.0 - smoothstep(0.16, 0.3, m)) * warm * (1.0 - smoothstep(500.0, 1100.0, e)) * 1.35,
                n(4) * smoothstep(0.52, 0.7, t) * smoothstep(0.5, 0.68, m) * 1.05,
                n(5) * (1.0 - smoothstep(0.26, 0.48, t)) * smoothstep(0.22, 0.42, m) * 1.25,
                n(6) * smoothstep(0.42, 0.6, t) * smoothstep(0.16, 0.3, m) * (1.0 - smoothstep(0.55, 0.72, m)) * 1.1,
                (1.0 - smoothstep(0.16, 0.34, t)) * smoothstep(0.03, 0.22, o) * 1.7,
            ]
        };
        let weights = par_cells_many(grid, &|c| {
            let s = score_of(c);
            let powered = s.map(|v| v.max(0.0).powi(5));
            let sum: f32 = powered.iter().sum::<f32>().max(1e-9);
            powered.map(|v| v / sum)
        });
        let fields = std::array::from_fn(|i| {
            let values: Vec<f32> = weights.iter().map(|w| w[i]).collect();
            CubeField::from_cells(grid, &values)
        });
        let hue = layer(30);
        let palette = par_cells(grid, &|c| (hue.fbm(grid.center(c) * 5.0, 3) * 1.9 + 0.5).clamp(0.0, 1.0));
        let leaves = layer(31);
        let autumn = par_cells(grid, &|c| smoothstep(-0.02, 0.24, leaves.fbm(grid.center(c) * 7.0, 3)));
        Self { fields, palette: CubeField::from_cells(grid, &palette), autumn: CubeField::from_cells(grid, &autumn) }
    }

    /// Poids de chaque type de relief au lieu `dir`, interpoles entre les cellules.
    #[inline]
    pub fn at(&self, dir: DVec3) -> Province {
        let v: [f32; 8] =
            CubeField::sample_linear_many(std::array::from_fn(|i| &self.fields[i]), dir).map(|v| v.max(0.0));
        Province {
            hills: v[0],
            mesa: v[1],
            badlands: v[2],
            dunes: v[3],
            karst: v[4],
            moor: v[5],
            inselberg: v[6],
            glacial: v[7],
        }
    }
}

fn par_cells_many(grid: CubeGrid, f: &(dyn Fn(usize) -> [f32; 8] + Sync)) -> Vec<[f32; 8]> {
    let mut out = vec![[0.0f32; 8]; grid.cells()];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = grid.cells() / threads + 1;
    let radius = crate::world::generator::planet_radius();
    std::thread::scope(|s| {
        for (i, slice) in out.chunks_mut(chunk).enumerate() {
            s.spawn(move || {
                crate::world::generator::worker_thread(radius);
                for (k, v) in slice.iter_mut().enumerate() {
                    *v = f(i * chunk + k);
                }
            });
        }
    });
    out
}

/// Un volcan : un cone concave, un cratere, et des ravines rayonnantes que le filtre d'erosion y creuse seul.
#[derive(Clone, Copy, Debug)]
pub struct Volcano {
    pub dir: DVec3,
    /// Altitude du sommet, en metres.
    pub summit: f32,
    /// Rayon du pied, en metres.
    pub radius: f32,
}

pub struct Volcanoes {
    pub list: Vec<Volcano>,
    grid: CubeGrid,
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Volcanoes {
    /// Sur les arcs de subduction, les dorsales et quelques points chauds, dont les iles forment des chaines.
    pub fn build(seed: u64, plates: &Tectonics, elevation: &CubeField) -> Self {
        let mut rng = Rng(hash_seed(seed, 0x701C, 3));
        let mut list: Vec<Volcano> = Vec::new();
        let add = |list: &mut Vec<Volcano>, dir: DVec3, summit: f32, radius: f32| {
            if list.iter().all(|v| v.dir.dot(dir) < (130_000.0 / planet_radius()).cos()) {
                list.push(Volcano { dir, summit, radius });
            }
        };
        for c in 0..plates.grid.cells() {
            let chance = match plates.boundary[c] {
                Boundary::Subduction => 0.09,
                Boundary::Collision => 0.012,
                Boundary::Ridge => 0.006,
                Boundary::None => 0.0003,
            };
            if rng.f32() >= chance {
                continue;
            }
            let dir = plates.grid.center(c);
            let e = elevation.sample(dir);
            let summit = e.max(0.0) + rng.range(1500.0, 3600.0);
            let radius = rng.range(14_000.0, 30_000.0) * if e < -600.0 { 1.5 } else { 1.0 };
            add(&mut list, dir, summit, radius);
        }
        // Chaines d'iles : un point chaud fixe, une plaque qui derive au-dessus.
        for _ in 0..14 {
            let y = rng.range(-0.8, 0.8) as f64;
            let a = rng.range(0.0, std::f32::consts::TAU) as f64;
            let r = (1.0 - y * y).sqrt();
            let mut dir = DVec3::new(a.cos() * r, y, a.sin() * r);
            let side = dir.any_orthonormal_vector();
            for k in 0..rng.range(5.0, 11.0) as u32 {
                if elevation.sample(dir) < -300.0 {
                    let summit = (2600.0 - k as f32 * 240.0).max(380.0) * rng.range(0.7, 1.15);
                    add(&mut list, dir, summit, rng.range(26_000.0, 44_000.0));
                }
                dir = (dir + side * (rng.range(150_000.0, 260_000.0) as f64 / planet_radius())).normalize();
            }
        }

        let grid = CubeGrid { n: 48 };
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for (i, v) in list.iter().enumerate() {
            let s = v.dir.any_orthonormal_vector();
            let o = v.dir.cross(s);
            let mut cells = vec![grid.cell_of(v.dir) as u32];
            for ring in [0.5, 1.0] {
                for k in 0..12 {
                    let a = k as f64 / 12.0 * std::f64::consts::TAU;
                    let d = (v.radius as f64 + 2.0e5) * ring / planet_radius();
                    cells.push(grid.cell_of((v.dir + (s * a.cos() + o * a.sin()) * d).normalize()) as u32);
                }
            }
            cells.sort_unstable();
            cells.dedup();
            pairs.extend(cells.into_iter().map(|c| (c, i as u32)));
        }
        pairs.sort_unstable();
        let mut start = vec![0u32; grid.cells() + 1];
        for (c, _) in &pairs {
            start[*c as usize + 1] += 1;
        }
        for i in 0..grid.cells() {
            start[i + 1] += start[i];
        }
        Self { list, grid, start, items: pairs.into_iter().map(|p| p.1).collect() }
    }

    /// Le relief volcanique en un point : l'altitude que le volcan impose et sa part, de 0 a 1.
    #[inline]
    pub fn at(&self, dir: DVec3) -> Option<(&Volcano, f32)> {
        let c = self.grid.cell_of(dir);
        let mut best: Option<(&Volcano, f32)> = None;
        for &i in &self.items[self.start[c] as usize..self.start[c + 1] as usize] {
            let v = &self.list[i as usize];
            let d = (dir - v.dir).length() * planet_radius();
            let x = d as f32 / v.radius;
            if x < 1.0 {
                let share = cone(x);
                if best.is_none_or(|b| share > b.1) {
                    best = Some((v, share));
                }
            }
        }
        best
    }
}

/// Profil d'un volcan, de 1 au bord du cratere a 0 au pied. Le cratere redescend au centre.
fn cone(x: f32) -> f32 {
    const K: f32 = 3.4;
    const RIM: f32 = 0.045;
    let far = (-K).exp();
    let at = |x: f32| ((-K * x).exp() - far) / (1.0 - far);
    if x > RIM {
        at(x - RIM)
    } else {
        // Une cuvette, un peu moins haute que le rebord.
        1.0 - 0.1 * smoothstep(RIM, RIM * 0.45, x)
    }
}

/// Vents dominants : alizes d'est sous 30 degres, vents d'ouest jusqu'a 60, vents polaires au-dela.
fn wind(dir: DVec3) -> DVec3 {
    let lat = dir.y.abs().asin().to_degrees();
    let east = DVec3::Y.cross(dir).normalize_or_zero();
    let north = dir.cross(east);
    let toward_equator = -north * dir.y.signum();
    if lat < 30.0 {
        -east * 0.9 + toward_equator * 0.4
    } else if lat < 60.0 {
        east - toward_equator * 0.3
    } else {
        -east * 0.8 + toward_equator * 0.4
    }
}

/// L'humidite portee par les vents depuis la mer. Elle se perd en chemin, surtout en franchissant
/// une chaine : on obtient des cotes pluvieuses, des interieurs secs et des deserts d'abri.
pub fn moisture_field(plates: &Tectonics) -> CubeField {
    let grid = plates.grid;
    let spacing = planet_radius() * std::f64::consts::FRAC_PI_2 / grid.n as f64;
    let height = |c: usize| plates.orogeny[c].clamp(0.0, 1.4).powf(1.8) * 4200.0;
    let upstream: Vec<u32> = (0..grid.cells())
        .map(|c| {
            let dir = grid.center(c);
            grid.cell_of((dir - wind(dir).normalize_or_zero() * (spacing * 1.1 / planet_radius())).normalize()) as u32
        })
        .collect();
    let sea: Vec<bool> = plates.crust.iter().map(|c| *c < 0.5).collect();
    let mut wet: Vec<f32> = sea.iter().map(|s| if *s { 1.0 } else { 0.4 }).collect();
    let mut around = [0u32; 8];
    for _ in 0..70 {
        for c in 0..grid.cells() {
            if sea[c] {
                continue;
            }
            let u = upstream[c] as usize;
            let climb = (height(c) - height(u)).max(0.0);
            let loss = 0.018 + climb * 0.00028;
            let mut carried = wet[u];
            // Un peu de diffusion : les fronts de pluie debordent du vent moyen.
            let count = grid.neighbours(c, &mut around);
            let side: f32 = around[..count].iter().map(|n| wet[*n as usize]).sum::<f32>() / count.max(1) as f32;
            carried = carried * 0.8 + side * 0.2;
            wet[c] = (carried * (1.0 - loss)).clamp(0.0, 1.0);
        }
    }
    CubeField::from_cells(grid, &wet)
}

#[cfg(test)]
mod tests {
    #[test]
    fn fast_turn_sin_cos_matches_the_libm() {
        for i in -4000..4000 {
            let t = i as f32 * 0.000_731 + i as f32 * 0.001_3;
            let (s, c) = turn_sin_cos(t);
            let (es, ec) = (t as f64 * std::f64::consts::TAU).sin_cos();
            assert!((s as f64 - es).abs() < 1e-6 && (c as f64 - ec).abs() < 1e-6, "{t}");
        }
    }

    #[test]
    fn corner_hashes_are_the_plain_hashes() {
        let c = Cells::new(42);
        for (x, y, z) in [(0i64, 0i64, 0i64), (-3, 7, 12345), (99_999, -88_888, 7)] {
            let h = c.corner_hashes(x, y, z);
            for k in 0..8i64 {
                assert_eq!(h[k as usize], c.hash(x + (k & 1), y + ((k >> 1) & 1), z + ((k >> 2) & 1)));
            }
        }
    }

    use super::*;

    #[test]
    fn terraces_only_climb_and_stay_continuous() {
        let mut last = terrace(0.0, 100.0, 0.2);
        for i in 1..20_000 {
            let v = terrace(i as f32 * 0.05, 100.0, 0.2);
            assert!(v >= last - 1e-3, "{last} puis {v}");
            assert!(v - last < 5.0, "saut de {}", v - last);
            last = v;
        }
    }

    #[test]
    fn overlapping_towers_never_jump() {
        let cells = Cells::new(5);
        let mut last = cells.towers(DVec3::new(0.3, 0.4, 0.5), 0.9, 0.7, 0.5, 100.0, 300.0);
        for i in 1..200_000 {
            let q = DVec3::new(0.3 + i as f64 * 0.0001, 0.4 + i as f64 * 0.00003, 0.5);
            let v = cells.towers(q, 0.9, 0.7, 0.5, 100.0, 300.0);
            assert!((v - last).abs() < 1.0, "saut de {} a {q}", (v - last).abs());
            last = v;
        }
    }

    /// Le filtre d'erosion doit rester continu : les ravines s'effacent la ou leur direction change d'un coup.
    #[test]
    fn erosion_is_continuous_and_follows_the_slope() {
        let cells = Cells::new(9);
        let up = Vec3::Y;
        let slope = Vec3::new(0.3, 0.0, 0.1);
        let at = |x: f64, z: f64| cells.erosion(DVec3::new(x, 6.0e6, z), up, slope, 6000.0, 9, 0.0, 1.0).offset;
        let mut last = at(0.0, 0.0);
        for i in 1..50_000 {
            let v = at(i as f64 * 0.1, 0.0);
            assert!((v - last).abs() < 1.5, "saut de {} m", (v - last).abs());
            last = v;
        }
        // Les ravines courent dans le sens de la pente : le relief varie bien plus en travers qu'en long.
        let (mut along, mut across) = (0.0, 0.0);
        let dir = slope.normalize();
        for i in 0..4000 {
            let (x, z) = (i as f64 * 53.0, i as f64 * -91.0);
            let v = at(x, z);
            along += (at(x + dir.x as f64 * 300.0, z + dir.z as f64 * 300.0) - v).abs();
            across += (at(x - dir.z as f64 * 300.0, z + dir.x as f64 * 300.0) - v).abs();
        }
        assert!(across > along * 1.3, "en travers {across}, en long {along}");
    }

    #[test]
    fn volcanoes_have_a_crater_and_a_foot() {
        assert!(cone(0.0) < cone(0.045));
        assert!(cone(0.99) < 0.01);
        assert!(cone(0.1) > cone(0.5));
    }
}
