//! Circuits de course : un par ville, sur le trace d'un vrai circuit de Formule 1 (voir `circuit_data`).
//!
//! Chaque circuit est pose a l'echelle reelle hors de la ville, sur une terre seche et assez plate, loin
//! des autoroutes et des autres villes. Une route d'acces le relie a une avenue de la ville : la ville
//! recoit une sortie de plus, que son plan prolonge jusqu'a son contour. Le sol se plie au profil en
//! long du circuit, lisse sur 300 m, avec des degagements plats de chaque cote.

use super::circuit_data::CIRCUITS;
use super::city::City;
use super::fast_hash::FastMap;
use super::generator::{planet_radius, smoothstep, Detail, Generator, Surface};
use super::highway::Highways;
use super::noise::hash_seed;
use glam::{DVec2, DVec3};
use std::sync::OnceLock;

/// Demi-largeur de la piste (14 m) et de la route d'acces.
pub const TRACK_HALF: f64 = 7.0;
pub const ACCESS_HALF: f64 = 4.0;
/// Degagements de chaque cote de la piste : herbe rase, bacs a gravier, puis les glissieres.
pub const RUNOFF: f64 = 26.0;
/// Portee de l'influence sur le sol, depuis l'axe.
const INFLUENCE: f64 = 75.0;
/// Mailles de l'index, par cote de face du cube (environ 600 m sur une Terre).
const CELLS: f64 = 16384.0;
/// Rayon de la fenetre de lissage du profil en long, en points (8 m chacun).
const SMOOTH: usize = 18;

/// Un trace : la boucle de la piste, ou la route d'acces.
pub struct Path {
    /// Points sur la sphere, tous les 8 m environ.
    pub points: Vec<DVec3>,
    pub closed: bool,
    pub half: f64,
    /// Abscisse curviligne de chaque point.
    pub s: Vec<f64>,
}

impl Path {
    fn new(points: Vec<DVec3>, closed: bool, half: f64) -> Self {
        let r = planet_radius();
        let mut s = vec![0.0];
        let mut along = 0.0;
        for w in points.windows(2) {
            along += (w[1] - w[0]).length() * r;
            s.push(along);
        }
        Self { points, closed, half, s }
    }

    /// Nombre de segments : autant que de points pour une boucle, un de moins sinon.
    pub fn segments(&self) -> usize {
        if self.closed {
            self.points.len()
        } else {
            self.points.len() - 1
        }
    }

    /// Indice du point qui suit `k`, en bouclant sur le premier.
    pub fn next(&self, k: usize) -> usize {
        (k + 1) % self.points.len()
    }

    /// Longueur totale (boucle fermee comprise).
    pub fn length(&self) -> f64 {
        let n = self.points.len();
        if self.closed {
            self.s[n - 1] + (self.points[0] - self.points[n - 1]).length() * planet_radius()
        } else {
            self.s[n - 1]
        }
    }

    /// Tangente au point k, moyenne des deux segments voisins.
    pub fn tangent(&self, k: usize) -> DVec3 {
        let n = self.points.len();
        let prev = if k > 0 {
            k - 1
        } else if self.closed {
            n - 1
        } else {
            0
        };
        let next = if k + 1 < n {
            k + 1
        } else if self.closed {
            0
        } else {
            n - 1
        };
        let d = self.points[next] - self.points[prev];
        let up = self.points[k];
        (d - up * d.dot(up)).normalize_or(up.any_orthonormal_vector())
    }

    /// Courbure au point k (1/m), signee : positive quand la piste tourne a gauche.
    pub fn curvature(&self, k: usize) -> f64 {
        let n = self.points.len();
        if !self.closed && (k == 0 || k + 1 >= n) {
            return 0.0;
        }
        let a = self.tangent((k + n - 2) % n);
        let b = self.tangent((k + 2) % n);
        let up = self.points[k];
        let turn = a.cross(b).dot(up).asin();
        let len = (self.points[(k + 2) % n] - self.points[(k + n - 2) % n]).length() * planet_radius();
        turn / len.max(1.0)
    }
}

pub struct Circuit {
    /// Index dans `CIRCUITS`.
    pub shape: usize,
    pub city: usize,
    pub center: DVec3,
    /// Rayon du disque qui contient la piste.
    pub radius: f64,
    /// La piste, puis la route d'acces s'il y en a une.
    pub paths: Vec<Path>,
    /// Point de la piste ou arrive la route d'acces.
    pub junction: Option<usize>,
    /// Profil en long de chaque trace, calcule a la premiere demande (le sol des villes doit exister).
    levels: OnceLock<Vec<Vec<f64>>>,
}

impl Circuit {
    /// Nom du circuit reel dont le trace est repris.
    pub fn name(&self) -> &'static str {
        CIRCUITS[self.shape].name
    }

    /// Ville du circuit reel.
    pub fn location(&self) -> &'static str {
        CIRCUITS[self.shape].location
    }

    /// Altitude de la chaussee au point k du trace p.
    pub fn level(&self, gen: &Generator, p: usize, k: usize) -> f64 {
        self.levels(gen)[p][k]
    }

    /// Altitude de la chaussee en chaque point de chaque trace : le terrain naturel, lisse le long de
    /// la piste. Calcule a la premiere demande, puis garde.
    pub fn levels(&self, gen: &Generator) -> &Vec<Vec<f64>> {
        self.levels.get_or_init(|| {
            let detail = Detail::for_spacing(8.0);
            let natural = |d: DVec3| gen.surface_natural(d, &detail).height as f64;
            let track = &self.paths[0];
            let raw: Vec<f64> = track.points.iter().map(|d| natural(*d)).collect();
            let mut out = vec![limit_grade(smooth(&raw, SMOOTH, true), &track.points, true)];
            if let Some(access) = self.paths.get(1) {
                let raw: Vec<f64> = access.points.iter().map(|d| natural(*d)).collect();
                let mut lv = limit_grade(smooth(&raw, SMOOTH / 2, false), &access.points, false);
                // Raccords : au sol de la ville d'un cote, a la piste de l'autre.
                let n = lv.len();
                let end = self.junction.map_or(raw[n - 1], |j| out[0][j]);
                for (i, v) in lv.iter_mut().enumerate() {
                    let a = smoothstep(0.0, 12.0, i as f32) as f64;
                    let b = smoothstep(0.0, 15.0, (n - 1 - i) as f32) as f64;
                    *v = raw[0] + (*v - raw[0]) * a;
                    *v = end + (*v - end) * b;
                }
                out.push(lv);
            }
            out
        })
    }

    /// Point de depart : la ligne d'arrivee et le sens de la course.
    pub fn start(&self) -> (DVec3, DVec3) {
        let track = &self.paths[0];
        (track.points[0], track.tangent(0))
    }
}

/// Pente maximale d'une piste (Spa monte a 18 % dans le raidillon, la plupart restent sous 10 %).
const MAX_GRADE: f64 = 0.12;

/// Ecrete les pentes trop fortes : deux points voisins trop eloignes en altitude se rapprochent.
fn limit_grade(mut v: Vec<f64>, points: &[DVec3], closed: bool) -> Vec<f64> {
    let n = v.len();
    let pairs = if closed { n } else { n - 1 };
    let r = super::generator::planet_radius();
    for _ in 0..2000 {
        let mut worst: f64 = 0.0;
        for k in 0..pairs {
            let j = (k + 1) % n;
            let step = MAX_GRADE * (points[j] - points[k]).length() * r;
            let d = v[j] - v[k];
            if d.abs() > step {
                let fix = (d.abs() - step) * 0.5 * d.signum();
                v[k] += fix;
                v[j] -= fix;
                worst = worst.max(d.abs() - step);
            }
        }
        if worst < 0.001 {
            break;
        }
    }
    // Ecretage final aller-retour : garantit la pente maximale meme si la relaxation n'a pas converge.
    let step = |k: usize, j: usize| MAX_GRADE * (points[j] - points[k]).length() * r * 0.999;
    for _ in 0..4 {
        for k in 0..pairs {
            let j = (k + 1) % n;
            let m = step(k, j);
            v[j] = v[j].clamp(v[k] - m, v[k] + m);
        }
        for k in (0..pairs).rev() {
            let j = (k + 1) % n;
            let m = step(k, j);
            v[k] = v[k].clamp(v[j] - m, v[j] + m);
        }
    }
    v
}

/// Moyenne glissante appliquee deux fois (boucle ou trace ouvert).
fn smooth(v: &[f64], r: usize, closed: bool) -> Vec<f64> {
    let n = v.len();
    let mut cur = v.to_vec();
    for _ in 0..2 {
        let mut next = vec![0.0; n];
        for (i, x) in next.iter_mut().enumerate() {
            let (mut sum, mut count) = (0.0, 0.0);
            for o in -(r as i64)..=(r as i64) {
                let j = i as i64 + o;
                let j = if closed {
                    j.rem_euclid(n as i64) as usize
                } else if j < 0 || j >= n as i64 {
                    continue;
                } else {
                    j as usize
                };
                sum += cur[j];
                count += 1.0;
            }
            *x = sum / count;
        }
        cur = next;
    }
    cur
}

/// Le point le plus proche sur un trace.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub circuit: usize,
    pub path: usize,
    /// Segment, et position sur ce segment (0 a 1).
    pub k: usize,
    pub t: f64,
    /// Distance a l'axe, en metres.
    pub dist: f64,
    /// Cote : positif a gauche du sens de la course.
    pub side: f64,
}

/// Morceaux de piste par maille de la sphere : (circuit, trace, segment).
type TrackIndex = FastMap<(u8, i32, i32), Vec<(u32, u8, u32)>>;

pub struct Circuits {
    pub list: Vec<Circuit>,
    index: TrackIndex,
}

fn key(dir: DVec3) -> (u8, i32, i32) {
    let a = dir.abs();
    let (face, u, v) = if a.x >= a.y && a.x >= a.z {
        (if dir.x > 0.0 { 0 } else { 1 }, dir.y / a.x, dir.z / a.x)
    } else if a.y >= a.z {
        (if dir.y > 0.0 { 2 } else { 3 }, dir.x / a.y, dir.z / a.y)
    } else {
        (if dir.z > 0.0 { 4 } else { 5 }, dir.x / a.z, dir.y / a.z)
    };
    let q = |t: f64| ((t + 1.0) * 0.5 * CELLS).floor().clamp(0.0, CELLS - 1.0) as i32;
    (face, q(u), q(v))
}

/// Repere tangent en un point : est, nord.
fn tangent_frame(dir: DVec3) -> (DVec3, DVec3) {
    let east = if dir.y.abs() > 0.999 { DVec3::X } else { DVec3::Y.cross(dir).normalize() };
    (east, dir.cross(east))
}

/// Ecart angulaire entre deux droites passant par le centre de la ville.
fn line_gap(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::PI);
    d.min(std::f64::consts::PI - d)
}

/// Un emplacement possible.
#[derive(Clone)]
struct Candidate {
    city: usize,
    shape: usize,
    angle: f64,
    center: DVec3,
    radius: f64,
    track: Vec<DVec3>,
    cost: f64,
}

impl Circuits {
    /// Aucun circuit : pour une planete sans villes, ou pendant sa construction.
    pub fn empty() -> Self {
        Self { list: Vec::new(), index: FastMap::default() }
    }

    /// Place un circuit pres de chaque ville qui a la place, et lui ouvre une sortie.
    pub fn build(gen: &Generator, cities: &mut [City], highways: &Highways) -> Self {
        let r = planet_radius();
        let seed = gen.params.seed;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).max(1);
        let n = cities.len();
        let cities_ref: &[City] = cities;
        let mut found: Vec<Vec<Candidate>> = vec![Vec::new(); n];
        std::thread::scope(|scope| {
            let chunks: Vec<_> = (0..threads)
                .map(|t| {
                    scope.spawn(move || {
                        let mut out = Vec::new();
                        for i in (t..n).step_by(threads) {
                            out.push((i, Self::candidates(gen, cities_ref, highways, i, seed)));
                        }
                        out
                    })
                })
                .collect();
            for c in chunks {
                for (i, list) in crate::sync::join(c) {
                    found[i] = list;
                }
            }
        });

        // Les villes choisissent tour a tour, sans que deux circuits se touchent.
        let mut list: Vec<Circuit> = Vec::new();
        for (i, options) in found.into_iter().enumerate() {
            let Some(c) = options
                .into_iter()
                .find(|c| list.iter().all(|o| (o.center - c.center).length() * r > o.radius + c.radius + 400.0))
            else {
                continue;
            };
            let city = &cities[i];
            // Route d'acces : du contour de la ville au point de piste le plus proche.
            let from = city.dir_of(DVec2::from_angle(c.angle) * (city.reach(c.angle) - 6.0));
            let (j, _) = c.track.iter().enumerate().fold((0, f64::MAX), |b, (k, p)| {
                let d = (*p - from).length();
                if d < b.1 {
                    (k, d)
                } else {
                    b
                }
            });
            let to = c.track[j];
            let len = (to - from).length() * r;
            let steps = (len / 8.0).ceil().max(2.0) as usize;
            let access: Vec<DVec3> = (0..=steps).map(|k| from.lerp(to, k as f64 / steps as f64).normalize()).collect();
            let detail = Detail::for_spacing(40.0);
            let clear = access.iter().enumerate().all(|(k, d)| {
                let s = gen.surface_natural(*d, &detail);
                let near_end = k + 3 >= access.len() || k < 3;
                (near_end || s.height > s.water_level + 1.0)
                    && highways.clearance(*d).is_none_or(|c| c > ACCESS_HALF + 6.0)
            });
            let mut paths = vec![Path::new(c.track.clone(), true, TRACK_HALF)];
            let mut junction = None;
            if clear && len < 4000.0 {
                // L'avenue de sortie part de la piste : la ville la prolonge jusqu'a son contour.
                paths.push(Path::new(access, false, ACCESS_HALF));
                junction = Some(j);
                cities[i].exits.push(c.angle);
            }
            list.push(Circuit {
                shape: c.shape,
                city: i,
                center: c.center,
                radius: c.radius,
                paths,
                junction,
                levels: OnceLock::new(),
            });
        }

        let mut index = TrackIndex::default();
        for (ci, c) in list.iter().enumerate() {
            for (pi, p) in c.paths.iter().enumerate() {
                for k in 0..p.segments() {
                    let (a, b) = (p.points[k], p.points[p.next(k)]);
                    let (e, nn) = tangent_frame(a);
                    let mut keys = Vec::with_capacity(10);
                    for q in [a, b] {
                        keys.push(key(q));
                        for (dx, dy) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                            keys.push(key((q + (e * dx + nn * dy) * (INFLUENCE / r)).normalize()));
                        }
                    }
                    keys.sort_unstable();
                    keys.dedup();
                    for kk in keys {
                        index.entry(kk).or_default().push((ci as u32, pi as u8, k as u32));
                    }
                }
            }
        }
        Self { list, index }
    }

    /// Les meilleurs emplacements pour le circuit d'une ville, du meilleur au moins bon.
    fn candidates(gen: &Generator, cities: &[City], highways: &Highways, i: usize, seed: u64) -> Vec<Candidate> {
        let r = planet_radius();
        let city = &cities[i];
        let count = CIRCUITS.len();
        let first = (hash_seed(seed ^ 0xF1, i as i64, 0) % count as u64) as usize;
        let detail = Detail::for_spacing(120.0);
        let mut best: Vec<Candidate> = Vec::new();
        for attempt in 0..3 {
            let shape = (first + attempt * 13) % count;
            let pts = CIRCUITS[shape].points;
            let radius = pts.iter().map(|p| (p[0] as f64).hypot(p[1] as f64)).fold(0.0, f64::max);
            for a in 0..12 {
                let angle = (a as f64 + 0.5) / 12.0 * std::f64::consts::TAU;
                if !city.exit_clear(angle) || city.exits.iter().any(|e| line_gap(*e, angle) < 0.3) {
                    continue;
                }
                for gap in [600.0, 1500.0] {
                    let dist = city.reach(angle) + radius + gap;
                    let center = city.dir_of(DVec2::from_angle(angle) * dist);
                    // Loin des villes (la sienne comprise).
                    if cities.iter().any(|o| (o.dir - center).length() * r < radius + o.radius * 1.35 + 250.0) {
                        continue;
                    }
                    let (east, north) = tangent_frame(center);
                    for rot in 0..4 {
                        let turn = rot as f64 * std::f64::consts::FRAC_PI_2
                            + (hash_seed(seed, i as i64, rot) % 1000) as f64 * 0.0006;
                        let (sn, cs) = turn.sin_cos();
                        let track: Vec<DVec3> = pts
                            .iter()
                            .map(|p| {
                                let (x, y) = (p[0] as f64 * cs - p[1] as f64 * sn, p[0] as f64 * sn + p[1] as f64 * cs);
                                (center * r + east * x + north * y).normalize()
                            })
                            .collect();
                        if track
                            .iter()
                            .step_by(4)
                            .any(|d| highways.clearance(*d).is_some_and(|c| c < TRACK_HALF + RUNOFF + 15.0))
                        {
                            continue;
                        }
                        // Terre seche, et le moins de denivele possible.
                        let worst = best.last().map_or(f64::MAX, |b| b.cost);
                        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
                        let mut ok = true;
                        for d in track.iter().step_by(6) {
                            let s = gen.surface_natural(*d, &detail);
                            if s.height < s.water_level + 3.0 || s.fresh {
                                ok = false;
                                break;
                            }
                            lo = lo.min(s.height as f64);
                            hi = hi.max(s.height as f64);
                            if hi - lo + gap * 0.01 > worst && best.len() >= 3 {
                                ok = false;
                                break;
                            }
                        }
                        if !ok {
                            continue;
                        }
                        let cost = hi - lo + gap * 0.01 + attempt as f64 * 40.0;
                        best.push(Candidate { city: i, shape, angle, center, radius, track, cost });
                        best.sort_by(|a, b| a.cost.total_cmp(&b.cost));
                        best.truncate(3);
                    }
                }
            }
            if !best.is_empty() {
                break;
            }
        }
        let _ = best.iter().map(|c| c.city);
        best
    }

    /// Le point le plus proche de chaque trace a moins de `reach` metres.
    pub fn near(&self, dir: DVec3, reach: f64, mut f: impl FnMut(Hit)) {
        if self.index.is_empty() {
            return;
        }
        let Some(segs) = self.index.get(&key(dir)) else { return };
        let r = planet_radius();
        let mut best: FastMap<(u32, u8), Hit> = FastMap::default();
        for &(c, p, k) in segs {
            let path = &self.list[c as usize].paths[p as usize];
            let (a, b) = (path.points[k as usize], path.points[path.next(k as usize)]);
            let ab = b - a;
            let t = ((dir - a).dot(ab) / ab.length_squared().max(1e-30)).clamp(0.0, 1.0);
            let q = a + ab * t;
            let dist = (dir - q).length() * r;
            if dist < reach && best.get(&(c, p)).is_none_or(|h| dist < h.dist) {
                let side = ab.cross(dir - q).dot(dir).signum();
                best.insert((c, p), Hit { circuit: c as usize, path: p as usize, k: k as usize, t, dist, side });
            }
        }
        for h in best.into_values() {
            f(h);
        }
    }

    /// Altitude de la chaussee au point le plus proche.
    pub fn level_at(&self, gen: &Generator, h: &Hit) -> f64 {
        let c = &self.list[h.circuit];
        let path = &c.paths[h.path];
        let lv = &c.levels(gen)[h.path];
        lv[h.k] + (lv[path.next(h.k)] - lv[h.k]) * h.t
    }

    /// Le sol se plie au circuit : la piste et ses degagements a plat, des talus au-dela.
    pub fn shape_ground(&self, gen: &Generator, dir: DVec3, s: &mut Surface) {
        if self.index.is_empty() || !self.index.contains_key(&key(dir)) {
            return;
        }
        let (mut weight, mut level) = (0.0f32, 0.0f32);
        self.near(dir, INFLUENCE, |hit| {
            let c = &self.list[hit.circuit];
            let lv = self.level_at(gen, &hit);
            let flat = c.paths[hit.path].half + if hit.path == 0 { RUNOFF } else { 3.0 };
            let dh = (lv - s.height as f64).abs().min(40.0);
            let w = 1.0 - smoothstep(flat as f32, (flat + 8.0 + dh * 1.5) as f32, hit.dist as f32);
            if w > weight {
                (weight, level) = (w, lv as f32 - 0.02);
            }
        });
        if weight > 0.0 {
            s.height += (level - s.height) * weight;
            s.ridge *= 1.0 - weight;
            // Plus d'eau ici (une mare sur la piste n'aurait pas de sens).
            if weight > 0.5 && s.water_level > s.height && s.water_level > 0.5 {
                s.water_level = s.height - 1.0;
            }
        }
    }

    /// Distance au bord de piste ou de route le plus proche (negative dessus).
    pub fn clearance(&self, dir: DVec3) -> Option<f64> {
        let mut best: Option<f64> = None;
        self.near(dir, INFLUENCE, |hit| {
            let d = hit.dist - self.list[hit.circuit].paths[hit.path].half;
            best = Some(best.map_or(d, |b: f64| b.min(d)));
        });
        best
    }

    /// Vrai sur la chaussee, ou a moins de 30 cm de son bord.
    pub fn on_track(&self, dir: DVec3) -> bool {
        self.clearance(dir).is_some_and(|d| d < 0.3)
    }

    /// Couleur du sol : asphalte, herbe rase et bacs a gravier a l'exterieur des virages.
    pub fn paint(&self, dir: DVec3, spacing: f64) -> Option<([f32; 3], f32)> {
        let mut out: Option<([f32; 3], f32)> = None;
        self.near(dir, INFLUENCE, |hit| {
            let c = &self.list[hit.circuit];
            let path = &c.paths[hit.path];
            let half = path.half;
            let paint = if hit.dist < half + spacing * 0.5 {
                let k = if spacing < half { 1.0 } else { (half * 4.0 / spacing).min(0.85) as f32 };
                Some(([0.045, 0.045, 0.05], k))
            } else if hit.path == 0 && hit.dist < half + RUNOFF {
                // Courbure positive : la piste tourne a gauche, l'exterieur est a droite.
                let bend = path.curvature(hit.k);
                let outside = bend * hit.side < 0.0 && bend.abs() > 1.0 / 160.0;
                if outside && hit.dist > half + 5.0 {
                    Some(([0.36, 0.32, 0.24], 0.95))
                } else {
                    Some(([0.06, 0.13, 0.035], 0.6))
                }
            } else {
                None
            };
            if let Some(p) = paint {
                if out.is_none_or(|o| p.1 > o.1) {
                    out = Some(p);
                }
            }
        });
        out
    }

    /// Le circuit le plus proche d'un point.
    pub fn nearest(&self, dir: DVec3) -> Option<&Circuit> {
        self.list.iter().max_by(|a, b| a.center.dot(dir).total_cmp(&b.center.dot(dir)))
    }

    /// Morceaux de trace dans la maille d'un point : (circuit, trace, segment).
    pub fn segments_at(&self, dir: DVec3) -> &[(u32, u8, u32)] {
        self.index.get(&key(dir)).map_or(&[], |v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::generator::PlanetParams;

    #[test]
    fn the_shapes_are_real_f1_circuits_at_scale() {
        assert!(CIRCUITS.len() >= 30);
        for c in CIRCUITS {
            let n = c.points.len();
            let len: f64 = (0..n)
                .map(|k| {
                    let (a, b) = (c.points[k], c.points[(k + 1) % n]);
                    ((a[0] - b[0]) as f64).hypot((a[1] - b[1]) as f64)
                })
                .sum();
            // Le lissage raccourcit un peu les virages serres.
            assert!((len / c.length as f64 - 1.0).abs() < 0.06, "{} : {len:.0} m pour {} m", c.name, c.length);
        }
    }

    #[test]
    fn cities_get_circuits_on_flat_dry_land_with_an_access_road() {
        let gen = Generator::new(PlanetParams { seed: 3, cities: 40, ..Default::default() });
        let circuits = &gen.cities.circuits;
        assert!(circuits.list.len() >= 20, "{} circuits pour 40 villes", circuits.list.len());
        let linked = circuits.list.iter().filter(|c| c.paths.len() == 2).count();
        assert!(linked * 10 >= circuits.list.len() * 7, "{linked} routes d'acces sur {}", circuits.list.len());
        for c in &circuits.list {
            let track = &c.paths[0];
            for (k, d) in track.points.iter().enumerate().step_by(10) {
                let s = gen.surface(*d);
                let lv = c.level(&gen, 0, k);
                assert!((s.height as f64 - lv).abs() < 0.6, "{} : sol {} piste {lv}", c.name(), s.height);
                assert!(s.height > s.water_level, "{} sous l'eau", c.name());
                assert!(circuits.on_track(*d));
            }
            // Une pente de circuit, pas de montagne russe.
            let lv = &c.levels(&gen)[0];
            for k in 0..lv.len() {
                let j = (k + 1) % lv.len();
                let run = (track.points[j] - track.points[k]).length() * crate::world::generator::planet_radius();
                let grade = (lv[j] - lv[k]).abs() / run.max(1.0);
                assert!(grade < 0.13, "{} : pente {grade:.2}", c.name());
            }
        }
    }
}
