//! Autoroutes : les villes d'un meme continent sont toutes reliees.
//!
//! - Le reseau est le graphe de voisinage relatif des villes (Toussaint, 1980) : deux villes sont
//!   reliees si aucune autre n'est a la fois plus proche de l'une et de l'autre. Il contient l'arbre
//!   couvrant minimal, donc il relie tout, et deux autoroutes quittent une ville a 60 degres au moins.
//!   Les liaisons impossibles (un bras de mer) sont remplacees par d'autres, entre les morceaux restes
//!   separes, tant qu'un chemin par la terre existe.
//! - Chaque liaison est cherchee par A* sur une grille posee sur le relief : les pentes, l'altitude, les
//!   lacs et les villes coutent cher, la mer est interdite. Le chemin est simplifie (Douglas-Peucker),
//!   puis arrondi en grandes courbes (Chaikin).
//! - Le profil en long suit le terrain lisse sur 400 m, monte a 7.5 m au-dessus de l'eau pour les ponts,
//!   et enjambe les autres autoroutes. Le terrain se plie a la chaussee : remblais et deblais, sauf sous
//!   les viaducs.
//! - Une autoroute part d'une sortie de la ville : une avenue de son plan prolongee jusqu'au bord.

use super::city::City;
use super::fast_hash::FastMap;
use super::generator::{planet_radius, smoothstep, Detail, Generator, Surface};
use glam::{DVec2, DVec3};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::OnceLock;

/// Ecart entre deux points du trace, en metres.
pub const POINT_STEP: f64 = 100.0;
/// Ecart entre deux points du profil en long.
pub const STEP: f64 = 20.0;
const SECTION: usize = 200;
const MARGIN: usize = 64;
/// Portee de l'influence d'une autoroute sur le terrain, depuis son axe.
pub const INFLUENCE: f64 = 180.0;
/// Mailles de l'index par face du cube : 2.4 km sur une planete comme la Terre.
const CELLS: f64 = 4096.0;
/// Hauteur du tablier au-dessus de l'eau, et au-dessus d'une autre autoroute.
const WATER_CLEARANCE: f64 = 7.5;
const OVERPASS: f64 = 10.5;
/// Pente des rampes d'acces aux ponts.
const RAMP: f64 = 0.045;
/// Au-dela de cette hauteur au-dessus du sol naturel, la chaussee passe sur un viaduc.
pub const VIADUCT: f64 = 9.0;
/// Distance de la sortie au bord de la ville, ou commence la recherche du chemin.
const LEAD: f64 = 1500.0;

/// Le profil en long en un point.
#[derive(Clone, Copy, Debug)]
pub struct Profile {
    /// Altitude de la chaussee.
    pub road: f64,
    /// Sol naturel sous l'axe (le fond, sous l'eau).
    pub ground: f64,
    /// De l'eau sous l'axe.
    pub wet: bool,
}

impl Profile {
    /// Vrai si la chaussee passe sur un pont : au-dessus de l'eau, ou trop haut pour un remblai.
    pub fn bridge(&self) -> bool {
        self.wet || self.road - self.ground > VIADUCT
    }
}

pub struct Highway {
    pub from: usize,
    pub to: usize,
    /// Voies par sens.
    pub lanes: u8,
    points: Vec<DVec3>,
    length: f64,
    sections: Vec<OnceLock<Vec<[f32; 3]>>>,
    /// Passages au-dessus d'autres autoroutes : (abscisse ici, autoroute, abscisse sur elle).
    over: Vec<(f64, usize, f64)>,
}

impl Highway {
    fn new(from: usize, to: usize, lanes: u8, line: &[DVec3]) -> Self {
        let points = resample(line, POINT_STEP);
        let r = planet_radius();
        let length = points.windows(2).map(|w| (w[1] - w[0]).length() * r).sum::<f64>();
        let samples = (length / STEP).ceil() as usize + 1;
        let sections = (0..(samples - 1).div_ceil(SECTION).max(1)).map(|_| OnceLock::new()).collect();
        Self { from, to, lanes, points, length, sections, over: Vec::new() }
    }

    /// Longueur de l'autoroute, en metres.
    pub fn length(&self) -> f64 {
        self.length
    }

    /// Le trace : directions depuis le centre de la planete, a pas constant.
    pub fn points(&self) -> &[DVec3] {
        &self.points
    }

    /// Demi-terre-plein central.
    pub fn median(&self) -> f64 {
        1.6
    }

    /// Largeur d'une chaussee : bande de guidage, voies de 3.5 m, bande d'arret d'urgence.
    pub fn carriage(&self) -> f64 {
        0.9 + self.lanes as f64 * 3.5 + 2.6
    }

    /// Demi-largeur totale, accotement compris.
    pub fn half_width(&self) -> f64 {
        self.median() + self.carriage() + 1.0
    }

    fn samples(&self) -> usize {
        (self.length / STEP).ceil() as usize + 1
    }

    fn seg_len(&self, i: usize) -> f64 {
        if i + 2 >= self.points.len() {
            (self.length - i as f64 * POINT_STEP).max(1e-3)
        } else {
            POINT_STEP
        }
    }

    /// Direction du point d'abscisse s.
    pub fn dir_at(&self, s: f64) -> DVec3 {
        let n = self.points.len();
        let i = ((s / POINT_STEP).floor().max(0.0) as usize).min(n - 2);
        let t = ((s - i as f64 * POINT_STEP) / self.seg_len(i)).clamp(0.0, 1.0);
        self.points[i].lerp(self.points[i + 1], t).normalize()
    }

    /// Direction de la route, dans le plan tangent.
    pub fn tangent_at(&self, s: f64) -> DVec3 {
        let (a, b) = (self.dir_at((s - 15.0).max(0.0)), self.dir_at((s + 15.0).min(self.length)));
        let up = self.dir_at(s);
        let t = b - a;
        (t - up * t.dot(up)).normalize_or(up.any_orthonormal_vector())
    }
}

/// Un point proche d'une autoroute.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub hw: usize,
    /// Abscisse le long de l'autoroute.
    pub s: f64,
    /// Distance a l'axe, en metres.
    pub dist: f64,
}

/// Morceaux de trace par maille de la sphere : (autoroute, premier segment, dernier segment).
type RouteIndex = FastMap<(u8, i32, i32), Vec<(u32, u32, u32)>>;

pub struct Highways {
    pub list: Vec<Highway>,
    index: RouteIndex,
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

/// Polyligne sur la sphere, re-echantillonnee a pas constant (le dernier pas est plus court).
fn resample(line: &[DVec3], step: f64) -> Vec<DVec3> {
    let (Some(&first), Some(&end)) = (line.first(), line.last()) else { return Vec::new() };
    let r = planet_radius();
    let mut out = vec![first];
    let mut carry = 0.0;
    for w in line.windows(2) {
        let len = (w[1] - w[0]).length() * r;
        let mut s = step - carry;
        while s < len {
            out.push(w[0].lerp(w[1], s / len).normalize());
            s += step;
        }
        carry = len - (s - step);
    }
    // Un dernier pas tombe sur l'arrivee, a un metre pres, est remplace par elle.
    let replace = carry <= 1.0 && out.len() >= 2;
    match out.last_mut() {
        Some(last) if replace => *last = end,
        _ => out.push(end),
    }
    out
}

/// Moyenne glissante, fenetre tronquee aux bouts.
fn box_filter(v: &[f64], r: usize) -> Vec<f64> {
    let n = v.len();
    let mut prefix = vec![0.0; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + v[i];
    }
    (0..n)
        .map(|i| {
            let (a, b) = (i.saturating_sub(r), (i + r + 1).min(n));
            (prefix[b] - prefix[a]) / (b - a) as f64
        })
        .collect()
}

impl Highways {
    /// Aucune autoroute : pour une planete sans villes, ou pendant sa construction.
    pub fn empty() -> Self {
        Self { list: Vec::new(), index: FastMap::default() }
    }

    /// Le profil en long d'une autoroute, calcule par sections de 4 km a l'approche.
    pub fn profile(&self, gen: &Generator, h: usize, s: f64) -> Profile {
        let hw = &self.list[h];
        let n = hw.samples();
        let x = (s / STEP).clamp(0.0, (n - 1) as f64);
        let i = (x.floor() as usize).min(n - 2);
        let t = (x - i as f64).clamp(0.0, 1.0);
        let sec = (i / SECTION).min(hw.sections.len() - 1);
        let data = hw.sections[sec].get_or_init(|| self.compute(gen, h, sec));
        let (a, b) = (data[i - sec * SECTION], data[(i + 1 - sec * SECTION).min(data.len() - 1)]);
        let lerp = |k: usize| a[k] as f64 + (b[k] as f64 - a[k] as f64) * t;
        Profile { road: lerp(0), ground: lerp(1), wet: a[2].max(b[2]) > 0.5 }
    }

    fn compute(&self, gen: &Generator, h: usize, sec: usize) -> Vec<[f32; 3]> {
        let hw = &self.list[h];
        let n = hw.samples();
        let first = sec * SECTION;
        let last = ((sec + 1) * SECTION).min(n - 1);
        let lo = first.saturating_sub(MARGIN);
        let hi = (last + MARGIN).min(n - 1);
        let detail = Detail::for_spacing(STEP);
        let raw: Vec<(f64, f64, bool)> = (lo..=hi)
            .map(|i| {
                let s = (i as f64 * STEP).min(hw.length);
                let sf = gen.surface_natural(hw.dir_at(s), &detail);
                (sf.height as f64, sf.water_level as f64, sf.height < sf.water_level + 0.5)
            })
            .collect();
        // Le terrain lisse sur 400 m ; l'eau compte a son niveau, pas a son fond.
        let base: Vec<f64> = raw.iter().map(|r| if r.2 { r.1 } else { r.0 }).collect();
        let smooth = box_filter(&box_filter(&base, 10), 10);
        let mut target = vec![f64::MIN; raw.len()];
        for (k, r) in raw.iter().enumerate() {
            if r.2 {
                target[k] = r.1 + WATER_CLEARANCE;
            }
        }
        for &(s, other, so) in &hw.over {
            let k = (s / STEP).round() as isize - lo as isize;
            if k + 4 < 0 || k - 4 >= raw.len() as isize {
                continue;
            }
            let level = self.profile(gen, other, so).road + OVERPASS;
            for d in -4..=4 {
                let j = k + d;
                if (0..raw.len() as isize).contains(&j) {
                    target[j as usize] = target[j as usize].max(level);
                }
            }
        }
        let any = target.iter().any(|t| *t > f64::MIN);
        (first..=last)
            .map(|i| {
                let k = i - lo;
                let mut road = smooth[k];
                if any {
                    let (a, b) = (k.saturating_sub(MARGIN), (k + MARGIN).min(raw.len() - 1));
                    for (j, t) in target.iter().enumerate().take(b + 1).skip(a) {
                        if *t > f64::MIN {
                            road = road.max(t - (k as f64 - j as f64).abs() * STEP * RAMP);
                        }
                    }
                }
                [road as f32, raw[k].0 as f32, raw[k].2 as u8 as f32]
            })
            .collect()
    }

    /// Les autoroutes a moins de `reach` metres : leur point le plus proche, par morceau de trace.
    pub fn near(&self, dir: DVec3, reach: f64, mut f: impl FnMut(Hit)) {
        if self.index.is_empty() {
            return;
        }
        let Some(runs) = self.index.get(&key(dir)) else { return };
        let r = planet_radius();
        for &(h, first, last) in runs {
            let hw = &self.list[h as usize];
            let mut best: Option<Hit> = None;
            for k in first..=last {
                let k = k as usize;
                let (a, b) = (hw.points[k], hw.points[k + 1]);
                let ab = b - a;
                let t = ((dir - a).dot(ab) / ab.length_squared().max(1e-30)).clamp(0.0, 1.0);
                let dist = (dir - (a + ab * t)).length() * r;
                if dist < reach && best.is_none_or(|b| dist < b.dist) {
                    best = Some(Hit { hw: h as usize, s: k as f64 * POINT_STEP + t * hw.seg_len(k), dist });
                }
            }
            if let Some(hit) = best {
                f(hit);
            }
        }
    }

    /// Le terrain se plie aux autoroutes : remblai ou deblai sous la chaussee, talus de chaque cote.
    pub fn shape_ground(&self, gen: &Generator, dir: DVec3, s: &mut Surface) {
        if self.index.is_empty() || s.height < s.water_level + 0.5 {
            return;
        }
        let (mut weight, mut level) = (0.0f32, 0.0f32);
        self.near(dir, INFLUENCE, |hit| {
            let hw = &self.list[hit.hw];
            let p = self.profile(gen, hit.hw, hit.s);
            let dh = (p.road - s.height as f64).abs().min(60.0);
            let inner = hw.half_width() + 1.0;
            let mut w = 1.0 - smoothstep(inner as f32, (inner + 10.0 + dh * 1.5) as f32, hit.dist as f32);
            // Sous un viaduc, le sol reste naturel ; une tranchee trop profonde s'efface.
            w *= 1.0 - smoothstep(VIADUCT as f32 - 1.0, VIADUCT as f32 + 2.0, (p.road - p.ground) as f32);
            w *= 1.0 - smoothstep(30.0, 45.0, (p.ground - p.road) as f32);
            if w > weight {
                (weight, level) = (w, p.road as f32 - 0.3);
            }
        });
        if weight > 0.0 {
            s.height += (level - s.height) * weight;
            s.ridge *= 1.0 - weight;
        }
    }

    /// Distance au bord de la chaussee la plus proche (negative dessus), si une autoroute est a portee.
    pub fn clearance(&self, dir: DVec3) -> Option<f64> {
        let mut best: Option<f64> = None;
        self.near(dir, INFLUENCE, |hit| {
            let d = hit.dist - self.list[hit.hw].half_width();
            best = Some(best.map_or(d, |b: f64| b.min(d)));
        });
        best
    }

    /// Altitude du tablier sous un point, s'il est sur une autoroute.
    pub fn deck(&self, gen: &Generator, dir: DVec3) -> Option<f64> {
        let mut best: Option<(f64, f64)> = None;
        self.near(dir, INFLUENCE, |hit| {
            if hit.dist < self.list[hit.hw].half_width() - 0.8 && best.is_none_or(|b| hit.dist < b.0) {
                best = Some((hit.dist, self.profile(gen, hit.hw, hit.s).road + 0.1));
            }
        });
        best.map(|b| b.1)
    }

    /// Couleur de la chaussee vue de loin, et sa part dans une maille de cet ecart.
    pub fn paint(&self, dir: DVec3, spacing: f64) -> Option<([f32; 3], f32)> {
        let mut best: Option<f32> = None;
        self.near(dir, INFLUENCE, |hit| {
            let half = self.list[hit.hw].half_width();
            if hit.dist < half + spacing * 0.5 {
                // De loin, un trait plus fin qu'une maille : sa couleur ne compte que pour sa part.
                // L'emprise deboisee et les accotements clairs se voient de plus loin que la chaussee seule.
                let k = if spacing < half { 1.0 } else { (half * 5.0 / spacing).min(0.85) as f32 };
                best = Some(best.map_or(k, |b| b.max(k)));
            }
        });
        best.map(|k| ([0.12, 0.118, 0.11], k))
    }

    /// Morceaux de trace dans une maille : (autoroute, premier segment, dernier segment).
    pub fn runs_at(&self, dir: DVec3) -> &[(u32, u32, u32)] {
        self.index.get(&key(dir)).map_or(&[], |v| v.as_slice())
    }

    fn build_index(&mut self) {
        let r = planet_radius();
        let m = (INFLUENCE + 60.0) / r;
        let mut index = RouteIndex::default();
        for (h, hw) in self.list.iter().enumerate() {
            for k in 0..hw.points.len() - 1 {
                let (a, b) = (hw.points[k], hw.points[k + 1]);
                let left = a.cross(b - a).normalize_or(a.any_orthonormal_vector());
                let along = (b - a).normalize_or(left.cross(a));
                let mut keys = Vec::with_capacity(12);
                for p in [a, b] {
                    for o in [DVec3::ZERO, left * m, -left * m, along * m, -along * m] {
                        keys.push(key((p + o).normalize()));
                    }
                }
                keys.sort_unstable();
                keys.dedup();
                for kk in keys {
                    let runs = index.entry(kk).or_default();
                    match runs.last_mut() {
                        Some(run) if run.0 == h as u32 && run.2 + 1 >= k as u32 => run.2 = k as u32,
                        _ => runs.push((h as u32, k as u32, k as u32)),
                    }
                }
            }
        }
        self.index = index;
    }

    /// Croisements : la plus recente des deux autoroutes passe au-dessus.
    fn find_crossings(&mut self) {
        let r = planet_radius();
        let mut found: Vec<(usize, f64, usize, f64)> = Vec::new();
        for runs in self.index.values() {
            for (i, ra) in runs.iter().enumerate() {
                for rb in &runs[i + 1..] {
                    if ra.0 == rb.0 {
                        continue;
                    }
                    let (lo, hi) = if ra.0 < rb.0 { (ra, rb) } else { (rb, ra) };
                    let (a, b) = (&self.list[lo.0 as usize], &self.list[hi.0 as usize]);
                    for ka in lo.1..=lo.2 {
                        let (p0, p1) = (a.points[ka as usize], a.points[ka as usize + 1]);
                        let up = p0;
                        let e = (p1 - p0 - up * (p1 - p0).dot(up)).normalize_or(up.any_orthonormal_vector());
                        let n = up.cross(e);
                        let flat = |q: DVec3| DVec2::new((q - p0).dot(e), (q - p0).dot(n)) * r;
                        let (a0, a1) = (DVec2::ZERO, flat(p1));
                        for kb in hi.1..=hi.2 {
                            let (b0, b1) = (flat(b.points[kb as usize]), flat(b.points[kb as usize + 1]));
                            let (da, db) = (a1 - a0, b1 - b0);
                            let den = da.perp_dot(db);
                            if den.abs() < 1e-9 {
                                continue;
                            }
                            let ta = (b0 - a0).perp_dot(db) / den;
                            let tb = (b0 - a0).perp_dot(da) / den;
                            if (0.0..=1.0).contains(&ta) && (0.0..=1.0).contains(&tb) {
                                let sa = ka as f64 * POINT_STEP + ta * a.seg_len(ka as usize);
                                let sb = kb as f64 * POINT_STEP + tb * b.seg_len(kb as usize);
                                if !found
                                    .iter()
                                    .any(|f| f.0 == hi.0 as usize && f.2 == lo.0 as usize && (f.1 - sb).abs() < 60.0)
                                {
                                    found.push((hi.0 as usize, sb, lo.0 as usize, sa));
                                }
                            }
                        }
                    }
                }
            }
        }
        for (upper, s, lower, so) in found {
            self.list[upper].over.push((s, lower, so));
        }
    }

    /// Relie les villes. Fixe aussi leurs sorties, que leur plan prolonge en avenues.
    pub fn build(gen: &Generator, cities: &mut [City]) -> Self {
        let n = cities.len();
        let mut out = Self::empty();
        if n < 2 {
            return out;
        }
        let r = planet_radius();
        let profile = std::env::var("TERRA_PROFILE").is_ok();
        let mut clock = std::time::Instant::now();
        let mut step = |name: &str| {
            if profile {
                println!("      autoroutes : {name} {:.2} s", clock.elapsed().as_secs_f64());
            }
            clock = std::time::Instant::now();
        };
        let max_dist = 1_600_000.0 * (r / super::generator::EARTH_RADIUS).max(0.4);
        let arc = |a: usize, b: usize| cities[a].dir.dot(cities[b].dir).clamp(-1.0, 1.0).acos() * r;
        let dist: Vec<f64> = (0..n * n).map(|k| arc(k / n, k % n)).collect();
        let d = |a: usize, b: usize| dist[a * n + b];

        // Graphe de voisinage relatif.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                let ab = d(a, b);
                if ab > max_dist {
                    continue;
                }
                if !(0..n).any(|c| c != a && c != b && d(a, c).max(d(b, c)) < ab) {
                    edges.push((a, b));
                }
            }
        }
        edges.sort_by(|x, y| d(x.0, x.1).total_cmp(&d(y.0, y.1)));

        // Les sorties possibles de chaque ville, en parallele.
        let options: Vec<Vec<(f64, f64)>> = {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let cities_ref: &[City] = cities;
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(n);
            let mut all: Vec<(usize, Vec<(f64, f64)>)> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let next = &next;
                        s.spawn(move || {
                            super::generator::worker_thread(r);
                            let mut out = Vec::new();
                            loop {
                                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if i >= n {
                                    break;
                                }
                                out.push((i, exit_costs(gen, &cities_ref[i])));
                            }
                            out
                        })
                    })
                    .collect();
                handles.into_iter().flat_map(crate::sync::join).collect()
            });
            all.sort_by_key(|x| x.0);
            all.into_iter().map(|x| x.1).collect()
        };
        step("voisinage et sorties");
        let mut used: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut routes: Vec<Route> = Vec::new();
        let route_all = |pairs: &[(usize, usize)], used: &mut Vec<Vec<f64>>| -> Vec<Route> {
            // Les sorties d'abord, une ville apres l'autre ; puis les chemins, en parallele.
            let mut jobs = Vec::new();
            for &(a, b) in pairs {
                let (Some(ea), Some(eb)) = (
                    pick_exit(&options[a], &cities[a], cities[b].dir, &used[a]),
                    pick_exit(&options[b], &cities[b], cities[a].dir, &used[b]),
                ) else {
                    if std::env::var("TERRA_DEBUG_HW").is_ok() {
                        eprintln!("pas de sortie {a}-{b}");
                    }
                    continue;
                };
                used[a].push(ea);
                used[b].push(eb);
                jobs.push((a, b, ea, eb));
            }
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(jobs.len().max(1));
            let next = std::sync::atomic::AtomicUsize::new(0);
            let cities_ref: &[City] = cities;
            let done: Vec<Option<Route>> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let (jobs, next) = (&jobs, &next);
                        s.spawn(move || {
                            super::generator::worker_thread(r);
                            let mut out = Vec::new();
                            loop {
                                // Les plus longues liaisons d'abord (elles sont en fin de liste) : aucun
                                // thread ne finit seul sur un long trace pendant que les autres attendent.
                                let n = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if n >= jobs.len() {
                                    break;
                                }
                                let k = jobs.len() - 1 - n;
                                let (a, b, ea, eb) = jobs[k];
                                out.push((
                                    k,
                                    link(gen, cities_ref, a, b, ea, eb, None).map(|line| (a, b, ea, eb, line)),
                                ));
                            }
                            out
                        })
                    })
                    .collect();
                let mut all: Vec<(usize, Option<Route>)> = handles.into_iter().flat_map(crate::sync::join).collect();
                all.sort_by_key(|x| x.0);
                all.into_iter().map(|x| x.1).collect()
            });
            // Une liaison impossible rend ses sorties.
            for (job, res) in jobs.iter().zip(&done) {
                if res.is_none() {
                    used[job.0].retain(|e| *e != job.2);
                    used[job.1].retain(|e| *e != job.3);
                }
            }
            done.into_iter().flatten().collect()
        };
        routes.extend(route_all(&edges, &mut used));
        step("traces");
        if std::env::var("TERRA_DEBUG_HW").is_ok() {
            eprintln!("voisinage : {} liaisons, {} tracees", edges.len(), routes.len());
        }

        // Les morceaux restes separes se rejoignent par leurs villes les plus proches.
        // Plus loin que le voisinage : une ile proche d'un continent y est reliee par un long pont.
        let repair_dist = max_dist * 2.0;
        let mut tried: std::collections::HashSet<(usize, usize)> = edges.iter().copied().collect();
        for _ in 0..8 {
            let mut comp: Vec<usize> = (0..n).collect();
            fn find(c: &mut [usize], x: usize) -> usize {
                let mut x = x;
                while c[x] != x {
                    c[x] = c[c[x]];
                    x = c[x];
                }
                x
            }
            for (a, b, ..) in &routes {
                let (ra, rb) = (find(&mut comp, *a), find(&mut comp, *b));
                comp[ra] = rb;
            }
            let roots: Vec<usize> = (0..n).map(|i| find(&mut comp, i)).collect();
            // Pour chaque couple de morceaux, les trois paires de villes les plus proches pas encore essayees.
            /// Paires de villes candidates : (distance, ville, ville).
            type Pairs = Vec<(f64, usize, usize)>;
            let mut best: FastMap<(usize, usize), Pairs> = FastMap::default();
            for a in 0..n {
                for b in a + 1..n {
                    if roots[a] == roots[b] || tried.contains(&(a, b)) || d(a, b) > repair_dist {
                        continue;
                    }
                    let key = (roots[a].min(roots[b]), roots[a].max(roots[b]));
                    let list = best.entry(key).or_default();
                    list.push((d(a, b), a, b));
                    list.sort_by(|x, y| x.0.total_cmp(&y.0));
                    list.truncate(3);
                }
            }
            if best.is_empty() {
                break;
            }
            // Les couples de morceaux les plus proches d'abord ; une seule paire par ville a chaque tour.
            let mut groups: Vec<Vec<(f64, usize, usize)>> = best.into_values().collect();
            groups.sort_by(|x, y| x[0].0.total_cmp(&y[0].0));
            let mut busy = vec![false; n];
            let mut pairs: Vec<(usize, usize)> = Vec::new();
            for g in groups.iter().take(48) {
                for &(_, a, b) in g {
                    if !busy[a] && !busy[b] {
                        busy[a] = true;
                        busy[b] = true;
                        pairs.push((a, b));
                    }
                }
            }
            tried.extend(pairs.iter().copied());
            let more = route_all(&pairs, &mut used);
            if std::env::var("TERRA_DEBUG_HW").is_ok() {
                eprintln!("reparation : {} essais, {} tracees", pairs.len(), more.len());
                for (a, b) in &pairs {
                    eprintln!("  {} - {} : {:.0} km", cities[*a].name, cities[*b].name, d(*a, *b) / 1000.0);
                }
            }
            if more.is_empty() {
                // Rien de nouveau : les morceaux restants sont separes par la mer profonde.
                break;
            }
            routes.extend(more);
        }

        step("reparations");
        for (i, c) in cities.iter_mut().enumerate() {
            c.exits = used[i].clone();
        }
        // Deux autoroutes ne se suivent pas cote a cote : la plus recente est retracee en evitant l'autre.
        let mut draft = Self::empty();
        draft.list = routes.iter().map(|r| Highway::new(r.0, r.1, 2, &r.4)).collect();
        draft.build_index();
        let crowded: Vec<usize> = {
            let count = draft.list.len();
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
            let per = count.div_ceil(threads).max(1);
            let draft = &draft;
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..count)
                    .step_by(per)
                    .map(|from| {
                        s.spawn(move || {
                            super::generator::worker_thread(r);
                            (from..(from + per).min(count)).filter(|&i| draft.shared(i) > 300.0).collect::<Vec<_>>()
                        })
                    })
                    .collect();
                handles.into_iter().flat_map(crate::sync::join).collect()
            })
        };
        step("index provisoire");
        for i in crowded {
            let (a, b, ea, eb, _) = routes[i];
            if let Some(line) = link(gen, cities, a, b, ea, eb, Some((&draft, i))) {
                routes[i].4 = line;
            }
        }

        step("retraces");
        for (a, b, _, _, line) in routes {
            let lanes = if cities[a].size >= 0.7 && cities[b].size >= 0.7 { 3 } else { 2 };
            out.list.push(Highway::new(a, b, lanes, &line));
        }
        out.build_index();
        step("index");
        out.find_crossings();
        step("croisements");
        out
    }
}

/// Une liaison tracee : villes, sorties, trace.
type Route = (usize, usize, f64, f64, Vec<DVec3>);

impl Highways {
    /// Longueur d'une autoroute posee sur une autre plus ancienne, hors des abords des villes.
    fn shared(&self, i: usize) -> f64 {
        let h = &self.list[i];
        let skip = 20.min(h.points.len() / 2);
        let mut shared = 0.0;
        for p in &h.points[skip..h.points.len() - skip] {
            let mut on = false;
            self.near(*p, 2.0 * h.half_width(), |hit| on |= hit.hw < i);
            if on {
                shared += POINT_STEP;
            }
        }
        shared
    }
}

/// Angle d'une direction vue d'une ville, dans son plan.
fn bearing(city: &City, to: DVec3) -> f64 {
    let (e, n) = city.axes();
    let t = to - city.dir * to.dot(city.dir);
    t.dot(n).atan2(t.dot(e))
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::TAU);
    d.min(std::f64::consts::TAU - d)
}

/// Les sorties possibles d'une ville et leur cout : l'avenue qui y mene doit rester dans la ville
/// d'un bout a l'autre, et de preference a sec, jusqu'au depart de l'autoroute.
fn exit_costs(gen: &Generator, city: &City) -> Vec<(f64, f64)> {
    const STEP: f64 = 120.0;
    let coarse = Detail::for_spacing(STEP);
    city.exit_options()
        .into_iter()
        .filter(|(a, _)| city.style == super::city::Style::Radial || city.exit_clear(*a))
        .map(|(a, bias)| {
            let d = DVec2::from_angle(a);
            let end = city.reach(a) + LEAD;
            let (mut run, mut longest, mut any) = (0.0f64, 0.0f64, false);
            let mut s = city.radius * 0.1;
            while s < end {
                let w = gen.surface_natural(city.dir_of(d * s), &coarse);
                if w.height < w.water_level + 1.0 {
                    run += STEP;
                    longest = longest.max(run);
                    any = true;
                } else {
                    run = 0.0;
                }
                s += STEP;
            }
            let wet = if longest > 250.0 {
                4.0
            } else if any {
                1.0
            } else {
                0.0
            };
            (a, bias + wet)
        })
        .collect()
}

/// La sortie de la ville vers une autre : une direction de son plan, libre, au plus pres du cap, et a sec.
fn pick_exit(options: &[(f64, f64)], city: &City, to: DVec3, used: &[f64]) -> Option<f64> {
    let want = bearing(city, to);
    options
        .iter()
        .filter(|(a, _)| used.iter().all(|u| angle_diff(*u, *a) > 0.3))
        .filter_map(|&(a, cost)| {
            let turn = angle_diff(a, want);
            (turn <= 2.1).then_some((a, turn + cost))
        })
        .min_by(|x, y| x.1.total_cmp(&y.1))
        .map(|x| x.0)
}

/// Repere gnomonique local : x vers la ville d'arrivee.
struct Plane {
    m: DVec3,
    ex: DVec3,
    ey: DVec3,
    r: f64,
}

impl Plane {
    fn local(&self, d: DVec3) -> DVec2 {
        DVec2::new(d.dot(self.ex), d.dot(self.ey)) * (self.r / d.dot(self.m).max(1e-6))
    }

    fn dir(&self, p: DVec2) -> DVec3 {
        (self.m * self.r + self.ex * p.x + self.ey * p.y).normalize()
    }
}

/// Trace d'une ville a l'autre : de la sortie de l'une a celle de l'autre.
fn link(
    gen: &Generator,
    cities: &[City],
    a: usize,
    b: usize,
    ea: f64,
    eb: f64,
    avoid: Option<(&Highways, usize)>,
) -> Option<Vec<DVec3>> {
    let (ca, cb) = (&cities[a], &cities[b]);
    let gate = |c: &City, e: f64, extra: f64| c.dir_of(DVec2::from_angle(e) * (c.reach(e) + extra));
    let (start, end) = (gate(ca, ea, 0.0), gate(cb, eb, 0.0));
    let (lead_a, lead_b) = (gate(ca, ea, LEAD), gate(cb, eb, LEAD));
    let path = search(gen, cities, lead_a, lead_b, avoid)?;
    let mut line = vec![start];
    line.extend(path);
    line.push(end);
    // De grandes courbes : quatre passes de Chaikin, les deux bouts fixes.
    for _ in 0..4 {
        let mut next = vec![line[0]];
        for w in line.windows(2) {
            next.push(w[0].lerp(w[1], 0.25).normalize());
            next.push(w[0].lerp(w[1], 0.75).normalize());
        }
        next.push(end);
        line = next;
    }
    Some(line)
}

/// A* sur le relief, de a a b (directions). Rend le chemin simplifie, a et b compris.
/// `avoid` : des autoroutes deja tracees, dont on s'ecarte (sauf pour les croiser).
fn search(
    gen: &Generator,
    cities: &[City],
    a: DVec3,
    b: DVec3,
    avoid: Option<(&Highways, usize)>,
) -> Option<Vec<DVec3>> {
    let r = planet_radius();
    let m = (a + b).normalize();
    let x = b - a;
    let ex = (x - m * x.dot(m)).normalize_or(m.any_orthonormal_vector());
    let plane = Plane { m, ex, ey: m.cross(ex), r };
    let (pa, pb) = (plane.local(a), plane.local(b));
    let len = (pb - pa).length();
    // Mailles plus grossieres pour les tres longues liaisons (reparations entre iles) : la recherche reste bornee.
    let cell = (len / 220.0).clamp(500.0, 6000.0).max(len / 270.0);
    let half = (len * 0.32).max(15_000.0);
    let (x0, y0) = (pa.x.min(pb.x) - 15_000.0, -half);
    let (nx, ny) =
        (((pa.x.max(pb.x) + 15_000.0 - x0) / cell).ceil() as usize + 1, ((2.0 * half) / cell).ceil() as usize + 1);
    let center = |i: usize, j: usize| DVec2::new(x0 + i as f64 * cell, y0 + j as f64 * cell);
    let cell_of = |p: DVec2| (((p.x - x0) / cell).round() as usize, ((p.y - y0) / cell).round() as usize);
    let detail = Detail::for_spacing(cell);
    let reach = (len * 0.5 + half + 60_000.0) / r;
    let nearby: Vec<&City> = cities.iter().filter(|c| c.dir.dot(m) > reach.cos()).collect();

    // Les mailles deja traversees par une autre autoroute.
    let mut taken = vec![false; nx * ny];
    if let Some((h, me)) = avoid {
        for (j, other) in h.list.iter().enumerate() {
            if j == me {
                continue;
            }
            for p in &other.points {
                if p.dot(m) > reach.cos() {
                    let (i, jj) = cell_of(plane.local(*p));
                    if i < nx && jj < ny {
                        taken[jj * nx + i] = true;
                    }
                }
            }
        }
    }
    // Sol de chaque maille, a la demande : altitude, et 0 terre, 1 eau douce, 2 mer, 3 ville.
    let mut ground: Vec<(f32, u8)> = vec![(f32::NAN, 0); nx * ny];
    let mut sample = |i: usize, j: usize| -> (f32, u8) {
        let k = j * nx + i;
        if ground[k].0.is_nan() {
            let dir = plane.dir(center(i, j));
            let s = gen.surface_natural(dir, &detail);
            let kind = if s.height < s.water_level + 1.0 {
                if s.fresh {
                    1
                } else if s.water_level - s.height < 70.0 {
                    // Mer peu profonde : un long pont reste possible, a grands frais.
                    5
                } else {
                    2
                }
            } else if nearby.iter().any(|c| c.extent(c.local(dir)) < 1.1) {
                3
            } else if taken[k] {
                4
            } else {
                0
            };
            ground[k] = (s.height.max(s.water_level), kind);
        }
        ground[k]
    };

    let (si, sj) = cell_of(pa);
    let (gi, gj) = cell_of(pb);
    if si >= nx || sj >= ny || gi >= nx || gj >= ny {
        return None;
    }
    let goal = gj * nx + gi;
    let mut cost = vec![f64::INFINITY; nx * ny];
    let mut from = vec![u32::MAX; nx * ny];
    let mut open = BinaryHeap::new();
    cost[sj * nx + si] = 0.0;
    open.push(Reverse((0u64, (sj * nx + si) as u32)));
    let h = |i: usize, j: usize| (center(i, j) - pb).length();
    let mut found = false;
    while let Some(Reverse((_, u))) = open.pop() {
        let u = u as usize;
        if u == goal {
            found = true;
            break;
        }
        let (ui, uj) = (u % nx, u / nx);
        let (hu, _) = sample(ui, uj);
        let gu = cost[u];
        for (di, dj) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
            let (vi, vj) = (ui as i32 + di, uj as i32 + dj);
            if vi < 0 || vj < 0 || vi >= nx as i32 || vj >= ny as i32 {
                continue;
            }
            let (vi, vj) = (vi as usize, vj as usize);
            let v = vj * nx + vi;
            let (hv, kind) = sample(vi, vj);
            // Les bouts sont toujours accessibles : ils sont au pied d'une ville.
            if kind == 2 && v != goal {
                continue;
            }
            let step = if di != 0 && dj != 0 { cell * std::f64::consts::SQRT_2 } else { cell };
            let grade = (hv - hu).abs() as f64 / step;
            let mut k = 1.0 + grade * 25.0 + (grade - 0.05).max(0.0) * 600.0 + (hv as f64 - 1200.0).max(0.0) / 800.0;
            if kind == 1 {
                k += 5.0;
            } else if kind == 3 && v != goal {
                k += 6.0;
            } else if kind == 4 {
                k += 3.0;
            } else if kind == 5 {
                k += 30.0;
            }
            let g = gu + step * k;
            if g < cost[v] {
                cost[v] = g;
                from[v] = u as u32;
                // A* pondere : un chemin a peine plus long, trouve bien plus vite.
                let f = g + h(vi, vj) * 1.3;
                open.push(Reverse((f.to_bits(), v as u32)));
            }
        }
    }
    if !found {
        return None;
    }
    let mut cells = vec![goal];
    let mut c = goal;
    while from[c] != u32::MAX {
        c = from[c] as usize;
        cells.push(c);
    }
    cells.reverse();
    let mut pts: Vec<DVec2> = cells.iter().map(|&c| center(c % nx, c / nx)).collect();
    pts[0] = pa;
    if let Some(last) = pts.last_mut() {
        *last = pb;
    }
    let pts = douglas_peucker(&pts, cell * 0.75);
    Some(pts.into_iter().map(|p| plane.dir(p)).collect())
}

fn douglas_peucker(p: &[DVec2], tol: f64) -> Vec<DVec2> {
    if p.len() < 3 {
        return p.to_vec();
    }
    let (a, b) = (p[0], p[p.len() - 1]);
    let ab = b - a;
    let len = ab.length().max(1e-9);
    let (mut worst, mut at) = (0.0, 0);
    for (i, q) in p.iter().enumerate().take(p.len() - 1).skip(1) {
        let d = (ab.perp_dot(*q - a) / len).abs();
        if d > worst {
            (worst, at) = (d, i);
        }
    }
    if worst <= tol {
        return vec![a, b];
    }
    let mut left = douglas_peucker(&p[..=at], tol);
    let right = douglas_peucker(&p[at..], tol);
    left.pop();
    left.extend(right);
    left
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::generator::PlanetParams;

    #[test]
    fn resampling_keeps_the_ends_and_the_step() {
        let r = planet_radius();
        let a = DVec3::X;
        let b = (DVec3::X + DVec3::Y * (1234.0 / r)).normalize();
        let p = resample(&[a, b], POINT_STEP);
        assert_eq!(p[0], a);
        assert_eq!(*p.last().unwrap(), b);
        assert_eq!(p.len(), 14);
        assert!(((p[1] - p[0]).length() * r - POINT_STEP).abs() < 0.01);
    }

    #[test]
    fn every_city_of_a_continent_is_reachable() {
        let gen = Generator::new(PlanetParams { seed: 3, ..Default::default() });
        let hw = &gen.cities.highways;
        assert!(hw.list.len() as f64 >= gen.cities.list.len() as f64 * 0.8, "{} autoroutes", hw.list.len());
        // Chaque autoroute part d'une ville et arrive a une autre, sans saut.
        let r = planet_radius();
        for h in &hw.list {
            assert!(h.points.windows(2).all(|w| (w[1] - w[0]).length() * r < POINT_STEP + 1.0));
            let ca = &gen.cities.list[h.from];
            assert!(ca.dir.dot(h.points[0]).acos() * r < ca.radius * 1.6);
        }
        // Le terrain se plie a la chaussee.
        let h = &hw.list[0];
        let s = h.length() * 0.37;
        let p = hw.profile(&gen, 0, s);
        if !p.bridge() {
            let ground = gen.height(h.dir_at(s)) as f64;
            assert!((ground - (p.road - 0.3)).abs() < 1.0, "sol {ground}, chaussee {}", p.road);
        }
    }
}
