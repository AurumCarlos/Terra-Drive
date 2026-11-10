//! Routes de montagne : de chaque ville au pied d'un relief, une route etroite monte en lacets
//! jusqu'a un belvedere, ou elle finit en raquette autour d'un ilot d'herbe.
//!
//! - Le sommet vise est le plus haut point sec a moins de 14 km du bord de la ville, au moins 350 m
//!   au-dessus d'elle, un peu penalise par la distance. La route sort de la ville par une avenue
//!   tournee vers lui.
//! - Le trace est cherche par A* sur une grille de 40 m dont chaque etat est une case et un cap parmi
//!   32, dans l'esprit de Galin et al. 2010 (« Procedural Generation of Roads ») : la route tourne d'au
//!   plus un cran par pas et ne monte jamais plus de 10 % d'une case a l'autre. Sur une pente plus
//!   raide, la seule facon de monter est l'epingle : un demi-tour qui decale la route d'une ou deux
//!   cases sur le cote, cher, donc reserve aux pentes ou il n'y a pas d'autre moyen. L'heuristique est
//!   la hauteur qui reste a gravir divisee par la pente maximale : la longueur minimale de route.
//! - Le trace est lisse sans toucher aux epingles, puis le profil en long suit le terrain, lisse sur
//!   70 m et ecrete a 12 %. Le sol se plie a la chaussee : deblais cote montagne, remblais cote vide.

use super::circuit::{key, limit_grade_to, smooth, tangent_frame, Circuits, Path, RUNOFF};
use super::city::City;
use super::fast_hash::FastMap;
use super::generator::{par_map, planet_radius, smoothstep, Detail, Generator, Surface};
use super::highway::Highways;
use glam::{DVec2, DVec3};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::f64::consts::{PI, TAU};
use std::sync::OnceLock;

/// Demi-largeur de la chaussee : deux voies etroites de 3.3 m.
pub const ROAD_HALF: f64 = 3.3;
/// Accotement plat de chaque cote, avant le talus.
const SHOULDER: f64 = 1.5;
/// Portee de l'influence sur le sol, depuis l'axe.
const INFLUENCE: f64 = 65.0;
/// Ecart minimal entre deux routes de montagne.
const SEPARATION: f64 = 40.0;
/// Rayon de la raquette du belvedere.
pub const LOOP_RADIUS: f64 = 15.0;
/// Maille de la recherche du trace.
const CELL: f64 = 40.0;
/// Ecart des noeuds ou le relief est lu, en cases.
const LATTICE: usize = 3;
/// Pente maximale d'une case a l'autre, et dans une epingle.
const SEARCH_GRADE: f64 = 0.10;
const HAIRPIN_GRADE: f64 = 0.22;
/// Pente du terrain en dessous de laquelle une epingle ne sert a rien.
const HAIRPIN_SLOPE: f64 = 0.2;
/// Cases droites au moins apres une epingle.
const LEG: usize = 3;
/// Pente maximale du profil en long.
pub const LEVEL_GRADE: f64 = 0.12;
/// Denivele minimal pour qu'une route de montagne vaille la peine.
const MIN_RISE: f64 = 300.0;
/// Routes de montagne au plus par ville, vers des sommets ecartes d'au moins PEAK_SPREAD radians.
pub const ROADS_PER_CITY: usize = 4;
const PEAK_SPREAD: f64 = 1.1;
/// Distance maximale du sommet au bord de la ville.
const REACH: f64 = 14_000.0;
/// Couts, en metres de route equivalents : une epingle, un changement de cap.
const HAIRPIN_COST: f64 = 320.0;
const TURN_COST: f64 = 6.0;
/// Etats sortis de la file au plus, pour tous les essais d'une route : au-dela, la route s'arrete au
/// plus haut point atteint.
const MAX_POPS: usize = 400_000;
/// Poids de l'heuristique (A* pondere) : un trace a peine plus long, trouve bien plus vite.
const HEURISTIC_WEIGHT: f64 = 1.5;
/// Distance au sommet a laquelle la route peut finir.
const NEAR_SUMMIT: f64 = 350.0;

/// Nombre de caps de la grille.
const N: usize = 32;
/// Les caps de la grille, dans le sens trigonometrique, tous les 11 a 14 degres : une route peut ainsi
/// monter a 10 % en travers d'une pente de 40 %.
const DIRS: [(i32, i32); N] = [
    (1, 0),
    (4, 1),
    (2, 1),
    (4, 3),
    (1, 1),
    (3, 4),
    (1, 2),
    (1, 4),
    (0, 1),
    (-1, 4),
    (-1, 2),
    (-3, 4),
    (-1, 1),
    (-4, 3),
    (-2, 1),
    (-4, 1),
    (-1, 0),
    (-4, -1),
    (-2, -1),
    (-4, -3),
    (-1, -1),
    (-3, -4),
    (-1, -2),
    (-1, -4),
    (0, -1),
    (1, -4),
    (1, -2),
    (3, -4),
    (1, -1),
    (4, -3),
    (2, -1),
    (4, -1),
];

/// Decalages possibles d'une epingle, a gauche puis a droite du cap `d` : d'une puis de deux cases, au
/// plus pres de la perpendiculaire (35 degres au plus).
fn hairpin_offsets(d: usize) -> Vec<(i32, i32)> {
    let f = DVec2::new(DIRS[d].0 as f64, DIRS[d].1 as f64).normalize();
    let mut out = Vec::new();
    for left in [1.0, -1.0] {
        let perp = f.perp() * left;
        // Parmi les decalages courts (une case), puis longs (deux cases) : le plus perpendiculaire, et le
        // plus en biais, qui monte moins raide sur une forte pente.
        for (lo, hi) in [(1.0, 1.5), (1.9, 2.3)] {
            let mut found: Vec<((i32, i32), f64)> = Vec::new();
            for a in -2i32..=2 {
                for b in -2i32..=2 {
                    let v = DVec2::new(a as f64, b as f64);
                    let len = v.length();
                    let cos = v.dot(perp) / len.max(1e-9);
                    if (lo..=hi).contains(&len) && cos >= 35f64.to_radians().cos() {
                        found.push(((a, b), cos));
                    }
                }
            }
            found.sort_by(|x, y| y.1.total_cmp(&x.1));
            if let (Some(first), Some(last)) = (found.first(), found.last()) {
                out.push(first.0);
                if last.0 != first.0 {
                    out.push(last.0);
                }
            }
        }
    }
    out
}

/// Pas d'une case, au plus pres du cap `d`.
fn unit(d: usize) -> (i32, i32) {
    let f = DVec2::new(DIRS[d].0 as f64, DIRS[d].1 as f64).normalize();
    (f.x.round() as i32, f.y.round() as i32)
}

pub struct MountainRoad {
    pub name: String,
    pub city: usize,
    /// La route, puis la raquette du belvedere.
    pub paths: Vec<Path>,
    /// Sommets des epingles.
    pub hairpins: Vec<DVec3>,
    /// Parapets de pierre plutot que glissieres d'acier.
    pub stone: bool,
    /// Profil en long de chaque trace, calcule a la premiere demande.
    levels: OnceLock<Vec<Vec<f64>>>,
}

impl MountainRoad {
    /// Altitude de la chaussee en chaque point de chaque trace : le terrain naturel, lisse le long de la
    /// route et ecrete ; la raquette reste a l'altitude du bout de la route.
    pub fn levels(&self, gen: &Generator) -> &Vec<Vec<f64>> {
        self.levels.get_or_init(|| {
            let detail = Detail::for_spacing(8.0);
            let road = &self.paths[0];
            let raw: Vec<f64> = road.points.iter().map(|d| gen.surface_natural(*d, &detail).height as f64).collect();
            let mut lv = smooth(&raw, 4, false);
            // Au depart, le sol de la ville.
            for (i, v) in lv.iter_mut().enumerate().take(13) {
                let a = smoothstep(0.0, 12.0, i as f32) as f64;
                *v = raw[0] + (*v - raw[0]) * a;
            }
            let lv = limit_grade_to(lv, &road.points, false, LEVEL_GRADE);
            let end = lv[lv.len() - 1];
            let ring = vec![end; self.paths[1].points.len()];
            vec![lv, ring]
        })
    }

    /// Point de depart (au bord de la ville) et sens de la montee.
    pub fn start(&self) -> (DVec3, DVec3) {
        let road = &self.paths[0];
        let k = 3.min(road.points.len() - 1);
        (road.points[k], road.tangent(k))
    }

    /// Denivele de la route, en metres.
    pub fn rise(&self, gen: &Generator) -> f64 {
        let lv = &self.levels(gen)[0];
        lv[lv.len() - 1] - lv[0]
    }
}

/// Le point le plus proche sur un trace.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub road: usize,
    pub path: usize,
    /// Segment, et position sur ce segment (0 a 1).
    pub k: usize,
    pub t: f64,
    /// Distance a l'axe, en metres.
    pub dist: f64,
}

/// Morceaux de trace par maille de la sphere : (route, trace, segment).
type RoadIndex = FastMap<(u8, i32, i32), Vec<(u32, u8, u32)>>;

pub struct MountainRoads {
    pub list: Vec<MountainRoad>,
    index: RoadIndex,
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(TAU);
    d.min(TAU - d)
}

/// Repere gnomonique local, centre sur la sortie de la ville, x vers le sommet.
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

/// Une grille de recherche : sa taille, son origine (la case de depart est en (0, 0)).
struct Grid {
    nx: usize,
    ny: usize,
    i0: usize,
    j0: usize,
}

impl Grid {
    fn center(&self, c: usize) -> DVec2 {
        let (i, j) = (c % self.nx, c / self.nx);
        DVec2::new((i as f64 - self.i0 as f64) * CELL, (j as f64 - self.j0 as f64) * CELL)
    }

    fn offset(&self, c: usize, (di, dj): (i32, i32)) -> Option<usize> {
        let (i, j) = ((c % self.nx) as i32 + di, (c / self.nx) as i32 + dj);
        (i >= 0 && j >= 0 && (i as usize) < self.nx && (j as usize) < self.ny).then(|| j as usize * self.nx + i as usize)
    }
}

fn dir_len(d: usize) -> f64 {
    let (x, y) = DIRS[d];
    ((x * x + y * y) as f64).sqrt()
}

/// Un pas du trace : la case atteinte, son cap, et vrai si on y arrive par une epingle.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Step {
    cell: usize,
    dir: usize,
    hairpin: bool,
}

/// A* sur la grille, de la case de depart (cap `d0`) jusqu'a une case d'altitude `goal` au moins, ou
/// a moins de `NEAR_SUMMIT` du sommet `summit` (en metres dans le plan de la grille).
/// `ground(c)` : altitude de la case, ou None si on ne peut pas y passer (eau, ville, autre route).
/// Rend les pas du trace, depart compris. Si le sommet est hors d'atteinte, s'arrete au plus haut
/// point atteint, pourvu qu'il ait gravi au moins `min_rise`.
fn search(
    grid: &Grid,
    d0: usize,
    goal: f64,
    summit: DVec2,
    min_rise: f64,
    budget: &mut usize,
    mut ground: impl FnMut(usize) -> Option<f64>,
) -> Option<Vec<Step>> {
    let cells = grid.nx * grid.ny;
    let start = grid.j0 * grid.nx + grid.i0;
    let h_start = ground(start)?;
    let mut cost = vec![f32::INFINITY; cells * N];
    let mut from = vec![u32::MAX; cells * N];
    let mut open = BinaryHeap::new();
    let offsets: Vec<Vec<(i32, i32)>> = (0..N).map(hairpin_offsets).collect();
    // La longueur de route qu'il faut encore au moins pour gravir ce qui reste, ou pour rejoindre le
    // sommet : la plus grande des deux guide la recherche vers lui (A* pondere, un peu optimiste).
    let heuristic = |c: usize, h: f64| {
        let climb = (goal - h).max(0.0) / SEARCH_GRADE;
        let reach = ((grid.center(c) - summit).length() - NEAR_SUMMIT).max(0.0);
        climb.max(reach) * HEURISTIC_WEIGHT
    };
    let arrived = |c: usize, h: f64| h >= goal || (grid.center(c) - summit).length() <= NEAR_SUMMIT;
    // Au depart, tous les caps, le plus cher le plus loin de `d0` : la route peut partir de biais.
    for d in 0..N {
        let turn = (d + N - d0) % N;
        let g = turn.min(N - turn) as f64 * TURN_COST * 2.0;
        cost[start * N + d] = g as f32;
        open.push(Reverse((((g + heuristic(start, h_start)) as f32).to_bits(), (start * N + d) as u32)));
    }
    // Pente du terrain : la plus forte difference avec les cases voisines, gardee pour chaque case.
    let mut slopes = vec![f32::NAN; cells];
    let mut side = |c: usize, h: f64, ground: &mut dyn FnMut(usize) -> Option<f64>| -> f64 {
        if slopes[c].is_nan() {
            let mut worst: f64 = 0.0;
            for o in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                if let Some(n) = grid.offset(c, o) {
                    if let Some(hn) = ground(n) {
                        worst = worst.max((hn - h).abs() / CELL);
                    }
                }
            }
            slopes[c] = worst as f32;
        }
        slopes[c] as f64
    };
    // Un pas long ne saute ni une riviere ni une autoroute : toutes les cases qu'il traverse sont libres.
    let clear = |c: usize, (dx, dy): (i32, i32), ground: &mut dyn FnMut(usize) -> Option<f64>| -> bool {
        let n = dx.abs().max(dy.abs());
        (1..n).all(|t| {
            let o = ((dx * t) as f64 / n as f64).round() as i32;
            let p = ((dy * t) as f64 / n as f64).round() as i32;
            grid.offset(c, (o, p)).and_then(&mut *ground).is_some()
        })
    };
    let mut best = (h_start, start * N + d0);
    let mut found = None;
    while let Some(Reverse((f_bits, s))) = open.pop() {
        let s = s as usize;
        let (c, d) = (s / N, s % N);
        let g = cost[s] as f64;
        let Some(h) = ground(c) else { continue };
        if (f32::from_bits(f_bits) as f64) > g + heuristic(c, h) + 1.0 {
            continue; // deja sorti avec un meilleur cout
        }
        *budget = budget.saturating_sub(1);
        if h > best.0 {
            best = (h, s);
        }
        if arrived(c, h) {
            found = Some(s);
            break;
        }
        if *budget == 0 {
            break;
        }
        let mut relax = |c2: usize, d2: usize, step: f64, open: &mut BinaryHeap<Reverse<(u32, u32)>>, h2: f64| {
            let s2 = c2 * N + d2;
            let g2 = g + step;
            if g2 < cost[s2] as f64 {
                cost[s2] = g2 as f32;
                from[s2] = s as u32;
                open.push(Reverse((((g2 + heuristic(c2, h2)) as f32).to_bits(), s2 as u32)));
            }
        };
        // Tout droit, ou un cran a gauche ou a droite.
        for turn in [N - 1, 0, 1] {
            let d2 = (d + turn) % N;
            let Some(c2) = grid.offset(c, DIRS[d2]) else { continue };
            let Some(h2) = ground(c2) else { continue };
            let len = dir_len(d2) * CELL;
            let grade = (h2 - h).abs() / len;
            if grade > SEARCH_GRADE || !clear(c, DIRS[d2], &mut ground) {
                continue;
            }
            let steep = side(c2, h2, &mut ground);
            if steep > 1.25 {
                continue; // une falaise
            }
            let k = 1.0 + 1.5 * grade + (steep - 0.45).max(0.0) * 3.0;
            relax(c2, d2, len * k + if turn != 0 { TURN_COST } else { 0.0 }, &mut open, h2);
        }
        // Epingle : demi-tour, decale d'une ou deux cases sur le cote, seulement sur une vraie pente
        // (ailleurs, un demi-tour ne fait rien gagner). Le virage deborde d'une case en avant, qui doit
        // etre praticable.
        if side(c, h, &mut ground) < HAIRPIN_SLOPE {
            continue;
        }
        let Some(apex) = grid.offset(c, unit(d)) else { continue };
        if ground(apex).is_none() {
            continue;
        }
        let back = (d + N / 2) % N;
        for &o in &offsets[d] {
            let Some(c2) = grid.offset(c, o) else { continue };
            let Some(h2) = ground(c2) else { continue };
            let width = ((o.0 * o.0 + o.1 * o.1) as f64).sqrt() * CELL;
            let len = PI * 0.5 * width;
            let grade = (h2 - h).abs() / len;
            if grade > HAIRPIN_GRADE {
                continue;
            }
            // Puis une branche droite d'au moins LEG pas : pas de lacets en ressort.
            let (mut cur, mut hc, mut extra) = (c2, h2, 0.0);
            let leg = dir_len(back) * CELL;
            let mut ok = true;
            for _ in 0..LEG {
                let next = grid.offset(cur, DIRS[back]).and_then(|n| ground(n).map(|hn| (n, hn)));
                match next {
                    Some((n, hn)) if (hn - hc).abs() <= leg * SEARCH_GRADE && clear(cur, DIRS[back], &mut ground) => {
                        extra += leg * (1.0 + 1.5 * (hn - hc).abs() / leg);
                        (cur, hc) = (n, hn);
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                relax(cur, back, HAIRPIN_COST + len * (1.0 + 1.5 * grade) + extra, &mut open, hc);
            }
        }
    }
    let end = match found {
        Some(s) => s,
        None if best.0 - h_start >= min_rise => best.1,
        None => return None,
    };
    let mut steps = Vec::new();
    let mut s = end;
    loop {
        let (c, d) = (s / N, s % N);
        let prev = from[s];
        let hairpin = prev != u32::MAX && (d + N - prev as usize % N) % N == N / 2;
        if hairpin {
            // Le demi-tour et sa branche droite : un seul coup de la recherche, LEG + 1 pas du trace.
            let (dx, dy) = DIRS[d];
            for j in 0..=LEG as i32 {
                let cell = grid.offset(c, (-dx * j, -dy * j)).expect("branche dans la grille");
                steps.push(Step { cell, dir: d, hairpin: j == LEG as i32 });
            }
        } else {
            steps.push(Step { cell: c, dir: d, hairpin: false });
        }
        if prev == u32::MAX {
            break;
        }
        s = prev as usize;
    }
    steps.reverse();
    Some(steps)
}

/// Le trace en metres dans le plan de la grille : centres des cases, demi-cercles aux epingles, lisse.
/// Rend aussi les sommets des epingles.
fn polyline(grid: &Grid, steps: &[Step]) -> (Vec<DVec2>, Vec<DVec2>) {
    let mut pts = vec![grid.center(steps[0].cell)];
    let mut pinned = vec![true];
    let mut apexes = Vec::new();
    for w in steps.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (p1, p2) = (grid.center(a.cell), grid.center(b.cell));
        if b.hairpin {
            let m = (p1 + p2) * 0.5;
            let rad = (p2 - p1).length() * 0.5;
            // Demi-cercle sur le decalage, bombe vers l'avant de la branche qui arrive.
            let (fx, fy) = DIRS[a.dir];
            let ahead = DVec2::new(fx as f64, fy as f64);
            let mut f = (p2 - p1).perp().normalize_or(ahead.normalize());
            if f.dot(ahead) < 0.0 {
                f = -f;
            }
            if let Some(last) = pinned.last_mut() {
                *last = true;
            }
            let n = (rad / 3.0).ceil().max(6.0) as usize;
            for q in 1..n {
                let t = q as f64 / n as f64 * PI;
                pts.push(m + (p1 - m) * t.cos() + f * rad * t.sin());
                pinned.push(true);
            }
            apexes.push(m + f * rad);
            pts.push(p2);
            pinned.push(true);
        } else {
            pts.push(p2);
            pinned.push(false);
        }
    }
    if let Some(last) = pinned.last_mut() {
        *last = true;
    }
    // Lissage de Laplace : les crans de la grille deviennent des courbes, les epingles restent rondes.
    for _ in 0..14 {
        let prev = pts.clone();
        for i in 1..pts.len().saturating_sub(1) {
            if !pinned[i] {
                pts[i] = prev[i] * 0.5 + (prev[i - 1] + prev[i + 1]) * 0.25;
            }
        }
    }
    (resample(&pts, 8.0), apexes)
}

/// Polyligne plane re-echantillonnee a pas constant, les deux bouts gardes.
fn resample(p: &[DVec2], step: f64) -> Vec<DVec2> {
    let mut out = vec![p[0]];
    let mut carry = 0.0;
    for w in p.windows(2) {
        let len = (w[1] - w[0]).length();
        let mut s = step - carry;
        while s < len {
            out.push(w[0].lerp(w[1], s / len));
            s += step;
        }
        carry = len - (s - step);
    }
    let end = p[p.len() - 1];
    if carry < step * 0.4 && out.len() >= 2 {
        let n = out.len();
        out[n - 1] = end;
    } else {
        out.push(end);
    }
    out
}

/// Le trace de la route de montagne numero `rank` de la ville `i`, vers son `rank`-ieme meilleur sommet
/// (chacun dans une direction a part) : (angle de sortie, points, sommets des epingles).
fn plan_road(
    gen: &Generator,
    cities: &[City],
    i: usize,
    rank: usize,
    highways: &Highways,
    circuits: &Circuits,
) -> Option<(f64, Vec<DVec3>, Vec<DVec3>)> {
    let city = &cities[i];
    let r = planet_radius();
    let coarse = Detail::for_spacing(200.0);
    let h0 = gen.surface_natural(city.dir, &coarse).height as f64;
    let others: Vec<&City> = cities
        .iter()
        .enumerate()
        .filter(|(j, o)| *j != i && (o.dir - city.dir).length() * r < REACH + city.radius * 2.0 + o.radius * 2.0 + 20_000.0)
        .map(|(_, o)| o)
        .collect();
    let in_other = |d: DVec3| others.iter().any(|o| o.extent(o.local(d)) < 1.15);

    // Les sommets candidats : le meilleur de chaque direction, puis les meilleurs bien ecartes les uns des
    // autres. La route `rank` vise le `rank`-ieme.
    let mut by_dir: Vec<(f64, f64, DVec3)> = Vec::new();
    for k in 0..48 {
        let mut best: Option<(f64, f64, DVec3)> = None;
        let a = k as f64 / 48.0 * TAU;
        let edge = city.reach(a);
        for q in 1..=10 {
            let dist = q as f64 * REACH / 10.0;
            let d = city.dir_of(DVec2::from_angle(a) * (edge + dist));
            let s = gen.surface_natural(d, &coarse);
            if s.height < s.water_level + 2.0 {
                continue;
            }
            let rise = s.height as f64 - h0;
            let score = rise - dist * 0.015;
            if rise >= MIN_RISE && best.is_none_or(|b| score > b.0) && !in_other(d) {
                best = Some((score, a, d));
            }
        }
        by_dir.extend(best);
    }
    by_dir.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut chosen: Vec<(f64, f64, DVec3)> = Vec::new();
    for c in by_dir {
        if chosen.iter().all(|o| angle_diff(o.1, c.1) >= PEAK_SPREAD) {
            chosen.push(c);
        }
    }
    let (_, _, peak) = *chosen.get(rank)?;

    // La sortie de la ville, au plus pres du cap du sommet, loin des autres sorties.
    let to = city.local(peak);
    let want = to.y.atan2(to.x);
    let angle = (0..17)
        .map(|k| want + (k as f64 - 8.0) * 0.1)
        .filter(|a| city.exit_clear(*a) && city.exits.iter().all(|e| angle_diff(*e, *a) > 0.35))
        .min_by(|a, b| angle_diff(*a, want).total_cmp(&angle_diff(*b, want)))?;
    let edge = city.reach(angle);
    let inside = city.dir_of(DVec2::from_angle(angle) * (edge - 6.0));
    let gate = city.dir_of(DVec2::from_angle(angle) * edge);

    // Plan local : origine a la sortie, x vers le sommet.
    let x = peak - gate;
    let ex = (x - gate * x.dot(gate)).normalize_or(gate.any_orthonormal_vector());
    let plane = Plane { m: gate, ex, ey: gate.cross(ex), r };
    let pp = plane.local(peak);
    let margin = 2400.0;
    let (lo, hi) = (pp.min(DVec2::ZERO) - DVec2::splat(margin), pp.max(DVec2::ZERO) + DVec2::splat(margin));
    let (i0, j0) = ((-lo.x / CELL).ceil() as usize, ((-lo.y).max(3000.0) / CELL).ceil() as usize);
    let grid = Grid {
        nx: i0 + (hi.x / CELL).ceil() as usize + 1,
        ny: j0 + (hi.y.max(3000.0) / CELL).ceil() as usize + 1,
        i0,
        j0,
    };

    // Les cases ou la route ne peut pas passer : eau, villes, autoroutes, circuits. Le relief est lu
    // sans ses bosses de quelques dizaines de metres, que la route traverse en deblai ou en remblai.
    let detail = Detail::for_spacing(CELL * LATTICE as f64);
    let own: Vec<f64> = (0..64).map(|k| city.reach(k as f64 / 64.0 * TAU)).collect();
    let in_own = |d: DVec3| {
        let p = city.local(d);
        let a = p.y.atan2(p.x).rem_euclid(TAU) / TAU * 64.0;
        let (k, t) = (a.floor() as usize % 64, a.fract());
        p.length() < own[k] + (own[(k + 1) % 64] - own[k]) * t
    };
    // Ce relief lisse est lu sur un reseau de noeuds tous les LATTICE cases, puis interpole : neuf fois
    // moins de lectures. Une case est dans l'eau si un de ses quatre noeuds l'est.
    let start = grid.j0 * grid.nx + grid.i0;
    let (lx, ly) = (grid.nx.div_ceil(LATTICE) + 1, grid.ny.div_ceil(LATTICE) + 1);
    let mut nodes: Vec<f32> = vec![f32::NAN; lx * ly];
    let mut node = |a: usize, b: usize| -> f32 {
        let k = b * lx + a;
        if nodes[k].is_nan() {
            let p = DVec2::new(
                (a as f64 * LATTICE as f64 - grid.i0 as f64) * CELL,
                (b as f64 * LATTICE as f64 - grid.j0 as f64) * CELL,
            );
            let s = gen.surface_natural(plane.dir(p), &detail);
            nodes[k] = if s.height < s.water_level + 1.5 { f32::INFINITY } else { s.height };
        }
        nodes[k]
    };
    let mut cache: Vec<f32> = vec![f32::NAN; grid.nx * grid.ny];
    let ground = |c: usize| -> Option<f64> {
        if cache[c].is_nan() {
            let (i, j) = (c % grid.nx, c / grid.nx);
            let (a, b) = (i / LATTICE, j / LATTICE);
            let (u, v) = ((i % LATTICE) as f32 / LATTICE as f32, (j % LATTICE) as f32 / LATTICE as f32);
            let (h00, h10, h01, h11) = (node(a, b), node(a + 1, b), node(a, b + 1), node(a + 1, b + 1));
            let h = (h00 * (1.0 - u) + h10 * u) * (1.0 - v) + (h01 * (1.0 - u) + h11 * u) * v;
            let d = plane.dir(grid.center(c));
            let blocked = !h.is_finite()
                || (c != start && in_own(d))
                || in_other(d)
                || highways.clearance(d).is_some_and(|k| k < 30.0)
                || circuits.clearance(d).is_some_and(|k| k < RUNOFF + 30.0);
            cache[c] = if blocked { f32::INFINITY } else { h };
        }
        cache[c].is_finite().then_some(cache[c] as f64)
    };

    // Premier cap : vers l'exterieur de la ville.
    let out = plane.local(city.dir_of(DVec2::from_angle(angle) * (edge + 200.0))).normalize_or(DVec2::X);
    let d0 = (0..N)
        .max_by(|a, b| {
            let v = |d: usize| DVec2::new(DIRS[d].0 as f64, DIRS[d].1 as f64).normalize().dot(out);
            v(*a).total_cmp(&v(*b))
        })
        .unwrap_or(0);
    let h_gate = gen.surface_natural(gate, &detail).height as f64;
    let peak_h = gen.surface_natural(peak, &detail).height as f64;
    let goal = h_gate + (peak_h - h_gate) * 0.85;
    // Une route qui se recroise, ou dont deux branches se frolent, recommence en evitant l'endroit.
    let mut ground = ground;
    let mut banned = vec![false; grid.nx * grid.ny];
    let mut budget = MAX_POPS;
    for _ in 0..4 {
        let steps =
            search(&grid, d0, goal, pp, MIN_RISE * 0.7, &mut budget, |c| if banned[c] { None } else { ground(c) })?;
        if steps.len() < 10 {
            return None;
        }
        let (pts, apexes) = polyline(&grid, &steps);
        let clashes = clashes(&pts);
        if clashes.is_empty() {
            // Un trace qui ne monte presque pas n'est pas une route de montagne.
            let top = steps.iter().map(|st| ground(st.cell).unwrap_or(f64::MIN)).fold(f64::MIN, f64::max);
            if top - h_gate < MIN_RISE * 0.7 {
                return None;
            }
            let mut line = vec![inside];
            line.extend(pts.iter().map(|p| plane.dir(*p)));
            return Some((angle, line, apexes.iter().map(|p| plane.dir(*p)).collect()));
        }
        for p in clashes {
            let (i, j) = ((p.x / CELL).round() as i32 + grid.i0 as i32, (p.y / CELL).round() as i32 + grid.j0 as i32);
            for (di, dj) in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (a, b) = (i + di, j + dj);
                if a >= 0 && b >= 0 && (a as usize) < grid.nx && (b as usize) < grid.ny {
                    banned[b as usize * grid.nx + a as usize] = true;
                }
            }
        }
        banned[grid.j0 * grid.nx + grid.i0] = false;
    }
    None
}

/// Points d'un trace trop proches d'un morceau plus ancien du meme trace (a plus de 80 m en arriere) :
/// les deux chaussees et leurs accotements se toucheraient.
fn clashes(pts: &[DVec2]) -> Vec<DVec2> {
    const NEAR: f64 = 2.0 * (ROAD_HALF + SHOULDER) + 4.0;
    let key = |p: DVec2| ((p.x / NEAR).floor() as i64, (p.y / NEAR).floor() as i64);
    let mut cells: FastMap<(i64, i64), Vec<usize>> = FastMap::default();
    let mut out = Vec::new();
    for (i, p) in pts.iter().enumerate() {
        let (x, y) = key(*p);
        let hit = (-1..=1).any(|dx| {
            (-1..=1).any(|dy| {
                cells.get(&(x + dx, y + dy)).is_some_and(|v| v.iter().any(|&j| j + 10 < i && (pts[j] - *p).length() < NEAR))
            })
        });
        if hit {
            out.push(*p);
        }
        cells.entry((x, y)).or_default().push(i);
    }
    out
}

impl MountainRoads {
    /// Aucune route : pour une planete sans villes, ou pendant sa construction.
    pub fn empty() -> Self {
        Self { list: Vec::new(), index: FastMap::default() }
    }

    /// Jusqu'a ROADS_PER_CITY routes de montagne pour chaque ville entouree de reliefs, chacune vers un
    /// sommet dans une direction differente ; la ville leur ouvre des sorties.
    pub fn build(gen: &Generator, cities: &mut [City], highways: &Highways, circuits: &Circuits) -> Self {
        let r = planet_radius();
        let found = {
            let cities_ref: &[City] = cities;
            let debug = std::env::var("TERRA_DEBUG_MOUNTAIN").is_ok();
            // Les routes principales de toutes les villes d'abord : elles passent avant les suivantes.
            let n = cities_ref.len();
            par_map(n * ROADS_PER_CITY, |job| {
                let (rank, i) = (job / n, job % n);
                let t = std::time::Instant::now();
                let road = plan_road(gen, cities_ref, i, rank, highways, circuits);
                if debug {
                    let ms = t.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("route {rank} de {} : {ms:.0} ms, {}", cities_ref[i].name, road.is_some());
                }
                (i, rank, road)
            })
        };
        let mut list: Vec<MountainRoad> = Vec::new();
        // Deux routes de montagne ne se croisent pas (elles n'ont pas de pont) : la seconde est abandonnee.
        let cell = SEPARATION / r;
        let key3 = |d: DVec3| ((d.x / cell).floor() as i64, (d.y / cell).floor() as i64, (d.z / cell).floor() as i64);
        let mut taken: FastMap<(i64, i64, i64), Vec<DVec3>> = FastMap::default();
        // Les noms, du premier au dernier sommet d'une ville.
        const NAMES: [&str; ROADS_PER_CITY] = ["Route du mont", "Route du col de", "Route des cretes de", "Route du pic de"];
        for (i, rank, f) in found {
            let Some((angle, line, hairpins)) = f else { continue };
            // Deux routes de la meme ville ne sortent pas au meme endroit.
            if cities[i].exits.iter().any(|e| angle_diff(*e, angle) < 0.35) {
                continue;
            }
            let crosses = line.iter().any(|p| {
                let (x, y, z) = key3(*p);
                (-1..=1).any(|dx| {
                    (-1..=1).any(|dy| {
                        (-1..=1).any(|dz| {
                            taken
                                .get(&(x + dx, y + dy, z + dz))
                                .is_some_and(|v| v.iter().any(|q| (*q - *p).length() * r < SEPARATION))
                        })
                    })
                })
            });
            if crosses {
                continue;
            }
            for p in &line {
                taken.entry(key3(*p)).or_default().push(*p);
            }
            cities[i].exits.push(angle);
            // La raquette du belvedere, dans le prolongement de la route.
            let n = line.len();
            let end = line[n - 1];
            let d = end - line[n - 2];
            let fwd = (d - end * d.dot(end)).normalize_or(end.any_orthonormal_vector());
            let center = (end + fwd * (LOOP_RADIUS / r)).normalize();
            let (e, nn) = tangent_frame(center);
            let v = end - center;
            let a0 = v.dot(nn).atan2(v.dot(e));
            let ring: Vec<DVec3> = (0..24)
                .map(|k| {
                    let a = a0 + k as f64 / 24.0 * TAU;
                    (center + (e * a.cos() + nn * a.sin()) * (LOOP_RADIUS / r)).normalize()
                })
                .collect();
            list.push(MountainRoad {
                name: format!("{} {}", NAMES[rank], cities[i].name),
                city: i,
                paths: vec![Path::new(line, false, ROAD_HALF), Path::new(ring, true, ROAD_HALF)],
                hairpins,
                stone: list.len() % 2 == 1,
                levels: OnceLock::new(),
            });
        }

        let mut index = RoadIndex::default();
        for (ri, road) in list.iter().enumerate() {
            for (pi, p) in road.paths.iter().enumerate() {
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
                        index.entry(kk).or_default().push((ri as u32, pi as u8, k as u32));
                    }
                }
            }
        }
        Self { list, index }
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
            let dist = (dir - (a + ab * t)).length() * r;
            if dist < reach && best.get(&(c, p)).is_none_or(|h| dist < h.dist) {
                best.insert((c, p), Hit { road: c as usize, path: p as usize, k: k as usize, t, dist });
            }
        }
        for h in best.into_values() {
            f(h);
        }
    }

    /// Altitude de la chaussee au point le plus proche.
    pub fn level_at(&self, gen: &Generator, h: &Hit) -> f64 {
        let road = &self.list[h.road];
        let path = &road.paths[h.path];
        let lv = &road.levels(gen)[h.path];
        lv[h.k] + (lv[path.next(h.k)] - lv[h.k]) * h.t
    }

    /// Le sol se plie a la route : chaussee et accotements a plat, talus au-dela. Entre deux branches d'un
    /// lacet, le terrain passe continument de l'une a l'autre.
    pub fn shape_ground(&self, gen: &Generator, dir: DVec3, s: &mut Surface) {
        if self.index.is_empty() || !self.index.contains_key(&key(dir)) {
            return;
        }
        let (mut most, mut sum, mut weighted) = (0.0f64, 0.0f64, 0.0f64);
        self.near(dir, INFLUENCE, |hit| {
            let lv = self.level_at(gen, &hit);
            let flat = ROAD_HALF + SHOULDER;
            let dh = (lv - s.height as f64).abs().min(38.0);
            let w = 1.0 - smoothstep(flat as f32, (flat + 4.0 + dh * 1.4) as f32, hit.dist as f32) as f64;
            if w > 0.0 {
                // Poids tres tranche : la branche la plus proche l'emporte, sans saut a mi-chemin.
                let k = w.powi(8);
                most = most.max(w);
                sum += k;
                weighted += k * lv;
            }
        });
        if most > 0.0 {
            let level = (weighted / sum) as f32 - 0.02;
            let w = most as f32;
            s.height += (level - s.height) * w;
            s.ridge *= 1.0 - w;
            if w > 0.5 && s.water_level > s.height && s.water_level > 0.5 {
                s.water_level = s.height - 1.0;
            }
        }
    }

    /// Distance au bord de chaussee le plus proche (negative dessus).
    pub fn clearance(&self, dir: DVec3) -> Option<f64> {
        let mut best: Option<f64> = None;
        self.near(dir, INFLUENCE, |hit| {
            let d = hit.dist - ROAD_HALF;
            best = Some(best.map_or(d, |b: f64| b.min(d)));
        });
        best
    }

    /// Vrai sur la chaussee, ou a moins de 30 cm de son bord.
    pub fn on_road(&self, dir: DVec3) -> bool {
        self.clearance(dir).is_some_and(|d| d < 0.3)
    }

    /// Couleur du sol sous la chaussee, et sa part dans une maille de cet ecart.
    pub fn paint(&self, dir: DVec3, spacing: f64) -> Option<([f32; 3], f32)> {
        let mut best: Option<f32> = None;
        self.near(dir, INFLUENCE, |hit| {
            if hit.dist < ROAD_HALF + spacing * 0.5 {
                let k = if spacing < ROAD_HALF { 1.0 } else { (ROAD_HALF * 4.0 / spacing).min(0.8) as f32 };
                best = Some(best.map_or(k, |b| b.max(k)));
            }
        });
        best.map(|k| ([0.05, 0.05, 0.055], k))
    }

    /// La route la plus proche d'un point, et la distance a son trace (a 100 m pres).
    pub fn nearest(&self, dir: DVec3) -> Option<(&MountainRoad, f64)> {
        let r = planet_radius();
        self.list
            .iter()
            .map(|road| {
                let d = road.paths[0].points.iter().step_by(12).map(|p| (*p - dir).length()).fold(f64::MAX, f64::min);
                (road, d * r)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Morceaux de trace dans la maille d'un point : (route, trace, segment).
    pub fn segments_at(&self, dir: DVec3) -> &[(u32, u8, u32)] {
        self.index.get(&key(dir)).map_or(&[], |v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sur une pente de 40 %, la seule facon de monter a 10 % est le lacet.
    #[test]
    fn a_steep_slope_is_climbed_in_hairpins() {
        let grid = Grid { nx: 120, ny: 30, i0: 5, j0: 15 };
        let slope = 0.4;
        let steps = search(&grid, 0, 300.0, DVec2::new(3000.0, 0.0), 1e9, &mut { usize::MAX }, |c| Some(grid.center(c).x.max(0.0) * slope)).expect("un trace");
        let pins = steps.iter().filter(|s| s.hairpin).count();
        assert!(pins >= 2, "{pins} epingles");
        let h = |c: usize| grid.center(c).x.max(0.0) * slope;
        let mut length = 0.0;
        for w in steps.windows(2) {
            let (a, b) = (grid.center(w[0].cell), grid.center(w[1].cell));
            let len = if w[1].hairpin { (b - a).length() * PI * 0.5 } else { (b - a).length() };
            let limit = if w[1].hairpin { HAIRPIN_GRADE } else { SEARCH_GRADE };
            assert!((h(w[1].cell) - h(w[0].cell)).abs() <= limit * len + 1e-6);
            length += len;
        }
        assert!(length >= 300.0 / SEARCH_GRADE * 0.95, "{length:.0} m de route");
        assert!(h(steps.last().unwrap().cell) >= 300.0);
        // Le trace lisse garde des points reguliers, et ses epingles.
        let (pts, apexes) = polyline(&grid, &steps);
        assert_eq!(apexes.len(), pins);
        assert!(pts.windows(2).all(|w| (w[1] - w[0]).length() < 8.0 * 1.4 + 1e-6));
    }

    /// Sur un terrain doux, pas d'epingle : la route monte tout droit, ou presque.
    #[test]
    fn a_gentle_slope_needs_no_hairpin() {
        let grid = Grid { nx: 200, ny: 40, i0: 5, j0: 20 };
        let steps = search(&grid, 0, 300.0, DVec2::new(5500.0, 0.0), 1e9, &mut { usize::MAX }, |c| Some(grid.center(c).x.max(0.0) * 0.06)).expect("un trace");
        assert!(steps.iter().all(|s| !s.hairpin));
    }

    /// Sur une vraie planete : des routes au pied des reliefs, qui montent haut, a pente raisonnable,
    /// avec des lacets, sur une chaussee posee sur le sol remodele.
    #[test]
    fn mountain_cities_get_winding_roads_up_to_a_viewpoint() {
        use crate::world::generator::PlanetParams;
        let gen = Generator::new(PlanetParams { seed: 3, ..Default::default() });
        let roads = &gen.cities.mountains;
        let r = planet_radius();
        let mut pins = 0;
        for road in &roads.list {
            let path = &road.paths[0];
            let lv = &road.levels(&gen)[0];
            let (lat, lon) = crate::world::generator::dir_to_lat_lon(road.start().0);
            println!(
                "{} : {:.1} km, {:.0} m de denivele, {} epingles, depart {lat:.4},{lon:.4}",
                road.name,
                path.length() / 1000.0,
                road.rise(&gen),
                road.hairpins.len()
            );
            pins += road.hairpins.len();
            // La route ne se recroise pas et ses branches ne se frolent pas.
            let o = path.points[0];
            let (e, nn) = tangent_frame(o);
            let flat: Vec<DVec2> = path.points.iter().map(|p| DVec2::new((*p - o).dot(e), (*p - o).dot(nn)) * r).collect();
            assert!(clashes(&flat[1..]).is_empty(), "{} se recroise", road.name);
            assert!(road.rise(&gen) > MIN_RISE * 0.5, "{} : {:.0} m", road.name, road.rise(&gen));
            for k in 0..lv.len() - 1 {
                let run = (path.points[k + 1] - path.points[k]).length() * r;
                assert!((lv[k + 1] - lv[k]).abs() <= run * LEVEL_GRADE + 0.05, "{} : pente au point {k}", road.name);
            }
            for (k, d) in path.points.iter().enumerate().step_by(7).skip(3) {
                assert!(roads.on_road(*d));
                let s = gen.surface(*d);
                assert!(s.height > s.water_level, "{} sous l'eau", road.name);
                // Le sol suit la chaussee, sauf la ou deux branches d'une epingle se partagent le terrain.
                let pinned = road.hairpins.iter().any(|h| (*h - *d).length() * r < 60.0);
                assert!(pinned || (s.height as f64 - lv[k]).abs() < 0.8, "{} : sol {} route {}", road.name, s.height, lv[k]);
            }
        }
        if let Ok(file) = std::env::var("TERRA_DUMP_MOUNTAIN") {
            // Traces dans le plan local de leur depart, pour les dessiner : route, x, y, altitude.
            let mut out = String::new();
            for (i, road) in roads.list.iter().enumerate() {
                let o = road.paths[0].points[0];
                let (e, n) = tangent_frame(o);
                for (k, p) in road.paths[0].points.iter().enumerate() {
                    let q = (*p - o) * r;
                    out += &format!("{i},{:.1},{:.1},{:.1}
", q.dot(e), q.dot(n), road.levels(&gen)[0][k]);
                }
            }
            std::fs::write(file, out).unwrap();
        }
        assert!(roads.list.len() >= 20, "{} routes", roads.list.len());
        assert!(pins >= roads.list.len(), "{pins} epingles");
    }

    /// Un sommet hors d'atteinte : la route s'arrete au plus haut point, ou n'existe pas.
    #[test]
    fn an_unreachable_summit_stops_at_the_highest_point() {
        let grid = Grid { nx: 60, ny: 30, i0: 5, j0: 15 };
        // Un mur infranchissable a x = 1000 m.
        let wall = |c: usize| (grid.center(c).x < 1000.0).then(|| grid.center(c).x.max(0.0) * 0.08);
        let steps = search(&grid, 0, 500.0, DVec2::new(2000.0, 0.0), 50.0, &mut { usize::MAX }, wall).expect("un trace");
        assert!(grid.center(steps.last().unwrap().cell).x < 1000.0);
        assert!(search(&grid, 0, 500.0, DVec2::new(2000.0, 0.0), 200.0, &mut { usize::MAX }, wall).is_none());
    }
}
