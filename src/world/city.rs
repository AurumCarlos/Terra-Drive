//! Villes : ou elles naissent sur la planete, et leur plan (rues, ilots, parcelles, batiments).
//!
//! Les plans reprennent ceux de villes reelles :
//! - la grille de Manhattan (plan de 1811) : avenues de 30 m tous les 260 m, rues de 18 m tous les 80 m ;
//! - l'Eixample de Cerda a Barcelone : ilots carres de 113 m aux angles coupes, rues de 20 m, deux diagonales ;
//! - Paris : une place en etoile, des avenues rayonnantes et des boulevards en anneaux ;
//! - les vieilles villes : des ruelles tortueuses autour d'une eglise.
//!
//! Les rues naissent en decoupant la ville recursivement (Parish et Muller, 2001), les parcelles en
//! decoupant les ilots le long de leur plus grand axe (Vanegas et al., 2012). Tout est deterministe :
//! le plan d'une ville ne depend que de la graine, et n'est calcule qu'a l'approche.

use super::generator::{smoothstep, Detail, Generator, planet_radius, SEA_LEVEL};
use super::highway::Highways;
use super::noise::{hash_seed, Rng};
use glam::{DVec2, DVec3};
use super::fast_hash::FastMap;
use std::sync::OnceLock;

/// Cote d'un panier de la table qui retrouve les villes proches d'un point, en mailles par face du cube.
const BUCKETS: f64 = 64.0;
/// Les villes au-dela n'aplanissent plus le relief.
const FLAT_EDGE: f64 = 1.25;
/// Maille de la grille d'acces aux elements d'un plan, en metres.
const GRID_CELL: f64 = 64.0;
const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// Grille de Manhattan.
    Grid,
    /// Ilots de Cerda a Barcelone.
    Eixample,
    /// Etoile et boulevards de Paris.
    Radial,
    /// Vieille ville aux rues tortueuses.
    Medieval,
}

impl Style {
    /// Nom du plan, pour l'affichage.
    pub fn label(self) -> &'static str {
        match self {
            Style::Grid => "en damier",
            Style::Eixample => "a ilots",
            Style::Radial => "en etoile",
            Style::Medieval => "ancienne",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Zone {
    /// Quartier d'affaires : tours et bureaux.
    Tower,
    /// Immeubles en bordure d'ilot.
    Dense,
    /// Maisons de ville serrees.
    Old,
    /// Maisons et jardins.
    Houses,
    /// Entrepots, cuves, cheminees.
    Industry,
    Park,
    Plaza,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    House,
    Garage,
    Pool,
    Townhouse,
    Haussmann,
    Apartments,
    Office,
    Tower,
    Warehouse,
    Tank,
    Chimney,
    Church,
    Monument,
    Fountain,
}

/// Ce qui couvre le sol en un point d'une ville.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ground {
    Wild,
    Road,
    Walk,
    Building,
    Paved,
    Yard,
    Garden,
    Park,
}

pub struct Road {
    pub a: DVec2,
    pub b: DVec2,
    /// Largeur de facade a facade, trottoirs compris.
    pub width: f32,
    pub walk: f32,
    /// Retrait de la chaussee a chaque bout, la ou la rue debouche sur une autre.
    pub trim: [f32; 2],
    /// Biais du bout de chaussee : en debouchant de travers, la chaussee s'arrete le long du bord de la
    /// rue croisee, pas a angle droit (sinon un triangle de terre reste entre les deux).
    pub skew: [f32; 2],
    /// Retrait des trottoirs : ils s'arretent au bord de la rue croisee.
    pub trim_walk: [f32; 2],
    /// Debouches d'autres rues : abscisse, demi-largeur, cote (-1, 1, ou 0 des deux cotes).
    /// Le trottoir de ce cote s'y interrompt.
    pub gaps: Vec<(f32, f32, f32)>,
    /// Alignement d'arbres sur les trottoirs.
    pub trees: bool,
}

impl Road {
    pub fn length(&self) -> f64 {
        (self.b - self.a).length()
    }

    pub fn dir(&self) -> DVec2 {
        (self.b - self.a).normalize_or_zero()
    }

    /// Demi-largeur de la chaussee.
    pub fn carriage(&self) -> f64 {
        (self.width * 0.5 - self.walk) as f64
    }

    /// Retrait de la chaussee au bout `k`, a la distance `x` de l'axe (vers `dir().perp()`).
    /// Il vaut `trim[k]` du cote le plus long et moins de l'autre : le bout suit le bord de la rue croisee.
    pub fn cut(&self, k: usize, x: f64) -> f64 {
        let sk = self.skew[k] as f64;
        (self.trim[k] as f64 - sk.abs() * self.carriage() + sk * x).max(0.0)
    }

    /// Le trottoir est-il interrompu a cette abscisse ?
    pub fn in_gap(&self, s: f32) -> bool {
        self.gaps.iter().any(|&(at, half, _)| (s - at).abs() < half)
    }

    /// Le trottoir de ce cote (-1 ou 1) est-il interrompu a cette abscisse ?
    pub fn in_gap_side(&self, s: f32, side: f32) -> bool {
        self.gaps.iter().any(|&(at, half, k)| (k == 0.0 || k == side) && (s - at).abs() < half)
    }
}

pub struct Block {
    /// Contour, dans le sens trigonometrique, au bord des trottoirs.
    pub poly: Vec<DVec2>,
    pub zone: Zone,
}

pub struct Building {
    /// Emprise dans le sens trigonometrique ; le premier cote donne sur la rue.
    pub foot: [DVec2; 4],
    pub kind: Kind,
    pub floors: u16,
    pub seed: u32,
    /// Mitoyen : ses cotes 1 et 3 touchent ses voisins et restent aveugles.
    pub row: bool,
}

impl Building {
    pub fn center(&self) -> DVec2 {
        (self.foot[0] + self.foot[1] + self.foot[2] + self.foot[3]) * 0.25
    }

    /// Hauteur approximative, pour choisir ce qui se voit de loin.
    pub fn height(&self) -> f32 {
        let floors = self.floors as f32;
        match self.kind {
            Kind::House | Kind::Garage => floors * 2.9 + 3.0,
            Kind::Pool | Kind::Fountain => 1.0,
            Kind::Townhouse => floors * 3.0 + 4.0,
            Kind::Haussmann => floors * 3.1 + 5.0,
            Kind::Apartments => floors * 2.9 + 1.5,
            Kind::Office => floors * 3.7 + 4.0,
            Kind::Tower => floors * 3.8 + 30.0,
            Kind::Warehouse => 11.0,
            Kind::Tank => 16.0,
            Kind::Chimney => 55.0,
            Kind::Church => 60.0,
            Kind::Monument => 32.0,
        }
    }
}

/// Cloture ou haie entre deux jardins.
pub struct Fence {
    pub a: DVec2,
    pub b: DVec2,
    /// 0 haie, 1 palissade blanche, 2 muret.
    pub style: u8,
}

/// Grille d'acces : les elements qui touchent chaque maille.
struct Grid {
    half: f64,
    n: usize,
    roads: Vec<Vec<u32>>,
    blocks: Vec<Vec<u32>>,
    buildings: Vec<Vec<u32>>,
}

impl Grid {
    fn new(half: f64) -> Self {
        let n = (half * 2.0 / GRID_CELL).ceil() as usize + 1;
        Self { half, n, roads: vec![Vec::new(); n * n], blocks: vec![Vec::new(); n * n], buildings: vec![Vec::new(); n * n] }
    }

    fn coords(&self, p: DVec2) -> (i64, i64) {
        (((p.x + self.half) / GRID_CELL).floor() as i64, ((p.y + self.half) / GRID_CELL).floor() as i64)
    }

    fn cell(&self, p: DVec2) -> Option<usize> {
        let (x, y) = self.coords(p);
        let n = self.n as i64;
        (x >= 0 && y >= 0 && x < n && y < n).then(|| (y * n + x) as usize)
    }

    /// Mailles couvertes par un rectangle.
    fn cells(&self, min: DVec2, max: DVec2) -> impl Iterator<Item = usize> + '_ {
        let n = self.n as i64;
        let (x0, y0) = self.coords(min);
        let (x1, y1) = self.coords(max);
        let (x0, y0, x1, y1) = (x0.max(0), y0.max(0), x1.min(n - 1), y1.min(n - 1));
        (y0..=y1).flat_map(move |y| (x0..=x1).map(move |x| (y * n + x) as usize))
    }
}

fn bounds(points: &[DVec2]) -> (DVec2, DVec2) {
    points.iter().fold((DVec2::splat(f64::MAX), DVec2::splat(f64::MIN)), |(lo, hi), p| (lo.min(*p), hi.max(*p)))
}

pub struct Plan {
    pub roads: Vec<Road>,
    pub blocks: Vec<Block>,
    pub buildings: Vec<Building>,
    pub fences: Vec<Fence>,
    grid: Grid,
    /// Profil des ponts, calcule rue par rue a l'approche : surelevation du tablier tous les `DECK_STEP` m.
    pub decks: std::sync::Mutex<FastMap<u32, std::sync::Arc<Vec<f32>>>>,
}

pub struct City {
    pub name: String,
    pub dir: DVec3,
    east: DVec3,
    north: DVec3,
    /// Rayon moyen, en metres.
    pub radius: f64,
    pub style: Style,
    /// Orientation de la grille, en radians depuis l'est.
    pub angle: f64,
    /// De 0 (village) a 1 (metropole).
    pub size: f32,
    pub seed: u64,
    /// Ellipse : demi-axes relatifs et orientation.
    shape: (f64, f64, f64),
    /// Sorties vers les autoroutes : angles dans le plan, depuis l'est. Le plan les prolonge en avenues.
    pub exits: Vec<f64>,
    plan: OnceLock<Plan>,
}

// ---------------------------------------------------------------------------------------------
// Geometrie plane

fn area(p: &[DVec2]) -> f64 {
    let n = p.len();
    (0..n).map(|i| p[i].perp_dot(p[(i + 1) % n])).sum::<f64>() * 0.5
}

fn centroid(p: &[DVec2]) -> DVec2 {
    p.iter().copied().sum::<DVec2>() / p.len().max(1) as f64
}

/// Vrai si `q` est dans le polygone convexe `p` (sens trigonometrique).
pub fn inside(p: &[DVec2], q: DVec2) -> bool {
    let n = p.len();
    n >= 3 && (0..n).all(|i| (p[(i + 1) % n] - p[i]).perp_dot(q - p[i]) >= -1e-6)
}

/// Coupe un polygone convexe par une droite (point, direction). Chaque sommet porte la rue qui longe
/// le cote qui en part. Rend la partie a gauche, celle a droite, et les deux points de coupe avec la
/// rue de leur cote. Le nouveau cote, de part et d'autre, recoit la rue `cut`.
fn split(poly: &[(DVec2, u32)], p: DVec2, d: DVec2, cut: u32) -> (Vec<(DVec2, u32)>, Vec<(DVec2, u32)>, Vec<(DVec2, u32)>) {
    let side = |v: DVec2| d.perp_dot(v - p);
    let (mut l, mut r, mut hits) = (Vec::new(), Vec::new(), Vec::new());
    let n = poly.len();
    for i in 0..n {
        let (a, ra) = poly[i];
        let b = poly[(i + 1) % n].0;
        let (sa, sb) = (side(a), side(b));
        if sa >= 0.0 {
            l.push((a, ra));
        } else {
            r.push((a, ra));
        }
        if (sa >= 0.0) != (sb >= 0.0) {
            let x = a + (b - a) * (sa / (sa - sb));
            hits.push((x, ra));
            if sa >= 0.0 {
                l.push((x, cut));
                r.push((x, ra));
            } else {
                r.push((x, cut));
                l.push((x, ra));
            }
        }
    }
    (l, r, hits)
}

fn plain(p: &[DVec2]) -> Vec<(DVec2, u32)> {
    p.iter().map(|v| (*v, NONE)).collect()
}

fn points(p: &[(DVec2, u32)]) -> Vec<DVec2> {
    p.iter().map(|v| v.0).collect()
}

/// Garde la partie a gauche d'une droite.
fn clip_left(p: &[DVec2], at: DVec2, d: DVec2) -> Vec<DVec2> {
    let (l, _, _) = split(&plain(p), at, d, NONE);
    points(&l)
}

/// Retrecit un polygone convexe : chaque cote recule de sa propre distance.
fn inset(p: &[DVec2], by: &[f64]) -> Vec<DVec2> {
    let mut out = p.to_vec();
    let n = p.len();
    for i in 0..n {
        if by[i] <= 0.0 {
            continue;
        }
        let d = (p[(i + 1) % n] - p[i]).normalize_or_zero();
        if d == DVec2::ZERO {
            continue;
        }
        out = clip_left(&out, p[i] + d.perp() * by[i], d);
        if out.len() < 3 {
            return Vec::new();
        }
    }
    out
}

/// Coupe les angles d'un polygone, comme les ilots de l'Eixample.
fn chamfer(p: &[DVec2], cut: f64) -> Vec<DVec2> {
    let mut out = p.to_vec();
    for i in 0..p.len() {
        let v = p[i];
        let a = (p[(i + p.len() - 1) % p.len()] - v).normalize_or_zero();
        let b = (p[(i + 1) % p.len()] - v).normalize_or_zero();
        // Seulement les angles proches du droit : les autres sont deja ouverts ou trop aigus.
        if a.dot(b).abs() > 0.4 {
            continue;
        }
        let bis = (a + b).normalize_or_zero();
        let at = v + bis * cut * 0.7071;
        out = clip_left(&out, at, -bis.perp());
        if out.len() < 3 {
            return p.to_vec();
        }
    }
    out
}

/// Rectangle oriente d'aire minimale : centre, grand axe, demi-longueurs (grand puis petit cote).
fn obb(p: &[DVec2]) -> (DVec2, DVec2, f64, f64) {
    let mut best = (f64::MAX, DVec2::ZERO, DVec2::X, 0.0, 0.0);
    for i in 0..p.len() {
        let ax = (p[(i + 1) % p.len()] - p[i]).normalize_or_zero();
        if ax == DVec2::ZERO {
            continue;
        }
        let ay = ax.perp();
        let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for v in p {
            let (x, y) = (v.dot(ax), v.dot(ay));
            (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
        }
        let a = (x1 - x0) * (y1 - y0);
        if a < best.0 {
            let c = ax * (x0 + x1) * 0.5 + ay * (y0 + y1) * 0.5;
            best = if x1 - x0 >= y1 - y0 { (a, c, ax, (x1 - x0) * 0.5, (y1 - y0) * 0.5) } else { (a, c, ay, (y1 - y0) * 0.5, (x1 - x0) * 0.5) };
        }
    }
    (best.1, best.2, best.3, best.4)
}

/// Rectangle aligne sur `axis` qui tient dans le polygone, retreci de `margin` sur chaque cote.
/// Le premier cote est le plus proche du bord `front` du polygone.
fn fit_rect(p: &[DVec2], axis: DVec2, margin: f64) -> Option<[DVec2; 4]> {
    let ay = axis.perp();
    let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for v in p {
        let (x, y) = (v.dot(axis), v.dot(ay));
        (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
    }
    let c = centroid(p);
    let (cx, cy) = (c.dot(axis), c.dot(ay));
    let (mut hx0, mut hx1, mut hy0, mut hy1) = (cx - x0 - margin, x1 - cx - margin, cy - y0 - margin, y1 - cy - margin);
    for _ in 0..24 {
        if hx0 + hx1 < 3.0 || hy0 + hy1 < 3.0 {
            return None;
        }
        let corners = [(cx - hx0, cy - hy0), (cx + hx1, cy - hy0), (cx + hx1, cy + hy1), (cx - hx0, cy + hy1)]
            .map(|(x, y)| axis * x + ay * y);
        let shrunk = inset(p, &vec![margin; p.len()]);
        if corners.iter().all(|q| inside(&shrunk, *q)) {
            return Some(corners);
        }
        (hx0, hx1, hy0, hy1) = (hx0 * 0.93, hx1 * 0.93, hy0 * 0.93, hy1 * 0.93);
    }
    None
}

fn rotate(v: DVec2, a: f64) -> DVec2 {
    DVec2::from_angle(a).rotate(v)
}

fn hash01(seed: u64, a: i64, b: i64) -> f64 {
    (hash_seed(seed, a, b) >> 11) as f64 / (1u64 << 53) as f64
}

/// Bruit de valeur plan, dans [0, 1].
fn vnoise(seed: u64, p: DVec2) -> f64 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * (DVec2::splat(3.0) - f * 2.0);
    let (x, y) = (i.x as i64, i.y as i64);
    let a = hash01(seed, x, y) + (hash01(seed, x + 1, y) - hash01(seed, x, y)) * u.x;
    let b = hash01(seed, x, y + 1) + (hash01(seed, x + 1, y + 1) - hash01(seed, x, y + 1)) * u.x;
    a + (b - a) * u.y
}

/// Trottoirs selon la largeur de la rue.
fn walk_for(width: f32) -> f32 {
    match width {
        w if w >= 40.0 => 6.0,
        w if w >= 24.0 => 4.5,
        w if w >= 16.0 => 3.5,
        w if w >= 11.0 => 2.4,
        _ => 1.2,
    }
}

// ---------------------------------------------------------------------------------------------
// Une ville

impl City {
    fn new(dir: DVec3, size: f32, style: Style, angle: f64, seed: u64) -> Self {
        let east = if dir.y.abs() > 0.999 { DVec3::X } else { DVec3::Y.cross(dir).normalize() };
        let north = dir.cross(east);
        let radius = match size {
            s if s > 0.9 => 8500.0,
            s if s > 0.5 => 5000.0,
            s if s > 0.3 => 2600.0,
            _ => 950.0,
        } * (0.85 + 0.3 * hash01(seed, 1, 0));
        let stretch = 0.8 + 0.25 * hash01(seed, 2, 0);
        Self {
            name: city_name(seed, style),
            dir,
            east,
            north,
            radius,
            style,
            angle,
            size,
            seed,
            shape: (1.0 / stretch, stretch, hash01(seed, 3, 0) * std::f64::consts::PI),
            exits: Vec::new(),
            plan: OnceLock::new(),
        }
    }

    /// Coordonnees dans le plan tangent de la ville, en metres : x vers l'est, y vers le nord.
    pub fn local(&self, dir: DVec3) -> DVec2 {
        let d = dir.dot(self.dir).max(1e-6);
        DVec2::new(dir.dot(self.east), dir.dot(self.north)) * (planet_radius() / d)
    }

    pub fn dir_of(&self, p: DVec2) -> DVec3 {
        (self.dir * planet_radius() + self.east * p.x + self.north * p.y).normalize()
    }

    /// Axes est et nord du plan, dans l'espace.
    pub fn axes(&self) -> (DVec3, DVec3) {
        (self.east, self.north)
    }

    /// Distance relative au centre : 1 sur la limite de la ville.
    pub fn extent(&self, p: DVec2) -> f64 {
        let q = rotate(p, -self.shape.2);
        (DVec2::new(q.x / self.shape.0, q.y / self.shape.1)).length() / self.radius
    }

    pub fn plan(&self, gen: &Generator) -> &Plan {
        self.plan.get_or_init(|| Plan::build(self, gen))
    }

    /// Plan s'il est deja calcule : de loin, on ne le calcule pas pour si peu.
    pub fn plan_if_ready(&self) -> Option<&Plan> {
        self.plan.get()
    }

    fn grid_axes(&self) -> (DVec2, DVec2) {
        let ax = DVec2::from_angle(self.angle);
        (ax, ax.perp())
    }

    /// Part du rayon occupee par le quartier des tours.
    fn downtown(&self) -> f64 {
        if self.size >= 0.5 && self.style != Style::Medieval {
            0.14 + 0.12 * self.size as f64
        } else {
            0.0
        }
    }

    /// Centre du quartier d'affaires. Une ville en etoile garde son coeur ancien et pousse ses
    /// tours au bout de l'axe principal, comme la Defense au bout de l'axe historique de Paris.
    fn business_center(&self) -> DVec2 {
        if self.style == Style::Radial {
            let ax = DVec2::from_angle(self.angle);
            // Sur l'avenue, a 62 % du rayon de la ville dans cette direction.
            let reach = self.radius * 0.62 / self.extent(ax * self.radius).max(0.3);
            ax * reach
        } else {
            DVec2::ZERO
        }
    }

    /// Distance relative au quartier d'affaires, a l'echelle de `downtown()`.
    fn business_distance(&self, p: DVec2) -> f64 {
        if self.style == Style::Radial {
            // Un quartier compact : 2.4 fois plus petit que le centre d'une ville en damier.
            (p - self.business_center()).length() / self.radius * 2.4
        } else {
            self.extent(p)
        }
    }

    /// Quartier d'un point, sans le plan : sert aussi a colorer la ville vue de loin.
    pub fn zone_at(&self, p: DVec2) -> Zone {
        let r = self.extent(p);
        let n = vnoise(self.seed ^ 0x51, p / 1400.0);
        let n2 = vnoise(self.seed ^ 0x77, p / 900.0);
        if self.size < 0.3 {
            return if r < 0.38 + 0.12 * n { Zone::Old } else { Zone::Houses };
        }
        if self.style == Style::Medieval && r < 0.3 + 0.1 * n {
            return Zone::Old;
        }
        if self.business_distance(p) < self.downtown() * (0.8 + 0.4 * n) {
            return Zone::Tower;
        }
        let dense = if self.size > 0.5 { 0.48 } else { 0.32 } + 0.18 * n;
        if r < dense {
            return Zone::Dense;
        }
        if n2 > 0.72 && r > 0.4 {
            return Zone::Industry;
        }
        Zone::Houses
    }

    /// Distance du centre au contour dans une direction (angle depuis l'est), en metres.
    pub fn reach(&self, angle: f64) -> f64 {
        let d = DVec2::from_angle(angle);
        let boundary = self.boundary();
        (0..boundary.len())
            .filter_map(|k| {
                let (a, b) = (boundary[k], boundary[(k + 1) % boundary.len()]);
                let e = b - a;
                let den = d.perp_dot(e);
                if den.abs() < 1e-9 {
                    return None;
                }
                let t = a.perp_dot(e) / den;
                let s = a.perp_dot(d) / den;
                (t > 0.0 && (0.0..=1.0).contains(&s)).then_some(t)
            })
            .fold(0.0, f64::max)
    }

    /// La droite du centre dans cette direction ne coupe le contour qu'une fois de chaque cote :
    /// une avenue peut la suivre d'un bord a l'autre sans ressortir de la ville.
    pub fn exit_clear(&self, angle: f64) -> bool {
        let boundary = self.boundary();
        [angle, angle + std::f64::consts::PI].iter().all(|&a| {
            let d = DVec2::from_angle(a);
            (0..boundary.len())
                .filter(|&k| {
                    let (a, b) = (boundary[k], boundary[(k + 1) % boundary.len()]);
                    let e = b - a;
                    let den = d.perp_dot(e);
                    if den.abs() < 1e-9 {
                        return false;
                    }
                    let t = a.perp_dot(e) / den;
                    let s = a.perp_dot(d) / den;
                    t > 0.0 && (0.0..1.0).contains(&s)
                })
                .count()
                == 1
        })
    }

    /// Les directions possibles d'une sortie vers une autoroute, et leur cout : les axes d'une grille,
    /// puis ses diagonales ; les avenues d'une etoile ; huit directions pour une vieille ville.
    pub fn exit_options(&self) -> Vec<(f64, f64)> {
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, TAU};
        match self.style {
            Style::Grid | Style::Eixample => {
                let diag = if self.style == Style::Grid { 0.5 } else { 0.9 };
                (0..4).map(|k| (self.angle + k as f64 * FRAC_PI_2, 0.0)).chain((0..4).map(|k| (self.angle + FRAC_PI_4 + k as f64 * FRAC_PI_2, diag))).collect()
            }
            Style::Radial => {
                let n = self.avenues();
                (0..n).map(|i| (self.angle + i as f64 / n as f64 * TAU, 0.0)).collect()
            }
            Style::Medieval => (0..8).map(|k| (self.angle + k as f64 * FRAC_PI_4, 0.0)).collect(),
        }
    }

    /// Nombre d'avenues d'une ville en etoile.
    fn avenues(&self) -> usize {
        if self.size > 0.5 {
            12
        } else {
            8
        }
    }

    /// Les sorties, une droite par direction (modulo un demi-tour).
    fn exit_lines(&self) -> Vec<f64> {
        let mut lines: Vec<f64> = Vec::new();
        for e in &self.exits {
            let a = e.rem_euclid(std::f64::consts::PI);
            if !lines.iter().any(|l| line_gap(*l, a) < 0.05) {
                lines.push(a);
            }
        }
        lines
    }

    /// Limite batie, en part du contour : des lobes et des echancrures, de 0.5 a 1.
    pub fn edge_at(&self, p: DVec2) -> f64 {
        let big = vnoise(self.seed ^ 0xED6E, p / (self.radius * 0.42));
        let small = vnoise(self.seed ^ 0xED6F, p / (self.radius * 0.16));
        (0.22 + 0.78 * big + 0.18 * small).clamp(0.42, 1.0)
    }

    /// Part batie d'un point, de 0 a 1 : 1 dans la ville, 0 dans les champs.
    pub fn urban(&self, p: DVec2) -> f32 {
        let (e, edge) = (self.extent(p), self.edge_at(p));
        1.0 - smoothstep((edge - 0.04) as f32, (edge + 0.04) as f32, e as f32)
    }

    /// Le contour de la ville : une ellipse, convexe.
    fn boundary(&self) -> Vec<DVec2> {
        (0..40)
            .map(|i| {
                let a = i as f64 / 40.0 * std::f64::consts::TAU;
                let q = DVec2::new(a.cos() * self.shape.0, a.sin() * self.shape.1) * self.radius;
                rotate(q, self.shape.2)
            })
            .collect()
    }
}

/// Distance d'un point a un segment.
fn seg_dist(p: DVec2, a: DVec2, b: DVec2) -> f64 {
    let e = b - a;
    let t = ((p - a).dot(e) / e.length_squared().max(1e-9)).clamp(0.0, 1.0);
    (a + e * t - p).length()
}

/// Les deux segments se coupent-ils ? Rend l'abscisse sur le premier.
fn seg_cross(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> Option<f64> {
    let (e, f) = (b - a, d - c);
    let den = e.perp_dot(f);
    if den.abs() < 1e-9 {
        return None;
    }
    let t = (c - a).perp_dot(f) / den;
    let u = (c - a).perp_dot(e) / den;
    ((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)).then(|| t * e.length())
}

/// Deux rues se touchent-elles ? Le bout de l'une sur la chaussee de l'autre, ou un croisement.
fn roads_touch(r: &Road, o: &Road) -> bool {
    let (wr, wo) = (r.width as f64 * 0.5 + 3.0, o.width as f64 * 0.5 + 3.0);
    seg_dist(r.a, o.a, o.b) < wo
        || seg_dist(r.b, o.a, o.b) < wo
        || seg_dist(o.a, r.a, r.b) < wr
        || seg_dist(o.b, r.a, r.b) < wr
        || seg_cross(r.a, r.b, o.a, o.b).is_some()
}

/// Abscisse, sur la rue `r`, de son contact avec la rue `o`.
fn contact_at(r: &Road, o: &Road) -> f64 {
    if let Some(t) = seg_cross(r.a, r.b, o.a, o.b) {
        return t;
    }
    let (d, len) = (r.dir(), r.length());
    let mut best = (f64::MAX, 0.0);
    for p in [o.a, o.b] {
        let s = (p - r.a).dot(d).clamp(0.0, len);
        let dd = (r.a + d * s - p).length();
        if dd < best.0 {
            best = (dd, s);
        }
    }
    for (s, p) in [(0.0, r.a), (len, r.b)] {
        let dd = seg_dist(p, o.a, o.b);
        if dd < best.0 {
            best = (dd, s);
        }
    }
    best.1
}

/// Pour chaque rue, les rues qu'elle touche.
pub fn road_contacts(roads: &[Road]) -> Vec<Vec<u32>> {
    const CELL: f64 = 80.0;
    let mut cells: FastMap<(i32, i32), Vec<u32>> = FastMap::default();
    for (i, r) in roads.iter().enumerate() {
        let m = r.width as f64 * 0.5 + 4.0;
        let (lo, hi) = (r.a.min(r.b) - DVec2::splat(m), r.a.max(r.b) + DVec2::splat(m));
        for x in (lo.x / CELL).floor() as i32..=(hi.x / CELL).floor() as i32 {
            for y in (lo.y / CELL).floor() as i32..=(hi.y / CELL).floor() as i32 {
                cells.entry((x, y)).or_default().push(i as u32);
            }
        }
    }
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); roads.len()];
    for list in cells.values() {
        for (k, &i) in list.iter().enumerate() {
            for &j in &list[k + 1..] {
                if !out[i as usize].contains(&j) && roads_touch(&roads[i as usize], &roads[j as usize]) {
                    out[i as usize].push(j);
                    out[j as usize].push(i);
                }
            }
        }
    }
    out
}

impl Plan {
    /// Nombre de morceaux du reseau de rues (1 : tout est relie).
    pub fn road_networks(&self) -> usize {
        let touch = road_contacts(&self.roads);
        let mut seen = vec![false; self.roads.len()];
        let mut count = 0;
        for s in 0..self.roads.len() {
            if seen[s] {
                continue;
            }
            count += 1;
            let mut stack = vec![s];
            seen[s] = true;
            while let Some(u) = stack.pop() {
                for &v in &touch[u] {
                    if !seen[v as usize] {
                        seen[v as usize] = true;
                        stack.push(v as usize);
                    }
                }
            }
        }
        count
    }

    /// Distance d'un point du plan au bord de la rue la plus proche (negative sur la rue).
    pub fn road_distance(&self, p: DVec2) -> f64 {
        self.roads.iter().map(|r| seg_dist(p, r.a, r.b) - r.width as f64 * 0.5).fold(f64::MAX, f64::min)
    }
}

/// Ecart entre deux droites, d'angles donnes (modulo un demi-tour).
fn line_gap(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::PI);
    d.min(std::f64::consts::PI - d)
}

fn city_name(seed: u64, style: Style) -> String {
    const START: [&str; 24] = [
        "Val", "Mor", "Bel", "Cor", "Sar", "Lun", "Ar", "Ost", "Mel", "Tor", "Vir", "Cal", "Bra", "Esk", "Ner", "Pal", "Rho", "Sel", "Tal",
        "Zan", "Ven", "Quel", "Dor", "Lis",
    ];
    const MID: [&str; 16] = ["a", "e", "i", "o", "en", "ar", "il", "or", "an", "e", "u", "ol", "ia", "er", "on", "y"];
    const END: [&str; 20] = [
        "ville", "bourg", "mont", "port", "lac", "nes", "dun", "grad", "polis", "heim", "via", "cia", "ra", "tes", "sac", "rive", "court", "mar",
        "lon", "zia",
    ];
    let mut rng = Rng(hash_seed(seed, 0x4E4D, 0));
    let pick = |r: &mut Rng, n: usize| (r.f32() * n as f32) as usize % n;
    let mut name = String::new();
    if style == Style::Medieval && rng.f32() < 0.25 {
        name.push_str("Saint-");
    }
    name.push_str(START[pick(&mut rng, START.len())]);
    if rng.f32() < 0.6 {
        name.push_str(MID[pick(&mut rng, MID.len())]);
    }
    name.push_str(END[pick(&mut rng, END.len())]);
    name
}

// ---------------------------------------------------------------------------------------------
// Le plan

struct PlanBuilder<'a> {
    city: &'a City,
    rng: Rng,
    roads: Vec<Road>,
    finals: Vec<Vec<(DVec2, u32)>>,
    /// Morceaux deja attribues a un quartier (la place centrale).
    fixed: Vec<(Vec<(DVec2, u32)>, Zone)>,
    /// Pour chaque rue, les rues sur lesquelles elle debouche a chaque bout.
    ends: Vec<[u32; 2]>,
    /// Debouches : (rue interrompue, indice du debouche, rue qui y arrive, bout de celle-ci).
    links: Vec<(u32, usize, u32, usize)>,
    /// Refuse les rues qui croisent les autres trop en biais (decoupage libre).
    strict: bool,
}

impl<'a> PlanBuilder<'a> {
    fn width(&self, r: u32) -> f64 {
        if r == NONE {
            0.0
        } else {
            self.roads[r as usize].width as f64
        }
    }

    /// Ajoute une rue entre deux points de coupe, et la fait deboucher proprement sur les rues qu'elle rejoint.
    /// En biais, la chaussee et le trottoir s'arretent assez tot pour ne pas empieter sur la rue croisee.
    fn add_road(&mut self, a: (DVec2, u32), b: (DVec2, u32), width: f32, trees: bool) -> u32 {
        let id = self.roads.len() as u32;
        let walk = walk_for(width);
        let dir = (b.0 - a.0).normalize_or_zero();
        let mut road = Road { a: a.0, b: b.0, width, walk, trim: [0.0; 2], skew: [0.0; 2], trim_walk: [0.0; 2], gaps: Vec::new(), trees };
        let cw = (width * 0.5 - walk) as f64;
        for (k, (p, other)) in [a, b].into_iter().enumerate() {
            if other != NONE {
                let o = &mut self.roads[other as usize];
                let od = o.dir();
                let sin = od.perp_dot(dir).abs().max(0.3);
                let cos = od.dot(dir).abs();
                let (ocw, ow2) = ((o.width * 0.5 - o.walk) as f64, o.width as f64 * 0.5);
                road.trim[k] = ((ocw + cw * cos) / sin) as f32;
                road.trim_walk[k] = ((ow2 + width as f64 * 0.5 * cos) / sin) as f32;
                let s = (p - o.a).dot(od) as f32;
                // La rue arrive d'un seul cote de l'autre : celui de son autre bout.
                let far = if k == 0 { b.0 } else { a.0 };
                let side = od.perp().dot(far - o.a).signum() as f32;
                // Bord de la rue croisee de ce cote : le retrait varie lineairement en travers.
                road.skew[k] = (-(side as f64) * dir.perp().dot(od.perp()) / sin) as f32;
                let half = cw / sin + ow2 * cos / sin;
                o.gaps.push((s, half as f32, side));
                let gi = o.gaps.len() - 1;
                self.links.push((other, gi, id, k));
            }
        }
        self.roads.push(road);
        self.ends.push([a.1, b.1]);
        id
    }

    /// Trace une rue a travers un morceau de ville.
    fn cut(&mut self, poly: &[(DVec2, u32)], p: DVec2, d: DVec2, width: f32, trees: bool) -> Option<(Vec<(DVec2, u32)>, Vec<(DVec2, u32)>)> {
        let id = self.roads.len() as u32;
        let (l, r, hits) = split(poly, p, d, id);
        if hits.len() != 2 || l.len() < 3 || r.len() < 3 || (hits[0].0 - hits[1].0).length() < 6.0 {
            return None;
        }
        // Un carrefour trop aigu donne des pointes d'ilots et des chaussees qui se chevauchent.
        if self.strict && hits.iter().any(|h| h.1 != NONE && self.roads[h.1 as usize].dir().perp_dot(d.normalize_or_zero()).abs() < 0.55) {
            return None;
        }
        // Pas de morceau trop mince de part et d'autre.
        if area(&points(&l)).abs() < 200.0 || area(&points(&r)).abs() < 200.0 {
            return None;
        }
        self.add_road(hits[0], hits[1], width, trees);
        Some((l, r))
    }

    /// Trace une meme droite a travers tous les morceaux qu'elle traverse.
    fn cut_all(&mut self, pieces: Vec<Vec<(DVec2, u32)>>, p: DVec2, d: DVec2, width: f32, trees: bool) -> Vec<Vec<(DVec2, u32)>> {
        let mut out = Vec::new();
        for piece in pieces {
            match self.cut(&piece, p, d, width, trees) {
                Some((l, r)) => {
                    out.push(l);
                    out.push(r);
                }
                None => out.push(piece),
            }
        }
        out
    }

    fn organic_width(&self, depth: u32, zone: Zone) -> f32 {
        if self.city.size < 0.3 {
            return if depth < 2 { 12.0 } else if zone == Zone::Old { 6.5 } else { 9.0 };
        }
        match (depth, zone) {
            (0..=1, _) => 24.0,
            (2..=3, Zone::Old) => 10.0,
            (2..=3, _) => 16.0,
            (_, Zone::Old) => 6.5,
            (_, Zone::Houses) => 11.0,
            (_, Zone::Dense) => 13.0,
            (_, Zone::Tower) => 18.0,
            _ => 14.0,
        }
    }

    /// Choisit la prochaine rue d'un morceau : point, direction, largeur, arbres. Aucune : c'est un ilot.
    fn choose_cut(&mut self, pts: &[DVec2], zone: Zone, depth: u32) -> Option<(DVec2, DVec2, f32, bool)> {
        let city = self.city;
        let lattice = match city.style {
            Style::Grid => Some((260.0, 80.0)),
            Style::Eixample => Some((133.3, 133.3)),
            _ => None,
        };
        // Le quadrillage s'arrete avant la limite batie : les faubourgs poussent librement.
        let core = self.on_lattice(pts, zone);
        if let (Some((px, py)), true) = (lattice, core) {
            // Grille : les rues tombent sur un reseau regulier, commun a toute la ville.
            let (ax, ay) = city.grid_axes();
            let step = if zone == Zone::Industry { 2 } else { 1 };
            let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
            for v in pts {
                let (x, y) = (v.dot(ax), v.dot(ay));
                (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
            }
            let pick = |lo: f64, hi: f64, pitch: f64| -> Option<i64> {
                let pitch = pitch * step as f64;
                let m = pitch * 0.3;
                let (k0, k1) = (((lo + m) / pitch).ceil() as i64, ((hi - m) / pitch).floor() as i64);
                (k0 <= k1).then(|| (((lo + hi) * 0.5 / pitch).round() as i64).clamp(k0, k1) * step)
            };
            let (rx, ry) = ((x1 - x0) / px, (y1 - y0) / py);
            let (kx, ky) = (pick(x0, x1, px), pick(y0, y1, py));
            let along_x = match (kx, ky) {
                (Some(_), Some(_)) => rx >= ry,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => return None,
            };
            let suburb = zone == Zone::Houses;
            return Some(if along_x {
                let k = kx.unwrap();
                let w = match city.style {
                    Style::Grid => if suburb { 20.0 } else { 30.0 },
                    _ => if k % 6 == 0 { 30.0 } else { 20.0 },
                };
                (ax * (k as f64 * px), ay, w, city.style == Style::Eixample || w >= 30.0)
            } else {
                let k = ky.unwrap();
                let w = match city.style {
                    Style::Grid => if k % 10 == 0 { 30.0 } else if suburb { 13.0 } else { 18.0 },
                    _ => if k % 6 == 0 { 30.0 } else { 20.0 },
                };
                (ay * (k as f64 * py), ax, w, city.style == Style::Eixample || w >= 30.0)
            });
        }

        // Decoupage libre, perpendiculaire au grand axe, un peu de travers dans les vieilles villes.
        let (c, axis, hx, hy) = obb(pts);
        let (long, short) = match zone {
            Zone::Tower => (120.0, 70.0),
            Zone::Dense => (150.0, 90.0),
            Zone::Old => (68.0, 42.0),
            Zone::Houses => (170.0, 62.0),
            _ => (260.0, 150.0),
        };
        let along_major = if hx * 2.0 > long {
            true
        } else if hy * 2.0 > short * 1.7 {
            false
        } else {
            return None;
        };
        let (jitter, spread) = match city.style {
            Style::Medieval => (0.3, 0.2),
            Style::Radial => (0.06, 0.12),
            _ => (0.1, 0.12),
        };
        let jitter = if zone == Zone::Old { 0.35 } else { jitter };
        // Faubourgs : des rues dans toutes les directions, des ilots de tailles variees.
        let outskirts = city.extent(centroid(pts)) > city.edge_at(centroid(pts)) * 0.72;
        let (jitter, spread) = if outskirts && zone != Zone::Old { (f32::max(jitter, 0.3), f32::max(spread, 0.22)) } else { (jitter, spread) };
        let axis_split = if along_major { axis } else { axis.perp() };
        let ext = if along_major { hx } else { hy };
        let t = self.rng.range(0.5 - spread as f32, 0.5 + spread as f32) as f64;
        let p = c + axis_split * (ext * (2.0 * t - 1.0));
        let d = rotate(axis_split.perp(), self.rng.range(-jitter, jitter) as f64);
        let w = self.organic_width(depth, zone);
        Some((p, d, w, w >= 22.0))
    }

    /// Ce morceau est-il decoupe par le quadrillage ? Seulement au coeur d'une ville en damier ou a ilots.
    fn on_lattice(&self, pts: &[DVec2], zone: Zone) -> bool {
        let c = centroid(pts);
        matches!(self.city.style, Style::Grid | Style::Eixample) && zone != Zone::Old && self.city.extent(c) < self.city.edge_at(c) * 0.72
    }

    fn subdivide(&mut self, start: Vec<(Vec<(DVec2, u32)>, u32)>) {
        let mut stack = start;
        while let Some((poly, depth)) = stack.pop() {
            let pts = points(&poly);
            let a = area(&pts);
            if a < 900.0 || depth > 20 {
                self.finals.push(poly);
                continue;
            }
            let zone = self.city.zone_at(centroid(&pts));
            let lattice = self.on_lattice(&pts, zone);
            let mut cut = None;
            // Une rue libre trop en biais est refusee : on en tire une autre.
            for _ in 0..if lattice { 1 } else { 4 } {
                let Some((p, d, w, trees)) = self.choose_cut(&pts, zone, depth) else { break };
                self.strict = !lattice;
                cut = self.cut(&poly, p, d, w, trees);
                self.strict = false;
                if cut.is_some() {
                    break;
                }
            }
            match cut {
                Some((l, r)) => {
                    stack.push((l, depth + 1));
                    stack.push((r, depth + 1));
                }
                None => self.finals.push(poly),
            }
        }
    }

    /// Paris : une place en etoile, des avenues rayonnantes, des boulevards en anneaux.
    fn radial(&mut self, boundary: &[DVec2]) -> Vec<(Vec<(DVec2, u32)>, u32)> {
        let city = self.city;
        let n = city.avenues();
        let plaza = if city.size > 0.5 { 75.0 } else { 55.0 };
        let rings: Vec<f64> = match city.size {
            s if s > 0.9 => vec![0.2, 0.4, 0.62],
            s if s > 0.5 => vec![0.28, 0.58],
            _ => vec![0.45],
        };
        let angles: Vec<f64> = (0..n).map(|i| city.angle + i as f64 / n as f64 * std::f64::consts::TAU).collect();

        // Les avenues, de la place jusqu'au bord de la ville.
        let first: Vec<u32> = angles
            .iter()
            .map(|&a| {
                let d = DVec2::from_angle(a);
                // Sortie du rayon par le contour.
                let far = (0..boundary.len())
                    .filter_map(|k| {
                        let (a, b) = (boundary[k], boundary[(k + 1) % boundary.len()]);
                        let e = b - a;
                        let den = d.perp_dot(e);
                        if den.abs() < 1e-9 {
                            return None;
                        }
                        let t = a.perp_dot(e) / den;
                        let s = a.perp_dot(d) / den;
                        (t > 0.0 && (0.0..=1.0).contains(&s)).then_some(t)
                    })
                    .fold(0.0, f64::max);
                let id = self.roads.len() as u32;
                let start = d * (plaza + 11.0);
                let width = if city.size > 0.5 { 34.0 } else { 26.0 };
                self.roads.push(Road {
                    a: start,
                    b: d * far,
                    width,
                    walk: walk_for(width),
                    trim: [0.0; 2],
                    skew: [0.0; 2],
                    trim_walk: [0.0; 2],
                    gaps: Vec::new(),
                    trees: true,
                });
                self.ends.push([NONE, NONE]);
                id
            })
            .collect();

        let mut out = Vec::new();
        for i in 0..n {
            let (a0, a1) = (angles[i], angles[(i + 1) % n]);
            let (d0, d1) = (DVec2::from_angle(a0), DVec2::from_angle(a1));
            // Le secteur : a gauche de l'avenue i, a droite de l'avenue i + 1.
            let (l, _, _) = split(&plain(boundary), DVec2::ZERO, d0, first[i]);
            let (wedge, _, _) = split(&l, DVec2::ZERO, -d1, first[(i + 1) % n]);
            let mut inner = wedge;
            let mut radii: Vec<f64> = rings.iter().map(|r| r * city.radius).collect();
            radii.push(plaza);
            radii.sort_by(|a, b| b.partial_cmp(a).unwrap());
            for (k, r) in radii.iter().enumerate() {
                let (p0, p1) = (d0 * *r, d1 * *r);
                // L'anneau exterieur est le plus large, comme un boulevard peripherique.
                let w = if *r == plaza { 22.0 } else if k == 0 && rings.len() > 1 { 40.0 } else { 30.0 };
                // Le cote exterieur de la corde est a droite quand on va de l'avenue i a l'avenue i + 1.
                match self.cut(&inner, p0, p1 - p0, w as f32, true) {
                    Some((l, r)) => {
                        out.push((r, 3));
                        inner = l;
                    }
                    None => break,
                }
            }
            self.fixed.push((inner, Zone::Plaza));
        }
        out
    }

    fn build(mut self, gen: &Generator) -> Plan {
        let city = self.city;
        let profile = std::env::var("TERRA_PROFILE").is_ok();
        let mut clock = std::time::Instant::now();
        let mut step = |name: &str| {
            if profile {
                println!("    plan de {} : {name} {:.3} s", city.name, clock.elapsed().as_secs_f64());
            }
            clock = std::time::Instant::now();
        };
        let boundary = city.boundary();
        let mut rng = Rng(hash_seed(city.seed, 0x51A7, 0));
        let start: Vec<(Vec<(DVec2, u32)>, u32)> = match city.style {
            Style::Radial => self.radial(&boundary),
            Style::Grid => {
                let mut pieces = vec![plain(&boundary)];
                // Les sorties d'abord : des avenues de 30 m, du centre jusqu'aux autoroutes.
                let exits = city.exit_lines();
                for &a in &exits {
                    pieces = self.cut_all(pieces, DVec2::ZERO, DVec2::from_angle(a), 30.0, true);
                }
                let broadway = city.angle + 0.42;
                if city.size >= 0.5 && exits.iter().all(|e| line_gap(*e, broadway) > 0.45) {
                    // Une diagonale, comme Broadway, de travers sur la grille.
                    let p = DVec2::new(rng.range(-0.2, 0.2) as f64, rng.range(-0.2, 0.2) as f64) * city.radius;
                    pieces = self.cut_all(pieces, p, DVec2::from_angle(city.angle + 0.42), 26.0, false);
                }
                pieces.into_iter().map(|p| (p, 0)).collect()
            }
            Style::Eixample => {
                let mut pieces = vec![plain(&boundary)];
                let exits = city.exit_lines();
                for &a in &exits {
                    pieces = self.cut_all(pieces, DVec2::ZERO, DVec2::from_angle(a), 30.0, true);
                }
                if city.size >= 0.3 {
                    let p = DVec2::new(rng.range(-0.1, 0.1) as f64, rng.range(-0.1, 0.1) as f64) * city.radius;
                    // Les deux diagonales, sauf si une sortie suit deja leur direction.
                    for (at, a, w) in [(p, city.angle + 0.785, 50.0), (-p * 0.5, city.angle - 0.6, 30.0)] {
                        if exits.iter().all(|e| line_gap(*e, a) > 0.35) {
                            pieces = self.cut_all(pieces, at, DVec2::from_angle(a), w, true);
                        }
                    }
                }
                pieces.into_iter().map(|p| (p, 0)).collect()
            }
            Style::Medieval => {
                // De vieilles routes convergent vers le centre.
                let mut pieces = vec![plain(&boundary)];
                let roads = 3 + (rng.f32() * 3.0) as usize;
                let width = if city.size < 0.3 { 11.0 } else { 16.0 };
                // Les routes des autoroutes passent par le centre ; les autres croisent franchement.
                let mut lines: Vec<(f64, DVec2)> = city.exit_lines().into_iter().map(|a| (a, DVec2::ZERO)).collect();
                for i in 0..roads {
                    let a = city.angle + i as f64 / roads as f64 * std::f64::consts::PI + rng.range(-0.12, 0.12) as f64;
                    let p = DVec2::new(rng.range(-0.06, 0.06) as f64, rng.range(-0.06, 0.06) as f64) * city.radius;
                    if lines.len() < roads.max(2) && lines.iter().all(|l| line_gap(l.0, a) > 0.5) {
                        lines.push((a, p));
                    }
                }
                for (a, p) in lines {
                    pieces = self.cut_all(pieces, p, DVec2::from_angle(a), width, false);
                }
                pieces.into_iter().map(|p| (p, 2)).collect()
            }
        };
        step("trace");
        self.subdivide(start);
        step("subdivision");

        // Ilots : au bord des trottoirs.
        let mut blocks = Vec::new();
        let central_park = if city.style == Style::Grid && city.size > 0.5 {
            let (ax, ay) = city.grid_axes();
            let c = ay * city.radius * 0.3;
            Some((c, ax, ay))
        } else {
            None
        };
        let mut block_roads: Vec<Vec<u32>> = Vec::new();
        let mut pieces: Vec<(Vec<(DVec2, u32)>, Option<Zone>)> = self.finals.drain(..).map(|p| (p, None)).collect();
        pieces.extend(self.fixed.drain(..).map(|(p, z)| (p, Some(z))));
        for (k, (poly, fixed)) in pieces.iter().enumerate() {
            let pts = points(poly);
            if pts.len() < 3 {
                continue;
            }
            let c = centroid(&pts);
            if city.extent(c) > 1.0 {
                continue;
            }
            let mut zone = fixed.unwrap_or_else(|| city.zone_at(c));
            let roll = hash01(city.seed ^ 0xB10C, k as i64, 0);
            // Au-dela de la limite batie : des champs, quelques hameaux, et des maisons le long des grandes routes.
            if fixed.is_none() && city.extent(c) > city.edge_at(c) {
                let arterial = poly.iter().any(|v| v.1 != NONE && self.roads[v.1 as usize].width >= 24.0);
                // Des hameaux groupes, pas une poussiere de maisons isolees.
                let hamlet = vnoise(city.seed ^ 0x4A41, c / 650.0) > 0.82;
                let ribbon = arterial && city.extent(c) < city.edge_at(c) + 0.22 && roll < 0.8;
                if !hamlet && !ribbon {
                    continue;
                }
                zone = if zone == Zone::Industry { Zone::Industry } else { Zone::Houses };
            }
            if let Some((pc, ax, ay)) = central_park {
                let q = c - pc;
                if q.dot(ax).abs() < 270.0 && q.dot(ay).abs() < 650.0 {
                    zone = Zone::Park;
                }
            }
            if matches!(zone, Zone::Dense | Zone::Houses) && roll < 0.05 {
                zone = Zone::Park;
            } else if zone == Zone::Tower && roll < 0.07 {
                zone = Zone::Plaza;
            }
            let shape = if zone == Zone::Plaza && fixed.is_some() {
                pts.clone()
            } else {
                let by: Vec<f64> = poly.iter().map(|v| self.width(v.1) * 0.5).collect();
                inset(&pts, &by)
            };
            let shape = if city.style == Style::Eixample && matches!(zone, Zone::Dense | Zone::Tower) { chamfer(&shape, 10.0) } else { shape };
            if shape.len() < 3 || area(&shape) < 150.0 {
                continue;
            }
            blocks.push(Block { poly: shape, zone });
            block_roads.push(poly.iter().map(|v| v.1).filter(|r| *r != NONE).collect::<Vec<u32>>());
        }

        step("ilots");
        // L'eau : pas d'ilot dans la mer ou une riviere, pas de rue au large.
        let detail = Detail::for_spacing(25.0);
        let wet = |p: DVec2| {
            let s = gen.surface_detail(city.dir_of(p), &detail);
            s.height < s.water_level + 0.5
        };
        let dry: Vec<bool> = super::generator::par_map(blocks.len(), |i| {
            let b = &blocks[i];
            let c = centroid(&b.poly);
            let probes = std::iter::once(c).chain(b.poly.iter().map(|v| c + (*v - c) * 0.8));
            probes.filter(|p| wet(*p)).count() * 2 < b.poly.len() + 1
        });
        step("eau des ilots");
        let mut kept = dry.iter();
        blocks.retain(|_| *kept.next().unwrap());
        let mut kept = dry.iter();
        block_roads.retain(|_| *kept.next().unwrap());
        let n = self.roads.len();
        // Les rues qui bordent un ilot sont gardees, ainsi que les avenues et les routes des autoroutes.
        let exits = city.exit_lines();
        let on_exit = |r: &Road| {
            exits.iter().any(|&a| {
                let nrm = DVec2::from_angle(a).perp();
                nrm.dot(r.a).abs() < 2.0 && nrm.dot(r.b).abs() < 2.0
            })
        };
        let exit_road: Vec<bool> = self.roads.iter().map(on_exit).collect();
        let major: Vec<bool> = self.roads.iter().zip(&exit_road).map(|(r, e)| r.width >= 30.0 || *e).collect();
        let mut keep = vec![false; n];
        for r in block_roads.iter().flatten() {
            keep[*r as usize] = true;
        }
        for i in 0..n {
            keep[i] |= major[i];
        }
        let sea = |p: DVec2| {
            let s = gen.surface_detail(city.dir_of(p), &detail);
            s.height < s.water_level + 0.5 || (s.height < SEA_LEVEL && !s.fresh)
        };
        // Un pont court traverse un bras de mer ; au-dela, la rue s'arrete au rivage.
        // Une rue courte posee surtout sur l'eau (un bout de damier au large) disparait aussi :
        // un vrai pont a ses deux culees sur la terre ferme. Toutes les rues sont testees ensemble, sur
        // tous les coeurs : le reseau ci-dessous en consulte une bonne part.
        let roads_ref = &self.roads;
        let test_road = |i: usize| -> bool {
            {
                let r = &roads_ref[i];
                let k = (r.length() / 30.0).ceil().max(1.0) as usize;
                let (mut run, mut longest, mut wet_count) = (0usize, 0usize, 0usize);
                let mut ends_dry = true;
                for j in 0..=k {
                    let w = sea(r.a + (r.b - r.a) * (j as f64 / k as f64));
                    if w {
                        run += 1;
                        wet_count += 1;
                        longest = longest.max(run);
                        if j == 0 || j == k {
                            ends_dry = false;
                        }
                    } else {
                        run = 0;
                    }
                }
                longest as f64 * 30.0 > 260.0 || (!ends_dry && (wet_count * 2 > k + 1 || longest as f64 * 30.0 > 60.0))
            }
        };
        let wet_all = super::generator::par_map(n, test_road);
        let drowned = |i: usize| -> bool { wet_all[i] };
        for i in 0..n {
            // Une avenue vers une autoroute traverse l'eau sur un pont, quelle que soit sa longueur.
            if keep[i] && !exit_road[i] && drowned(i) {
                keep[i] = false;
            }
        }

        // Un seul reseau : chaque morceau de rues isole est relie au reste par le plus court chemin
        // de rues existantes (des chemins de campagne vers les hameaux). Sinon il disparait.
        step("eau");
        let touch = road_contacts(roads_ref);
        let lengths: Vec<f64> = roads_ref.iter().map(|r| r.length()).collect();
        let groups = |keep: &[bool]| -> Vec<usize> {
            fn find(c: &mut [usize], x: usize) -> usize {
                let mut x = x;
                while c[x] != x {
                    c[x] = c[c[x]];
                    x = c[x];
                }
                x
            }
            let mut comp: Vec<usize> = (0..n).collect();
            for i in 0..n {
                if keep[i] {
                    for &j in &touch[i] {
                        if keep[j as usize] {
                            let (a, b) = (find(&mut comp, i), find(&mut comp, j as usize));
                            comp[a] = b;
                        }
                    }
                }
            }
            (0..n).map(|i| find(&mut comp, i)).collect()
        };
        let mut main = vec![false; n];
        {
            let g = groups(&keep);
            let mut total: FastMap<usize, f64> = FastMap::default();
            for i in (0..n).filter(|&i| keep[i]) {
                *total.entry(g[i]).or_default() += lengths[i] * if major[i] { 4.0 } else { 1.0 };
            }
            if let Some((&root, _)) = total.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                for i in 0..n {
                    main[i] = keep[i] && g[i] == root;
                }
            }
        }
        // Puis, pour les avenues des autoroutes seulement : au besoin par des ponts.
        let mut bridging = false;
        loop {
            let g = groups(&keep);
            // Distances depuis le reseau principal, en longueur de rues a rouvrir.
            let mut dist = vec![f64::INFINITY; n];
            let mut prev = vec![u32::MAX; n];
            let mut open = std::collections::BinaryHeap::new();
            for i in (0..n).filter(|&i| main[i]) {
                dist[i] = 0.0;
                open.push(std::cmp::Reverse((0u64, i as u32)));
            }
            while let Some(std::cmp::Reverse((dk, u))) = open.pop() {
                let u = u as usize;
                if f64::from_bits(dk) > dist[u] {
                    continue;
                }
                for &v in &touch[u] {
                    let v = v as usize;
                    let step = if keep[v] { 0.0 } else { lengths[v] };
                    let nd = dist[u] + step;
                    if nd < dist[v] && nd < if bridging { 12_000.0 } else { 1500.0 } && (keep[v] || bridging || !drowned(v)) {
                        dist[v] = nd;
                        prev[v] = u as u32;
                        open.push(std::cmp::Reverse((nd.to_bits(), v as u32)));
                    }
                }
            }
            // Le morceau isole le plus proche.
            let Some(target) = (0..n)
                .filter(|&i| keep[i] && !main[i] && dist[i].is_finite() && (!bridging || exit_road[i]))
                .min_by(|a, b| dist[*a].partial_cmp(&dist[*b]).unwrap())
            else {
                if !bridging && (0..n).any(|i| exit_road[i] && keep[i] && !main[i]) {
                    bridging = true;
                    continue;
                }
                break;
            };
            let mut v = target;
            while !main[v] {
                keep[v] = true;
                main[v] = true;
                if prev[v] == u32::MAX {
                    break;
                }
                v = prev[v] as usize;
            }
            let root = g[target];
            for i in 0..n {
                if keep[i] && g[i] == root {
                    main[i] = true;
                }
            }
        }
        if std::env::var("TERRA_DEBUG_PLAN").is_ok_and(|v| v == city.name) {
            eprintln!("sorties {:?} lignes {:?}", city.exits, exits);
            for i in (0..n).filter(|&i| exit_road[i]) {
                let r = &roads_ref[i];
                eprintln!("  rue {i} {:?} -> {:?} garde {} principal {}", r.a, r.b, keep[i], main[i]);
            }
        }
        step("reseau");
        // Ce qui reste isole disparait, avec ses ilots.
        for i in 0..n {
            keep[i] &= main[i];
        }
        {
            let flags: Vec<bool> = block_roads.iter().map(|rs| rs.iter().any(|r| keep[*r as usize])).collect();
            let mut it = flags.iter();
            blocks.retain(|_| *it.next().unwrap());
            let mut it = flags.iter();
            block_roads.retain(|_| *it.next().unwrap());
        }

        // Une rue de faubourg s'arrete apres la derniere maison : pas de longues routes vides dans les champs.
        // Elle garde toujours ses carrefours avec les rues gardees.
        let mut span = vec![(f64::MAX, f64::MIN); n];
        for (b, rs) in blocks.iter().zip(&block_roads) {
            for &r in rs {
                let road = &self.roads[r as usize];
                let d = road.dir();
                for v in &b.poly {
                    let s = (*v - road.a).dot(d);
                    let sp = &mut span[r as usize];
                    *sp = (sp.0.min(s), sp.1.max(s));
                }
            }
        }
        for i in 0..n {
            if !keep[i] {
                continue;
            }
            for &j in &touch[i] {
                if keep[j as usize] {
                    let s = contact_at(&self.roads[i], &self.roads[j as usize]);
                    let sp = &mut span[i];
                    *sp = (sp.0.min(s), sp.1.max(s));
                }
            }
        }
        let mut detached = vec![[false; 2]; n];
        for (i, r) in self.roads.iter_mut().enumerate() {
            if !keep[i] || major[i] || span[i].0 > span[i].1 {
                continue;
            }
            let (len, d) = (r.length(), r.dir());
            let (s0, s1) = (span[i].0 - 10.0, span[i].1 + 10.0);
            let cut0 = span[i].0 > r.trim_walk[0] as f64 + 25.0;
            let cut1 = span[i].1 < len - r.trim_walk[1] as f64 - 25.0;
            let start = if cut0 { s0 } else { 0.0 };
            let end = if cut1 { s1 } else { len };
            if cut0 {
                r.trim[0] = 0.0;
                r.skew[0] = 0.0;
                r.trim_walk[0] = 0.0;
                detached[i][0] = true;
                for g in &mut r.gaps {
                    g.0 -= start as f32;
                }
            }
            if cut1 {
                r.trim[1] = 0.0;
                r.skew[1] = 0.0;
                r.trim_walk[1] = 0.0;
                detached[i][1] = true;
            }
            let a = r.a;
            r.a = a + d * start;
            r.b = a + d * end;
        }
        for &(other, gi, by, k) in &self.links {
            if detached[by as usize][k] {
                self.roads[other as usize].gaps[gi].1 = 0.0;
            }
        }

        let drop: Vec<bool> = keep.iter().map(|k| !k).collect();
        // Les rues restantes ne s'interrompent plus pour une rue disparue.
        for &(other, gi, by, _) in &self.links {
            if drop[by as usize] {
                self.roads[other as usize].gaps[gi].1 = 0.0;
            }
        }
        for (i, ends) in self.ends.iter().enumerate() {
            for k in 0..2 {
                if ends[k] != NONE && drop[ends[k] as usize] {
                    self.roads[i].trim[k] = 0.0;
                    self.roads[i].skew[k] = 0.0;
                    self.roads[i].trim_walk[k] = 0.0;
                }
            }
        }
        let mut roads: Vec<Road> = Vec::new();
        for (i, r) in std::mem::take(&mut self.roads).into_iter().enumerate() {
            if !drop[i] {
                roads.push(r);
            }
        }

        // Une eglise sur la place de la vieille ville.
        if city.style == Style::Medieval || city.size < 0.3 {
            if let Some(b) = blocks
                .iter_mut()
                .filter(|b| b.zone == Zone::Old)
                .min_by(|a, b| centroid(&a.poly).length().partial_cmp(&centroid(&b.poly).length()).unwrap())
            {
                b.zone = Zone::Plaza;
            }
        }

        let mut plan = Plan {
            roads,
            blocks,
            buildings: Vec::new(),
            fences: Vec::new(),
            grid: Grid::new(city.radius * 1.5),
            decks: Default::default(),
        };
        step("nettoyage");
        for i in 0..plan.blocks.len() {
            plan.fill_block(city, i);
        }
        step("batiments");
        plan.index();
        step("index");
        plan
    }
}

impl Plan {
    fn build(city: &City, gen: &Generator) -> Plan {
        PlanBuilder { city, rng: Rng(hash_seed(city.seed, 0x9A4, 1)), roads: Vec::new(), finals: Vec::new(), fixed: Vec::new(), ends: Vec::new(), links: Vec::new(), strict: false }.build(gen)
    }

    fn push(&mut self, foot: [DVec2; 4], kind: Kind, floors: u16, rng: &mut Rng) {
        self.buildings.push(Building { foot, kind, floors: floors.max(1), seed: rng.next_u64() as u32, row: false });
    }

    /// Batiments en bordure d'ilot, mitoyens, autour d'une cour.
    fn perimeter(&mut self, poly: &[DVec2], depth: f64, unit: (f32, f32), rng: &mut Rng, mut kind_floors: impl FnMut(&mut Rng) -> (Kind, u16)) {
        let n = poly.len();
        let offset: Vec<(DVec2, DVec2)> = (0..n)
            .map(|i| {
                let d = (poly[(i + 1) % n] - poly[i]).normalize_or_zero();
                (poly[i] + d.perp() * depth, d)
            })
            .collect();
        // Sommet interieur : croisement des cotes decales voisins.
        let inner: Vec<Option<DVec2>> = (0..n)
            .map(|i| {
                let (p0, d0) = offset[(i + n - 1) % n];
                let (p1, d1) = offset[i];
                let den = d0.perp_dot(d1);
                (den.abs() > 1e-6).then(|| p0 + d0 * ((p1 - p0).perp_dot(d1) / den))
            })
            .collect();
        let valid = inner.iter().all(|v| v.is_some()) && {
            let inner: Vec<DVec2> = inner.iter().map(|v| v.unwrap()).collect();
            (0..n).all(|i| {
                let (a, b) = (inner[i], inner[(i + 1) % n]);
                (b - a).dot(poly[(i + 1) % n] - poly[i]) > 1.0
            }) && area(&inner) > 60.0
        };
        if !valid {
            let (_, axis, _, _) = obb(poly);
            if let Some(rect) = fit_rect(poly, axis, 0.0) {
                let (kind, floors) = kind_floors(rng);
                self.push(rect, kind, floors, rng);
            }
            return;
        }
        let inner: Vec<DVec2> = inner.into_iter().map(|v| v.unwrap()).collect();
        for i in 0..n {
            let (o0, o1, n0, n1) = (poly[i], poly[(i + 1) % n], inner[i], inner[(i + 1) % n]);
            let len = (o1 - o0).length();
            let count = (len / rng.range(unit.0, unit.1) as f64).round().max(1.0) as usize;
            for j in 0..count {
                let (t0, t1) = (j as f64 / count as f64, (j + 1) as f64 / count as f64);
                let foot = [o0.lerp(o1, t0), o0.lerp(o1, t1), n0.lerp(n1, t1), n0.lerp(n1, t0)];
                let (kind, floors) = kind_floors(rng);
                self.push(foot, kind, floors, rng);
                self.buildings.last_mut().unwrap().row = true;
            }
        }
    }

    /// Le cote d'un lot qui donne sur la rue : celui qui longe le bord de l'ilot, le plus long.
    fn front_edge(lot: &[DVec2], block: &[DVec2]) -> Option<usize> {
        let on_edge = |q: DVec2| {
            (0..block.len()).any(|i| {
                let (a, b) = (block[i], block[(i + 1) % block.len()]);
                let d = (b - a).normalize_or_zero();
                d.perp_dot(q - a).abs() < 0.6 && (q - a).dot(d) > -0.5 && (q - a).dot(d) < (b - a).length() + 0.5
            })
        };
        (0..lot.len())
            .filter(|&i| on_edge(lot[i].lerp(lot[(i + 1) % lot.len()], 0.5)))
            .max_by(|&a, &b| {
                let la = (lot[(a + 1) % lot.len()] - lot[a]).length();
                let lb = (lot[(b + 1) % lot.len()] - lot[b]).length();
                la.partial_cmp(&lb).unwrap()
            })
    }

    /// Maisons et jardins : lots dos a dos, le long des rues.
    fn houses(&mut self, block: &[DVec2], rng: &mut Rng, fence_style: u8) {
        let (c, ax, _, hy) = obb(block);
        let halves = if hy * 2.0 > 46.0 {
            let (l, r, _) = split(&plain(block), c, ax, NONE);
            vec![points(&l), points(&r)]
        } else {
            vec![block.to_vec()]
        };
        for half in halves {
            let (lo, hi) = half.iter().fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v.dot(ax)), hi.max(v.dot(ax))));
            let mut rest = half;
            let mut x = lo;
            loop {
                let frontage = rng.range(15.0, 23.0) as f64;
                let lot = if hi - x > frontage * 1.6 {
                    x += frontage;
                    let at = ax * x;
                    let (l, r, _) = split(&plain(&rest), at, ax.perp(), NONE);
                    rest = points(&r);
                    points(&l)
                } else {
                    std::mem::take(&mut rest)
                };
                if lot.len() >= 3 {
                    self.house(&lot, block, rng, fence_style);
                }
                if rest.len() < 3 {
                    break;
                }
            }
        }
    }

    fn house(&mut self, lot: &[DVec2], block: &[DVec2], rng: &mut Rng, fence_style: u8) {
        let Some(fi) = Self::front_edge(lot, block) else { return };
        let n = lot.len();
        let (a, b) = (lot[fi], lot[(fi + 1) % n]);
        let ex = (b - a).normalize_or_zero();
        let ey = ex.perp();
        let frontage = (b - a).length();
        let depth = lot.iter().map(|v| (*v - a).dot(ey)).fold(0.0, f64::max);
        let setback = rng.range(3.0, 9.0) as f64;
        // Du pavillon a la grande villa.
        let size = rng.f32();
        let grow = if size > 0.85 { 1.35 } else if size < 0.2 { 0.8 } else { 1.0 };
        let w = (rng.range(8.0, 13.0) as f64 * grow).min(frontage - 3.5);
        let d = (rng.range(7.5, 11.5) as f64 * grow).min(depth - setback - 5.0);
        if w < 6.0 || d < 6.0 {
            return;
        }
        let x = 1.5 + (frontage - w - 3.0) * rng.range(0.15, 0.6) as f64;
        let rect = |x0: f64, y0: f64, x1: f64, y1: f64| [a + ex * x0 + ey * y0, a + ex * x1 + ey * y0, a + ex * x1 + ey * y1, a + ex * x0 + ey * y1];
        let fits = |r: &[DVec2; 4]| r.iter().all(|q| inside(lot, *q));
        let home = rect(x, setback, x + w, setback + d);
        if !fits(&home) {
            return;
        }
        // Parfois un petit immeuble de rapport au milieu des maisons.
        if rng.f32() < 0.05 && w > 10.0 && d > 9.0 {
            self.push(home, Kind::Apartments, 3 + (rng.f32() * 2.0) as u16, rng);
            return;
        }
        let floors = if rng.f32() < 0.55 { 2 } else { 1 };
        self.push(home, Kind::House, floors, rng);
        // Une aile en retour vers le jardin : maison en L, toits croises.
        if rng.f32() < 0.45 {
            let ww = (w * rng.range(0.38, 0.55) as f64).max(4.5);
            let wd = rng.range(4.5, 8.0) as f64;
            let left = rng.f32() < 0.5;
            let x0 = if left { x } else { x + w - ww };
            let wing = rect(x0, setback + d - 0.6, x0 + ww, setback + d + wd);
            if fits(&wing) {
                self.push(wing, Kind::House, 1, rng);
            }
        }
        // Un garage a cote, une piscine derriere.
        if frontage - x - w > 5.0 {
            let garage = rect(x + w + 0.4, setback + 1.0, x + w + 4.0, setback + 7.0);
            if fits(&garage) {
                self.push(garage, Kind::Garage, 1, rng);
            }
        }
        if depth - setback - d > 15.0 && rng.f32() < 0.3 {
            let px = x + w * 0.5 + rng.range(-2.0, 2.0) as f64;
            let pool = rect(px - 2.2, setback + d + 4.0, px + 2.2, setback + d + 12.0);
            if fits(&pool) {
                self.push(pool, Kind::Pool, 1, rng);
            }
        }
        for i in 0..n {
            if i != fi {
                self.fences.push(Fence { a: lot[i], b: lot[(i + 1) % n], style: fence_style });
            }
        }
    }

    fn fill_block(&mut self, city: &City, bi: usize) {
        let mut rng = Rng(hash_seed(city.seed ^ 0xF111, bi as i64, 7));
        let poly = self.blocks[bi].poly.clone();
        let zone = self.blocks[bi].zone;
        let c = centroid(&poly);
        let r = city.extent(c);
        let (_, axis, hx, hy) = obb(&poly);
        let square = |c: DVec2, half: f64| [c + DVec2::new(-half, -half), c + DVec2::new(half, -half), c + DVec2::new(half, half), c + DVec2::new(-half, half)];
        // Caractere de la ville : basse et etalee, ou haute et serree.
        let tall = 0.7 + 0.7 * hash01(city.seed, 0, 99);
        match zone {
            Zone::Park => {
                if hx * hy > 3000.0 && rng.f32() < 0.6 {
                    self.push(square(c, 7.0), Kind::Fountain, 1, &mut rng);
                }
            }
            Zone::Plaza => {
                // Une seule des pointes de la place en etoile porte le monument.
                let center = inside(&poly, DVec2::from_angle(city.angle + 0.02) * 5.0);
                if city.style == Style::Medieval || city.size < 0.3 {
                    if let Some(rect) = fit_rect(&poly, axis, 6.0) {
                        // La nef suit le grand axe ; le clocher se dresse sur la facade.
                        let (mid, half) = ((rect[0] + rect[2]) * 0.5, (rect[1] - rect[0]).length().min(48.0) * 0.5);
                        let depth = ((rect[3] - rect[0]).length() * 0.5).min(11.0);
                        let ex = (rect[1] - rect[0]).normalize();
                        let ey = ex.perp();
                        let foot = [mid - ex * half - ey * depth, mid + ex * half - ey * depth, mid + ex * half + ey * depth, mid - ex * half + ey * depth];
                        self.push(foot, Kind::Church, 1, &mut rng);
                    }
                } else if city.style == Style::Radial && center {
                    self.push(square(DVec2::ZERO, 9.0), Kind::Monument, 1, &mut rng);
                } else if hx * hy > 800.0 {
                    self.push(square(c, 6.0), Kind::Fountain, 1, &mut rng);
                }
            }
            Zone::Tower => {
                let count = if hx * hy > 2400.0 { 1 + (rng.f32() * 2.4) as usize } else { 1 };
                let mut lots = vec![poly.clone()];
                for k in 1..count {
                    let last = lots.pop().unwrap();
                    let (lc, la, lhx, _) = obb(&last);
                    let t = 1.0 / (count - k + 1) as f64;
                    let (l, rr, _) = split(&plain(&last), lc - la * lhx * (1.0 - 2.0 * t), la.perp(), NONE);
                    lots.push(points(&l));
                    lots.push(points(&rr));
                }
                let core = city.downtown();
                let lhx_ok = |r: &[DVec2; 4]| (r[1] - r[0]).length() > 44.0;
                for lot in lots {
                    if lot.len() < 3 {
                        continue;
                    }
                    let (_, la, _, _) = obb(&lot);
                    let Some(rect) = fit_rect(&lot, la, 2.0) else { continue };
                    let rb = city.business_distance(c);
                    let closeness = (1.0 - rb / core.max(0.01)).clamp(0.0, 1.0);
                    if rb < core * 0.75 || rng.f32() < 0.3 {
                        let floors = (22.0 + 78.0 * closeness.powf(1.3) * rng.range(0.5, 1.15) as f64) * (0.6 + 0.4 * city.size as f64) * tall;
                        let floors = floors.clamp(16.0, 120.0);
                        let shrink = |r: &[DVec2; 4], k: f64| {
                            let c = (r[0] + r[2]) * 0.5;
                            r.map(|q| c + (q - c) * k)
                        };
                        let roll = rng.f32();
                        if roll < 0.35 {
                            // Socle de bureaux sur tout le lot, la tour en retrait au-dessus.
                            self.push(rect, Kind::Office, rng.range(3.0, 6.0) as u16, &mut rng);
                            self.push(shrink(&rect, 0.68), Kind::Tower, floors as u16, &mut rng);
                        } else if roll < 0.6 {
                            // Gradins : chaque etage de la tour plus etroit que le precedent (art deco, zoning de 1916).
                            self.push(rect, Kind::Office, (floors * rng.range(0.25, 0.4) as f64) as u16, &mut rng);
                            self.push(shrink(&rect, 0.78), Kind::Office, (floors * rng.range(0.5, 0.65) as f64) as u16, &mut rng);
                            self.push(shrink(&rect, 0.56), Kind::Tower, floors as u16, &mut rng);
                        } else if roll < 0.75 && lhx_ok(&rect) {
                            // Deux tours jumelles sur le meme lot.
                            let ax2 = (rect[1] - rect[0]) * 0.5;
                            let half = [rect[0], rect[0] + ax2, rect[3] + ax2, rect[3]];
                            let other = half.map(|q| q + ax2);
                            let f2 = floors * rng.range(0.7, 1.0) as f64;
                            self.push(shrink(&half, 0.82), Kind::Tower, floors as u16, &mut rng);
                            self.push(shrink(&other, 0.82), Kind::Tower, f2.max(16.0) as u16, &mut rng);
                        } else {
                            self.push(rect, Kind::Tower, floors as u16, &mut rng);
                        }
                    } else {
                        self.push(rect, Kind::Office, (rng.range(6.0, 22.0) * tall as f32) as u16, &mut rng);
                    }
                }
            }
            Zone::Dense => {
                let classic = matches!(city.style, Style::Radial | Style::Eixample);
                let depth = if city.style == Style::Eixample { rng.range(17.0, 22.0) } else { rng.range(12.0, 16.0) } as f64;
                let base = ((5.0 + (1.0 - r) * 5.0) * tall) as f32;
                let office = city.business_distance(c) < city.downtown() * 1.5;
                self.perimeter(&poly, depth, if classic { (13.0, 20.0) } else { (16.0, 30.0) }, &mut rng, |rng| {
                    if classic && rng.f32() < 0.75 {
                        (Kind::Haussmann, if city.style == Style::Eixample { 6 } else { 6 + (rng.f32() < 0.3) as u16 })
                    } else if office && rng.f32() < 0.35 {
                        (Kind::Office, rng.range(7.0, 14.0) as u16)
                    } else if city.style == Style::Medieval && rng.f32() < 0.5 {
                        (Kind::Townhouse, rng.range(3.0, 5.99) as u16)
                    } else if rng.f32() < 0.12 {
                        // Un immeuble qui depasse : la ligne des toits n'est jamais droite.
                        (Kind::Apartments, (base + rng.range(3.0, 7.0)) as u16)
                    } else {
                        (Kind::Apartments, (base + rng.range(-2.5, 2.5)).max(3.0) as u16)
                    }
                });
            }
            Zone::Old => {
                let depth = rng.range(9.0, 12.5) as f64;
                self.perimeter(&poly, depth, (6.0, 10.5), &mut rng, |rng| (Kind::Townhouse, rng.range(2.0, 4.99) as u16 + 1));
            }
            Zone::Houses => {
                let style = (hash01(city.seed, bi as i64, 3) * 3.0) as u8;
                self.houses(&poly, &mut rng, style);
            }
            Zone::Industry => {
                let mut lots = vec![poly.clone()];
                while let Some(i) = lots.iter().position(|l| obb(l).2 > 75.0) {
                    let l = lots.swap_remove(i);
                    let (lc, la, _, _) = obb(&l);
                    let (a, b, _) = split(&plain(&l), lc, la.perp(), NONE);
                    lots.push(points(&a));
                    lots.push(points(&b));
                }
                for lot in lots {
                    if lot.len() < 3 {
                        continue;
                    }
                    let (lc, la, lhx, lhy) = obb(&lot);
                    if rng.f32() < 0.25 && lhx * lhy > 900.0 {
                        // Cuves rondes et cheminee.
                        let tanks = 1 + (rng.f32() * 3.0) as usize;
                        for k in 0..tanks {
                            let at = lc + la * ((k as f64 - (tanks - 1) as f64 * 0.5) * 19.0);
                            let foot = square(at, rng.range(5.0, 8.0) as f64);
                            if foot.iter().all(|q| inside(&lot, *q)) {
                                self.push(foot, Kind::Tank, 1, &mut rng);
                            }
                        }
                    } else if let Some(rect) = fit_rect(&lot, la, 6.0) {
                        self.push(rect, Kind::Warehouse, 1, &mut rng);
                        if rng.f32() < 0.12 {
                            let at = rect[3].lerp(rect[2], 0.15) + la.perp() * 4.0;
                            let foot = square(at, 2.2);
                            if foot.iter().all(|q| inside(&lot, *q)) {
                                self.push(foot, Kind::Chimney, 1, &mut rng);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Range chaque element dans les mailles qu'il touche.
    fn index(&mut self) {
        let mut grid = Grid::new(self.grid.half);
        for (i, r) in self.roads.iter().enumerate() {
            let half = DVec2::splat(r.width as f64 * 0.5 + 1.0);
            let (lo, hi) = bounds(&[r.a, r.b]);
            for c in grid.cells(lo - half, hi + half).collect::<Vec<_>>() {
                grid.roads[c].push(i as u32);
            }
        }
        for (i, b) in self.blocks.iter().enumerate() {
            let (lo, hi) = bounds(&b.poly);
            for c in grid.cells(lo, hi).collect::<Vec<_>>() {
                grid.blocks[c].push(i as u32);
            }
        }
        for (i, b) in self.buildings.iter().enumerate() {
            let (lo, hi) = bounds(&b.foot);
            for c in grid.cells(lo, hi).collect::<Vec<_>>() {
                grid.buildings[c].push(i as u32);
            }
        }
        self.grid = grid;
    }

    /// Indices des batiments dont le centre tombe dans un rectangle.
    pub fn buildings_in(&self, min: DVec2, max: DVec2) -> Vec<u32> {
        let mut out = Vec::new();
        for c in self.grid.cells(min, max) {
            for &i in &self.grid.buildings[c] {
                let p = self.buildings[i as usize].center();
                if p.x >= min.x && p.y >= min.y && p.x < max.x && p.y < max.y {
                    out.push(i);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Indices des rues qui touchent un rectangle.
    pub fn roads_in(&self, min: DVec2, max: DVec2) -> Vec<u32> {
        let mut out: Vec<u32> = self.grid.cells(min, max).flat_map(|c| self.grid.roads[c].iter().copied()).collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn fences_in(&self, min: DVec2, max: DVec2) -> impl Iterator<Item = &Fence> {
        self.fences.iter().filter(move |f| {
            let m = (f.a + f.b) * 0.5;
            m.x >= min.x && m.y >= min.y && m.x < max.x && m.y < max.y
        })
    }

    /// Ce qui couvre le sol en un point du plan.
    pub fn ground(&self, p: DVec2) -> Ground {
        let Some(cell) = self.grid.cell(p) else { return Ground::Wild };
        // La chaussee l'emporte sur le trottoir d'une autre rue ; un trottoir s'interrompt aux debouches.
        let mut walk = false;
        for &ri in &self.grid.roads[cell] {
            let r = &self.roads[ri as usize];
            let (d, len) = (r.dir(), r.length());
            let s = (p - r.a).dot(d);
            let x = d.perp().dot(p - r.a);
            let off = x.abs();
            if off >= r.width as f64 * 0.5 || s < 0.0 || s > len {
                continue;
            }
            let carriage = r.carriage();
            if off < carriage && s >= r.cut(0, x) && s <= len - r.cut(1, x) {
                return Ground::Road;
            }
            if off >= carriage && s >= r.trim_walk[0] as f64 && s <= len - r.trim_walk[1] as f64 && !r.in_gap_side(s as f32, x.signum() as f32) {
                walk = true;
            }
        }
        if walk {
            return Ground::Walk;
        }
        for &bi in &self.grid.blocks[cell] {
            let b = &self.blocks[bi as usize];
            if !inside(&b.poly, p) {
                continue;
            }
            for &i in &self.grid.buildings[cell] {
                let f = &self.buildings[i as usize];
                if inside(&f.foot, p) {
                    return if f.kind == Kind::Pool { Ground::Paved } else { Ground::Building };
                }
            }
            return match b.zone {
                Zone::Park => Ground::Park,
                Zone::Plaza | Zone::Tower => Ground::Paved,
                Zone::Houses => Ground::Garden,
                Zone::Dense if bi % 2 == 0 => Ground::Garden,
                _ => Ground::Yard,
            };
        }
        Ground::Wild
    }

    /// Arbres d'alignement des boulevards, dans un rectangle.
    pub fn street_trees(&self, min: DVec2, max: DVec2) -> Vec<DVec2> {
        let mut out = Vec::new();
        for ri in self.roads_in(min, max) {
            let r = &self.roads[ri as usize];
            if !r.trees {
                continue;
            }
            let (d, len) = (r.dir(), r.length());
            let off = (r.width * 0.5 - r.walk * 0.45) as f64;
            let step = 11.0;
            let mut s = r.trim_walk[0] as f64 + 6.0;
            while s < len - r.trim_walk[1] as f64 - 6.0 {
                if !r.in_gap(s as f32) && !r.gaps.iter().any(|&(at, half, _)| (s as f32 - at).abs() < half + 5.0) {
                    for side in [-1.0, 1.0] {
                        let p = r.a + d * s + d.perp() * off * side;
                        if p.x >= min.x && p.y >= min.y && p.x < max.x && p.y < max.y {
                            out.push(p);
                        }
                    }
                }
                s += step;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------
// Toutes les villes d'une planete

pub struct Cities {
    pub list: Vec<City>,
    buckets: FastMap<(u8, i32, i32), Vec<u32>>,
    /// Les autoroutes qui les relient.
    pub highways: Highways,
    /// Un circuit de course par ville, quand la place le permet.
    pub circuits: super::circuit::Circuits,
}

fn bucket(dir: DVec3) -> (u8, i32, i32) {
    let a = dir.abs();
    let (face, u, v) = if a.x >= a.y && a.x >= a.z {
        (if dir.x > 0.0 { 0 } else { 1 }, dir.y / a.x, dir.z / a.x)
    } else if a.y >= a.z {
        (if dir.y > 0.0 { 2 } else { 3 }, dir.x / a.y, dir.z / a.y)
    } else {
        (if dir.z > 0.0 { 4 } else { 5 }, dir.x / a.z, dir.y / a.z)
    };
    let q = |t: f64| ((t + 1.0) * 0.5 * BUCKETS).floor().clamp(0.0, BUCKETS - 1.0) as i32;
    (face, q(u), q(v))
}

/// Couleur d'un sol de ville, en lineaire.
fn ground_paint(g: Ground) -> Option<([f32; 3], f32)> {
    match g {
        Ground::Wild => None,
        Ground::Road => Some(([0.03, 0.03, 0.033], 1.0)),
        Ground::Walk => Some(([0.2, 0.19, 0.175], 1.0)),
        Ground::Building => Some(([0.07, 0.065, 0.06], 1.0)),
        Ground::Paved => Some(([0.25, 0.23, 0.2], 1.0)),
        Ground::Yard => Some(([0.15, 0.14, 0.125], 1.0)),
        Ground::Garden => Some(([0.06, 0.12, 0.035], 0.55)),
        Ground::Park => Some(([0.05, 0.11, 0.03], 0.5)),
    }
}

/// Teinte d'un quartier vu de loin : toits, rues et jardins melanges.
fn zone_paint(z: Zone) -> [f32; 3] {
    match z {
        Zone::Tower => [0.09, 0.09, 0.095],
        Zone::Dense => [0.14, 0.125, 0.11],
        Zone::Old => [0.19, 0.11, 0.075],
        Zone::Houses => [0.1, 0.1, 0.065],
        Zone::Industry => [0.16, 0.16, 0.155],
        Zone::Park => [0.05, 0.11, 0.03],
        Zone::Plaza => [0.25, 0.23, 0.2],
    }
}

impl Cities {
    pub fn empty() -> Self {
        Self { list: Vec::new(), buckets: FastMap::default(), highways: Highways::empty(), circuits: super::circuit::Circuits::empty() }
    }

    /// Choisit les sites : des terres basses et peu accidentees, au climat doux, pres d'une cote ou
    /// d'une riviere ; les plus beaux deviennent des metropoles.
    pub fn place(gen: &Generator) -> Self {
        let seed = gen.params.seed;
        let wanted = gen.params.cities as usize;
        if wanted == 0 {
            return Self::empty();
        }
        // Assez de sites candidats pour en garder le nombre voulu, espaces.
        let n = (26_000).max(wanted * 220).min(140_000);
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        // Assez fin pour voir les vallees larges de quelques kilometres.
        let coarse = Detail::for_spacing(150.0);
        let probe = |dir: DVec3, k: usize, dist: f64| {
            let east = if dir.y.abs() > 0.999 { DVec3::X } else { DVec3::Y.cross(dir).normalize() };
            let north = dir.cross(east);
            let a = k as f64 / 12.0 * std::f64::consts::TAU;
            (dir + (east * a.cos() + north * a.sin()) * (dist / planet_radius())).normalize()
        };
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let per = n / threads + 1;
        let radius = planet_radius();
        let profile = std::env::var("TERRA_PROFILE").is_ok();
        let mut clock = std::time::Instant::now();
        let mut step = |name: &str| {
            if profile {
                println!("    villes : {name} {:.2} s", clock.elapsed().as_secs_f64());
            }
            clock = std::time::Instant::now();
        };
        let mut candidates: Vec<(f64, DVec3, f64)> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    s.spawn(move || {
                        crate::world::generator::worker_thread(radius);
                        let mut out = Vec::new();
                        for i in t * per..((t + 1) * per).min(n) {
                            let y = 1.0 - (i as f64 + 0.5) / n as f64 * 2.0;
                            let r = (1.0 - y * y).sqrt();
                            let th = golden * i as f64;
                            let dir = DVec3::new(th.cos() * r, y, th.sin() * r);
                            let base = gen.water.elevation.sample(dir);
                            if !(6.0..1100.0).contains(&base) || gen.orogeny.sample_linear(dir) > 0.3 {
                                continue;
                            }
                            let c = gen.climate(dir, base);
                            if !(0.24..0.95).contains(&c.temperature) || c.moisture < 0.12 {
                                continue;
                            }
                            let w = gen.provinces.at(dir);
                            let rough = w.karst + w.dunes + w.glacial + w.badlands * 0.7 + w.mesa * 0.6 + w.inselberg * 0.3;
                            if rough > 0.8 {
                                continue;
                            }
                            let center = gen.surface_detail(dir, &coarse);
                            if center.height < center.water_level + 2.0 {
                                continue;
                            }
                            let (mut lo, mut hi, mut coast, mut river) = (center.height, center.height, 0.0f64, center.valley as f64);
                            // Deux couronnes de sondes : une ville veut un site plat sur toute son etendue.
                            for k in 0..12 {
                                let s = gen.surface_detail(probe(dir, k, if k % 2 == 0 { 1500.0 } else { 3500.0 }), &coarse);
                                lo = lo.min(s.height);
                                hi = hi.max(s.height);
                                river = river.max(s.valley as f64 * 0.8);
                                if k % 2 == 0 && gen.water.elevation.sample(probe(dir, k, 14_000.0)) < 0.0 {
                                    coast = 1.0;
                                }
                            }
                            let range = (hi - lo) as f64;
                            if range > 110.0 {
                                continue;
                            }
                            let mild = 1.0 - ((c.temperature as f64 - 0.62).abs() * 2.0).min(1.0);
                            let score = 1.0 + coast * 1.1 + river.min(1.0) * 1.4 + mild * 0.6 - rough as f64 * 1.5 - range / 70.0
                                + hash01(seed ^ 0xC17, i as i64, 0) * 0.8;
                            out.push((score, dir, coast));
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        step("sites");

        // Ecart entre deux villes : 140 km pour 140 villes sur une planete comme la Terre ; plus serre
        // s'il en faut davantage ou si la planete est petite. On le resserre tant qu'il manque des villes.
        let earth = planet_radius() / super::generator::EARTH_RADIUS;
        let mut gap = 140_000.0 * (140.0 / wanted as f64).sqrt() * earth;
        let mut chosen: Vec<(DVec3, f32, f64)> = Vec::new();
        for _ in 0..12 {
            chosen.clear();
            let metro_gap = gap * 4.3;
            for &(_, dir, coast) in &candidates {
                if chosen.len() >= wanted {
                    break;
                }
                let k = chosen.len() as f64 / wanted as f64;
                // Quelques metropoles, des villes moyennes, beaucoup de petites villes.
                let size = if k < 0.07 { 1.0 } else if k < 0.28 { 0.7 } else if k < 0.85 { 0.4 } else { 0.18 };
                let mine = if size > 0.9 { metro_gap } else { gap };
                if chosen.iter().any(|c| c.0.dot(dir) > (mine.max(if c.1 > 0.9 { metro_gap } else { 0.0 }) / planet_radius()).cos()) {
                    continue;
                }
                chosen.push((dir, size, coast));
            }
            if chosen.len() >= wanted || gap < 12_000.0 {
                break;
            }
            gap *= 0.75;
        }

        let mut list: Vec<City> = Vec::new();
        for (dir, size, coast) in chosen {
            let k = list.len();
            let id = hash_seed(seed, 0xC1, k as i64);
            let roll = hash01(id, 5, 0);
            let style = match size {
                s if s > 0.9 => if roll < 0.45 { Style::Grid } else if roll < 0.75 { Style::Radial } else { Style::Eixample },
                s if s > 0.5 => [Style::Grid, Style::Eixample, Style::Radial, Style::Medieval][(roll * 4.0) as usize % 4],
                s if s > 0.3 => if roll < 0.5 { Style::Medieval } else if roll < 0.8 { Style::Grid } else { Style::Radial },
                _ => Style::Medieval,
            };
            // Les rues d'une ville de cote suivent le rivage.
            let angle = if coast > 0.0 {
                let east = if dir.y.abs() > 0.999 { DVec3::X } else { DVec3::Y.cross(dir).normalize() };
                let north = dir.cross(east);
                let e = 9000.0 / planet_radius();
                let h = |d: DVec3| gen.water.elevation.sample((dir + d * e).normalize()) as f64;
                let g = DVec2::new(h(east) - h(-east), h(north) - h(-north));
                g.y.atan2(g.x)
            } else {
                hash01(id, 6, 0) * std::f64::consts::PI
            };
            let mut city = City::new(dir, size, style, angle, id);
            // Deux villes ne portent pas le meme nom.
            let mut attempt = 0;
            while list.iter().any(|c| c.name == city.name) {
                attempt += 1;
                city.name = city_name(hash_seed(id, 0x4A3E, attempt), style);
            }
            list.push(city);
        }

        step("choix");
        let highways = Highways::build(gen, &mut list);
        step("autoroutes");
        let circuits = super::circuit::Circuits::build(gen, &mut list, &highways);
        step("circuits");
        let mut buckets: FastMap<(u8, i32, i32), Vec<u32>> = FastMap::default();
        for (i, c) in list.iter().enumerate() {
            let mut keys = vec![bucket(c.dir)];
            let (e, n) = (c.east, c.north);
            for k in 0..16 {
                let a = k as f64 / 16.0 * std::f64::consts::TAU;
                let d = (c.dir + (e * a.cos() + n * a.sin()) * (c.radius * 1.6 / planet_radius())).normalize();
                keys.push(bucket(d));
            }
            keys.sort_unstable();
            keys.dedup();
            for key in keys {
                buckets.entry(key).or_default().push(i as u32);
            }
        }
        Self { list, buckets, highways, circuits }
    }

    /// Villes dont l'influence peut atteindre ce point.
    pub fn near(&self, dir: DVec3) -> impl Iterator<Item = &City> {
        self.buckets
            .get(&bucket(dir))
            .into_iter()
            .flatten()
            .map(|&i| &self.list[i as usize])
            .filter(move |c| c.dir.dot(dir) > (c.radius * 1.4 / planet_radius()).cos())
    }

    /// Villes qui touchent un disque (centre, rayon en metres).
    pub fn touching(&self, dir: DVec3, radius: f64) -> Vec<&City> {
        self.list.iter().filter(|c| c.dir.dot(dir) > ((c.radius * 1.3 + radius) / planet_radius()).min(3.0).cos()).collect()
    }

    /// Part de relief aplani par une ville, de 0 a 1.
    pub fn flatten(&self, dir: DVec3) -> f32 {
        self.near(dir)
            .map(|c| 1.0 - smoothstep(0.92, FLAT_EDGE as f32, c.extent(c.local(dir)) as f32))
            .fold(0.0, f32::max)
    }

    /// Le sol en un point. Calcule le plan de la ville au besoin.
    pub fn ground(&self, dir: DVec3, gen: &Generator) -> Ground {
        for c in self.near(dir) {
            let p = c.local(dir);
            if c.extent(p) < 1.05 {
                let g = c.plan(gen).ground(p);
                if g != Ground::Wild {
                    return g;
                }
            }
        }
        Ground::Wild
    }

    /// Couleur du sol de ville et sa part. De pres, rue par rue ; de loin, la teinte des quartiers.
    pub fn paint(&self, dir: DVec3, spacing: f64, gen: &Generator) -> Option<([f32; 3], f32)> {
        for c in self.near(dir) {
            let p = c.local(dir);
            let e = c.extent(p);
            if e > 1.02 {
                continue;
            }
            if spacing < 30.0 {
                if let Some(paint) = ground_paint(c.plan(gen).ground(p)) {
                    return Some(paint);
                }
                continue;
            }
            let edge = (1.0 - smoothstep(0.85, 1.02, e as f32)) * c.urban(p);
            if edge <= 0.0 {
                continue;
            }
            let zone = c.plan_if_ready().and_then(|plan| plan.block_zone(p)).unwrap_or_else(|| c.zone_at(p));
            return Some((zone_paint(zone), 0.85 * edge));
        }
        None
    }

    /// Repousse un cercle vertical (position, rayon) hors des batiments dont il est plus bas que le toit.
    /// `ground` : altitude du sol sous lui, celle des pieds des batiments voisins dans une ville aplanie.
    pub fn push_out(&self, pos: &mut DVec3, radius: f64, ground: f64) {
        let up = pos.normalize_or(DVec3::Y);
        let alt = pos.length() - planet_radius() - ground;
        for c in self.near(up) {
            let Some(plan) = c.plan_if_ready() else { continue };
            let p = c.local(up);
            let reach = DVec2::splat(radius + 80.0);
            let mut push = DVec2::ZERO;
            for bi in plan.buildings_in(p - reach, p + reach) {
                let b = &plan.buildings[bi as usize];
                if matches!(b.kind, Kind::Pool | Kind::Fountain) || alt > b.height() as f64 + 1.0 {
                    continue;
                }
                // Distance signee au contour convexe : le cote le plus eloigne vers l'exterieur.
                let q = p + push;
                let (mut best, mut normal) = (f64::MIN, DVec2::ZERO);
                for i in 0..4 {
                    let (a, e) = (b.foot[i], b.foot[(i + 1) % 4] - b.foot[i]);
                    let out = DVec2::new(e.y, -e.x).normalize_or_zero();
                    let d = (q - a).dot(out);
                    if d > best {
                        (best, normal) = (d, out);
                    }
                }
                if best < radius {
                    push += normal * (radius - best);
                }
            }
            if push != DVec2::ZERO {
                let r = pos.length();
                *pos = c.dir_of(p + push) * r;
                return;
            }
        }
    }

    /// Un point de rue pres d'une direction : le milieu de la chaussee la plus proche.
    pub fn street_near(&self, dir: DVec3, gen: &Generator) -> Option<DVec3> {
        let c = self.near(dir).find(|c| c.extent(c.local(dir)) < 1.0)?;
        let plan = c.plan(gen);
        let p = c.local(dir);
        let r = DVec2::splat(150.0);
        let best = plan
            .roads_in(p - r, p + r)
            .into_iter()
            .map(|i| &plan.roads[i as usize])
            .filter(|road| road.length() > 30.0)
            .map(|road| {
                let s = (p - road.a).dot(road.dir()).clamp(road.trim[0] as f64 + 5.0, road.length() - road.trim[1] as f64 - 5.0);
                road.a + road.dir() * s
            })
            .min_by(|a, b| (*a - p).length().partial_cmp(&(*b - p).length()).unwrap())?;
        Some(c.dir_of(best))
    }

    /// Altitude du tablier d'un pont de ville sous un point, s'il y en a un.
    pub fn deck(&self, dir: DVec3, gen: &Generator) -> Option<f64> {
        for c in self.near(dir) {
            let Some(plan) = c.plan_if_ready() else { continue };
            let p = c.local(dir);
            let reach = DVec2::splat(30.0);
            for ri in plan.roads_in(p - reach, p + reach) {
                let r = &plan.roads[ri as usize];
                let (d, len) = (r.dir(), r.length());
                let s = (p - r.a).dot(d);
                if s < 0.0 || s > len || d.perp_dot(p - r.a).abs() > r.width as f64 * 0.5 {
                    continue;
                }
                let deck = super::city_mesh::road_deck(plan, c, gen, ri);
                let lift = super::city_mesh::deck_lift(&deck, s);
                if lift > 0.05 {
                    let axis = gen.surface_detail(c.dir_of(r.a + d * s), &Detail::for_spacing(super::city_mesh::DECK_STEP));
                    return Some(axis.height as f64 + lift);
                }
            }
        }
        None
    }

    /// La chaussee la plus proche, a moins de `reach` metres : le milieu de la voie de droite,
    /// la direction de la rue (du cote de `heading`) et la distance.
    pub fn road_near(&self, dir: DVec3, gen: &Generator, heading: DVec3, reach: f64) -> Option<(DVec3, DVec3, f64)> {
        let mut best: Option<(DVec3, DVec3, f64)> = None;
        for c in self.near(dir) {
            let p = c.local(dir);
            if c.extent(p) > 1.2 {
                continue;
            }
            let plan = c.plan(gen);
            let ahead = c.local((dir + heading * (10.0 / planet_radius())).normalize()) - p;
            let span = DVec2::splat(reach);
            for ri in plan.roads_in(p - span, p + span) {
                let r = &plan.roads[ri as usize];
                let len = r.length();
                if len < 15.0 {
                    continue;
                }
                let d = r.dir();
                let s = (p - r.a).dot(d).clamp(r.trim[0] as f64 + 4.0, (len - r.trim[1] as f64 - 4.0).max(r.trim[0] as f64 + 4.0));
                let q = r.a + d * s;
                let dist = (q - p).length();
                if dist > reach || best.is_some_and(|b| b.2 <= dist) {
                    continue;
                }
                let f = if d.dot(ahead) >= 0.0 { d } else { -d };
                let cw = (r.width * 0.5 - r.walk) as f64;
                let lane = q + DVec2::new(f.y, -f.x) * (cw * 0.5);
                let at = c.dir_of(lane);
                let fwd = c.dir_of(lane + f * 5.0) - at;
                best = Some((at, (fwd - at * fwd.dot(at)).normalize(), dist));
            }
        }
        best
    }

    /// La ville la plus proche : indice et distance au centre, en metres.
    pub fn nearest(&self, dir: DVec3) -> Option<(usize, f64)> {
        self.list
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.dir.dot(dir).clamp(-1.0, 1.0).acos() * planet_radius()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }
}

impl Plan {
    /// Quartier de l'ilot qui contient ce point.
    pub fn block_zone(&self, p: DVec2) -> Option<Zone> {
        let cell = self.grid.cell(p)?;
        self.grid.blocks[cell].iter().map(|&i| &self.blocks[i as usize]).find(|b| inside(&b.poly, p)).map(|b| b.zone)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::generator::PlanetParams;

    fn square(half: f64) -> Vec<DVec2> {
        vec![DVec2::new(-half, -half), DVec2::new(half, -half), DVec2::new(half, half), DVec2::new(-half, half)]
    }

    #[test]
    fn split_keeps_the_area_and_marks_the_new_edge() {
        let poly = plain(&square(50.0));
        let (l, r, hits) = split(&poly, DVec2::new(10.0, 0.0), DVec2::Y, 7);
        let (al, ar) = (area(&points(&l)), area(&points(&r)));
        assert!((al + ar - 10_000.0).abs() < 1e-6, "{al} + {ar}");
        assert!(al > 0.0 && ar > 0.0, "les deux moities restent dans le sens trigonometrique");
        assert_eq!(hits.len(), 2);
        assert!(l.iter().any(|v| v.1 == 7) && r.iter().any(|v| v.1 == 7));
    }

    #[test]
    fn inset_moves_each_edge_by_its_own_distance() {
        let out = inset(&square(50.0), &[10.0, 0.0, 5.0, 0.0]);
        let (lo, hi) = bounds(&out);
        assert!((lo.y + 40.0).abs() < 1e-6 && (hi.y - 45.0).abs() < 1e-6, "{lo} {hi}");
        assert!((lo.x + 50.0).abs() < 1e-6 && (hi.x - 50.0).abs() < 1e-6);
    }

    #[test]
    fn fitted_rectangle_stays_inside() {
        let tri = vec![DVec2::new(0.0, 0.0), DVec2::new(80.0, 0.0), DVec2::new(20.0, 60.0)];
        let rect = fit_rect(&tri, DVec2::X, 2.0).expect("un rectangle tient dans le triangle");
        assert!(rect.iter().all(|q| inside(&tri, *q)));
        assert!(area(&rect) > 200.0);
    }

    #[test]
    fn every_style_gives_a_coherent_plan() {
        let gen = Generator::new(crate::world::generator::PlanetParams { seed: 3, ..Default::default() });
        for style in [Style::Grid, Style::Eixample, Style::Radial, Style::Medieval] {
            let dir = gen.cities.list.first().map(|c| c.dir).unwrap_or(DVec3::X);
            let city = City::new(dir, 0.7, style, 0.3, 42);
            let plan = Plan::build(&city, &gen);
            assert!(plan.roads.len() > 30, "{style:?} : {} rues", plan.roads.len());
            assert!(plan.buildings.len() > 200, "{style:?} : {} batiments", plan.buildings.len());
            // Aucun batiment ne mord sur une chaussee.
            for b in plan.buildings.iter().step_by(7) {
                let g = plan.ground(b.center());
                assert!(!matches!(g, Ground::Road), "{style:?} : batiment {:?} sur la rue", b.kind);
            }
            for r in &plan.roads {
                assert!(r.trim[0] <= r.trim_walk[0] && r.length() > 0.0);
            }
        }
    }

    #[test]
    fn street_ends_meet_the_crossing_carriageway_without_a_gap() {
        let gen = Generator::new(crate::world::generator::PlanetParams { seed: 3, ..Default::default() });
        let (mut ends, mut skewed, mut gaps) = (0, 0, Vec::new());
        for style in [Style::Grid, Style::Eixample, Style::Radial, Style::Medieval] {
            let dir = gen.cities.list.first().map(|c| c.dir).unwrap_or(DVec3::X);
            let city = City::new(dir, 0.7, style, 0.3, 42);
            let plan = Plan::build(&city, &gen);
            for r in &plan.roads {
                let (d, len, cw) = (r.dir(), r.length(), r.carriage());
                for k in 0..2 {
                    // Seuls les debouches nets : angle franc (sin > 0,3) et bout pas raccourci.
                    if r.trim[k] < 0.5 || r.skew[k].abs() > 3.0 {
                        continue;
                    }
                    ends += 1;
                    skewed += (r.skew[k].abs() > 0.2) as i32;
                    for x in [-0.9 * cw, 0.0, 0.9 * cw] {
                        let c = r.cut(k, x);
                        // Juste avant le bout (dans la rue croisee) et juste apres (dans celle-ci) : asphalte.
                        for e in [-0.3, 0.3] {
                            let s = if k == 0 { c + e } else { len - c - e };
                            let p = r.a + d * s + d.perp() * x;
                            if plan.ground(p) != Ground::Road {
                                gaps.push((style, p, e));
                            }
                        }
                    }
                }
            }
        }
        if std::env::var("TRACE").is_ok() {
            // Quelques debouches en biais de Pallac (planete 3), pour les verifier a l'oeil.
            let city = gen.cities.list.iter().find(|c| c.name.starts_with("Pallac")).unwrap();
            let plan = city.plan(&gen);
            for r in plan.roads.iter().filter(|r| r.skew[0].abs() > 0.5 && r.skew[0].abs() < 2.0 && r.trim[0] > 5.0).take(4) {
                let p = r.a + r.dir() * r.trim[0] as f64;
                let (lat, lon) = crate::world::generator::dir_to_lat_lon(city.dir_of(p));
                let yaw = (-r.dir().x).atan2(r.dir().y).to_degrees();
                eprintln!("biais {:.2} : {lat:.5},{lon:.5} cap {yaw:.0}", r.skew[0]);
            }
        }
        assert!(ends > 200 && skewed > 20, "{ends} debouches dont {skewed} en biais");
        // Quelques bouts touchent des places ou des rues retirees ensuite : tolerance de 2 %.
        assert!(gaps.len() * 50 < ends * 6, "{} trous sur {ends} debouches : {:?}", gaps.len(), &gaps[..gaps.len().min(6)]);
    }

    #[test]
    fn walkers_are_pushed_out_of_houses_and_spawn_on_streets() {
        let gen = Generator::new(PlanetParams { seed: 3, ..Default::default() });
        let city = &gen.cities.list[0];
        let plan = city.plan(&gen);
        let b = plan.buildings.iter().find(|b| b.kind == Kind::House).expect("une maison");
        let dir = city.dir_of(b.center());
        let ground = gen.height(dir) as f64;
        let mut pos = dir * (planet_radius() + ground + 1.7);
        gen.cities.push_out(&mut pos, 0.4, ground);
        assert!(!inside(&b.foot, city.local(pos.normalize())), "toujours dans la maison");
        // Au-dessus du toit, on passe.
        let mut high = dir * (planet_radius() + ground + 40.0);
        let before = high;
        gen.cities.push_out(&mut high, 0.4, ground);
        assert_eq!(high, before);
        let street = gen.cities.street_near(dir, &gen).expect("une rue");
        assert_eq!(plan.ground(city.local(street)), Ground::Road);
    }

    #[test]
    fn radial_cities_keep_their_towers_out_of_the_old_center() {
        let gen = Generator::new(PlanetParams { seed: 3, ..Default::default() });
        let Some(city) = gen.cities.list.iter().find(|c| c.style == Style::Radial && c.size > 0.9) else { return };
        assert_ne!(city.zone_at(DVec2::new(300.0, 200.0)), Zone::Tower);
        assert_eq!(city.zone_at(city.business_center()), Zone::Tower);
    }

    #[test]
    fn cities_sit_on_dry_gentle_land() {
        let gen = Generator::new(crate::world::generator::PlanetParams { seed: 3, ..Default::default() });
        assert!(gen.cities.list.len() >= 40, "{} villes", gen.cities.list.len());
        for c in gen.cities.list.iter().take(20) {
            let s = gen.surface(c.dir);
            // Le centre peut tomber sur la riviere qui traverse la ville, jamais en mer.
            assert!(s.height > s.water_level || s.fresh, "{} en mer", c.name);
            assert!(gen.cities.near(c.dir).any(|o| std::ptr::eq(o, c)));
        }
    }
}
