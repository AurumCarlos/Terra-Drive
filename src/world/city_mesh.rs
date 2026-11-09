//! Maillages des villes, chunk par chunk.
//!
//! Rues : chaussee marquee, bordures de granit, trottoirs dalles interrompus aux debouches,
//! passages pietons, lampadaires, voitures garees, et ponts au-dessus de l'eau.
//! Batiments : chaque type a sa silhouette et ses details (toits a deux pans ou a croupe,
//! lucarnes, cheminees, balcons filants et toit mansarde haussmannien, acroteres, tours a
//! gradins ou de verre, clochers, cuves...). Les fenetres sont dessinees par le shader.
//!
//! Plus un chunk est grossier, moins il porte de details : les petits batiments disparaissent
//! au loin, les grands restent pour dessiner la silhouette de la ville.

use super::chunk::{NodeId, GRID};
use super::city::{Building, City, Fence, Kind, Road};
use super::generator::{Detail, Generator, planet_radius};
use super::highway::{Highway, Highways, Profile, POINT_STEP};
use super::noise::{hash_seed, Rng};
use bytemuck::{Pod, Zeroable};
use glam::{DVec2, DVec3};
use std::collections::HashSet;
use std::sync::Arc;

/// Materiaux, lus par city.wgsl.
pub mod mat {
    pub const PLAIN: u8 = 0;
    pub const FACADE: u8 = 1;
    pub const ROOF: u8 = 2;
    pub const TILES: u8 = 3;
    pub const METAL: u8 = 4;
    pub const ROAD: u8 = 5;
    pub const WALK: u8 = 6;
    pub const KERB: u8 = 7;
    pub const GLASS: u8 = 8;
    pub const LAMP: u8 = 9;
    pub const RAILING: u8 = 10;
    pub const CROSSING: u8 = 11;
    pub const ZINC: u8 = 12;
    pub const AWNING: u8 = 13;
    pub const WATER: u8 = 14;
    pub const BEACON: u8 = 15;
    pub const HEDGE: u8 = 16;
    pub const PAINT: u8 = 17;
}

/// Styles de fenetres (city.wgsl en garde la geometrie). `SHOP` : vitrines au rez-de-chaussee.
pub mod win {
    pub const RESIDENTIAL: u8 = 0;
    pub const HAUSSMANN: u8 = 1;
    pub const RIBBON: u8 = 2;
    pub const CURTAIN: u8 = 3;
    pub const OLD: u8 = 4;
    pub const INDUSTRIAL: u8 = 5;
    pub const HOUSE: u8 = 6;
    pub const CHURCH: u8 = 7;
    pub const SHOP: u8 = 0x80;
}

/// Largeur d'une travee par style, en metres : identique dans city.wgsl.
fn bay(style: u8) -> f64 {
    match style & 0x7F {
        0 => 3.0,
        1 => 3.2,
        2 => 1.6,
        3 => 1.5,
        4 => 2.6,
        5 => 6.0,
        6 => 3.4,
        7 => 5.0,
        _ => 3.0,
    }
}

/// Pas du profil des ponts le long d'une rue.
pub const DECK_STEP: f64 = 12.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CityVertex {
    /// Relative au centre du chunk.
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    /// Facades : metres le long du mur et hauteur ; rues : ecart a l'axe et abscisse.
    pub uv: [f32; 2],
    /// Couleur encodee en racine carree.
    pub color: [u8; 4],
    /// Materiau, puis ses parametres (style de fenetres, hauteur d'etage en dm, graine).
    pub mat: [u8; 4],
}

pub struct CityMesh {
    pub vertices: Vec<CityVertex>,
    pub indices: Vec<u32>,
    /// Boite englobante relative au centre du chunk.
    pub min: DVec3,
    pub max: DVec3,
    /// Distance au centre de la planete du point le plus haut.
    pub top: f64,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Tier {
    /// Tous les details : lucarnes, balcons, garde-corps, lampadaires, voitures.
    Full,
    /// Volumes, toits, corniches et balcons.
    Mid,
    /// Volumes et toits simples.
    Low,
}

/// Niveau de detail et hauteur minimale des batiments selon le LOD du chunk.
pub fn tier_for(lod: u8) -> Option<(Tier, f32)> {
    Some(match lod {
        0 | 1 => (Tier::Full, 0.0),
        2 | 3 => (Tier::Mid, 0.0),
        4 => (Tier::Low, 9.0),
        5 => (Tier::Low, 15.0),
        6 => (Tier::Low, 30.0),
        7 | 8 => (Tier::Low, 70.0),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------------------------
// Couleurs (albedo lineaire)

// Rangees par familles voisines : une ville tire surtout dans une fenetre de la liste (voir `pick_local`).
const PLASTER: [[f32; 3]; 15] = [
    [0.72, 0.7, 0.64],
    [0.66, 0.64, 0.58],
    [0.64, 0.62, 0.56],
    [0.6, 0.57, 0.5],
    [0.56, 0.5, 0.4],
    [0.6, 0.52, 0.33],
    [0.55, 0.42, 0.27],
    [0.58, 0.44, 0.36],
    [0.5, 0.33, 0.24],
    [0.55, 0.3, 0.2],
    [0.6, 0.45, 0.42],
    [0.62, 0.55, 0.3],
    [0.42, 0.5, 0.53],
    [0.36, 0.45, 0.5],
    [0.48, 0.5, 0.49],
];
const STONE: [[f32; 3]; 3] = [[0.55, 0.5, 0.4], [0.5, 0.47, 0.4], [0.58, 0.53, 0.44]];
const BRICK: [[f32; 3]; 3] = [[0.26, 0.11, 0.07], [0.3, 0.15, 0.09], [0.22, 0.11, 0.08]];
const TILE: [[f32; 3]; 7] = [[0.33, 0.14, 0.07], [0.3, 0.1, 0.05], [0.25, 0.08, 0.04], [0.16, 0.09, 0.06], [0.06, 0.06, 0.07], [0.09, 0.1, 0.12], [0.14, 0.15, 0.16]];
const MODERN: [[f32; 3]; 6] = [
    [0.62, 0.61, 0.58],
    [0.45, 0.45, 0.44],
    [0.52, 0.47, 0.38],
    [0.36, 0.32, 0.28],
    [0.38, 0.17, 0.1],
    [0.58, 0.55, 0.46],
];
const GLASS_TINT: [[f32; 3]; 5] = [[0.05, 0.09, 0.12], [0.04, 0.07, 0.07], [0.1, 0.08, 0.06], [0.07, 0.075, 0.09], [0.03, 0.06, 0.11]];
const METALS: [[f32; 3]; 5] = [[0.5, 0.5, 0.48], [0.22, 0.3, 0.4], [0.45, 0.4, 0.3], [0.25, 0.33, 0.25], [0.55, 0.53, 0.5]];
const CARS: [[f32; 3]; 9] = [
    [0.6, 0.6, 0.6],
    [0.02, 0.02, 0.02],
    [0.3, 0.3, 0.32],
    [0.4, 0.03, 0.02],
    [0.03, 0.08, 0.25],
    [0.65, 0.65, 0.62],
    [0.12, 0.13, 0.14],
    [0.5, 0.35, 0.05],
    [0.05, 0.18, 0.1],
];

fn enc(c: [f32; 3]) -> [u8; 4] {
    let e = |v: f32| (v.clamp(0.0, 1.0).sqrt() * 255.0 + 0.5) as u8;
    [e(c[0]), e(c[1]), e(c[2]), 255]
}

fn scale(c: [f32; 3], k: f32) -> [f32; 3] {
    c.map(|v| v * k)
}

#[derive(Clone, Copy)]
struct Paint {
    color: [u8; 4],
    mat: [u8; 4],
}

fn plain(c: [f32; 3]) -> Paint {
    Paint { color: enc(c), mat: [mat::PLAIN, 0, 0, 0] }
}

fn with(c: [f32; 3], m: u8) -> Paint {
    Paint { color: enc(c), mat: [m, 0, 0, 0] }
}

fn with_param(c: [f32; 3], m: u8, p: u8) -> Paint {
    Paint { color: enc(c), mat: [m, p, 0, 0] }
}

/// Facade : style de fenetres, hauteur d'etage et graine (allumage la nuit, volets, rideaux).
fn facade(c: [f32; 3], style: u8, floor_height: f64, seed: u32) -> Paint {
    // La couleur des volets voyage dans alpha.
    let mut color = enc(c);
    color[3] = (seed >> 8) as u8;
    Paint { color, mat: [mat::FACADE, style, (floor_height * 10.0).round().min(255.0) as u8, seed as u8] }
}

// ---------------------------------------------------------------------------------------------
// Sortie

struct Out {
    v: Vec<CityVertex>,
    i: Vec<u32>,
    origin: DVec3,
    min: DVec3,
    max: DVec3,
    top: f64,
}

impl Out {
    fn new(origin: DVec3) -> Self {
        Self { v: Vec::new(), i: Vec::new(), origin, min: DVec3::splat(f64::MAX), max: DVec3::splat(f64::MIN), top: 0.0 }
    }

    fn vert(&mut self, p: DVec3, n: DVec3, uv: [f64; 2], pt: Paint) -> u32 {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
        self.top = self.top.max((p + self.origin).length());
        self.v.push(CityVertex {
            pos: p.as_vec3().to_array(),
            normal: n.as_vec3().to_array(),
            uv: [uv[0] as f32, uv[1] as f32],
            color: pt.color,
            mat: pt.mat,
        });
        (self.v.len() - 1) as u32
    }

    /// Polygone convexe, en eventail.
    fn poly(&mut self, p: &[DVec3], n: DVec3, uv: &[[f64; 2]], pt: Paint) {
        let base = self.v.len() as u32;
        for k in 0..p.len() {
            self.vert(p[k], n, uv[k], pt);
        }
        for k in 1..p.len() as u32 - 1 {
            self.i.extend_from_slice(&[base, base + k, base + k + 1]);
        }
    }

    /// Quadrilatere aux normales par sommet (surfaces arrondies).
    fn quad_smooth(&mut self, p: [DVec3; 4], n: [DVec3; 4], uv: [[f64; 2]; 4], pt: Paint) {
        let base = self.v.len() as u32;
        for k in 0..4 {
            self.vert(p[k], n[k], uv[k], pt);
        }
        self.i.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Repere plan local d'un objet : la courbure de la planete est negligeable a son echelle.
#[derive(Clone, Copy)]
struct Frame {
    o: DVec3,
    ex: DVec3,
    ey: DVec3,
    up: DVec3,
    c: DVec2,
}

impl Frame {
    fn new(city: &City, c: DVec2, base: f64, origin: DVec3) -> Self {
        let up = city.dir_of(c);
        let ex = (city.dir_of(c + DVec2::X) - city.dir_of(c - DVec2::X)) * (planet_radius() * 0.5);
        let ey = (city.dir_of(c + DVec2::Y) - city.dir_of(c - DVec2::Y)) * (planet_radius() * 0.5);
        Self { o: up * (planet_radius() + base) - origin, ex, ey, up, c }
    }

    fn p(&self, q: DVec2, z: f64) -> DVec3 {
        self.o + self.ex * (q.x - self.c.x) + self.ey * (q.y - self.c.y) + self.up * z
    }

    fn dirv(&self, d: DVec2, z: f64) -> DVec3 {
        (self.ex * d.x + self.ey * d.y + self.up * z).normalize_or_zero()
    }
}

// ---------------------------------------------------------------------------------------------
// Geometrie plane

fn rect(c: DVec2, ax: DVec2, hx: f64, hy: f64) -> [DVec2; 4] {
    let ay = ax.perp();
    [c - ax * hx - ay * hy, c + ax * hx - ay * hy, c + ax * hx + ay * hy, c - ax * hx + ay * hy]
}

/// Retrait uniforme d'un polygone convexe (sens trigonometrique).
fn shrink(p: &[DVec2], d: f64) -> Vec<DVec2> {
    let n = p.len();
    (0..n)
        .map(|i| {
            let (a, b, c) = (p[(i + n - 1) % n], p[i], p[(i + 1) % n]);
            let n0 = (b - a).normalize_or_zero().perp();
            let n1 = (c - b).normalize_or_zero().perp();
            let bis = (n0 + n1).normalize_or_zero();
            b + bis * (d / bis.dot(n0).max(0.35))
        })
        .collect()
}

/// Coins coupes : un rectangle devient un octogone.
fn chamfer(p: &[DVec2], cut: f64) -> Vec<DVec2> {
    let n = p.len();
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let (a, b, c) = (p[(i + n - 1) % n], p[i], p[(i + 1) % n]);
        out.push(b + (a - b).normalize_or_zero() * cut);
        out.push(b + (c - b).normalize_or_zero() * cut);
    }
    out
}

fn scaled(p: &[DVec2], c: DVec2, k: f64) -> Vec<DVec2> {
    p.iter().map(|q| c + (*q - c) * k).collect()
}

fn center(p: &[DVec2]) -> DVec2 {
    p.iter().copied().sum::<DVec2>() / p.len() as f64
}

/// Normale exterieure d'un cote parcouru dans le sens trigonometrique.
fn outward(a: DVec2, b: DVec2) -> DVec2 {
    let d = (b - a).normalize_or_zero();
    DVec2::new(d.y, -d.x)
}

// ---------------------------------------------------------------------------------------------
// Constructeur de volumes dans le repere d'un objet

struct Ctx<'a> {
    o: &'a mut Out,
    f: Frame,
    rng: Rng,
    tier: Tier,
    /// Graine de la ville : sa palette dominante.
    region: u64,
}

impl Ctx<'_> {
    fn pick<T: Copy>(&mut self, list: &[T]) -> T {
        list[((self.rng.f32() * list.len() as f32) as usize).min(list.len() - 1)]
    }

    /// Couleur dominante de la ville : quatre fois sur cinq dans une fenetre de la liste propre a la ville
    /// (villes blanches, ocres, de brique ou pastel), sinon n'importe laquelle.
    fn pick_local<T: Copy>(&mut self, list: &[T]) -> T {
        if self.region == 0 || self.rng.f32() < 0.2 {
            return self.pick(list);
        }
        let n = list.len();
        let window = n.div_ceil(3).max(2);
        let start = (self.region % n as u64) as usize;
        list[(start + (self.rng.f32() * window as f32) as usize % window) % n]
    }

    /// Polygone plan quelconque (pans de toit, pignons) ; la normale est orientee vers `hint`.
    fn face(&mut self, pts: &[(DVec2, f64)], hint: DVec3, pt: Paint) {
        let w: Vec<DVec3> = pts.iter().map(|&(q, z)| self.f.p(q, z)).collect();
        let mut n = DVec3::ZERO;
        for i in 1..w.len() - 1 {
            n += (w[i] - w[0]).cross(w[i + 1] - w[0]);
        }
        let Some(mut n) = n.try_normalize() else { return };
        if n.dot(hint) < 0.0 {
            n = -n;
        }
        let Some(u) = (0..w.len()).map(|i| w[(i + 1) % w.len()] - w[i]).find_map(|e| e.try_normalize()) else { return };
        let mut v = n.cross(u);
        if v.dot(self.f.up) < 0.0 {
            v = -v;
        }
        let uv: Vec<[f64; 2]> = w.iter().map(|p| [(*p - w[0]).dot(u), (*p - w[0]).dot(v)]).collect();
        self.o.poly(&w, n, &uv, pt);
    }

    /// Mur vertical du cote a -> b ; une facade recoit un nombre entier de travees.
    fn wall(&mut self, a: DVec2, b: DVec2, z0: f64, z1: f64, pt: Paint) {
        let len = (b - a).length();
        if len < 0.05 || z1 <= z0 {
            return;
        }
        let n = self.f.dirv(outward(a, b), 0.0);
        let k = if pt.mat[0] == mat::FACADE {
            let bay = bay(pt.mat[1]);
            (len / bay).round().max(1.0) * bay / len
        } else {
            1.0
        };
        let w = [self.f.p(a, z0), self.f.p(b, z0), self.f.p(b, z1), self.f.p(a, z1)];
        self.o.poly(&w, n, &[[0.0, z0], [len * k, z0], [len * k, z1], [0.0, z1]], pt);
    }

    fn walls(&mut self, poly: &[DVec2], z0: f64, z1: f64, pt: Paint) {
        for i in 0..poly.len() {
            self.wall(poly[i], poly[(i + 1) % poly.len()], z0, z1, pt);
        }
    }

    /// Murs d'un batiment a quatre cotes : rue, cote, cour, cote.
    fn shell(&mut self, f: &[DVec2; 4], z0: f64, z1: f64, front: Paint, back: Paint, side: Paint) {
        self.wall(f[0], f[1], z0, z1, front);
        self.wall(f[1], f[2], z0, z1, side);
        self.wall(f[2], f[3], z0, z1, back);
        self.wall(f[3], f[0], z0, z1, side);
    }

    fn cap(&mut self, poly: &[DVec2], z: f64, pt: Paint) {
        let w: Vec<DVec3> = poly.iter().map(|q| self.f.p(*q, z)).collect();
        let uv: Vec<[f64; 2]> = poly.iter().map(|q| [q.x, q.y]).collect();
        self.o.poly(&w, self.f.up, &uv, pt);
    }

    fn cap_down(&mut self, poly: &[DVec2], z: f64, pt: Paint) {
        let w: Vec<DVec3> = poly.iter().map(|q| self.f.p(*q, z)).collect();
        let uv: Vec<[f64; 2]> = poly.iter().map(|q| [q.x, q.y]).collect();
        self.o.poly(&w, -self.f.up, &uv, pt);
    }

    fn prism(&mut self, poly: &[DVec2], z0: f64, z1: f64, side: Paint, top: Option<Paint>) {
        self.walls(poly, z0, z1, side);
        if let Some(top) = top {
            self.cap(poly, z1, top);
        }
    }

    /// Pave oriente, avec dessus (et dessous s'il flotte).
    fn cube(&mut self, c: DVec2, ax: DVec2, hx: f64, hy: f64, z0: f64, z1: f64, pt: Paint) {
        let r = rect(c, ax, hx, hy);
        self.prism(&r, z0, z1, pt, Some(pt));
        if z0 > 0.05 {
            self.cap_down(&r, z0, pt);
        }
    }

    /// Bandeau le long d'un cote : depasse de `out` vers l'exterieur, entre de `inn` dans le mur.
    fn strip(&mut self, a: DVec2, b: DVec2, out: f64, inn: f64, z0: f64, z1: f64, pt: Paint) {
        let o = outward(a, b);
        let r = [a + o * out, b + o * out, b - o * inn, a - o * inn];
        self.prism(&r, z0, z1, pt, Some(pt));
        if z0 > 0.05 {
            self.cap_down(&r, z0, pt);
        }
    }

    /// Cylindre ou tronc de cone, aux normales lissees.
    #[allow(clippy::too_many_arguments)]
    fn cylinder(&mut self, c: DVec2, r0: f64, r1: f64, z0: f64, z1: f64, segs: usize, side: Paint, top: Option<Paint>, inward: bool) {
        let f = self.f;
        let ring = |r: f64, k: usize| {
            let a = k as f64 / segs as f64 * std::f64::consts::TAU;
            c + DVec2::from_angle(a) * r
        };
        let slope = (r0 - r1) / (z1 - z0).max(1e-3);
        for k in 0..segs {
            let (a, b) = (k, k + 1);
            let n = |i: usize| {
                let d = DVec2::from_angle(i as f64 / segs as f64 * std::f64::consts::TAU);
                let v = f.dirv(d, slope);
                if inward { -v } else { v }
            };
            let arc = |i: usize| i as f64 / segs as f64 * std::f64::consts::TAU * r0.max(r1);
            self.o.quad_smooth(
                [f.p(ring(r0, a), z0), f.p(ring(r0, b), z0), f.p(ring(r1, b), z1), f.p(ring(r1, a), z1)],
                [n(a), n(b), n(b), n(a)],
                [[arc(a), z0], [arc(b), z0], [arc(b), z1], [arc(a), z1]],
                side,
            );
        }
        if let Some(top) = top {
            if r1 > 0.02 {
                let poly: Vec<DVec2> = (0..segs).map(|k| ring(r1, k)).collect();
                self.cap(&poly, z1, top);
            }
        }
    }

    /// Couronne horizontale entre deux rayons.
    fn annulus(&mut self, c: DVec2, r0: f64, r1: f64, z: f64, segs: usize, pt: Paint) {
        for k in 0..segs {
            let a = DVec2::from_angle(k as f64 / segs as f64 * std::f64::consts::TAU);
            let b = DVec2::from_angle((k + 1) as f64 / segs as f64 * std::f64::consts::TAU);
            self.face(&[(c + a * r0, z), (c + b * r0, z), (c + b * r1, z), (c + a * r1, z)], self.f.up, pt);
        }
    }

    fn pyramid(&mut self, poly: &[DVec2], z0: f64, apex: DVec2, z1: f64, pt: Paint) {
        let c = center(poly);
        for i in 0..poly.len() {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            let mid = (a + b) * 0.5 - c;
            self.face(&[(a, z0), (b, z0), (apex, z1)], self.f.dirv(mid.normalize_or_zero(), 0.5), pt);
        }
    }

    fn beacon(&mut self, at: DVec2, z: f64) {
        self.cube(at, DVec2::X, 0.3, 0.3, z, z + 0.6, with([0.5, 0.02, 0.01], mat::BEACON));
    }

    /// Toit a deux pans (faitage le long de r0 -> r1) ou a croupe. Renvoie l'altitude du faitage.
    #[allow(clippy::too_many_arguments)]
    fn gable(&mut self, r: &[DVec2; 4], z: f64, pitch: f64, over: f64, roof: Paint, end: Paint, hip: bool) -> f64 {
        let (lx, ly) = ((r[1] - r[0]).length(), (r[3] - r[0]).length());
        if lx < 0.1 || ly < 0.1 {
            return z;
        }
        let (ax, ay) = ((r[1] - r[0]) / lx, (r[3] - r[0]) / ly);
        let c = (r[0] + r[2]) * 0.5;
        let zt = z + ly * 0.5 * pitch;
        let ze = z - over * pitch;
        let e = |sx: f64, sy: f64| c + ax * (sx * (lx * 0.5 + over)) + ay * (sy * (ly * 0.5 + over));
        let up = self.f.up;
        if hip {
            let rh = (lx * 0.5 - ly * 0.5).max(0.0);
            let (r0, r1) = (c - ax * rh, c + ax * rh);
            self.face(&[(e(-1.0, -1.0), ze), (e(1.0, -1.0), ze), (r1, zt), (r0, zt)], up, roof);
            self.face(&[(e(1.0, 1.0), ze), (e(-1.0, 1.0), ze), (r0, zt), (r1, zt)], up, roof);
            self.face(&[(e(1.0, -1.0), ze), (e(1.0, 1.0), ze), (r1, zt)], up, roof);
            self.face(&[(e(-1.0, 1.0), ze), (e(-1.0, -1.0), ze), (r0, zt)], up, roof);
        } else {
            let (r0, r1) = (c - ax * (lx * 0.5 + over), c + ax * (lx * 0.5 + over));
            self.face(&[(e(-1.0, -1.0), ze), (e(1.0, -1.0), ze), (r1, zt), (r0, zt)], up, roof);
            self.face(&[(e(1.0, 1.0), ze), (e(-1.0, 1.0), ze), (r0, zt), (r1, zt)], up, roof);
            for sx in [-1.0, 1.0] {
                let w0 = c + ax * (sx * lx * 0.5) - ay * (ly * 0.5);
                let w1 = c + ax * (sx * lx * 0.5) + ay * (ly * 0.5);
                let tip = c + ax * (sx * lx * 0.5);
                self.face(&[(w0, z), (w1, z), (tip, zt)], self.f.dirv(ax * sx, 0.0), end);
            }
        }
        zt
    }

    /// Toit d'une maison de rang : pans vers la rue et vers la cour, pignons sur les mitoyens.
    /// `t` : part de la profondeur couverte par chaque pan (0.5 : deux pans ; moins : mansarde).
    #[allow(clippy::too_many_arguments)]
    fn row_roof(&mut self, f: &[DVec2; 4], z: f64, t: f64, h: f64, slope: Paint, top: Paint, side: Paint) {
        let t = t.min(0.5);
        let (a0, a1) = (f[0].lerp(f[3], t), f[1].lerp(f[2], t));
        let (b2, b3) = (f[2].lerp(f[1], t), f[3].lerp(f[0], t));
        let up = self.f.up;
        self.face(&[(f[0], z), (f[1], z), (a1, z + h), (a0, z + h)], up, slope);
        self.face(&[(f[2], z), (f[3], z), (b3, z + h), (b2, z + h)], up, slope);
        if t < 0.49 {
            self.cap(&[a0, a1, b2, b3], z + h, top);
            self.face(&[(f[1], z), (f[2], z), (b2, z + h), (a1, z + h)], self.f.dirv(outward(f[1], f[2]), 0.0), side);
            self.face(&[(f[3], z), (f[0], z), (a0, z + h), (b3, z + h)], self.f.dirv(outward(f[3], f[0]), 0.0), side);
        } else {
            self.face(&[(f[1], z), (f[2], z), (a1, z + h)], self.f.dirv(outward(f[1], f[2]), 0.0), side);
            self.face(&[(f[3], z), (f[0], z), (a0, z + h)], self.f.dirv(outward(f[3], f[0]), 0.0), side);
        }
    }

    /// Lucarne : un petit volume vitre en saillie du toit, coiffe de deux pans.
    #[allow(clippy::too_many_arguments)]
    fn dormer(&mut self, q: DVec2, z: f64, out: DVec2, w: f64, h: f64, depth: f64, wall: Paint, roof: Paint) {
        let d = out.perp();
        let (a, b) = (q - d * (w * 0.5), q + d * (w * 0.5));
        let r = [a, b, b - out * depth, a - out * depth];
        self.walls(&r, z - 1.2, z + h, wall);
        let (g0, g1) = (a + d * 0.22 + out * 0.03, b - d * 0.22 + out * 0.03);
        self.face(&[(g0, z + 0.2), (g1, z + 0.2), (g1, z + h - 0.15), (g0, z + h - 0.15)], self.f.dirv(out, 0.0), with([0.03, 0.035, 0.04], mat::GLASS));
        let rr = [b + out * 0.12, b - out * depth, a - out * depth, a + out * 0.12];
        self.gable(&rr, z + h, 0.95, 0.12, roof, wall, false);
    }

    /// Toit-terrasse ceint d'un acrotere.
    fn parapet(&mut self, poly: &[DVec2], z: f64, h: f64, wall: Paint, roof: Paint) {
        let inner = shrink(poly, 0.3);
        let coping = plain([0.42, 0.41, 0.39]);
        let n = poly.len();
        for i in 0..n {
            let (a, b) = (poly[i], poly[(i + 1) % n]);
            let (ia, ib) = (inner[i], inner[(i + 1) % n]);
            self.wall(a, b, z, z + h, wall);
            if self.tier != Tier::Low {
                self.wall(ib, ia, z, z + h, wall);
                self.face(&[(a, z + h), (b, z + h), (ib, z + h), (ia, z + h)], self.f.up, coping);
            }
        }
        self.cap(&inner, z + 0.02, roof);
    }

    fn railing(&mut self, a: DVec2, b: DVec2, z: f64, h: f64, pt: Paint) {
        let n = self.f.dirv(outward(a, b), 0.0);
        let len = (b - a).length();
        let w = [self.f.p(a, z), self.f.p(b, z), self.f.p(b, z + h), self.f.p(a, z + h)];
        self.o.poly(&w, n, &[[0.0, 0.0], [len, 0.0], [len, h], [0.0, h]], pt);
    }
}

// ---------------------------------------------------------------------------------------------
// Batiments

impl Ctx<'_> {
    fn building(&mut self, b: &Building) {
        let f = b.foot;
        let floors = b.floors as f64;
        let seed = b.seed;
        let full = self.tier == Tier::Full;
        let low = self.tier == Tier::Low;
        let len0 = (f[1] - f[0]).length();
        let ax = (f[1] - f[0]).normalize_or_zero();
        let out0 = outward(f[0], f[1]);
        match b.kind {
            Kind::House => {
                let top = 0.5 + floors * 2.9;
                let wall = self.pick_local(&PLASTER);
                let roof = with(self.pick_local(&TILE), mat::TILES);
                let face = facade(wall, win::HOUSE, 2.9, seed);
                if !low {
                    self.prism(&f, -2.5, 0.5, plain([0.3, 0.29, 0.27]), None);
                }
                self.walls(&f, if low { -2.5 } else { 0.5 }, top, face);
                let r = if len0 >= (f[3] - f[0]).length() { f } else { [f[1], f[2], f[3], f[0]] };
                let hip = self.rng.f32() < 0.4;
                let pitch = self.rng.range(0.55, 0.85) as f64;
                let zt = self.gable(&r, top, pitch, if low { 0.0 } else { 0.45 }, roof, plain(wall), hip);
                if full {
                    let rax = (r[1] - r[0]).normalize_or_zero();
                    let rc = center(&r);
                    let side = if self.rng.f32() < 0.5 { -1.0 } else { 1.0 };
                    let brick = plain(self.pick(&BRICK));
                    self.cube(rc + rax * ((r[1] - r[0]).length() * 0.22 * side), rax, 0.4, 0.5, top, zt + 0.9, brick);
                    let m = f[0].lerp(f[1], self.rng.range(0.3, 0.7) as f64);
                    let door = [0.12, 0.07, 0.04];
                    self.face(
                        &[(m - ax * 0.5 + out0 * 0.03, 0.5), (m + ax * 0.5 + out0 * 0.03, 0.5), (m + ax * 0.5 + out0 * 0.03, 2.65), (m - ax * 0.5 + out0 * 0.03, 2.65)],
                        self.f.dirv(out0, 0.0),
                        plain(door),
                    );
                    self.cube(m + out0 * 0.45, ax, 0.85, 0.45, -0.5, 0.45, plain([0.35, 0.34, 0.32]));
                    // Auvent au-dessus de la porte.
                    self.face(
                        &[(m - ax * 0.9, 2.95), (m + ax * 0.9, 2.95), (m + ax * 0.9 + out0 * 1.1, 2.6), (m - ax * 0.9 + out0 * 1.1, 2.6)],
                        self.f.up,
                        roof,
                    );
                }
            }
            Kind::Garage => {
                let wall = self.pick_local(&PLASTER);
                self.prism(&f, -1.0, 2.8, plain(wall), Some(with([0.2, 0.2, 0.2], mat::ROOF)));
                if full {
                    let m = f[0].lerp(f[1], 0.5) + out0 * 0.02;
                    self.face(&[(m - ax * 1.2, 0.0), (m + ax * 1.2, 0.0), (m + ax * 1.2, 2.2), (m - ax * 1.2, 2.2)], self.f.dirv(out0, 0.0), with([0.6, 0.6, 0.58], mat::METAL));
                }
            }
            Kind::Pool => {
                if !low {
                    let inner = shrink(&f, 0.35);
                    self.cap(&inner, 0.06, with([0.02, 0.12, 0.16], mat::WATER));
                    for i in 0..4 {
                        self.strip(f[i], f[(i + 1) % 4], 0.0, 0.35, -0.2, 0.12, plain([0.55, 0.53, 0.5]));
                    }
                }
            }
            Kind::Townhouse => self.townhouse(b),
            Kind::Haussmann => self.haussmann(b),
            Kind::Apartments => self.apartments(b),
            Kind::Office => self.office(b),
            Kind::Tower => self.tower(b),
            Kind::Warehouse => self.warehouse(b),
            Kind::Tank => {
                let c = center(&f);
                let r = (f[1] - f[0]).length() * 0.5;
                let h = 11.0 + self.rng.f32() as f64 * 6.0;
                let metal = with(self.pick(&[[0.6, 0.6, 0.58], [0.55, 0.55, 0.5], [0.45, 0.47, 0.5]]), mat::PAINT);
                let segs = if low { 10 } else { 22 };
                self.cylinder(c, r, r, -1.0, h, segs, metal, None, false);
                self.cylinder(c, r, 0.5, h, h + r * 0.25, segs, metal, Some(metal), false);
                if full {
                    // Rambarde du toit et escalier en colimacon figure par un poteau.
                    self.cylinder(c, r + 0.05, r + 0.05, h, h + 1.0, segs, with_param([0.1, 0.1, 0.1], mat::RAILING, 0), None, false);
                }
            }
            Kind::Chimney => {
                let c = center(&f);
                let segs = if low { 8 } else { 14 };
                let brick = plain(self.pick(&BRICK));
                self.cylinder(c, 2.1, 1.3, -1.0, 52.0, segs, brick, None, false);
                self.cylinder(c, 1.36, 1.3, 47.5, 52.0, segs, plain([0.04, 0.035, 0.03]), Some(plain([0.01, 0.01, 0.01])), false);
                if !low {
                    self.beacon(c + DVec2::X * 1.3, 52.0);
                }
            }
            Kind::Church => self.church(b),
            Kind::Monument => {
                let c = center(&f);
                let stone = plain([0.5, 0.46, 0.38]);
                for (k, half) in [9.0, 7.6, 6.2].into_iter().enumerate() {
                    self.cube(c, ax, half, half, -1.0, 0.45 * (k + 1) as f64, stone);
                }
                self.cube(c, ax, 2.4, 2.4, 1.35, 6.0, stone);
                self.cube(c, ax, 2.7, 2.7, 5.6, 6.2, stone);
                let base = rect(c, ax, 1.55, 1.55);
                let tip = rect(c, ax, 1.0, 1.0);
                for i in 0..4 {
                    let (a, b2, t0, t1) = (base[i], base[(i + 1) % 4], tip[i], tip[(i + 1) % 4]);
                    let n = self.f.dirv(outward(a, b2), 0.05);
                    self.face(&[(a, 6.2), (b2, 6.2), (t1, 29.0), (t0, 29.0)], n, stone);
                }
                self.pyramid(&tip, 29.0, c, 31.6, with([0.55, 0.4, 0.12], mat::PAINT));
            }
            Kind::Fountain => {
                if low {
                    return;
                }
                let c = center(&f);
                let r = (f[1] - f[0]).length() * 0.5 - 0.5;
                let stone = plain([0.5, 0.47, 0.42]);
                let water = with([0.02, 0.1, 0.12], mat::WATER);
                self.cylinder(c, r, r, -0.5, 0.6, 28, stone, None, false);
                self.cylinder(c, r - 0.35, r - 0.35, 0.0, 0.6, 28, stone, None, true);
                self.annulus(c, r - 0.35, r, 0.6, 28, stone);
                let disc: Vec<DVec2> = (0..28).map(|k| c + DVec2::from_angle(k as f64 / 28.0 * std::f64::consts::TAU) * (r - 0.35)).collect();
                self.cap(&disc, 0.42, water);
                self.cylinder(c, 0.5, 0.35, 0.0, 2.2, 12, stone, None, false);
                self.cylinder(c, 0.45, 1.7, 2.0, 2.45, 18, stone, Some(water), false);
                self.cylinder(c, 0.28, 0.12, 2.45, 3.3, 8, stone, Some(stone), false);
            }
        }
    }

    fn townhouse(&mut self, b: &Building) {
        let f = b.foot;
        let fh = 3.0;
        let top = 0.3 + b.floors as f64 * fh;
        let wall = self.pick_local(&PLASTER);
        let shop = self.rng.f32() < 0.45;
        let front = facade(wall, win::OLD | if shop { win::SHOP } else { 0 }, fh, b.seed);
        let back = facade(wall, win::OLD, fh, b.seed);
        let side = if b.row { plain(wall) } else { back };
        self.shell(&f, -2.5, top, front, back, side);
        let depth = ((f[3] + f[2]) * 0.5 - (f[0] + f[1]) * 0.5).length();
        let h = depth * 0.5 * self.rng.range(0.7, 1.0) as f64;
        let roof = with(self.pick_local(&TILE), mat::TILES);
        self.row_roof(&f, top, 0.5, h, roof, roof, plain(wall));
        if self.tier == Tier::Low {
            return;
        }
        let light = plain(scale(wall, 1.15));
        self.strip(f[0], f[1], 0.25, 0.05, top - 0.35, top, light);
        self.strip(f[0], f[1], 0.12, 0.05, fh + 0.3 - 0.2, fh + 0.3, light);
        if self.tier != Tier::Full {
            return;
        }
        let out = outward(f[0], f[1]);
        let len = (f[1] - f[0]).length();
        let count = ((len / 4.0).floor() as usize).min(2);
        for k in 0..count {
            let t = (k as f64 + 0.5) / count as f64;
            let q = f[0].lerp(f[1], t);
            let ridge = f[0].lerp(f[3], 0.5).lerp(f[1].lerp(f[2], 0.5), t);
            self.dormer(q.lerp(ridge, 0.32), top + h * 0.32, out, 1.2, 1.35, 1.6, plain(wall), roof);
        }
        let m = f[1].lerp(f[2], 0.5);
        let brick = plain(self.pick(&BRICK));
        self.cube(m, (f[2] - f[1]).normalize_or_zero(), 0.45, 0.3, top, top + h + 1.0, brick);
        if shop {
            let ax = (f[1] - f[0]).normalize_or_zero();
            let (a, c) = (f[0] + ax * 0.35, f[1] - ax * 0.35);
            let awning = with_param(self.pick(&[[0.4, 0.03, 0.02], [0.03, 0.18, 0.08], [0.04, 0.06, 0.2], [0.35, 0.25, 0.03]]), mat::AWNING, 0);
            self.face(&[(a, 3.35), (c, 3.35), (c + out * 1.3, 2.75), (a + out * 1.3, 2.75)], self.f.up, awning);
            self.face(&[(a + out * 1.3, 2.75), (c + out * 1.3, 2.75), (c + out * 1.3, 2.5), (a + out * 1.3, 2.5)], self.f.dirv(out, 0.0), awning);
        }
    }

    fn haussmann(&mut self, b: &Building) {
        let f = b.foot;
        let fh = 3.1;
        let floors = b.floors as f64;
        let top = 0.3 + floors * fh;
        let stone = self.pick(&STONE);
        let shop = self.rng.f32() < 0.8;
        let front = facade(stone, win::HAUSSMANN | if shop { win::SHOP } else { 0 }, fh, b.seed);
        let back = facade(scale(stone, 0.85), win::RESIDENTIAL, fh, b.seed);
        let side = if b.row { plain(scale(stone, 0.9)) } else { front };
        self.shell(&f, -2.5, top, front, back, side);
        let depth = ((f[3] + f[2]) * 0.5 - (f[0] + f[1]) * 0.5).length().max(4.0);
        let t = (2.5 / depth).min(0.45);
        let h = 3.4;
        let zinc = with([0.15, 0.17, 0.2], mat::ZINC);
        self.row_roof(&f, top, t, h, zinc, zinc, plain(stone));
        if self.tier == Tier::Low {
            return;
        }
        let carved = plain(scale(stone, 1.12));
        self.strip(f[0], f[1], 0.55, 0.05, top - 0.6, top, carved);
        self.strip(f[0], f[1], 0.15, 0.05, 0.3 + fh * 2.0 - 0.35, 0.3 + fh * 2.0 - 0.1, carved);
        let out = outward(f[0], f[1]);
        let ax = (f[1] - f[0]).normalize_or_zero();
        let iron = with_param([0.02, 0.02, 0.025], mat::RAILING, 0);
        // Balcons filants des deuxieme et cinquieme etages, comme sur les boulevards de Paris.
        for fl in [2.0, 5.0] {
            if fl >= floors {
                continue;
            }
            let z = 0.3 + fl * fh;
            self.strip(f[0], f[1], 0.9, 0.0, z - 0.22, z, carved);
            if self.tier == Tier::Full {
                self.railing(f[0] + out * 0.88, f[1] + out * 0.88, z, 1.0, iron);
                self.railing(f[0], f[0] + out * 0.88, z, 1.0, iron);
                self.railing(f[1] + out * 0.88, f[1], z, 1.0, iron);
            }
        }
        if self.tier != Tier::Full {
            return;
        }
        // Garde-corps des autres fenetres, et lucarnes du comble.
        let len = (f[1] - f[0]).length();
        let bays = (len / 3.2).round().max(1.0) as usize;
        for fl in 1..floors as usize {
            if fl == 2 || fl == 5 {
                continue;
            }
            let z = 0.3 + fl as f64 * fh + 0.35;
            for k in 0..bays {
                let m = f[0].lerp(f[1], (k as f64 + 0.5) / bays as f64) + out * 0.08;
                self.railing(m - ax * 0.7, m + ax * 0.7, z, 0.75, iron);
            }
        }
        let (a0, a1) = (f[0].lerp(f[3], t), f[1].lerp(f[2], t));
        for k in 0..bays {
            if bays > 3 && k % 2 == 1 {
                continue;
            }
            let tk = (k as f64 + 0.5) / bays as f64;
            let q = f[0].lerp(f[1], tk).lerp(a0.lerp(a1, tk), 0.38);
            self.dormer(q, top + h * 0.38, out, 1.15, 1.65, 1.3, plain(scale(stone, 1.05)), zinc);
        }
        let brick = plain(self.pick(&BRICK));
        for (p, q) in [(a1, a0), (a0, a1)] {
            let d = (q - p).normalize_or_zero();
            self.cube(p + d * 0.5, (f[2] - f[1]).normalize_or_zero(), 1.1, 0.35, top + h - 0.2, top + h + 1.5, brick);
        }
    }

    fn apartments(&mut self, b: &Building) {
        let f = b.foot;
        let fh = 2.9;
        let floors = b.floors as f64;
        let top = 0.3 + floors * fh;
        let wall = self.pick(&MODERN);
        let shop = floors >= 4.0 && self.rng.f32() < 0.35;
        let front = facade(wall, win::RESIDENTIAL | if shop { win::SHOP } else { 0 }, fh, b.seed);
        let back = facade(wall, win::RESIDENTIAL, fh, b.seed);
        let side = if b.row { plain(wall) } else { back };
        self.shell(&f, -2.5, top, front, back, side);
        let gravel = with([0.2, 0.19, 0.18], mat::ROOF);
        self.parapet(&f, top, 0.9, plain(scale(wall, 0.95)), gravel);
        if self.tier == Tier::Low {
            return;
        }
        let out = outward(f[0], f[1]);
        let ax = (f[1] - f[0]).normalize_or_zero();
        if self.rng.f32() < 0.55 {
            let len = (f[1] - f[0]).length();
            let bays = (len / 3.0).round().max(1.0) as usize;
            let glass = self.rng.f32() < 0.4;
            let rail = if glass { with_param([0.05, 0.07, 0.08], mat::RAILING, 2) } else { with_param([0.04, 0.04, 0.045], mat::RAILING, 0) };
            let slab = plain([0.5, 0.49, 0.47]);
            for fl in 1..floors as usize {
                let z = 0.3 + fl as f64 * fh;
                for k in (0..bays).filter(|k| k % 2 == fl % 2 || bays < 3) {
                    let m = f[0].lerp(f[1], (k as f64 + 0.5) / bays as f64);
                    self.cube(m + out * 0.65, ax, 1.25, 0.65, z - 0.18, z, slab);
                    if self.tier == Tier::Full {
                        let (p0, p1) = (m - ax * 1.25 + out * 1.28, m + ax * 1.25 + out * 1.28);
                        self.railing(p0, p1, z, 1.0, rail);
                        self.railing(m - ax * 1.25, p0, z, 1.0, rail);
                        self.railing(p1, m + ax * 1.25, z, 1.0, rail);
                    }
                }
            }
        }
        if self.tier == Tier::Full {
            let c = center(&f);
            self.cube(c, ax, 1.6, 1.4, top, top + 2.6, plain(scale(wall, 0.9)));
            self.cube(c + ax * 3.0, ax, 0.6, 0.5, top, top + 0.9, with([0.5, 0.5, 0.5], mat::METAL));
        }
    }

    fn office(&mut self, b: &Building) {
        let f = b.foot;
        let fh = 3.7;
        let floors = b.floors as f64;
        let top = 0.4 + floors * fh;
        let curtain = self.rng.f32() < 0.5;
        let (color, style) = if curtain { (self.pick(&GLASS_TINT), win::CURTAIN) } else { (self.pick(&MODERN), win::RIBBON) };
        let split = if floors > 12.0 && self.rng.f32() < 0.6 { (floors * 0.6).round() } else { floors };
        let z1 = 0.4 + split * fh;
        let front = facade(color, style | win::SHOP, fh, b.seed);
        let back = facade(color, style, fh, b.seed);
        let side = if b.row { plain(scale(color, 0.9)) } else { back };
        self.shell(&f, -2.5, z1, front, back, side);
        let roof = with([0.22, 0.22, 0.22], mat::ROOF);
        let mut crown = f.to_vec();
        if split < floors {
            let inner = shrink(&f, 3.0);
            self.cap(&f, z1, roof);
            self.walls(&inner, z1, top, back);
            crown = inner;
        }
        self.parapet(&crown, if split < floors { top } else { z1 }, 0.7, plain(scale(color, 0.8)), roof);
        if self.tier != Tier::Low && self.rng.f32() < 0.6 {
            let c = center(&crown);
            let ax = (f[1] - f[0]).normalize_or_zero();
            self.cube(c, ax, 3.0, 2.2, top, top + 3.2, with([0.4, 0.4, 0.4], mat::METAL));
        }
    }

    fn tower(&mut self, b: &Building) {
        let f = b.foot;
        let floors = b.floors as f64;
        let c = center(&f);
        let ax = (f[1] - f[0]).normalize_or_zero();
        let (lx, ly) = ((f[1] - f[0]).length(), (f[3] - f[0]).length());
        let pod_top = 0.4 + 4.0 * 4.2;
        let pod_color = self.pick(&[[0.3, 0.28, 0.25], [0.05, 0.06, 0.07], [0.45, 0.42, 0.36]]);
        self.walls(&f, -2.5, pod_top, facade(pod_color, win::CURTAIN | win::SHOP, 4.2, b.seed));
        let roof = with([0.2, 0.2, 0.2], mat::ROOF);
        self.cap(&f, pod_top, roof);
        let m = (lx.min(ly) * 0.14).clamp(2.0, 6.0);
        let shaft = rect(c, ax, (lx * 0.5 - m).max(6.0), (ly * 0.5 - m).max(6.0));
        let small = (lx.min(ly) * 0.5 - m).max(6.0);
        let top = pod_top + floors * 3.8;
        let low = self.tier == Tier::Low;
        let glass = self.pick(&GLASS_TINT);
        let curtain = facade(glass, win::CURTAIN, 3.8, b.seed);
        match b.seed % 4 {
            0 => {
                // Gradins Art deco, couronne en pyramide et fleche.
                let stone = self.pick(&STONE);
                let face = facade(stone, win::RIBBON, 3.8, b.seed);
                let (mut z, mut k) = (pod_top, 1.0);
                for share in [0.62, 0.24, 0.14] {
                    let poly = scaled(&shaft, c, k);
                    let z1 = z + (top - pod_top) * share;
                    self.walls(&poly, z, z1, face);
                    self.cap(&poly, z1, roof);
                    z = z1;
                    k *= 0.78;
                }
                let poly = scaled(&shaft, c, k / 0.78 * 0.92);
                let apex = top + small * 1.1;
                self.pyramid(&poly, top, c, apex, with([0.1, 0.2, 0.16], mat::PAINT));
                if !low {
                    self.cylinder(c, 0.7, 0.06, apex - 3.0, apex + 24.0, 6, with([0.5, 0.5, 0.5], mat::METAL), None, false);
                    self.beacon(c, apex + 24.0);
                }
            }
            1 => {
                // Boite de verre, etages techniques sombres, ecran de couronnement et mat.
                let band = plain([0.06, 0.06, 0.065]);
                let mut z = pod_top;
                while z < top {
                    let z1 = (z + 25.0 * 3.8).min(top);
                    self.walls(&shaft, z, z1, curtain);
                    if z1 < top && !low {
                        self.walls(&shaft, z1, z1 + 2.4, band);
                    }
                    z = z1 + if z1 < top { 2.4 } else { 0.0 };
                }
                self.cap(&shaft, top, roof);
                self.walls(&shaft, top, top + 6.0, plain(scale(glass, 2.5)));
                if !low {
                    self.cube(c, ax, 0.6, 0.6, top, top + 30.0, with([0.45, 0.45, 0.45], mat::METAL));
                    self.beacon(c, top + 30.0);
                    for q in shaft.iter() {
                        self.beacon(*q + (c - *q).normalize() * 0.6, top + 6.0);
                    }
                }
            }
            2 => {
                // Octogone et couronne en gradins.
                let oct = chamfer(&shaft, small * 0.42);
                self.walls(&oct, pod_top, top, curtain);
                let (mut z, mut k) = (top, 1.0);
                for _ in 0..2 {
                    self.cap(&scaled(&oct, c, k), z, roof);
                    k *= 0.78;
                    self.walls(&scaled(&oct, c, k), z, z + 9.0, curtain);
                    z += 9.0;
                }
                self.cap(&scaled(&oct, c, k), z, roof);
                if !low {
                    self.cylinder(c, 0.5, 0.2, z, z + 22.0, 6, with([0.5, 0.5, 0.5], mat::METAL), None, false);
                    self.beacon(c, z + 22.0);
                }
            }
            _ => {
                // Tour effilee : six troncons de plus en plus fins, toit en biseau.
                let face = if b.seed % 8 < 4 { curtain } else { facade(self.pick(&MODERN), win::RIBBON, 3.8, b.seed) };
                let mut poly = shaft.to_vec();
                for k in 0..6 {
                    let z0 = pod_top + (top - pod_top) * k as f64 / 6.0;
                    let z1 = pod_top + (top - pod_top) * (k + 1) as f64 / 6.0;
                    poly = scaled(&shaft, c, 1.0 - 0.055 * k as f64);
                    self.walls(&poly, z0, z1, face);
                    if k < 5 {
                        self.cap(&poly, z1, roof);
                    }
                }
                let apex = c + ax * (lx * 0.2);
                self.pyramid(&poly, top, apex, top + small * 1.3, with([0.35, 0.36, 0.38], mat::METAL));
                if !low {
                    self.beacon(apex, top + small * 1.3);
                }
            }
        }
    }

    fn warehouse(&mut self, b: &Building) {
        let f = b.foot;
        let h = 8.0 + self.rng.f32() as f64 * 3.0;
        let color = self.pick(&METALS);
        self.walls(&f, -1.5, h, facade(color, win::INDUSTRIAL, 25.0, b.seed));
        let roof_metal = with(scale(self.pick(&METALS), 0.8), mat::METAL);
        if self.tier == Tier::Low {
            self.cap(&f, h, roof_metal);
            return;
        }
        let r = if (f[1] - f[0]).length() >= (f[3] - f[0]).length() { f } else { [f[1], f[2], f[3], f[0]] };
        let (lx, ly) = ((r[1] - r[0]).length(), (r[3] - r[0]).length());
        let (ax, ay) = ((r[1] - r[0]) / lx, (r[3] - r[0]) / ly);
        if self.rng.f32() < 0.5 {
            // Sheds : dents de scie vitrees face au nord.
            let n = (lx / 9.0).round().max(2.0) as usize;
            let glass = with([0.05, 0.06, 0.07], mat::GLASS);
            let side = plain(color);
            for k in 0..n {
                let (x0, x1) = (k as f64 * lx / n as f64, (k + 1) as f64 * lx / n as f64);
                let p = |x: f64, y: f64| r[0] + ax * x + ay * y;
                self.face(&[(p(x0, 0.0), h), (p(x0, ly), h), (p(x0, ly), h + 3.0), (p(x0, 0.0), h + 3.0)], self.f.dirv(-ax, 0.0), glass);
                self.face(&[(p(x0, 0.0), h + 3.0), (p(x0, ly), h + 3.0), (p(x1, ly), h), (p(x1, 0.0), h)], self.f.up, roof_metal);
                self.face(&[(p(x0, 0.0), h), (p(x1, 0.0), h), (p(x0, 0.0), h + 3.0)], self.f.dirv(-ay, 0.0), side);
                self.face(&[(p(x0, ly), h), (p(x1, ly), h), (p(x0, ly), h + 3.0)], self.f.dirv(ay, 0.0), side);
            }
        } else {
            self.gable(&r, h, 0.16, 0.4, roof_metal, plain(color), false);
        }
        if self.tier == Tier::Full {
            let out = outward(f[0], f[1]);
            let ex = (f[1] - f[0]).normalize_or_zero();
            let len = (f[1] - f[0]).length();
            let docks = ((len / 11.0).floor() as usize).min(6);
            for k in 0..docks {
                let m = f[0].lerp(f[1], (k as f64 + 0.5) / docks as f64) + out * 0.03;
                self.face(&[(m - ex * 1.8, 0.0), (m + ex * 1.8, 0.0), (m + ex * 1.8, 4.3), (m - ex * 1.8, 4.3)], self.f.dirv(out, 0.0), with([0.15, 0.15, 0.16], mat::METAL));
                self.cube(m + out * 0.7, ex, 2.2, 0.7, 4.6, 4.8, plain([0.2, 0.2, 0.2]));
            }
        }
    }

    fn church(&mut self, b: &Building) {
        let f = b.foot;
        let (lx, ly) = ((f[1] - f[0]).length(), (f[3] - f[0]).length());
        let (ax, ay) = ((f[1] - f[0]) / lx, (f[3] - f[0]) / ly);
        let p = |x: f64, y: f64| f[0] + ax * x + ay * y;
        let stone_c = [0.48, 0.44, 0.36];
        let stone = plain(stone_c);
        let windows = facade(stone_c, win::CHURCH, 14.0, b.seed);
        let slate = with([0.06, 0.065, 0.075], mat::TILES);
        let tw = (ly * 0.62).min(9.0);
        let ar = ly * 0.5;
        let nave = [p(tw - 0.5, 0.0), p(lx - ar, 0.0), p(lx - ar, ly), p(tw - 0.5, ly)];
        self.walls(&nave, -2.5, 13.0, windows);
        self.gable(&nave, 13.0, 1.1, 0.4, slate, stone, false);
        // Abside en demi-octogone.
        let ac = p(lx - ar, ly * 0.5);
        let mut apse: Vec<DVec2> = (0..=4).map(|k| ac + DVec2::from_angle(-std::f64::consts::FRAC_PI_2 + k as f64 / 4.0 * std::f64::consts::PI).rotate(ax) * ar).collect();
        apse.push(ac);
        let ring: Vec<DVec2> = apse[..5].to_vec();
        for i in 0..4 {
            self.wall(ring[i], ring[i + 1], -2.5, 11.0, windows);
        }
        for i in 0..4 {
            let mid = ((ring[i] + ring[i + 1]) * 0.5 - ac).normalize_or_zero();
            self.face(&[(ring[i], 11.0), (ring[i + 1], 11.0), (ac, 11.0 + ar * 0.95)], self.f.dirv(mid, 0.6), slate);
        }
        // Clocher et fleche.
        let tc = p(tw * 0.5, ly * 0.5);
        let tower = rect(tc, ax, tw * 0.5, tw * 0.5);
        self.walls(&tower, -2.5, 31.0, windows);
        self.cap(&tower, 31.0, stone);
        let spire = chamfer(&shrink(&tower, 0.4), tw * 0.13);
        self.pyramid(&spire, 31.0, tc, 31.0 + tw * 2.7, slate);
        if self.tier == Tier::Low {
            return;
        }
        self.strip(tower[0], tower[1], 0.3, 0.05, 30.4, 31.2, stone);
        self.strip(tower[1], tower[2], 0.3, 0.05, 30.4, 31.2, stone);
        self.strip(tower[2], tower[3], 0.3, 0.05, 30.4, 31.2, stone);
        self.strip(tower[3], tower[0], 0.3, 0.05, 30.4, 31.2, stone);
        if self.tier == Tier::Full {
            // Contreforts le long de la nef, croix au sommet.
            let mut x = tw + 2.5;
            while x < lx - ar - 1.0 {
                self.cube(p(x, -0.45), ax, 0.4, 0.5, -1.0, 9.5, stone);
                self.cube(p(x, ly + 0.45), ax, 0.4, 0.5, -1.0, 9.5, stone);
                x += 5.5;
            }
            let z = 31.0 + tw * 2.7;
            let iron = with([0.3, 0.25, 0.1], mat::PAINT);
            self.cube(tc, ax, 0.08, 0.08, z - 0.2, z + 2.6, iron);
            self.cube(tc, ax, 0.7, 0.08, z + 1.5, z + 1.7, iron);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Sol : hauteurs du terrain deja calculees pour le chunk

/// Le relief du chunk tel qu'il est maille, pour poser rues et batiments exactement dessus.
pub struct Terrain<'a> {
    pub id: NodeId,
    /// Grille de (GRID + 3)^2 valeurs, une rangee de marge autour du chunk.
    pub heights: &'a [f32],
    pub levels: &'a [f32],
    pub gen: &'a Generator,
    pub detail: &'a Detail,
}

impl Terrain<'_> {
    /// Hauteur du sol et niveau de l'eau, interpoles sur les memes triangles que le maillage.
    pub fn at(&self, dir: DVec3) -> (f64, f64) {
        let side = GRID + 3;
        if let Some((s, t)) = self.id.locate(dir) {
            let n = GRID as f64;
            let (fx, fy) = (s * n + 1.0, t * n + 1.0);
            let max = (GRID + 2) as f64;
            if fx >= 0.0 && fy >= 0.0 && fx <= max && fy <= max {
                let (i, j) = ((fx.floor() as usize).min(GRID + 1), (fy.floor() as usize).min(GRID + 1));
                let (kx, ky) = (fx - i as f64, fy - j as f64);
                let interp = |a: &[f32]| {
                    let h = |di: usize, dj: usize| a[(j + dj) * side + i + di] as f64;
                    if kx > ky {
                        h(0, 0) + (h(1, 0) - h(0, 0)) * kx + (h(1, 1) - h(1, 0)) * ky
                    } else {
                        h(0, 0) + (h(0, 1) - h(0, 0)) * ky + (h(1, 1) - h(0, 1)) * kx
                    }
                };
                return (interp(self.heights), interp(self.levels));
            }
        }
        let s = self.gen.surface_detail(dir, self.detail);
        (s.height as f64, s.water_level as f64)
    }
}

/// Surelevation du tablier le long d'une rue, tous les `DECK_STEP` m ; vide si la rue reste au sol.
pub fn deck_profile(city: &City, gen: &Generator, r: &Road) -> Vec<f32> {
    let len = r.length();
    let k = (len / DECK_STEP).ceil().max(1.0) as usize;
    let detail = Detail::for_spacing(DECK_STEP);
    let samples: Vec<(f64, f64)> = (0..=k)
        .map(|i| {
            let s = (i as f64 * DECK_STEP).min(len);
            let sf = gen.surface_detail(city.dir_of(r.a + r.dir() * s), &detail);
            (sf.height as f64, sf.water_level as f64)
        })
        .collect();
    if !samples.iter().any(|&(h, l)| h < l + 0.3) {
        return Vec::new();
    }
    // Le tablier passe 4.5 m au-dessus de l'eau et rejoint le sol par des rampes a 7 %.
    let target: Vec<f64> = samples.iter().map(|&(h, l)| if h < l + 0.3 { l + 4.5 } else { f64::MIN }).collect();
    (0..=k)
        .map(|i| {
            let mut deck = samples[i].0;
            for (j, t) in target.iter().enumerate() {
                deck = deck.max(t - (i as f64 - j as f64).abs() * DECK_STEP * 0.07);
            }
            (deck - samples[i].0) as f32
        })
        .collect()
}

/// Le profil des ponts d'une rue, calcule une fois puis garde avec le plan.
pub fn road_deck(plan: &super::city::Plan, city: &City, gen: &Generator, ri: u32) -> Arc<Vec<f32>> {
    let cached = plan.decks.lock().unwrap().get(&ri).cloned();
    match cached {
        Some(d) => d,
        None => {
            let d = Arc::new(deck_profile(city, gen, &plan.roads[ri as usize]));
            plan.decks.lock().unwrap().insert(ri, d.clone());
            d
        }
    }
}

/// Surelevation du tablier a l'abscisse s.
pub fn deck_lift(profile: &[f32], s: f64) -> f64 {
    deck_at(profile, s)
}

fn deck_at(profile: &[f32], s: f64) -> f64 {
    if profile.is_empty() {
        return 0.0;
    }
    let x = (s / DECK_STEP).max(0.0);
    let i = (x.floor() as usize).min(profile.len() - 1);
    let j = (i + 1).min(profile.len() - 1);
    let t = x - i as f64;
    profile[i] as f64 * (1.0 - t) + profile[j] as f64 * t
}

// ---------------------------------------------------------------------------------------------
// Rues

struct RoadCtx<'a, 'b> {
    o: &'a mut Out,
    city: &'a City,
    terrain: &'a Terrain<'b>,
    tier: Tier,
}

impl RoadCtx<'_, '_> {
    fn ground(&self, q: DVec2) -> f64 {
        self.terrain.at(self.city.dir_of(q)).0
    }

    fn frame(&self, q: DVec2, base: f64) -> Frame {
        Frame::new(self.city, q, base, self.o.origin)
    }

    #[allow(clippy::too_many_arguments)]
    fn road(&mut self, r: &Road, ri: u32, deck: &[f32], own: &dyn Fn(DVec2) -> bool) {
        let len = r.length();
        if len < 0.5 {
            return;
        }
        let d = r.dir();
        let nrm = d.perp();
        let w2 = r.width as f64 * 0.5;
        let cw = w2 - r.walk as f64;
        let step = match self.tier {
            Tier::Full => 6.0,
            Tier::Mid => 12.0,
            Tier::Low => 24.0,
        };
        let (t0, t1) = (r.trim[0] as f64, len - r.trim[1] as f64);
        let (w0, w1) = (r.trim_walk[0] as f64, len - r.trim_walk[1] as f64);
        let mut breaks = vec![t0, t1, w0, w1];
        for &(at, half, _) in &r.gaps {
            breaks.push((at - half) as f64);
            breaks.push((at + half) as f64);
        }
        breaks.retain(|b| *b > 0.0 && *b < len);
        let walk_at = |s: f64, side: f64| r.walk > 0.0 && s >= w0 && s <= w1 && !r.in_gap_side(s as f32, side as f32);
        let lanes = if cw >= 9.0 { 2 } else if cw >= 4.5 { 1 } else { 0 };
        let asphalt = enc([0.045, 0.045, 0.05]);
        let city = self.city;
        let up_at = |s: f64| city.dir_of(r.a + d * s);

        let count = (len / step).ceil() as usize;
        for k in 0..count {
            let (s0, s1) = (k as f64 * step, ((k + 1) as f64 * step).min(len));
            if !own(r.a + d * ((s0 + s1) * 0.5)) {
                continue;
            }
            let mut cuts = vec![s0];
            cuts.extend(breaks.iter().copied().filter(|b| *b > s0 + 0.01 && *b < s1 - 0.01));
            cuts.push(s1);
            cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for w in cuts.windows(2) {
                let (sa, sb) = (w[0], w[1]);
                let m = (sa + sb) * 0.5;
                let (la, lb) = (deck_at(deck, sa), deck_at(deck, sb));
                let bridge = la.max(lb) > 0.25;
                if self.tier == Tier::Low && !bridge {
                    continue;
                }
                let (ha, hb) = (self.ground(r.a + d * sa) + la, self.ground(r.a + d * sb) + lb);
                let origin = self.o.origin;
                let p = |s: f64, h: f64, x: f64, z: f64| city.dir_of(r.a + d * s + nrm * x) * (planet_radius() + h + z) - origin;
                let up = up_at(m);
                let carriage = m >= t0 && m <= t1;
                if carriage {
                    let style = if r.in_gap(m as f32) { 0 } else { lanes };
                    let pt = Paint { color: asphalt, mat: [mat::ROAD, style, (cw * 4.0).min(255.0) as u8, (ri % 251) as u8] };
                    let z = 0.08;
                    self.o.poly(
                        &[p(sa, ha, -cw, z), p(sb, hb, -cw, z), p(sb, hb, cw, z), p(sa, ha, cw, z)],
                        up,
                        &[[-cw, sa], [-cw, sb], [cw, sb], [cw, sa]],
                        pt,
                    );
                }
                let walk_side = |side: f64| walk_at(m, side) && self.tier != Tier::Low;
                let walk = walk_side(-1.0) || walk_side(1.0);
                if walk {
                    let kerb = with([0.33, 0.33, 0.32], mat::KERB);
                    let slabs = with(if r.trees { [0.3, 0.27, 0.23] } else { [0.26, 0.25, 0.24] }, mat::WALK);
                    for side in [-1.0, 1.0] {
                        if !walk_side(side) {
                            continue;
                        }
                        let (x0, x1, x2) = (cw * side, (cw + 0.2) * side, w2 * side);
                        let n_in = (-nrm * side).extend_to(city, r.a + d * m);
                        self.o.poly(&[p(sa, ha, x0, 0.02), p(sb, hb, x0, 0.02), p(sb, hb, x0, 0.2), p(sa, ha, x0, 0.2)], n_in, &[[sa, 0.0], [sb, 0.0], [sb, 0.18], [sa, 0.18]], kerb);
                        self.o.poly(&[p(sa, ha, x0, 0.2), p(sb, hb, x0, 0.2), p(sb, hb, x1, 0.2), p(sa, ha, x1, 0.2)], up, &[[sa, 0.0], [sb, 0.0], [sb, 0.2], [sa, 0.2]], kerb);
                        self.o.poly(&[p(sa, ha, x1, 0.2), p(sb, hb, x1, 0.2), p(sb, hb, x2, 0.2), p(sa, ha, x2, 0.2)], up, &[[x1.abs(), sa], [x1.abs(), sb], [x2.abs(), sb], [x2.abs(), sa]], slabs);
                        let bottom = if bridge { -1.2 } else { -0.4 };
                        let n_out = (nrm * side).extend_to(city, r.a + d * m);
                        self.o.poly(&[p(sa, ha, x2, bottom), p(sb, hb, x2, bottom), p(sb, hb, x2, 0.2), p(sa, ha, x2, 0.2)], n_out, &[[sa, 0.0], [sb, 0.0], [sb, 1.0], [sa, 1.0]], kerb);
                        // Bout de trottoir au debouche d'une rue.
                        for (s, h, inside) in [(sa, ha, sa - 0.02), (sb, hb, sb + 0.02)] {
                            if !walk_at(inside, side) {
                                let dirn = if inside < s { -d } else { d };
                                let n = dirn.extend_to(city, r.a + d * s);
                                self.o.poly(&[p(s, h, x0, 0.02), p(s, h, x2, 0.02), p(s, h, x2, 0.2), p(s, h, x0, 0.2)], n, &[[0.0, 0.0], [1.0, 0.0], [1.0, 0.2], [0.0, 0.2]], kerb);
                            }
                        }
                    }
                }
                if bridge {
                    let concrete = plain([0.33, 0.32, 0.3]);
                    self.o.poly(&[p(sa, ha, -w2, -1.2), p(sb, hb, -w2, -1.2), p(sb, hb, w2, -1.2), p(sa, ha, w2, -1.2)], -up, &[[0.0, 0.0]; 4], concrete);
                    for side in [-1.0, 1.0] {
                        let (xo, xi) = (w2 * side, (w2 - 0.35) * side);
                        let n_out = (nrm * side).extend_to(city, r.a + d * m);
                        if !walk_side(side) {
                            self.o.poly(&[p(sa, ha, xo, -1.2), p(sb, hb, xo, -1.2), p(sb, hb, xo, 0.1), p(sa, ha, xo, 0.1)], n_out, &[[0.0, 0.0]; 4], concrete);
                            if !carriage {
                                self.o.poly(&[p(sa, ha, -w2, 0.1), p(sb, hb, -w2, 0.1), p(sb, hb, w2, 0.1), p(sa, ha, w2, 0.1)], up, &[[0.0, 0.0]; 4], concrete);
                            }
                        }
                        let zb = if walk_side(side) { 0.2 } else { 0.1 };
                        self.o.poly(&[p(sa, ha, xo, zb), p(sb, hb, xo, zb), p(sb, hb, xo, 1.25), p(sa, ha, xo, 1.25)], n_out, &[[0.0, 0.0]; 4], concrete);
                        self.o.poly(&[p(sa, ha, xi, zb), p(sb, hb, xi, zb), p(sb, hb, xi, 1.25), p(sa, ha, xi, 1.25)], -n_out, &[[0.0, 0.0]; 4], concrete);
                        self.o.poly(&[p(sa, ha, xi, 1.25), p(sb, hb, xi, 1.25), p(sb, hb, xo, 1.25), p(sa, ha, xo, 1.25)], up, &[[0.0, 0.0]; 4], concrete);
                    }
                }
            }
        }

        // Bouts de chaussee en biais : de la coupe droite jusqu'au bord de la rue croisee.
        for end in 0..2 {
            let (c0, c1) = (r.cut(end, -cw), r.cut(end, cw));
            let t = r.trim[end] as f64;
            if t < 0.5 || (c0 - c1).abs() < 0.05 {
                continue;
            }
            // Abscisses le long de la rue : la coupe droite (t) et le bord de la rue croisee (c).
            let at = |c: f64| if end == 0 { c } else { len - c };
            let (sl, sr, st) = (at(c0), at(c1), at(t));
            let mid = r.a + d * ((sl + sr + st) / 3.0);
            if !own(mid) {
                continue;
            }
            let h = |s: f64| self.ground(r.a + d * s) + deck_at(deck, s);
            if self.tier == Tier::Low && deck_at(deck, st) < 0.25 {
                continue;
            }
            let origin = self.o.origin;
            let p = |s: f64, x: f64| city.dir_of(r.a + d * s + nrm * x) * (planet_radius() + h(s) + 0.08) - origin;
            let pt = Paint { color: asphalt, mat: [mat::ROAD, 0, (cw * 4.0).min(255.0) as u8, (ri % 251) as u8] };
            let mut q = vec![(sl, -cw), (st, -cw), (st, cw), (sr, cw)];
            q.dedup_by(|a, b| (a.0 - b.0).abs() < 0.01 && a.1 == b.1);
            // Sens direct vu d'en haut, comme les quads de la chaussee.
            if end == 1 {
                q.reverse();
            }
            let pts: Vec<DVec3> = q.iter().map(|&(s, x)| p(s, x)).collect();
            let uv: Vec<[f64; 2]> = q.iter().map(|&(s, x)| [x, s]).collect();
            self.o.poly(&pts, up_at(st), &uv, pt);
        }

        // Piles des ponts.
        if !deck.is_empty() && self.tier != Tier::Low {
            for (k, lift) in deck.iter().enumerate() {
                let s = k as f64 * DECK_STEP;
                if k % 3 != 1 || *lift < 2.5 || s > len {
                    continue;
                }
                let q = r.a + d * s;
                if !own(q) {
                    continue;
                }
                let g = self.ground(q);
                let f = self.frame(q, g);
                let mut c = Ctx { o: &mut *self.o, f, rng: Rng(0), tier: self.tier, region: 0 };
                c.cube(q, d, 0.8, w2 * 0.7, -8.0, *lift as f64 - 1.2, plain([0.3, 0.29, 0.27]));
            }
        }

        if self.tier != Tier::Full || r.walk < 2.0 {
            return;
        }
        // Passages pietons et ligne d'arret aux bouts de la chaussee.
        for end in 0..2 {
            if r.trim[end] < 0.5 || cw < 2.5 {
                continue;
            }
            let (a, b) = if end == 0 { (t0 + 0.8, t0 + 4.3) } else { (t1 - 4.3, t1 - 0.8) };
            if a < 0.0 || b > len || a >= b {
                continue;
            }
            let mid = r.a + d * ((a + b) * 0.5);
            if !own(mid) {
                continue;
            }
            let (ha, hb) = (self.ground(r.a + d * a) + deck_at(deck, a), self.ground(r.a + d * b) + deck_at(deck, b));
            let origin = self.o.origin;
            let p = |s: f64, h: f64, x: f64| city.dir_of(r.a + d * s + nrm * x) * (planet_radius() + h + 0.1) - origin;
            let up = up_at((a + b) * 0.5);
            let white = with_param([0.6, 0.6, 0.58], mat::CROSSING, 0);
            self.o.poly(&[p(a, ha, -cw), p(b, hb, -cw), p(b, hb, cw), p(a, ha, cw)], up, &[[-cw, 0.0], [-cw, 3.5], [cw, 3.5], [cw, 0.0]], white);
            let (sa, sb) = if end == 0 { (b + 0.9, b + 1.3) } else { (a - 1.3, a - 0.9) };
            if sa > 0.0 && sb < len {
                let solid = with_param([0.6, 0.6, 0.58], mat::CROSSING, 1);
                let (h0, h1) = (self.ground(r.a + d * sa) + deck_at(deck, sa), self.ground(r.a + d * sb) + deck_at(deck, sb));
                // Ligne d'arret sur la voie de droite seulement.
                let (x0, x1) = if end == 0 { (-cw, 0.0) } else { (0.0, cw) };
                self.o.poly(&[p(sa, h0, x0), p(sb, h1, x0), p(sb, h1, x1), p(sa, h0, x1)], up, &[[0.0, 0.0]; 4], solid);
            }
        }

        // Lampadaires.
        let mut k = 0;
        let mut s = w0 + 9.0;
        while s < w1 - 4.0 {
            let sides: &[f64] = if w2 >= 12.0 { &[-1.0, 1.0] } else if k % 2 == 0 { &[1.0] } else { &[-1.0] };
            k += 1;
            let blocked = r.gaps.iter().any(|&(at, half, _)| (s - at as f64).abs() < half as f64 + 3.0);
            if !blocked {
                for &side in sides {
                    let q = r.a + d * s + nrm * (side * (cw + 0.55));
                    if !own(q) {
                        continue;
                    }
                    let base = self.ground(r.a + d * s) + deck_at(deck, s) + 0.2;
                    let f = self.frame(q, base);
                    let mut c = Ctx { o: &mut *self.o, f, rng: Rng(0), tier: self.tier, region: 0 };
                    let pole = with([0.06, 0.065, 0.07], mat::PAINT);
                    let toward = -nrm * side;
                    c.cylinder(q, 0.12, 0.07, 0.0, 7.6, 6, pole, None, false);
                    c.cube(q + toward * 0.8, toward, 0.85, 0.05, 7.35, 7.48, pole);
                    c.cube(q + toward * 1.5, toward, 0.38, 0.18, 7.15, 7.35, with([0.8, 0.75, 0.6], mat::LAMP));
                }
            }
            s += 26.0;
        }

        // Voitures garees le long des trottoirs.
        if cw < 3.4 {
            return;
        }
        let mut s = t0 + 7.0;
        let mut k = 0i64;
        while s < t1 - 7.0 {
            k += 1;
            let here = s;
            s += 6.2;
            if deck_at(deck, here) > 0.25 || r.gaps.iter().any(|&(at, half, _)| (here - at as f64).abs() < half as f64 + 4.5) {
                continue;
            }
            // Pas de voiture garee la ou l'autoroute debouche dans la rue.
            let at = r.a + d * here;
            if self.city.exits.iter().any(|&a| (DVec2::from_angle(a) * self.city.reach(a) - at).length() < 80.0) {
                continue;
            }
            for side in [-1.0f64, 1.0] {
                let roll = hash_seed(ri as u64 ^ 0xCA75, k, side as i64);
                if (roll % 100) >= 55 {
                    continue;
                }
                let q = r.a + d * here + nrm * (side * (cw - 1.1));
                if !own(q) {
                    continue;
                }
                let base = self.ground(r.a + d * here) + 0.08;
                let f = self.frame(q, base);
                let mut c = Ctx { o: &mut *self.o, f, rng: Rng(roll), tier: self.tier, region: 0 };
                c.car(q, d * -side);
            }
        }
    }
}

trait Extend2 {
    fn extend_to(self, city: &City, at: DVec2) -> DVec3;
}

impl Extend2 for DVec2 {
    /// Direction plane du plan de la ville, en 3D au point donne.
    fn extend_to(self, city: &City, at: DVec2) -> DVec3 {
        (city.dir_of(at + self) - city.dir_of(at - self)).normalize_or_zero()
    }
}

impl Ctx<'_> {
    fn car(&mut self, q: DVec2, ax: DVec2) {
        let color = self.pick(&CARS);
        let body = with(color, mat::PAINT);
        let ay = ax.perp();
        self.cube(q, ax, 2.2, 0.88, 0.32, 0.95, body);
        let cabin = rect(q - ax * 0.25, ax, 1.2, 0.8);
        self.walls(&cabin, 0.95, 1.45, with([0.02, 0.025, 0.03], mat::GLASS));
        self.cap(&cabin, 1.45, body);
        let tire = plain([0.015, 0.015, 0.015]);
        for (x, y) in [(1.35, 0.8), (1.35, -0.8), (-1.35, 0.8), (-1.35, -0.8)] {
            self.cube(q + ax * x + ay * y, ax, 0.33, 0.12, 0.0, 0.64, tire);
        }
    }

    fn fence(&mut self, fence: &Fence) {
        let (a, b) = (fence.a, fence.b);
        let len = (b - a).length();
        if len < 0.5 {
            return;
        }
        let ax = (b - a) / len;
        // Raccourcie aux bouts : pas de doublon aux coins.
        let (a, b) = (a + ax * 0.2, b - ax * 0.2);
        match fence.style {
            0 => self.cube((a + b) * 0.5, ax, len * 0.5 - 0.2, 0.35, -0.3, 1.35, with([0.05, 0.1, 0.035], mat::HEDGE)),
            1 => self.railing(a, b, 0.0, 1.05, with_param([0.6, 0.6, 0.58], mat::RAILING, 1)),
            _ => self.cube((a + b) * 0.5, ax, len * 0.5 - 0.2, 0.14, -0.3, 0.95, plain([0.4, 0.37, 0.32])),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Assemblage

/// Rues et batiments des villes sur ce chunk ; None s'il n'y en a pas.
pub fn build_city_mesh(gen: &Generator, id: NodeId, origin: DVec3, terrain: &Terrain) -> Option<CityMesh> {
    let (tier, min_height) = tier_for(id.lod)?;
    let (cdir, bound) = id.bounds();
    let mut out = Out::new(origin);
    for city in gen.cities.touching(cdir, bound) {
        if city.dir.dot(cdir) < 0.5 {
            continue;
        }
        let corners: Vec<DVec2> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)]
            .iter()
            .map(|&(s, t)| city.local(id.dir_at(s, t)))
            .collect();
        let (mut lo, mut hi) = (DVec2::splat(f64::MAX), DVec2::splat(f64::MIN));
        for c in &corners {
            lo = lo.min(*c);
            hi = hi.max(*c);
        }
        let reach = city.radius * 1.6;
        if lo.x > reach || lo.y > reach || hi.x < -reach || hi.y < -reach {
            continue;
        }
        let pad = (hi - lo).max_element() * 0.03 + 2.0;
        let (lo, hi) = (lo - DVec2::splat(pad), hi + DVec2::splat(pad));
        let plan = city.plan(gen);
        let own = |p: DVec2| id.locate(city.dir_of(p)).is_some_and(|(s, t)| (0.0..1.0).contains(&s) && (0.0..1.0).contains(&t));

        // Batiments : chacun appartient au chunk qui contient son centre.
        for bi in plan.buildings_in(lo - DVec2::splat(60.0), hi + DVec2::splat(60.0)) {
            let b = &plan.buildings[bi as usize];
            if b.height() < min_height || !own(b.center()) {
                continue;
            }
            let c = b.center();
            let base = b.foot.iter().map(|q| terrain.at(city.dir_of(*q)).0).fold(f64::MAX, f64::min);
            let f = Frame::new(city, c, base, origin);
            let mut ctx = Ctx { o: &mut out, f, rng: Rng(hash_seed(b.seed as u64, 0xB1D, 0)), tier, region: city.seed | 1 };
            ctx.building(b);
        }

        // Rues et ponts.
        if id.lod <= 5 {
            let road_tier = match id.lod {
                0 | 1 => Tier::Full,
                2 => Tier::Mid,
                _ => Tier::Low,
            };
            let roads = plan.roads_in(lo, hi);
            let mut rc = RoadCtx { o: &mut out, city, terrain, tier: road_tier };
            for ri in roads {
                let r = &plan.roads[ri as usize];
                let deck = road_deck(plan, city, gen, ri);
                rc.road(r, ri, &deck, &own);
            }
        }

        // Clotures et haies des jardins, sans doublon entre voisins.
        if tier != Tier::Low {
            let mut seen = HashSet::new();
            for fence in plan.fences_in(lo, hi) {
                let m = (fence.a + fence.b) * 0.5;
                let key = ((m.x * 4.0).round() as i64, (m.y * 4.0).round() as i64);
                if !seen.insert(key) || !own(m) {
                    continue;
                }
                let base = terrain.at(city.dir_of(m)).0;
                let f = Frame::new(city, m, base, origin);
                let mut ctx = Ctx { o: &mut out, f, rng: Rng(0), tier, region: 0 };
                ctx.fence(fence);
            }
        }
    }
    highway_mesh(&mut out, gen, id);
    circuit_mesh(&mut out, gen, id);
    if out.i.is_empty() {
        return None;
    }
    Some(CityMesh { vertices: out.v, indices: out.i, min: out.min, max: out.max, top: out.top })
}


// ---------------------------------------------------------------------------------------------
// Autoroutes

/// Une coupe en travers de l'autoroute.
#[derive(Clone, Copy)]
struct Station {
    s: f64,
    dir: DVec3,
    left: DVec3,
    fwd: DVec3,
    p: Profile,
}

fn station(hws: &Highways, gen: &Generator, h: usize, s: f64) -> Station {
    let hw = &hws.list[h];
    let dir = hw.dir_at(s);
    let fwd = hw.tangent_at(s);
    Station { s, dir, left: dir.cross(fwd).normalize(), fwd, p: hws.profile(gen, h, s) }
}

impl Out {
    /// Point d'une coupe : x vers la gauche, z au-dessus de la chaussee, d en avant.
    fn at(&self, st: &Station, x: f64, z: f64, d: f64) -> DVec3 {
        let r = planet_radius();
        (st.dir + st.left * (x / r) + st.fwd * (d / r)).normalize() * (r + st.p.road + z) - self.origin
    }

    /// Bande entre deux coupes, du point (x0, z0) au point (x1, z1).
    #[allow(clippy::too_many_arguments)]
    fn band(&mut self, a: &Station, b: &Station, x0: f64, z0: f64, x1: f64, z1: f64, uv: [[f64; 2]; 4], pt: Paint) {
        // Normale : perpendiculaire a la coupe, cote gauche du segment pour un profil parcouru de gauche a droite.
        let (ex, ez) = (x1 - x0, z1 - z0);
        // x croissant vers la gauche : parcourue de gauche a droite, la face regarde a gauche de son sens.
        let (nx, nz) = (ez, -ex);
        let len = (nx * nx + nz * nz).sqrt().max(1e-9);
        let n = (a.left * (nx / len) + a.dir * (nz / len)).normalize_or(a.dir);
        let p = [self.at(a, x0, z0, 0.0), self.at(b, x0, z0, 0.0), self.at(b, x1, z1, 0.0), self.at(a, x1, z1, 0.0)];
        self.poly(&p, n, &uv, pt);
    }

    /// Profil extrude entre deux coupes ; points de gauche a droite (x decroissant) par-dessus.
    fn profile_band(&mut self, a: &Station, b: &Station, pts: &[(f64, f64)], pt: Paint) {
        for w in pts.windows(2) {
            self.band(a, b, w[0].0, w[0].1, w[1].0, w[1].1, [[0.0, 0.0]; 4], pt);
        }
    }

    /// Pave pose sur une coupe : x de x0 a x1, d de d0 a d1, z de z0 a z1 (pas de face dessous).
    #[allow(clippy::too_many_arguments)]
    fn slab(&mut self, st: &Station, x0: f64, x1: f64, d0: f64, d1: f64, z0: f64, z1: f64, pt: Paint) {
        let c = |x: f64, d: f64, z: f64| self.at(st, x, z, d);
        let quads = [
            ([c(x0, d0, z1), c(x1, d0, z1), c(x1, d1, z1), c(x0, d1, z1)], st.dir),
            ([c(x0, d0, z0), c(x1, d0, z0), c(x1, d0, z1), c(x0, d0, z1)], -st.fwd),
            ([c(x0, d1, z0), c(x1, d1, z0), c(x1, d1, z1), c(x0, d1, z1)], st.fwd),
            ([c(x0, d0, z0), c(x0, d1, z0), c(x0, d1, z1), c(x0, d0, z1)], if x0 > x1 { st.left } else { -st.left }),
            ([c(x1, d0, z0), c(x1, d1, z0), c(x1, d1, z1), c(x1, d0, z1)], if x0 > x1 { -st.left } else { st.left }),
        ];
        for (q, n) in quads {
            self.poly(&q, n, &[[0.0, 0.0]; 4], pt);
        }
    }

    /// Un troncon d'autoroute entre deux coupes.
    fn highway_slice(&mut self, hw: &Highway, h: usize, a: &Station, b: &Station, tier: Tier) {
        let (med, cw, half) = (hw.median(), hw.carriage(), hw.half_width());
        let bridge = a.p.bridge() || b.p.bridge();
        let asphalt = enc([0.05, 0.05, 0.055]);
        for side in [-1.0, 1.0] {
            let (x0, x1) = (side * med, side * (med + cw));
            let pt = Paint { color: asphalt, mat: [mat::ROAD, 8 + hw.lanes, (cw * 4.0).min(255.0) as u8, (h % 251) as u8] };
            let (l, r) = if side > 0.0 { (x1, x0) } else { (x0, x1) };
            let (ul, ur) = if side > 0.0 { (cw, 0.0) } else { (0.0, cw) };
            self.band(a, b, l, 0.08, r, 0.08, [[ul, a.s], [ul, b.s], [ur, b.s], [ur, a.s]], pt);
            // Accotement de gravier, puis le flanc du remblai ou du tablier.
            let (g0, g1) = (side * (med + cw), side * half);
            let gravel = plain([0.2, 0.19, 0.17]);
            let (gl, gr) = if side > 0.0 { (g1, g0) } else { (g0, g1) };
            self.band(a, b, gl, 0.06, gr, 0.06, [[0.0, 0.0]; 4], gravel);
            let concrete = plain([0.36, 0.35, 0.33]);
            let top = if bridge && tier != Tier::Low { 1.05 } else { 0.06 };
            let bottom = if bridge { -1.7 } else { -1.6 };
            if side > 0.0 {
                self.band(a, b, g1, bottom, g1, top, [[0.0, 0.0]; 4], if bridge { concrete } else { gravel });
            } else {
                self.band(a, b, g1, top, g1, bottom, [[0.0, 0.0]; 4], if bridge { concrete } else { gravel });
            }
            if tier == Tier::Low {
                continue;
            }
            if bridge {
                // Parapet de beton.
                let xi = side * (half - 0.3);
                let pts = if side > 0.0 { [(g1, 1.05), (xi, 1.05), (xi, 0.06)] } else { [(xi, 0.06), (xi, 1.05), (g1, 1.05)] };
                self.profile_band(a, b, &pts, concrete);
            } else {
                // Glissiere de securite.
                let xr = side * (med + cw + 0.45);
                let steel = with([0.42, 0.43, 0.44], mat::PAINT);
                self.band(a, b, xr, 0.85, xr, 0.5, [[0.0, 0.0]; 4], steel);
                if tier == Tier::Full {
                    let k0 = (a.s / 4.0).ceil() as i64;
                    let k1 = (b.s / 4.0).ceil() as i64;
                    for k in k0..k1 {
                        let t = (k as f64 * 4.0 - a.s) / (b.s - a.s).max(1e-6);
                        let st = Station { s: k as f64 * 4.0, dir: a.dir.lerp(b.dir, t).normalize(), p: Profile { road: a.p.road + (b.p.road - a.p.road) * t, ..a.p }, ..*a };
                        self.slab(&st, xr + side * 0.18, xr + side * 0.06, -0.06, 0.06, 0.0, 0.8, steel);
                    }
                }
            }
        }
        // Terre-plein central.
        if tier == Tier::Low {
            self.band(a, b, med, 0.08, -med, 0.08, [[0.0, 0.0]; 4], Paint { color: asphalt, mat: [mat::PLAIN, 0, 0, 0] });
        } else {
            let verge = plain([0.13, 0.13, 0.12]);
            self.band(a, b, med, 0.07, 0.32, 0.07, [[0.0, 0.0]; 4], verge);
            self.band(a, b, -0.32, 0.07, -med, 0.07, [[0.0, 0.0]; 4], verge);
            // Separateur en beton, profil de Jersey.
            self.profile_band(a, b, &[(0.32, 0.07), (0.16, 0.36), (0.11, 0.86), (-0.11, 0.86), (-0.16, 0.36), (-0.32, 0.07)], plain([0.3, 0.295, 0.28]));
        }
        if bridge {
            self.band(a, b, -half, -1.7, half, -1.7, [[0.0, 0.0]; 4], plain([0.3, 0.29, 0.27]));
        }
    }

    /// Pile de pont sous une coupe : deux futs et un chevetre.
    fn pier(&mut self, hw: &Highway, st: &Station) {
        let depth = st.p.road - st.p.ground;
        if depth < 3.0 {
            return;
        }
        let concrete = plain([0.33, 0.32, 0.3]);
        let (med, cw, half) = (hw.median(), hw.carriage(), hw.half_width());
        self.slab(st, half - 0.6, -(half - 0.6), -0.9, 0.9, -2.7, -1.7, concrete);
        for side in [-1.0, 1.0] {
            let x = side * (med + cw * 0.5);
            self.slab(st, x + 0.8, x - 0.8, -0.8, 0.8, -depth - 3.0, -2.7, concrete);
        }
    }

    /// Portique de signalisation, au-dessus des deux chaussees.
    fn gantry(&mut self, hw: &Highway, st: &Station) {
        let (med, cw, half) = (hw.median(), hw.carriage(), hw.half_width());
        let steel = with([0.5, 0.51, 0.52], mat::PAINT);
        for side in [-1.0, 1.0] {
            let x = side * (half - 0.5);
            self.slab(st, x + 0.22, x - 0.22, -0.22, 0.22, 0.0, 7.6, steel);
            // Un panneau vert par sens, accroche a la poutre.
            let (p0, p1) = (side * (med + 1.2), side * (med + cw - 2.8));
            let (hi, lo) = if side > 0.0 { (p1, p0) } else { (p0, p1) };
            self.slab(st, hi, lo, 0.25, 0.4, 4.9, 7.1, with([0.015, 0.14, 0.05], mat::PAINT));
            self.slab(st, hi - 0.2 * side.signum().abs(), lo + 0.2 * side.signum().abs(), 0.4, 0.42, 5.1, 6.9, plain([0.55, 0.55, 0.52]));
        }
        self.slab(st, half - 0.5, -(half - 0.5), -0.25, 0.25, 7.0, 7.6, steel);
    }
}

/// Les autoroutes qui traversent un chunk.
fn highway_mesh(out: &mut Out, gen: &Generator, id: NodeId) {
    let hws = &gen.cities.highways;
    if hws.list.is_empty() || id.lod > 6 {
        return;
    }
    let tier = match id.lod {
        0 | 1 => Tier::Full,
        2 | 3 => Tier::Mid,
        _ => Tier::Low,
    };
    let step = match tier {
        Tier::Full => 10.0,
        Tier::Mid => 25.0,
        Tier::Low => 50.0,
    };
    // Les mailles de l'index qui touchent le chunk.
    let n = if id.lod >= 4 { 6 } else { 2 };
    let mut segs: HashSet<(u32, u32)> = HashSet::new();
    for i in 0..=n {
        for j in 0..=n {
            let d = id.dir_at(i as f64 / n as f64, j as f64 / n as f64);
            for &(h, f, l) in hws.runs_at(d) {
                segs.extend((f..=l).map(|k| (h, k)));
            }
        }
    }
    let own = |d: DVec3| id.locate(d).is_some_and(|(s, t)| (0.0..1.0).contains(&s) && (0.0..1.0).contains(&t));
    let mut segs: Vec<(u32, u32)> = segs.into_iter().collect();
    segs.sort_unstable();
    for (h, k) in segs {
        let (h, hw) = (h as usize, &hws.list[h as usize]);
        let s0 = k as f64 * POINT_STEP;
        let s1 = (s0 + POINT_STEP).min(hw.length());
        if s1 <= s0 {
            continue;
        }
        let count = ((s1 - s0) / step).ceil().max(1.0) as usize;
        let mut prev: Option<Station> = None;
        for q in 0..=count {
            let s = s0 + (s1 - s0) * q as f64 / count as f64;
            let st = station(hws, gen, h, s);
            if let Some(a) = prev {
                if own(hw.dir_at((a.s + st.s) * 0.5)) {
                    out.highway_slice(hw, h, &a, &st, tier);
                    if tier != Tier::Low {
                        // Piles tous les 40 m sous les ponts, portiques tous les 5 km.
                        for (every, gantry) in [(40.0, false), (5000.0, true)] {
                            let m = (a.s / every).ceil() * every;
                            if m < st.s && m >= a.s && m > 300.0 && m < hw.length() - 300.0 {
                                let at = station(hws, gen, h, m);
                                if gantry && !at.p.bridge() {
                                    out.gantry(hw, &at);
                                } else if !gantry && at.p.bridge() {
                                    out.pier(hw, &at);
                                }
                            }
                        }
                    }
                }
            }
            prev = Some(st);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Circuits

/// Les circuits qui traversent un chunk : piste, vibreurs, glissieres, ligne d'arrivee, grille, tribune.
fn circuit_mesh(out: &mut Out, gen: &Generator, id: NodeId) {
    use super::circuit::RUNOFF;
    let circuits = &gen.cities.circuits;
    if circuits.list.is_empty() || id.lod > 6 {
        return;
    }
    let tier = match id.lod {
        0 | 1 => Tier::Full,
        2 | 3 => Tier::Mid,
        _ => Tier::Low,
    };
    let stride = match tier {
        Tier::Full => 1,
        Tier::Mid => 2,
        Tier::Low => 4,
    };
    let cells = (id.size() / 250.0).ceil().clamp(2.0, 96.0) as usize;
    let mut segs: HashSet<(u32, u8, u32)> = HashSet::new();
    for i in 0..=cells {
        for j in 0..=cells {
            let d = id.dir_at(i as f64 / cells as f64, j as f64 / cells as f64);
            for &(c, p, k) in circuits.segments_at(d) {
                segs.insert((c, p, k - k % stride as u32));
            }
        }
    }
    let own = |d: DVec3| id.locate(d).is_some_and(|(s, t)| (0.0..1.0).contains(&s) && (0.0..1.0).contains(&t));
    let mut segs: Vec<_> = segs.into_iter().collect();
    segs.sort_unstable();
    for (c, p, k) in segs {
        let circuit = &circuits.list[c as usize];
        let path = &circuit.paths[p as usize];
        let (k, n) = (k as usize, path.points.len());
        let k1 = if k + stride >= n {
            if path.closed { 0 } else { n - 1 }
        } else {
            k + stride
        };
        if k1 == k || !own((path.points[k] + path.points[k1]).normalize()) {
            continue;
        }
        let levels = &circuit.levels(gen)[p as usize];
        let station = |q: usize| {
            let dir = path.points[q];
            let fwd = path.tangent(q);
            let s = if q == 0 && k != 0 { path.length() } else { path.s[q] };
            Station { s, dir, left: dir.cross(fwd).normalize(), fwd, p: Profile { road: levels[q], ground: levels[q], wet: false } }
        };
        let (a, b) = (station(k), station(k1));
        let half = path.half;
        let asphalt = enc([0.042, 0.042, 0.047]);
        if p != 0 {
            // Route d'acces : deux voies, axe en tirets, lignes de rive.
            let pt = Paint { color: asphalt, mat: [mat::ROAD, 1, (half * 4.0) as u8, 0] };
            out.band(&a, &b, half, 0.06, -half, 0.06, [[half, a.s], [half, b.s], [-half, b.s], [-half, a.s]], pt);
            continue;
        }

        // Piste : asphalte fin et lignes blanches de bord de piste (style 7 dans city.wgsl).
        let pt = Paint { color: asphalt, mat: [mat::ROAD, 7, (half * 4.0) as u8, 0] };
        out.band(&a, &b, half, 0.07, -half, 0.07, [[half, a.s], [half, b.s], [-half, b.s], [-half, a.s]], pt);
        if tier == Tier::Low {
            continue;
        }

        // Vibreurs rouges et blancs dans les virages, des deux cotes.
        if path.curvature(k).abs() > 1.0 / 200.0 {
            let stripes = ((b.s - a.s) / 2.0).round().max(1.0) as usize;
            for q in 0..stripes {
                let at = |t: f64| Station {
                    s: a.s + (b.s - a.s) * t,
                    dir: a.dir.lerp(b.dir, t).normalize(),
                    left: a.left.lerp(b.left, t).normalize(),
                    fwd: a.fwd,
                    p: Profile { road: a.p.road + (b.p.road - a.p.road) * t, ..a.p },
                };
                let (sa, sb) = (at(q as f64 / stripes as f64), at((q + 1) as f64 / stripes as f64));
                let red = ((sa.s / 2.0).round() as i64) % 2 == 0;
                let col = if red { [0.5, 0.02, 0.02] } else { [0.62, 0.62, 0.6] };
                for side in [-1.0, 1.0] {
                    let (x0, x1) = (side * half, side * (half + 1.3));
                    let (l, r) = if side > 0.0 { (x1, x0) } else { (x0, x1) };
                    out.band(&sa, &sb, l, 0.12, r, 0.12, [[0.0, 0.0]; 4], with(col, mat::PAINT));
                }
            }
        }

        // Glissieres au bout des degagements, sauf la ou arrive la route d'acces.
        let behind = (n - k) % n;
        let steel = with([0.45, 0.46, 0.47], mat::PAINT);
        for side in [-1.0f64, 1.0] {
            let open = circuit.junction.is_some_and(|j| {
                let d = (j as i64 - k as i64).rem_euclid(n as i64);
                let toward = circuit.paths[1].points[0] - a.dir;
                d.min(n as i64 - d) < 7 && toward.dot(a.left) * side > 0.0
            });
            // Les stands et la tribune, a gauche de la ligne droite : pas de glissiere.
            let pits = side > 0.0 && (behind <= 18 || k <= 10);
            if open || pits {
                continue;
            }
            let x = side * (half + RUNOFF);
            out.band(&a, &b, x, 0.95, x, 0.45, [[0.0, 0.0]; 4], steel);
            out.band(&a, &b, x, 0.45, x, 0.95, [[0.0, 0.0]; 4], steel);
        }

        // Ligne d'arrivee en damier.
        if k == 0 && tier == Tier::Full {
            let cols = 14;
            for row in 0..2 {
                for q in 0..cols {
                    let x0 = half - 2.0 * half * q as f64 / cols as f64;
                    let x1 = half - 2.0 * half * (q + 1) as f64 / cols as f64;
                    let col = if (q + row) % 2 == 0 { [0.7, 0.7, 0.68] } else { [0.015, 0.015, 0.015] };
                    out.slab(&a, x0, x1, row as f64, row as f64 + 1.0, 0.07, 0.09, with(col, mat::PAINT));
                }
            }
        }
        // Grille de depart : une case tous les 8 m derriere la ligne, en quinconce.
        if tier == Tier::Full && (1..=12).contains(&behind) {
            let side = if behind % 2 == 0 { 1.0 } else { -1.0 };
            let x = side * half * 0.45;
            out.slab(&a, x + 2.0, x - 2.0, 0.0, 0.25, 0.07, 0.09, with([0.7, 0.7, 0.68], mat::PAINT));
        }
        // Tribune couverte le long de la ligne droite des stands, a gauche ; muret des stands a droite.
        if behind <= 18 || k <= 10 {
            let concrete = plain([0.4, 0.39, 0.37]);
            let seats = with([0.05, 0.12, 0.35], mat::PAINT);
            let x0 = half + 14.0;
            let rows = 8;
            // Profil des gradins, de l'exterieur vers la piste (x decroissant) : la face regarde la piste.
            let mut pts: Vec<(f64, f64)> = Vec::new();
            for q in (0..=rows).rev() {
                let x = x0 + q as f64 * 1.6;
                let z = 1.0 + q as f64 * 1.1;
                pts.push((x, z));
                pts.push((x - 1.6, z));
            }
            pts.pop();
            pts.push((x0, 0.0));
            out.profile_band(&a, &b, &pts, seats);
            let back = x0 + rows as f64 * 1.6;
            out.band(&a, &b, back, -1.0, back, 10.0, [[0.0, 0.0]; 4], concrete);
            out.band(&a, &b, back, 10.0, back, -1.0, [[0.0, 0.0]; 4], concrete);
            out.band(&a, &b, back + 0.5, 14.2, x0 - 2.0, 13.0, [[0.0, 0.0]; 4], plain([0.55, 0.55, 0.56]));
            out.band(&a, &b, x0 - 2.0, 12.8, back + 0.5, 14.0, [[0.0, 0.0]; 4], plain([0.3, 0.3, 0.31]));
            let wall = -(half + 1.5);
            out.band(&a, &b, wall, 0.0, wall, 1.2, [[0.0, 0.0]; 4], plain([0.62, 0.62, 0.6]));
            out.band(&a, &b, wall, 1.2, wall, 0.0, [[0.0, 0.0]; 4], plain([0.62, 0.62, 0.6]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shrink_and_chamfer_keep_convex_shapes_inside() {
        let sq = rect(DVec2::ZERO, DVec2::X, 10.0, 10.0);
        let s = shrink(&sq, 1.0);
        assert!(s.iter().all(|p| p.x.abs() <= 9.0 + 1e-9 && p.y.abs() <= 9.0 + 1e-9));
        let o = chamfer(&sq, 3.0);
        assert_eq!(o.len(), 8);
        assert!(o.iter().all(|p| p.x.abs() <= 10.0 + 1e-9 && p.y.abs() <= 10.0 + 1e-9));
    }

    #[test]
    fn city_chunks_carry_valid_meshes_resting_on_the_ground() {
        use crate::world::chunk::build_chunk;
        use crate::world::generator::PlanetParams;
        let gen = Generator::new(PlanetParams { seed: 3, ..Default::default() });
        let city = &gen.cities.list[0];
        for (lod, offset) in [(1u8, DVec2::new(800.0, 300.0)), (3, DVec2::ZERO), (6, DVec2::ZERO)] {
            let dir = city.dir_of(offset);
            let mesh = build_chunk(&gen, NodeId::containing(dir, lod));
            let c = mesh.city.as_ref().unwrap_or_else(|| panic!("LOD {lod} : pas de ville"));
            assert!(c.indices.len() > 300, "LOD {lod} : {} indices", c.indices.len());
            assert!(c.indices.iter().all(|&i| (i as usize) < c.vertices.len()));
            // Les batiments se posent sur le relief du chunk, a sa finesse.
            let detail = Detail::for_spacing(NodeId::containing(dir, lod).spacing());
            for v in c.vertices.iter().step_by(37) {
                let p = mesh.center + DVec3::from_array(v.pos.map(|x| x as f64));
                let ground = gen.surface_detail(p.normalize(), &detail).height as f64;
                // Pieds des murs, pentes sous les grandes emprises, piles de ponts : un peu sous le sol.
                let alt = p.length() - planet_radius() - ground;
                assert!((-30.0..800.0).contains(&alt), "LOD {lod} : sommet a {alt} m du sol, materiau {:?}, {:?}", v.mat, crate::world::generator::dir_to_lat_lon(p.normalize()));
                assert!(v.normal.iter().all(|x| x.is_finite()));
            }
            // Le chunk englobe ses batiments : sans cela, ils seraient elimines a tort.
            let top = c.vertices.iter().map(|v| v.pos[1]).fold(f32::MIN, f32::max);
            assert!(mesh.aabb_max[1] >= top);
        }
    }

    #[test]
    fn far_chunks_keep_only_the_skyline() {
        assert_eq!(tier_for(0), Some((Tier::Full, 0.0)));
        assert!(tier_for(6).unwrap().1 >= 30.0);
        assert_eq!(tier_for(9), None);
    }

    #[test]
    fn bays_match_the_shader_table() {
        // city.wgsl : style_params().x
        for (style, b) in [(0u8, 3.0), (1, 3.2), (2, 1.6), (3, 1.5), (4, 2.6), (5, 6.0), (6, 3.4), (7, 5.0)] {
            assert_eq!(bay(style | win::SHOP), b);
        }
    }
}
