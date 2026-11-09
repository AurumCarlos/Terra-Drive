//! Le modele de la voiture, genere par code : une berline sportive de 4.62 m.
//!
//! Repere local : x vers la droite de la voiture, y vers le haut, z vers l'avant ; l'origine est
//! au sol, a mi-chemin des essieux. La carrosserie est une surface lissee (sections successives),
//! avec passages de roues ; le shader y dessine au pixel les joints de portes, poignees, vitres,
//! montants, phares, feux, calandre, plaques et diffuseur, a partir des coordonnees locales.
//! Les roues (pneu sculpte, jante a branches, disque, etrier), les retroviseurs, essuie-glaces,
//! echappements, antenne et tout l'habitacle (planche de bord, compteurs, ecran, volant, sieges,
//! console) sont des volumes a part.

use crate::world::city_mesh::CityVertex;
use glam::{Mat3, Vec3};
use std::f32::consts::{PI, TAU};
use std::ops::Range;

/// Materiaux propres a la voiture, lus par city.wgsl.
pub mod mat {
    /// Carrosserie : details au pixel. color = (x, y) locaux sur 16 bits, uv = (z, t) ;
    /// mat.y = teinte (4 bits) | 0x10 piece sans details ; mat.z = feux.
    pub const CAR: u8 = 18;
    /// Pneu : mat.y = 0 bande de roulement, 1 flanc.
    pub const TIRE: u8 = 19;
    /// Metal : mat.y = 0 chrome, 1 jante usinee, 2 disque de frein, 3 jante graphite.
    pub const CHROME: u8 = 20;
    /// Habitacle : mat.y = 0 cuir, 1 plastique, 2 compteurs, 3 ecran, 4 moquette, 5 alu ; mat.w = vitesse.
    pub const CABIN: u8 = 21;
    pub const GLASS: u8 = 8;
    pub const PAINT: u8 = 17;
}

/// Feux et etat, dans mat.z de la carrosserie.
pub mod lights {
    pub const HEAD: u8 = 1;
    pub const BRAKE: u8 = 2;
    pub const REVERSE: u8 = 4;
    pub const LEFT: u8 = 8;
    pub const RIGHT: u8 = 16;
    /// La camera est dans l'habitacle : les vitres laissent voir dehors.
    pub const COCKPIT: u8 = 32;
}

pub const LENGTH: f32 = 4.62;
pub const FRONT_AXLE: f32 = 1.40;
pub const REAR_AXLE: f32 = -1.36;
pub const WHEELBASE: f32 = FRONT_AXLE - REAR_AXLE;
pub const WHEEL_RADIUS: f32 = 0.337;
pub const TRACK: f32 = 0.80;
const ARCH: f32 = 0.405;

/// Centres des moyeux : avant gauche, avant droit, arriere gauche, arriere droit.
pub const WHEELS: [Vec3; 4] = [
    Vec3::new(-TRACK, WHEEL_RADIUS, FRONT_AXLE),
    Vec3::new(TRACK, WHEEL_RADIUS, FRONT_AXLE),
    Vec3::new(-TRACK, WHEEL_RADIUS, REAR_AXLE),
    Vec3::new(TRACK, WHEEL_RADIUS, REAR_AXLE),
];

/// Le volant : centre et axe de la colonne (vers le conducteur).
pub const STEERING_CENTER: Vec3 = Vec3::new(-0.37, 0.92, 0.40);
pub fn steering_axis() -> Vec3 {
    Vec3::new(0.0, 0.42, -0.9).normalize()
}

/// Les yeux du conducteur.
pub const DRIVER_EYE: Vec3 = Vec3::new(-0.37, 1.20, -0.32);

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Part {
    Body,
    /// Tourne avec la roue (pneu, jante, disque).
    Wheel(usize),
    /// Suit le braquage sans tourner (etrier).
    Hub(usize),
    Steering,
}

pub struct CarModel {
    /// Pieces mobiles : sommets relatifs a leur pivot (moyeu ou volant).
    pub vertices: Vec<CityVertex>,
    pub indices: Vec<u32>,
    pub parts: Vec<(Part, Range<usize>)>,
}

// ---------------------------------------------------------------------------------------------
// Construction

fn srgb(c: [f32; 3]) -> [u8; 4] {
    let e = |v: f32| (v.clamp(0.0, 1.0).sqrt() * 255.0 + 0.5) as u8;
    [e(c[0]), e(c[1]), e(c[2]), 255]
}

/// Position locale (x, y) sur 16 bits chacune, de -3 a 3 m.
fn enc_xy(x: f32, y: f32) -> [u8; 4] {
    let q = |v: f32| (((v + 3.0) / 6.0).clamp(0.0, 1.0) * 65535.0).round() as u32;
    let (a, b) = (q(x), q(y));
    [(a >> 8) as u8, (a & 255) as u8, (b >> 8) as u8, (b & 255) as u8]
}

#[derive(Clone, Copy)]
struct Paint {
    color: [u8; 4],
    mat: [u8; 4],
    /// Carrosserie : couleur calculee depuis la position.
    body: bool,
}

fn paint(c: [f32; 3], m: u8, p: u8) -> Paint {
    Paint { color: srgb(c), mat: [m, p, 0, 0], body: false }
}

struct B {
    v: Vec<CityVertex>,
    i: Vec<u32>,
    parts: Vec<(Part, Range<usize>)>,
    palette: u8,
}

impl B {
    fn push(&mut self, p: Vec3, n: Vec3, uv: [f32; 2], pt: Paint) -> u32 {
        let color = if pt.body { enc_xy(p.x, p.y) } else { pt.color };
        self.v.push(CityVertex { pos: p.to_array(), normal: n.normalize_or_zero().to_array(), uv, color, mat: pt.mat });
        (self.v.len() - 1) as u32
    }

    fn body_paint(&self, plain: bool) -> Paint {
        Paint { color: [0; 4], mat: [mat::CAR, self.palette | if plain { 0x10 } else { 0 }, 0, 0], body: true }
    }

    fn begin(&mut self) -> usize {
        self.v.len()
    }

    fn end(&mut self, part: Part, start: usize) {
        if self.v.len() > start {
            self.parts.push((part, start..self.v.len()));
        }
    }

    /// Surface en grille : rangees de points, colonnes eventuellement refermees. Normales lissees,
    /// orientees par `out` (une direction vers l'exterieur en chaque point).
    fn grid(&mut self, rows: &[Vec<Vec3>], wrap: bool, out: &dyn Fn(Vec3) -> Vec3, style: &dyn Fn(usize, usize, Vec3) -> ([f32; 2], Paint)) {
        let (nr, nc) = (rows.len(), rows[0].len());
        let base = self.v.len() as u32;
        for r in 0..nr {
            for c in 0..nc {
                let p = rows[r][c];
                let cm = if c > 0 { c - 1 } else if wrap { nc - 1 } else { c };
                let cp = if c + 1 < nc { c + 1 } else if wrap { 0 } else { c };
                let (rm, rp) = (r.saturating_sub(1), (r + 1).min(nr - 1));
                let du = rows[r][cp] - rows[r][cm];
                let dv = rows[rp][c] - rows[rm][c];
                let mut n = du.cross(dv);
                if n.length_squared() < 1e-12 {
                    n = out(p);
                }
                if n.dot(out(p)) < 0.0 {
                    n = -n;
                }
                let (uv, pt) = style(r, c, p);
                self.push(p, n, uv, pt);
            }
        }
        let cols = if wrap { nc } else { nc - 1 };
        for r in 0..nr - 1 {
            for c in 0..cols {
                let c1 = (c + 1) % nc;
                let a = base + (r * nc + c) as u32;
                let b = base + (r * nc + c1) as u32;
                let d = base + ((r + 1) * nc + c) as u32;
                let e = base + ((r + 1) * nc + c1) as u32;
                self.i.extend_from_slice(&[a, b, e, a, e, d]);
            }
        }
    }

    /// Polygone plat (convexe), en eventail.
    fn fan(&mut self, pts: &[Vec3], n: Vec3, style: &dyn Fn(Vec3) -> ([f32; 2], Paint)) {
        let base = self.v.len() as u32;
        for p in pts {
            let (uv, pt) = style(*p);
            self.push(*p, n, uv, pt);
        }
        for k in 1..pts.len() as u32 - 1 {
            self.i.extend_from_slice(&[base, base + k, base + k + 1]);
        }
    }

    /// Superellipsoide : un volume arrondi (exposant 1 : ellipsoide ; vers 0 : boite aux coins ronds).
    fn blob(&mut self, center: Vec3, radii: Vec3, e: f32, rot: Mat3, pt: Paint, uv: [f32; 2]) {
        let f = |c: f32, e: f32| c.signum() * c.abs().powf(e);
        let (nu, nv) = (14, 24);
        let rows: Vec<Vec<Vec3>> = (0..=nu)
            .map(|i| {
                let u = -PI * 0.5 + PI * (i as f32 / nu as f32).clamp(0.002, 0.998);
                (0..nv)
                    .map(|j| {
                        let v = -PI + TAU * j as f32 / nv as f32;
                        let p = Vec3::new(radii.x * f(u.cos(), e) * f(v.cos(), e), radii.y * f(u.sin(), e), radii.z * f(u.cos(), e) * f(v.sin(), e));
                        center + rot * p
                    })
                    .collect()
            })
            .collect();
        self.grid(&rows, true, &|p| p - center, &|_, _, _| (uv, pt));
    }

    /// Volume de revolution : profil (rayon, position sur l'axe), autour de `axis` passant par `center`.
    fn revolve(&mut self, center: Vec3, axis: Vec3, profile: &[(f32, f32)], segs: usize, style: &dyn Fn(usize, f32, f32, f32) -> ([f32; 2], Paint)) {
        let axis = axis.normalize();
        let e1 = axis.any_orthonormal_vector();
        let e2 = axis.cross(e1);
        let rows: Vec<Vec<Vec3>> = profile
            .iter()
            .map(|&(r, a)| {
                (0..segs)
                    .map(|j| {
                        let t = TAU * j as f32 / segs as f32;
                        center + axis * a + (e1 * t.cos() + e2 * t.sin()) * r
                    })
                    .collect()
            })
            .collect();
        // Vers l'exterieur : en s'eloignant de l'axe, et sur les faces planes, du cote du profil.
        let mid = profile.iter().map(|p| p.1).sum::<f32>() / profile.len() as f32;
        let out = move |p: Vec3| {
            let q = p - center;
            let along = q.dot(axis);
            let radial = q - axis * along;
            radial.normalize_or_zero() + axis * (along - mid).signum() * 0.3
        };
        let base_rows = rows.len();
        let _ = base_rows;
        self.grid(&rows, true, &out, &|r, c, p| {
            let t = TAU * c as f32 / segs as f32;
            let (rad, a) = profile[r];
            let _ = p;
            style(r, t, rad, a)
        });
    }

    /// Boite orientee (centre, demi-tailles, rotation), faces plates.
    fn cube(&mut self, c: Vec3, h: Vec3, rot: Mat3, pt: Paint, uv: [f32; 2]) {
        for (n, u, v) in [(Vec3::X, Vec3::Y, Vec3::Z), (-Vec3::X, Vec3::Z, Vec3::Y), (Vec3::Y, Vec3::Z, Vec3::X), (-Vec3::Y, Vec3::X, Vec3::Z), (Vec3::Z, Vec3::X, Vec3::Y), (-Vec3::Z, Vec3::Y, Vec3::X)] {
            let (n, u, v) = (n * h, u * h, v * h);
            let pts = [c + rot * (n - u - v), c + rot * (n + u - v), c + rot * (n + u + v), c + rot * (n - u + v)];
            self.fan(&pts, rot * n.normalize(), &|_| (uv, pt));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Carrosserie

/// Interpolation cubique monotone entre des cles (z, valeur).
fn curve(keys: &[(f32, f32)], z: f32) -> f32 {
    let n = keys.len();
    if z <= keys[0].0 {
        return keys[0].1;
    }
    if z >= keys[n - 1].0 {
        return keys[n - 1].1;
    }
    let k = keys.windows(2).position(|w| z <= w[1].0).unwrap();
    let slope = |i: usize| (keys[i + 1].1 - keys[i].1) / (keys[i + 1].0 - keys[i].0);
    let tangent = |i: usize| {
        if i == 0 {
            slope(0)
        } else if i == n - 1 {
            slope(n - 2)
        } else {
            let (a, b) = (slope(i - 1), slope(i));
            if a * b <= 0.0 { 0.0 } else { 2.0 / (1.0 / a + 1.0 / b) }
        }
    };
    let (z0, z1) = (keys[k].0, keys[k + 1].0);
    let h = z1 - z0;
    let t = (z - z0) / h;
    let (y0, y1, m0, m1) = (keys[k].1, keys[k + 1].1, tangent(k) * h, tangent(k + 1) * h);
    let (t2, t3) = (t * t, t * t * t);
    (2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * m0 + (-2.0 * t3 + 3.0 * t2) * y1 + (t3 - t2) * m1
}

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

const HALF: f32 = LENGTH * 0.5;
/// Ligne de toit : capot, pare-brise, pavillon, lunette, coffre.
const TOP: [(f32, f32); 16] = [
    (-HALF, 0.80),
    (-2.27, 0.93),
    (-2.15, 0.995),
    (-1.85, 1.025),
    (-1.72, 1.06),
    (-1.06, 1.36),
    (-0.75, 1.415),
    (-0.30, 1.435),
    (-0.04, 1.41),
    (0.22, 1.32),
    (0.96, 0.925),
    (1.10, 0.895),
    (1.80, 0.80),
    (2.12, 0.715),
    (2.26, 0.64),
    (HALF, 0.53),
];
/// Ligne de caisse (bas des vitres) ; sur le capot et le coffre, la carrosserie s'y bombe.
const BELT: [(f32, f32); 11] = [(-HALF, 0.78), (-2.25, 0.90), (-2.12, 0.965), (-1.75, 0.99), (-1.0, 0.985), (0.0, 0.955), (0.95, 0.905), (1.1, 0.875), (1.8, 0.775), (2.2, 0.66), (HALF, 0.52)];
/// Demi-largeur.
const WIDTH: [(f32, f32); 9] = [(-HALF, 0.74), (-2.26, 0.84), (-2.08, 0.905), (-1.6, 0.925), (0.0, 0.925), (1.6, 0.92), (2.05, 0.885), (2.22, 0.80), (HALF, 0.66)];
/// Bas de caisse.
const BOTTOM: [(f32, f32); 8] = [(-HALF, 0.40), (-2.25, 0.31), (-2.0, 0.235), (-1.8, 0.185), (1.8, 0.185), (2.1, 0.22), (2.25, 0.265), (HALF, 0.33)];
/// Parametre t des points de la demi-section (0 dessous, 0.5 ligne de caisse, 1 sommet).
const T: [f32; 12] = [0.0, 0.08, 0.12, 0.18, 0.26, 0.36, 0.44, 0.5, 0.62, 0.74, 0.87, 1.0];

/// Demi-section droite a l'abscisse z : (x, y, t).
fn half_section(z: f32) -> Vec<(f32, f32, f32)> {
    let w0 = curve(&WIDTH, z);
    let yb = curve(&BOTTOM, z);
    let top = curve(&TOP, z);
    let belt = curve(&BELT, z).min(top - 0.03);
    let cab = smooth(-1.85, -1.6, z) * (1.0 - smooth(0.85, 1.1, z));
    let wt = 0.62 * w0 + (0.70 - 0.62 * w0) * cab;
    let mut arch: Option<f32> = None;
    let mut flare = 0.0f32;
    for axle in [FRONT_AXLE, REAR_AXLE] {
        let dz = z - axle;
        if dz.abs() < ARCH {
            arch = Some(WHEEL_RADIUS + (ARCH * ARCH - dz * dz).sqrt());
        }
        flare += 0.024 * (1.0 - smooth(0.32, 0.8, dz.abs()));
    }
    let w = w0 + flare;
    let a = arch.unwrap_or(yb + 0.005).max(yb + 0.005);
    let p: [(f32, f32); 12] = {
        let p3 = (w - 0.05, a.max(yb + 0.08));
        let p4 = (w, (a + 0.04).max(yb + 0.28));
        let p5 = (w + flare * 0.3, (belt - 0.25).max(p4.1 + 0.03));
        let p6 = (w - 0.015, (belt - 0.07).max(p5.1 + 0.02));
        let p7 = (w - 0.06, belt.max(p6.1 + 0.01));
        let h = (top - p7.1).max(0.02);
        let p8 = (p7.0 - (p7.0 - wt) * 0.45, p7.1 + h * 0.45);
        let p9 = (wt + 0.02, top - (h * 0.25).min(0.06));
        let p10 = ((wt - 0.1).max(0.05), top - 0.006);
        [(0.0, yb), (w0 - 0.30, yb), (w0 - 0.30, a), p3, p4, p5, p6, p7, p8, p9, p10, (0.0, top)]
    };
    // Le passage de roue reste anguleux ; le reste est lisse (Catmull-Rom).
    let mut out = Vec::new();
    for k in 0..11 {
        let subs = if k < 3 { 2 } else { 4 };
        for s in 0..subs {
            let u = s as f32 / subs as f32;
            let (x, y) = if k < 3 {
                (p[k].0 + (p[k + 1].0 - p[k].0) * u, p[k].1 + (p[k + 1].1 - p[k].1) * u)
            } else {
                let q = |i: usize| p[i.clamp(3, 11)];
                let (a0, a1, a2, a3) = (q(k - 1), q(k), q(k + 1), q(k + 2));
                let cr = |c0: f32, c1: f32, c2: f32, c3: f32| {
                    0.5 * (2.0 * c1 + (-c0 + c2) * u + (2.0 * c0 - 5.0 * c1 + 4.0 * c2 - c3) * u * u + (-c0 + 3.0 * c1 - 3.0 * c2 + c3) * u * u * u)
                };
                (cr(a0.0, a1.0, a2.0, a3.0), cr(a0.1, a1.1, a2.1, a3.1))
            };
            out.push((x.max(0.0), y, T[k] + (T[k + 1] - T[k]) * u));
        }
    }
    out.push((0.0, p[11].1, 1.0));
    out
}

fn body(b: &mut B) {
    let n = 160;
    let stations: Vec<f32> = (0..=n).map(|i| -HALF + LENGTH * i as f32 / n as f32).collect();
    let mut rows: Vec<Vec<Vec3>> = Vec::new();
    let mut ts: Vec<f32> = Vec::new();
    for &z in &stations {
        let half = half_section(z);
        if ts.is_empty() {
            ts = half.iter().map(|h| h.2).chain(half.iter().rev().skip(1).take(half.len() - 2).map(|h| h.2)).collect();
        }
        let ring: Vec<Vec3> = half
            .iter()
            .map(|h| Vec3::new(h.0, h.1, z))
            .chain(half.iter().rev().skip(1).take(half.len() - 2).map(|h| Vec3::new(-h.0, h.1, z)))
            .collect();
        rows.push(ring);
    }
    let pt = b.body_paint(false);
    let ts2 = ts.clone();
    b.grid(&rows, true, &|p| Vec3::new(p.x, p.y - 0.65, p.z * 0.25), &|_, c, p| ([p.z, ts2[c]], pt));
    // Les deux bouts : le bouclier avant et la jupe arriere.
    for (row, n) in [(rows.last().unwrap().clone(), Vec3::Z), (rows[0].clone(), -Vec3::Z)] {
        let c = row.iter().copied().sum::<Vec3>() / row.len() as f32;
        let base = b.v.len() as u32;
        b.push(c, n, [c.z, 0.3], pt);
        for p in &row {
            b.push(*p, n, [p.z, 0.3], pt);
        }
        let m = row.len() as u32;
        for k in 0..m {
            b.i.extend_from_slice(&[base, base + 1 + k, base + 1 + (k + 1) % m]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Roues

fn wheel(b: &mut B, index: usize) {
    let side = if index % 2 == 1 { 1.0 } else { -1.0 };
    let axis = Vec3::X * side;
    let rubber = [0.03, 0.03, 0.032];
    let start = b.begin();
    // Pneu 235/40 R19 : flancs bombes, epaulements arrondis.
    let tire: [(f32, f32); 13] = [
        (0.243, -0.104),
        (0.265, -0.116),
        (0.300, -0.118),
        (0.322, -0.110),
        (0.333, -0.095),
        (0.336, -0.06),
        (WHEEL_RADIUS, 0.0),
        (0.336, 0.06),
        (0.333, 0.095),
        (0.322, 0.110),
        (0.300, 0.118),
        (0.265, 0.116),
        (0.243, 0.104),
    ];
    b.revolve(Vec3::ZERO, axis, &tire, 72, &|r, t, rad, a| {
        let flank = !(4..=8).contains(&r);
        if flank {
            ([t * 0.3, rad], paint(rubber, mat::TIRE, 1))
        } else {
            ([t * WHEEL_RADIUS, a], paint(rubber, mat::TIRE, 0))
        }
    });
    // Jante : rebord, fond, voile a dix branches, moyeu, ecrous, cache central.
    let rim = paint([0.62, 0.62, 0.64], mat::CHROME, 1);
    let lip: [(f32, f32); 4] = [(0.243, 0.104), (0.248, 0.098), (0.240, 0.088), (0.228, 0.086)];
    b.revolve(Vec3::ZERO, axis, &lip, 72, &|_, t, rad, _| ([t, rad], rim));
    let barrel: [(f32, f32); 3] = [(0.228, 0.086), (0.229, 0.0), (0.236, -0.104)];
    b.revolve(Vec3::ZERO, axis, &barrel, 48, &|_, t, rad, _| ([t, rad], paint([0.18, 0.18, 0.19], mat::CHROME, 3)));
    for k in 0..10 {
        let ang = TAU * k as f32 / 10.0 + if k % 2 == 1 { -0.13 } else { 0.0 } + (k / 2) as f32 * 0.0;
        let pair = (k / 2) as f32 * TAU / 5.0 + if k % 2 == 1 { 0.16 } else { -0.16 };
        let _ = ang;
        // Chaque branche part du moyeu et s'ouvre vers le rebord, legerement creusee.
        let dir = Vec3::new(0.0, pair.cos(), pair.sin());
        let along = (0.07 + 0.225) * 0.5;
        let rot = Mat3::from_cols(Vec3::X, dir, Vec3::X.cross(dir));
        let c = dir * along + axis * 0.083;
        b.cube(c, Vec3::new(0.014, 0.078, 0.013), rot * Mat3::from_rotation_x(if k % 2 == 1 { -0.12 } else { 0.12 }), rim, [0.0, along]);
    }
    let hub: [(f32, f32); 4] = [(0.0, 0.092), (0.035, 0.092), (0.078, 0.084), (0.08, 0.05)];
    b.revolve(Vec3::ZERO, axis, &hub, 36, &|r, t, rad, _| ([t, rad], if r < 2 { paint([0.05, 0.05, 0.055], mat::CHROME, 3) } else { rim }));
    for k in 0..5 {
        let a = TAU * k as f32 / 5.0 + 0.3;
        let c = Vec3::new(0.0, a.cos() * 0.055, a.sin() * 0.055);
        let nut: [(f32, f32); 3] = [(0.0, 0.097), (0.009, 0.097), (0.011, 0.086)];
        b.revolve(c, axis, &nut, 6, &|_, t, rad, _| ([t, rad], paint([0.7, 0.7, 0.72], mat::CHROME, 0)));
    }
    // Disque de frein ventile.
    let disc: [(f32, f32); 6] = [(0.09, 0.012), (0.19, 0.012), (0.192, 0.0), (0.19, -0.012), (0.09, -0.012), (0.088, 0.0)];
    b.revolve(Vec3::ZERO, axis, &disc, 48, &|_, t, rad, _| ([t, rad], paint([0.3, 0.29, 0.28], mat::CHROME, 2)));
    b.end(Part::Wheel(index), start);

    // Etrier rouge, en haut vers l'interieur de la voiture : il ne tourne pas avec la roue.
    let start = b.begin();
    let back = if index < 2 { -1.0 } else { 1.0 };
    let a = 0.45f32 * back;
    let c = Vec3::new(0.012 * side, a.cos() * 0.175, a.sin() * 0.175);
    let rot = Mat3::from_rotation_x(-a);
    b.blob(c, Vec3::new(0.04, 0.03, 0.075), 0.35, rot, paint([0.55, 0.02, 0.015], mat::PAINT, 0), [0.0, 0.0]);
    b.end(Part::Hub(index), start);
}

// ---------------------------------------------------------------------------------------------
// Accessoires exterieurs

fn exterior(b: &mut B) {
    let body = b.body_paint(true);
    let black = paint([0.02, 0.02, 0.022], mat::PAINT, 0);
    let matte = paint([0.025, 0.025, 0.027], mat::CABIN, 1);
    let start = b.begin();
    for s in [-1.0f32, 1.0] {
        // Retroviseur : pied, coque, miroir.
        let foot = Vec3::new(s * 0.86, 0.95, 0.78);
        b.blob(foot, Vec3::new(0.05, 0.025, 0.06), 0.5, Mat3::IDENTITY, black, [0.0, 0.0]);
        let shell = Vec3::new(s * 1.0, 1.0, 0.74);
        b.blob(shell, Vec3::new(0.13, 0.075, 0.07), 0.45, Mat3::from_rotation_y(s * 0.12), body, [0.74, 0.3]);
        let glass = Vec3::new(s * 1.0, 1.0, 0.668);
        b.fan(
            &[glass + Vec3::new(-0.115, -0.06, 0.0), glass + Vec3::new(0.115, -0.06, 0.0), glass + Vec3::new(0.115, 0.06, 0.0), glass + Vec3::new(-0.115, 0.06, 0.0)],
            -Vec3::Z,
            &|_| ([0.0, 0.0], paint([0.35, 0.37, 0.4], mat::GLASS, 0)),
        );
        // Echappement : deux sorties ovales en inox.
        let tip = Vec3::new(s * 0.52, 0.29, -HALF + 0.02);
        let pipe: [(f32, f32); 5] = [(0.0, -0.02), (0.035, -0.02), (0.045, 0.0), (0.045, 0.12), (0.038, 0.12)];
        b.revolve(tip, Vec3::Z, &pipe, 20, &|r, t, rad, _| ([t, rad], if r < 2 { black } else { paint([0.75, 0.74, 0.72], mat::CHROME, 0) }));
    }
    // Essuie-glaces au pied du pare-brise.
    for (x0, x1) in [(-0.72, 0.02), (0.06, 0.62)] {
        let a = Vec3::new(x0, 0.935, 0.93);
        let c = Vec3::new(x1, 0.975, 0.86);
        let d = (c - a).normalize();
        let rot = Mat3::from_cols(d, d.cross(Vec3::new(0.0, 0.45, 0.9).normalize()).normalize(), Vec3::new(0.0, 0.45, 0.9).normalize());
        b.cube((a + c) * 0.5, Vec3::new((c - a).length() * 0.5, 0.008, 0.006), rot, matte, [0.0, 0.0]);
    }
    // Antenne en aileron de requin.
    b.blob(Vec3::new(0.0, 1.38, -0.92), Vec3::new(0.03, 0.06, 0.1), 0.8, Mat3::IDENTITY, body, [0.0, 0.3]);
    // Becquet sur le bord du coffre.
    b.blob(Vec3::new(0.0, 1.02, -2.2), Vec3::new(0.68, 0.018, 0.06), 0.3, Mat3::from_rotation_x(-0.15), body, [-2.2, 0.3]);
    // Lame d'aero sous le bouclier avant.
    b.blob(Vec3::new(0.0, 0.17, 2.18), Vec3::new(0.72, 0.012, 0.09), 0.25, Mat3::IDENTITY, black, [0.0, 0.0]);
    b.end(Part::Body, start);
}

// ---------------------------------------------------------------------------------------------
// Habitacle

fn cabin(b: &mut B) {
    let leather = paint([0.05, 0.035, 0.03], mat::CABIN, 0);
    let plastic = paint([0.03, 0.03, 0.032], mat::CABIN, 1);
    let carpet = paint([0.03, 0.03, 0.03], mat::CABIN, 4);
    let alu = paint([0.5, 0.5, 0.52], mat::CABIN, 5);
    let start = b.begin();
    // Plancher.
    b.fan(&[Vec3::new(-0.82, 0.24, -1.7), Vec3::new(0.82, 0.24, -1.7), Vec3::new(0.82, 0.24, 0.95), Vec3::new(-0.82, 0.24, 0.95)], Vec3::Y, &|p| ([p.x, p.z], carpet));
    // Planche de bord, casquette des compteurs, bandeau alu.
    b.blob(Vec3::new(0.0, 0.84, 0.82), Vec3::new(0.86, 0.15, 0.24), 0.28, Mat3::IDENTITY, plastic, [0.0, 0.0]);
    b.blob(Vec3::new(-0.37, 1.0, 0.6), Vec3::new(0.21, 0.025, 0.1), 0.4, Mat3::IDENTITY, plastic, [0.0, 0.0]);
    b.blob(Vec3::new(0.0, 0.83, 0.585), Vec3::new(0.8, 0.012, 0.012), 0.4, Mat3::IDENTITY, alu, [0.0, 0.0]);
    // Compteurs et ecran central : quads emissifs, uv sur [0, 1].
    let quad = |b: &mut B, c: Vec3, w: f32, h: f32, tilt: f32, pt: Paint| {
        let rot = Mat3::from_rotation_x(tilt);
        let (u, v) = (rot * Vec3::X * w, rot * Vec3::Y * h);
        let n = rot * -Vec3::Z;
        let pts = [c - u - v, c + u - v, c + u + v, c - u + v];
        let base = b.v.len() as u32;
        for (k, p) in pts.iter().enumerate() {
            let uv = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]][k];
            b.push(*p, n, uv, pt);
        }
        b.i.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };
    quad(b, Vec3::new(-0.37, 0.915, 0.555), 0.17, 0.055, -0.35, paint([0.0, 0.0, 0.0], mat::CABIN, 2));
    quad(b, Vec3::new(0.0, 0.89, 0.56), 0.14, 0.085, -0.25, paint([0.0, 0.0, 0.0], mat::CABIN, 3));
    // Console centrale et levier.
    b.blob(Vec3::new(0.0, 0.47, 0.0), Vec3::new(0.11, 0.2, 0.6), 0.3, Mat3::IDENTITY, plastic, [0.0, 0.0]);
    b.blob(Vec3::new(0.0, 0.70, 0.12), Vec3::new(0.025, 0.04, 0.03), 0.6, Mat3::IDENTITY, alu, [0.0, 0.0]);
    // Sieges avant baquets, banquette arriere.
    for s in [-1.0f32, 1.0] {
        let x = s * 0.37;
        b.blob(Vec3::new(x, 0.42, -0.12), Vec3::new(0.25, 0.075, 0.26), 0.35, Mat3::IDENTITY, leather, [0.0, 0.0]);
        b.blob(Vec3::new(x - s * 0.2, 0.5, -0.12), Vec3::new(0.05, 0.09, 0.25), 0.5, Mat3::IDENTITY, leather, [0.0, 0.0]);
        b.blob(Vec3::new(x + s * 0.2, 0.5, -0.12), Vec3::new(0.05, 0.09, 0.25), 0.5, Mat3::IDENTITY, leather, [0.0, 0.0]);
        b.blob(Vec3::new(x, 0.78, -0.43), Vec3::new(0.25, 0.33, 0.075), 0.35, Mat3::from_rotation_x(-0.2), leather, [0.0, 0.0]);
        b.blob(Vec3::new(x, 1.17, -0.53), Vec3::new(0.13, 0.09, 0.05), 0.4, Mat3::from_rotation_x(-0.15), leather, [0.0, 0.0]);
        b.blob(Vec3::new(s * 0.36, 1.03, -1.36), Vec3::new(0.12, 0.08, 0.045), 0.4, Mat3::from_rotation_x(-0.25), leather, [0.0, 0.0]);
    }
    b.blob(Vec3::new(0.0, 0.43, -1.1), Vec3::new(0.66, 0.08, 0.26), 0.3, Mat3::IDENTITY, leather, [0.0, 0.0]);
    b.blob(Vec3::new(0.0, 0.76, -1.38), Vec3::new(0.66, 0.3, 0.075), 0.3, Mat3::from_rotation_x(-0.25), leather, [0.0, 0.0]);
    // Plage arriere, retroviseur interieur.
    b.fan(&[Vec3::new(-0.74, 1.0, -1.5), Vec3::new(0.74, 1.0, -1.5), Vec3::new(0.7, 1.0, -1.78), Vec3::new(-0.7, 1.0, -1.78)], Vec3::Y, &|p| ([p.x, p.z], carpet));
    b.blob(Vec3::new(0.0, 1.29, 0.12), Vec3::new(0.12, 0.035, 0.015), 0.4, Mat3::IDENTITY, plastic, [0.0, 0.0]);
    // Colonne de direction.
    let axis = steering_axis();
    b.revolve(STEERING_CENTER + axis * 0.06, -axis, &[(0.0, 0.0), (0.04, 0.0), (0.045, 0.25), (0.0, 0.25)], 16, &|_, _, _, _| ([0.0, 0.0], plastic));
    b.end(Part::Body, start);

    // Volant : jante gainee, trois branches, coussin central.
    let start = b.begin();
    let rim: Vec<(f32, f32)> = (0..=12).map(|k| {
        let a = TAU * k as f32 / 12.0;
        (0.185 + 0.021 * a.cos(), 0.021 * a.sin())
    }).collect();
    b.revolve(Vec3::ZERO, axis, &rim, 48, &|_, _, _, _| ([0.0, 0.0], leather));
    let e1 = Vec3::X;
    let e2 = axis.cross(e1).normalize();
    for (a, len) in [(0.0f32, 0.17f32), (PI, 0.17), (PI * 0.5, 0.17)] {
        let d = e1 * a.cos() + e2 * a.sin();
        let rot = Mat3::from_cols(d, axis, d.cross(axis));
        b.cube(d * len * 0.5 + axis * -0.01, Vec3::new(len * 0.5, 0.012, 0.022), rot, plastic, [0.0, 0.0]);
    }
    let pad = Mat3::from_cols(e1, e2, axis);
    b.blob(axis * -0.02, Vec3::new(0.075, 0.06, 0.035), 0.5, pad, plastic, [0.0, 0.0]);
    b.blob(axis * -0.05 - e2 * 0.0, Vec3::new(0.02, 0.012, 0.004), 0.8, pad, alu, [0.0, 0.0]);
    b.end(Part::Steering, start);
}

impl CarModel {
    /// La voiture, peinte de la teinte `palette` (0 a 15, voir city.wgsl).
    pub fn new(palette: u8) -> Self {
        let mut b = B { v: Vec::new(), i: Vec::new(), parts: Vec::new(), palette: palette & 15 };
        let start = b.begin();
        body(&mut b);
        b.end(Part::Body, start);
        exterior(&mut b);
        cabin(&mut b);
        for k in 0..4 {
            wheel(&mut b, k);
        }
        CarModel { vertices: b.v, indices: b.i, parts: b.parts }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_body_sections_keep_their_order() {
        for i in 0..=100 {
            let z = -HALF + LENGTH * i as f32 / 100.0;
            let s = half_section(z);
            assert_eq!(s.len(), 39);
            assert!(s.iter().all(|p| p.0.is_finite() && p.1.is_finite()));
            assert!(s.windows(2).all(|w| w[1].2 >= w[0].2), "t croissant a z = {z}");
            let top = s.last().unwrap().1;
            assert!(top > s[0].1, "le dessus est au-dessus du dessous a z = {z}");
        }
    }

    #[test]
    fn the_model_is_detailed_and_every_part_is_known() {
        let m = CarModel::new(0);
        assert!(m.vertices.len() > 20_000, "{} sommets", m.vertices.len());
        assert!(m.indices.iter().all(|&i| (i as usize) < m.vertices.len()));
        let covered: usize = m.parts.iter().map(|p| p.1.len()).sum();
        assert_eq!(covered, m.vertices.len(), "chaque sommet appartient a une piece");
        for k in 0..4 {
            assert!(m.parts.iter().any(|p| p.0 == Part::Wheel(k)));
        }
        assert!(m.parts.iter().any(|p| p.0 == Part::Steering));
    }

    #[test]
    fn local_positions_round_trip_through_the_color() {
        let c = enc_xy(-0.913, 1.272);
        let dec = |hi: u8, lo: u8| (hi as f32 * 256.0 + lo as f32) / 65535.0 * 6.0 - 3.0;
        assert!((dec(c[0], c[1]) + 0.913).abs() < 1e-3);
        assert!((dec(c[2], c[3]) - 1.272).abs() < 1e-3);
    }
}
