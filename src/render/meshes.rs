//! Maillages low-poly des objets, generes par code : aucun fichier externe a charger.
//! Chaque type a une version proche et une version lointaine, plus legere.

use crate::world::chunk::encode_color;
use crate::world::noise::Rng;
use crate::world::scatter::KIND_COUNT;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::ops::Range;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ObjectVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    /// rgb encode en racine carree, a = sensibilite au vent.
    pub color: [u8; 4],
}

#[derive(Default)]
struct Builder {
    verts: Vec<ObjectVertex>,
}

impl Builder {
    /// Triangle a normale plate orientee a l'oppose de `center`.
    fn tri(&mut self, a: Vec3, b: Vec3, c: Vec3, center: Vec3, color: [f32; 3], sway: f32) {
        let mut n = (b - a).cross(c - a).normalize_or_zero();
        if n.dot((a + b + c) / 3.0 - center) < 0.0 {
            n = -n;
        }
        self.push_tri([a, b, c], n, [color; 3], [sway; 3]);
    }

    fn push_tri(&mut self, p: [Vec3; 3], n: Vec3, colors: [[f32; 3]; 3], sway: [f32; 3]) {
        for i in 0..3 {
            let mut color = encode_color(colors[i]);
            color[3] = (sway[i].clamp(0.0, 1.0) * 255.0) as u8;
            self.verts.push(ObjectVertex { pos: p[i].to_array(), normal: n.to_array(), color });
        }
    }

    /// Triangle a normales et couleurs par sommet.
    fn smooth_tri(&mut self, p: [Vec3; 3], n: [Vec3; 3], colors: [[f32; 3]; 3], sway: f32) {
        for i in 0..3 {
            let mut color = encode_color(colors[i]);
            color[3] = (sway.clamp(0.0, 1.0) * 255.0) as u8;
            self.verts.push(ObjectVertex { pos: p[i].to_array(), normal: n[i].to_array(), color });
        }
    }

    /// Cone lisse, plus sombre en bas : le bas d'un feuillage recoit moins de lumiere.
    fn cone(&mut self, y0: f32, y1: f32, r0: f32, r1: f32, sides: u32, color: [f32; 3], sway: f32) {
        let tilt = (r0 - r1) / (y1 - y0);
        let at = |i: u32, y: f32, r: f32| {
            let a = i as f32 / sides as f32 * std::f32::consts::TAU;
            Vec3::new(a.cos() * r, y, a.sin() * r)
        };
        let normal = |i: f32| {
            let a = i / sides as f32 * std::f32::consts::TAU;
            Vec3::new(a.cos(), tilt, a.sin()).normalize()
        };
        let (low, high) = (color.map(|c| c * 0.68), color.map(|c| c * 1.12));
        let under = color.map(|c| c * 0.4);
        for i in 0..sides {
            let (a, b) = (at(i, y0, r0), at(i + 1, y0, r0));
            let (c, d) = (at(i, y1, r1), at(i + 1, y1, r1));
            let (na, nb) = (normal(i as f32), normal(i as f32 + 1.0));
            if r1 > 0.0 {
                self.smooth_tri([a, b, d], [na, nb, nb], [low, low, high], sway);
                self.smooth_tri([a, d, c], [na, nb, na], [low, high, high], sway);
            } else {
                let top = normal(i as f32 + 0.5);
                self.smooth_tri([a, b, d], [na, nb, top], [low, low, high], sway);
            }
            // Dessous, visible sous les etages de sapin.
            let down = Vec3::NEG_Y;
            self.smooth_tri([a, b, Vec3::new(0.0, y0, 0.0)], [down; 3], [under; 3], sway);
        }
    }

    /// Etage de sapin : une jupe en etoile dont les pointes retombent, sombre vers le tronc.
    fn fir_tier(&mut self, y0: f32, y1: f32, r: f32, branches: u32, turn: f32, color: [f32; 3], sway: f32) {
        let count = branches * 2;
        let at = |k: u32| {
            let a = turn + k as f32 / count as f32 * std::f32::consts::TAU;
            let (radius, y) = if k.is_multiple_of(2) { (r, y0) } else { (r * 0.5, y0 + (y1 - y0) * 0.16) };
            Vec3::new(a.cos() * radius, y, a.sin() * radius)
        };
        let tilt = r / (y1 - y0);
        let normal = |p: Vec3| {
            Vec3::new(p.x, 0.0, p.z).normalize_or_zero().lerp(Vec3::Y * tilt.min(1.0), 0.35).normalize_or_zero()
        };
        let apex = Vec3::new(0.0, y1, 0.0);
        let (tip, notch, top) = (color.map(|c| c * 1.25), color.map(|c| c * 0.55), color.map(|c| c * 0.9));
        let under = color.map(|c| c * 0.28);
        let center = Vec3::new(0.0, y0 + (y1 - y0) * 0.22, 0.0);
        for k in 0..count {
            let (a, c) = (at(k), at(k + 1));
            let shade = |p: u32| if p.is_multiple_of(2) { tip } else { notch };
            self.smooth_tri([a, c, apex], [normal(a), normal(c), Vec3::Y], [shade(k), shade(k + 1), top], sway);
            self.smooth_tri([a, c, center], [Vec3::NEG_Y; 3], [under; 3], sway);
        }
    }

    /// Icosaedre deforme, subdivise `subdiv` fois.
    fn blob(&mut self, c: Vec3, r: Vec3, subdiv: u32, jitter: f32, seed: u64, color: [f32; 3], sway: f32) {
        let t = (1.0 + 5f32.sqrt()) * 0.5;
        let v = [
            Vec3::new(-1.0, t, 0.0),
            Vec3::new(1.0, t, 0.0),
            Vec3::new(-1.0, -t, 0.0),
            Vec3::new(1.0, -t, 0.0),
            Vec3::new(0.0, -1.0, t),
            Vec3::new(0.0, 1.0, t),
            Vec3::new(0.0, -1.0, -t),
            Vec3::new(0.0, 1.0, -t),
            Vec3::new(t, 0.0, -1.0),
            Vec3::new(t, 0.0, 1.0),
            Vec3::new(-t, 0.0, -1.0),
            Vec3::new(-t, 0.0, 1.0),
        ]
        .map(|p| p.normalize());
        const F: [[usize; 3]; 20] = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        // Les rochers gardent leurs facettes.
        let facet = if sway > 0.0 { 0.0 } else { 0.85 };
        let mut tris: Vec<[Vec3; 3]> = F.iter().map(|f| [v[f[0]], v[f[1]], v[f[2]]]).collect();
        for _ in 0..subdiv {
            let mut next = Vec::with_capacity(tris.len() * 4);
            for [a, b, c] in tris {
                let (ab, bc, ca) = ((a + b).normalize(), (b + c).normalize(), (c + a).normalize());
                next.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
            }
            tris = next;
        }
        // Le deplacement depend de la position pour que les sommets partages restent soudes.
        let displace = |p: Vec3| {
            let q = (p * 1000.0).round();
            let key = (q.x as i64 as u64).wrapping_mul(73856093)
                ^ (q.y as i64 as u64).wrapping_mul(19349663)
                ^ (q.z as i64 as u64).wrapping_mul(83492791);
            let k = Rng(seed ^ key).f32();
            c + p * r * (1.0 + jitter * (k - 0.5))
        };
        // Normales rondes et ombre vers le bas : un feuillage, pas une pierre taillee.
        let height = |p: Vec3| ((p.y - c.y) / r.y * 0.5 + 0.5).clamp(0.0, 1.0);
        for [a, b, tc] in tris {
            let (pa, pb, pc) = (displace(a), displace(b), displace(tc));
            let mut flat = (pb - pa).cross(pc - pa).normalize_or_zero();
            if flat.dot((pa + pb + pc) / 3.0 - c) < 0.0 {
                flat = -flat;
            }
            let normal =
                |p: Vec3| (((p - c) / r).normalize_or_zero() * (1.0 - facet) + flat * facet).normalize_or_zero();
            let shade = |p: Vec3| color.map(|v| v * (0.6 + 0.62 * height(p)));
            self.smooth_tri(
                [pa, pb, pc],
                [normal(pa), normal(pb), normal(pc)],
                [shade(pa), shade(pb), shade(pc)],
                sway,
            );
        }
    }

    fn take(&mut self, out: &mut Vec<ObjectVertex>) -> Range<u32> {
        let start = out.len() as u32;
        out.append(&mut self.verts);
        start..out.len() as u32
    }
}

const BARK: [f32; 3] = [0.14, 0.095, 0.06];
const PINE: [f32; 3] = [0.024, 0.052, 0.032];
const LEAF: [f32; 3] = [0.06, 0.105, 0.032];
const BIRCH_LEAF: [f32; 3] = [0.125, 0.155, 0.045];

/// Vent presque nul au loin, mais assez pour etre traite comme du feuillage.
const FAR_SWAY: f32 = 0.08;

/// `lod` : 0 proche, 1 lointain, 2 tres lointain (une seule forme, sans tronc).
fn pine(b: &mut Builder, lod: u8) {
    if lod == 2 {
        b.cone(1.5, 15.0, 3.2, 0.0, 4, PINE, FAR_SWAY);
        return;
    }
    let far = lod > 0;
    b.cone(0.0, 15.0, 0.34, 0.05, if far { 3 } else { 6 }, BARK, 0.0);
    // Une silhouette dentelee : des etages de branches qui retombent, plus serres vers la cime.
    let (tiers, branches) = if far { (4, 5) } else { (10, 8) };
    for i in 0..tiers {
        let k = i as f32 / (tiers - 1) as f32;
        let y0 = 2.4 + 12.6 * k.powf(1.1);
        let height = 5.4 - 2.6 * k;
        let radius = 3.5 * (1.0 - k).powf(0.85) + 0.32;
        let shade = 0.82 + 0.5 * k;
        b.fir_tier(y0, y0 + height, radius, branches, i as f32 * 1.7, PINE.map(|c| c * shade), 0.08 + k * 0.4);
    }
}

fn oak(b: &mut Builder, lod: u8) {
    if lod == 2 {
        b.blob(Vec3::new(0.0, 6.5, 0.0), Vec3::new(4.4, 4.8, 4.4), 0, 0.2, 1, LEAF, FAR_SWAY);
        return;
    }
    let far = lod > 0;
    b.cone(0.0, 5.2, 0.48, 0.28, if far { 3 } else { 7 }, BARK, 0.0);
    if far {
        b.blob(Vec3::new(0.0, 7.6, 0.0), Vec3::new(4.4, 3.7, 4.4), 0, 0.2, 1, LEAF, 0.2);
        return;
    }
    // Un houppier fait de nombreux paquets de feuilles, plus clairs au sommet.
    let mut rng = Rng(11);
    for i in 0..24 {
        let k = i as f32 / 23.0;
        let a = i as f32 * 2.399963;
        let radius = 4.3 * (1.0 - k * k).sqrt().max(0.15) * rng.range(0.55, 1.0);
        let c = Vec3::new(a.cos() * radius, 5.6 + 5.0 * k + rng.range(-0.3, 0.3), a.sin() * radius);
        let r = rng.range(1.2, 1.9);
        let shade = 0.7 + 0.5 * k;
        b.blob(c, Vec3::new(r, r * 0.8, r), 1, 0.5, 11 + i, LEAF.map(|v| v * shade), 0.3);
    }
}

/// Acacia : un tronc qui fourche, un houppier plat comme une table.
fn acacia(b: &mut Builder, lod: u8) {
    if lod == 2 {
        b.blob(Vec3::new(0.0, 7.2, 0.0), Vec3::new(5.4, 1.5, 5.4), 0, 0.2, 41, LEAF, FAR_SWAY);
        return;
    }
    let far = lod > 0;
    b.cone(0.0, 4.2, 0.34, 0.2, if far { 3 } else { 6 }, BARK, 0.0);
    if far {
        b.blob(Vec3::new(0.0, 7.0, 0.0), Vec3::new(5.4, 1.3, 5.4), 0, 0.2, 41, LEAF, 0.2);
        return;
    }
    let mut rng = Rng(41);
    for i in 0..3 {
        // Les maitresses branches, en faisceau.
        let a = i as f32 * 2.1;
        let tip = Vec3::new(a.cos() * 2.8, 6.8, a.sin() * 2.8);
        let root = Vec3::new(0.0, 4.0, 0.0);
        let side = Vec3::new(-a.sin(), 0.0, a.cos()) * 0.13;
        let center = (root + tip) * 0.5;
        b.tri(root - side, root + side, tip, center + Vec3::X, BARK, 0.0);
        b.tri(root + side, root - side, tip, center - Vec3::X, BARK, 0.0);
    }
    for i in 0..18 {
        let k = (i as f32 + 0.5) / 18.0;
        let a = i as f32 * 2.399963;
        let radius = 5.2 * k.sqrt();
        let c = Vec3::new(a.cos() * radius, 7.0 + rng.range(-0.2, 0.25) - k * 0.5, a.sin() * radius);
        let r = rng.range(1.2, 1.8);
        b.blob(c, Vec3::new(r, r * 0.42, r), 1, 0.4, 41 + i, LEAF.map(|v| v * rng.range(0.85, 1.15)), 0.3);
    }
}

fn birch(b: &mut Builder, lod: u8) {
    if lod == 2 {
        b.blob(Vec3::new(0.0, 8.0, 0.0), Vec3::new(2.3, 6.0, 2.3), 0, 0.2, 31, BIRCH_LEAF, FAR_SWAY);
        return;
    }
    let far = lod > 0;
    b.cone(0.0, 11.0, 0.2, 0.06, if far { 3 } else { 6 }, [0.52, 0.5, 0.44], 0.0);
    if far {
        b.blob(Vec3::new(0.0, 8.0, 0.0), Vec3::new(2.3, 5.2, 2.3), 0, 0.2, 31, BIRCH_LEAF, 0.25);
        return;
    }
    for (i, (y, r, h)) in [(5.6, 2.3, 2.3), (8.2, 2.1, 2.6), (10.8, 1.3, 2.0)].iter().enumerate() {
        let c = Vec3::new(if i == 1 { 0.4 } else { -0.3 }, *y, if i == 0 { 0.4 } else { -0.2 });
        b.blob(c, Vec3::new(*r, *h, *r), 1, 0.3, 33 + i as u64, BIRCH_LEAF, 0.4);
    }
}

fn palm(b: &mut Builder, lod: u8) {
    // Tronc courbe : chaque sommet est decale selon sa hauteur, donc les troncons restent soudes.
    let sides = if lod == 0 { 6 } else { 3 };
    let segments = if lod == 0 { 5 } else { 2 };
    let height = 9.0;
    let lean = |y: f32| 1.4 * (y / height) * (y / height);
    for i in 0..segments {
        let y0 = height * i as f32 / segments as f32;
        let y1 = height * (i + 1) as f32 / segments as f32;
        let r = |y: f32| 0.26 - 0.12 * y / height;
        let start = b.verts.len();
        b.cone(y0, y1, r(y0), r(y1), sides, [0.13, 0.095, 0.06], 0.0);
        for v in &mut b.verts[start..] {
            v.pos[0] += lean(v.pos[1]);
        }
    }
    // Palmes qui retombent, en deux triangles chacune de pres.
    let top = Vec3::new(lean(height), height, 0.0);
    let fronds = if lod == 0 { 15 } else { 6 };
    let color = [0.06, 0.115, 0.03];
    for i in 0..fronds {
        let a = i as f32 * 2.399963;
        let (dir, side) = (Vec3::new(a.cos(), 0.0, a.sin()), Vec3::new(-a.sin(), 0.0, a.cos()));
        let droop = 0.5 + (i % 3) as f32 * 0.5;
        let at = |k: f32| top + dir * 4.6 * k + Vec3::Y * (1.1 * k - droop * 2.3 * k * k);
        let width = |k: f32| 0.34 * (k * (1.0 - k) * 4.0).sqrt().max(0.03);
        let n = (dir * 0.3 + Vec3::Y).normalize();
        let segments = if lod == 0 { 4 } else { 2 };
        for s in 0..segments {
            let (k0, k1) = (s as f32 / segments as f32, (s + 1) as f32 / segments as f32);
            let (p0, p1) = (at(k0), at(k1));
            let (w0, w1) = (side * width(k0), side * width(k1));
            let shade = color.map(|c| c * (0.65 + 0.5 * k1));
            b.smooth_tri([p0 - w0, p0 + w0, p1 + w1], [n; 3], [shade; 3], 0.5);
            b.smooth_tri([p0 - w0, p1 + w1, p1 - w1], [n; 3], [shade; 3], 0.5);
        }
    }
}

fn flower(b: &mut Builder, lod: u8) {
    let head = Vec3::new(0.0, 0.42, 0.0);
    let up = Vec3::Y;
    let white = [1.0; 3];
    let petals = if lod == 0 { 6 } else { 3 };
    for i in 0..petals {
        let a = i as f32 / petals as f32 * std::f32::consts::TAU;
        let (dir, side) = (Vec3::new(a.cos(), 0.1, a.sin()), Vec3::new(-a.sin(), 0.0, a.cos()));
        b.smooth_tri(
            [head, head + dir * 0.09 + side * 0.04, head + dir * 0.09 - side * 0.04],
            [up; 3],
            [white; 3],
            0.8,
        );
    }
    if lod == 0 {
        // Tige : la couleur sombre attenue la teinte de la fleur.
        let stem = [0.1, 0.16, 0.05];
        let s = Vec3::new(0.012, 0.0, 0.0);
        b.smooth_tri([Vec3::ZERO - s, Vec3::ZERO + s, head], [up; 3], [stem; 3], 0.4);
    }
}

fn bush(b: &mut Builder, lod: u8) {
    let far = lod > 0;
    let color = [0.075, 0.11, 0.04];
    b.blob(Vec3::new(0.0, 0.6, 0.0), Vec3::new(1.1, 0.8, 1.1), if far { 0 } else { 1 }, 0.35, 5, color, 0.3);
}

fn cactus(b: &mut Builder, lod: u8) {
    let far = lod > 0;
    let color = [0.08, 0.19, 0.07];
    let sides = if far { 4 } else { 7 };
    b.cone(0.0, 4.4, 0.34, 0.26, sides, color, 0.0);
    if !far {
        b.cone(4.4, 4.75, 0.26, 0.0, sides, color, 0.0);
        for (x, y0, y1) in [(0.75f32, 1.7f32, 3.3f32), (-0.7, 2.3, 3.6)] {
            let start = b.verts.len();
            b.cone(y0, y1, 0.2, 0.16, sides, color, 0.0);
            b.cone(y1, y1 + 0.2, 0.16, 0.0, sides, color, 0.0);
            for v in &mut b.verts[start..] {
                v.pos[0] += x;
            }
            // Coude reliant le bras au tronc.
            let y = y0;
            let (a, e) = (Vec3::new(0.0, y - 0.18, 0.0), Vec3::new(x, y + 0.14, 0.0));
            let side = Vec3::new(0.0, 0.0, 0.16);
            let up = Vec3::new(0.0, 0.16, 0.0);
            let center = (a + e) * 0.5;
            for (p, q) in [(side, up), (up, -side), (-side, -up), (-up, side)] {
                b.tri(a + p, a + q, e + q, center, color, 0.0);
                b.tri(a + p, e + q, e + p, center, color, 0.0);
            }
        }
    }
}

fn rock(b: &mut Builder, lod: u8) {
    let far = lod > 0;
    let color = [0.15, 0.14, 0.125];
    b.blob(Vec3::new(0.0, 0.45, 0.0), Vec3::new(1.5, 1.0, 1.2), if far { 0 } else { 1 }, 0.5, 21, color, 0.0);
}

/// Un brin courbe en `segments` troncons, qui s'affine jusqu'a sa pointe.
fn blade(b: &mut Builder, base: Vec3, side: Vec3, lean: Vec3, height: f32, segments: u32, colors: [[f32; 3]; 2]) {
    let up = Vec3::Y;
    let point = |k: f32| base + up * height * k + lean * k * k;
    let shade = |k: f32| std::array::from_fn::<f32, 3, _>(|c| colors[0][c] + (colors[1][c] - colors[0][c]) * k);
    let width = |k: f32| 1.0 - k * k;
    for i in 0..segments {
        let (k0, k1) = (i as f32 / segments as f32, (i + 1) as f32 / segments as f32);
        let (p0, p1) = (point(k0), point(k1));
        let (w0, w1) = (side * width(k0), side * width(k1));
        // Normale vers le haut : les brins prennent la meme lumiere que le sol.
        b.push_tri([p0 - w0, p0 + w0, p1 + w1], up, [shade(k0), shade(k0), shade(k1)], [k0, k0, k1]);
        if i + 1 < segments {
            b.push_tri([p0 - w0, p1 + w1, p1 - w1], up, [shade(k0), shade(k1), shade(k1)], [k0, k1, k1]);
        }
    }
}

const GRASS_BASE: [f32; 3] = [0.26, 0.28, 0.18];
const GRASS_TIP: [f32; 3] = [1.0, 1.0, 0.82];

/// Des brins fins, courbes et nombreux de pres, moins nombreux et plus larges au loin.
fn grass(b: &mut Builder, lod: u8) {
    let mut rng = Rng(99);
    let (blades, width, segments) = match lod {
        0 => (30, 0.014, 4),
        1 => (8, 0.055, 2),
        _ => (3, 0.11, 1),
    };
    for _ in 0..blades {
        let (s, c) = rng.range(0.0, std::f32::consts::TAU).sin_cos();
        let base = Vec3::new(rng.range(-0.6, 0.6), 0.0, rng.range(-0.6, 0.6));
        let side = Vec3::new(c, 0.0, s) * width;
        let lean = Vec3::new(-s, 0.0, c) * rng.range(-0.34, 0.34);
        blade(b, base, side, lean, rng.range(0.32, 0.8), segments, [GRASS_BASE, GRASS_TIP]);
    }
}

/// Hautes herbes a epis, qui cadrent le premier plan et s'embrasent a contre-jour.
fn tall_grass(b: &mut Builder, lod: u8) {
    let mut rng = Rng(123);
    let (stems, width, segments, grains) = match lod {
        0 => (11, 0.009, 6, 20),
        1 => (5, 0.024, 3, 4),
        _ => (2, 0.05, 1, 1),
    };
    let stem = ([0.2, 0.2, 0.09], [0.52, 0.44, 0.2]);
    let seed = [0.66, 0.36, 0.14];
    for _ in 0..stems {
        let (s, c) = rng.range(0.0, std::f32::consts::TAU).sin_cos();
        let base = Vec3::new(rng.range(-0.34, 0.34), 0.0, rng.range(-0.34, 0.34));
        let side = Vec3::new(c, 0.0, s) * width;
        let lean = Vec3::new(-s, 0.0, c) * rng.range(0.12, 0.42) * if rng.f32() < 0.5 { -1.0 } else { 1.0 };
        let height = rng.range(0.82, 1.42);
        blade(b, base, side, lean, height, segments, [stem.0, stem.1]);
        // L'epi : des grains allonges qui s'ecartent de la tige sur son dernier quart.
        for g in 0..grains {
            let k = 0.74 + 0.26 * (g as f32 + 0.5) / grains as f32;
            let at = base + Vec3::Y * height * k + lean * k * k;
            let (gs, gc) = rng.range(0.0, std::f32::consts::TAU).sin_cos();
            let out = Vec3::new(gc * 0.7, 1.0, gs * 0.7).normalize() * if lod == 0 { 0.042 } else { 0.1 };
            let wing = Vec3::new(-gs, 0.0, gc) * if lod == 0 { 0.006 } else { 0.03 };
            let tone = seed.map(|v| v * rng.range(0.8, 1.2));
            b.push_tri([at - wing, at + wing, at + out], Vec3::Y, [tone; 3], [k, k, 1.0]);
        }
    }
}

pub struct ObjectMeshes {
    pub vertices: Vec<ObjectVertex>,
    /// Plage de sommets par type : [proche, lointain, tres lointain].
    pub ranges: [[Range<u32>; 3]; KIND_COUNT],
}

/// Maillages des arbres, rochers et herbes, dans l'ordre de `Kind` : construits une fois au demarrage.
pub fn build_object_meshes() -> ObjectMeshes {
    let mut vertices = Vec::new();
    let mut b = Builder::default();
    // Dans l'ordre de `Kind`.
    let builders: [fn(&mut Builder, u8); KIND_COUNT] =
        [pine, oak, bush, cactus, rock, grass, flower, birch, palm, tall_grass, acacia];
    let ranges = builders.map(|f| {
        [0, 1, 2].map(|lod| {
            f(&mut b, lod);
            b.take(&mut vertices)
        })
    });
    ObjectMeshes { vertices, ranges }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_has_geometry_and_far_is_cheaper() {
        let m = build_object_meshes();
        for [near, far, farthest] in &m.ranges {
            assert!(!near.is_empty() && !far.is_empty() && !farthest.is_empty());
            assert!(far.len() <= near.len() && farthest.len() <= far.len());
            assert_eq!(near.len() % 3, 0);
        }
        for v in &m.vertices {
            assert!(v.pos.iter().chain(&v.normal).all(|c| c.is_finite()));
        }
    }
}
