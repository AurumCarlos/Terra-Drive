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
        self.shaded_cone(y0, y1, r0, r1, sides, color, sway, [0.68, 1.12]);
    }

    /// Cone dont la teinte passe de `shade[0]` en bas a `shade[1]` en haut.
    #[allow(clippy::too_many_arguments)]
    fn shaded_cone(&mut self, y0: f32, y1: f32, r0: f32, r1: f32, sides: u32, color: [f32; 3], sway: f32, shade: [f32; 2]) {
        let tilt = (r0 - r1) / (y1 - y0);
        let at = |i: u32, y: f32, r: f32| {
            let a = i as f32 / sides as f32 * std::f32::consts::TAU;
            Vec3::new(a.cos() * r, y, a.sin() * r)
        };
        let normal = |i: f32| {
            let a = i / sides as f32 * std::f32::consts::TAU;
            Vec3::new(a.cos(), tilt, a.sin()).normalize()
        };
        let (low, high) = (color.map(|c| c * shade[0]), color.map(|c| c * shade[1]));
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

    /// Branche ou tronc droit de `from` a `to`, qui s'affine de `r0` a `r1`.
    #[allow(clippy::too_many_arguments)]
    fn limb(&mut self, from: Vec3, to: Vec3, r0: f32, r1: f32, sides: u32, color: [f32; 3], sway: f32) {
        let axis = to - from;
        let length = axis.length();
        if length < 1e-4 {
            return;
        }
        let start = self.verts.len();
        // Teinte uniforme : les troncons d'une meme branche se raccordent sans marche.
        self.shaded_cone(0.0, length, r0, r1, sides, color, sway, [1.0, 1.0]);
        let turn = glam::Quat::from_rotation_arc(Vec3::Y, axis / length);
        for v in &mut self.verts[start..] {
            v.pos = (from + turn * Vec3::from_array(v.pos)).to_array();
            v.normal = (turn * Vec3::from_array(v.normal)).to_array();
        }
    }

    fn take(&mut self, out: &mut Vec<ObjectVertex>) -> Range<u32> {
        let start = out.len() as u32;
        out.append(&mut self.verts);
        start..out.len() as u32
    }
}

const BARK: [f32; 3] = [0.14, 0.095, 0.06];
const GREY_BARK: [f32; 3] = [0.25, 0.24, 0.22];
const PINE: [f32; 3] = [0.024, 0.052, 0.032];
const LEAF: [f32; 3] = [0.06, 0.105, 0.032];
const BIRCH_LEAF: [f32; 3] = [0.115, 0.15, 0.042];

/// Vent presque nul au loin, mais assez pour etre traite comme du feuillage.
const FAR_SWAY: f32 = 0.08;
/// Sensibilite au vent des feuillages proches. Au-dessus de 1/8, le shader traite un sommet en feuille :
/// les feuillus le perdent l'hiver.
const LEAF_SWAY: f32 = 0.3;

/// Un feuillu : tronc, maitresses branches qui partent de la fourche (ou le long d'une fleche) vers la
/// surface du houppier, rameaux secondaires, et paquets de feuilles au bout des branches. Le houppier est
/// un ellipsoide entre `base` et `height` : les trois niveaux de detail en gardent la silhouette.
#[derive(Clone, Copy)]
struct Broadleaf {
    seed: u64,
    /// Hauteur de la fourche, ou s'arrete le fut.
    fork: f32,
    /// Rayon du tronc au pied, et celui des maitresses branches par rapport au tronc.
    girth: f32,
    limb: f32,
    /// Bas et sommet du houppier, et son rayon.
    base: f32,
    height: f32,
    spread: f32,
    /// Maitresses branches.
    limbs: u32,
    /// Port des branches : 0 etalees (chene), 1 dressees (erable).
    rise: f32,
    /// Le tronc continue en fleche jusqu'au sommet (bouleau, hetre, pin) : les branches partent le long d'elle.
    leader: bool,
    /// Paquets de feuilles au bout de chaque maitresse branche, leur rayon et leur hauteur relative.
    tufts: u32,
    tuft: f32,
    flat: f32,
    /// Les paquets pendent sous le bout des rameaux (bouleau, eucalyptus).
    droop: f32,
    /// Paquets au coeur du houppier, pour qu'il ne paraisse pas creux.
    fill: u32,
    bark: [f32; 3],
    leaf: [f32; 3],
}

impl Broadleaf {
    fn center(&self) -> Vec3 {
        Vec3::new(0.0, (self.base + self.height) * 0.5, 0.0)
    }

    fn radii(&self) -> Vec3 {
        Vec3::new(self.spread, (self.height - self.base) * 0.5, self.spread)
    }

    /// Point de la surface du houppier, a l'angle polaire `phi` depuis le sommet et l'azimut `a`.
    fn surface(&self, phi: f32, a: f32, k: f32) -> Vec3 {
        let r = self.radii();
        self.center() + Vec3::new(phi.sin() * a.cos() * r.x, phi.cos() * r.y, phi.sin() * a.sin() * r.z) * k
    }

    /// Couleur d'un paquet de feuilles : plus sombre en bas et au coeur du houppier.
    fn shade(&self, p: Vec3, rng: &mut Rng) -> [f32; 3] {
        let r = self.radii();
        let up = ((p.y - self.base) / (self.height - self.base)).clamp(0.0, 1.0);
        let out = (Vec3::new(p.x / r.x, 0.0, p.z / r.z).length()).min(1.0);
        let k = (0.68 + 0.5 * up) * (0.85 + 0.25 * out) * rng.range(0.86, 1.14);
        self.leaf.map(|c| c * k)
    }
}

fn broadleaf(b: &mut Builder, lod: u8, t: &Broadleaf) {
    let (c, r) = (t.center(), t.radii());
    if lod == 2 {
        // Au loin, un fut nu sous un houppier haut perche (pin parasol, kapokier) reste visible.
        if t.base > 6.0 {
            b.cone(0.0, t.base + 1.0, t.girth * 0.9, t.girth * 0.5, 3, t.bark, 0.0);
        }
        b.blob(c, r * Vec3::new(1.0, 0.95, 0.92), 0, 0.3, t.seed, t.leaf, FAR_SWAY);
        return;
    }
    let mut rng = Rng(t.seed);
    if lod == 1 {
        let top = c.y;
        b.limb(Vec3::NEG_Y * 0.3, Vec3::Y * top, t.girth, t.girth * 0.55, 3, t.bark, 0.0);
        b.blob(c, r * 0.94, 0, 0.35, t.seed, t.leaf.map(|v| v * 0.92), LEAF_SWAY);
        let side = Vec3::new(r.x * 0.4, r.y * 0.3, -r.z * 0.25);
        b.blob(c + side, r * 0.55, 0, 0.35, t.seed + 1, t.leaf.map(|v| v * 1.1), LEAF_SWAY);
        return;
    }

    // Fut en deux troncons un peu coudes, evase au pied.
    let bend = Vec3::new(rng.range(-0.5, 0.5), 0.0, rng.range(-0.5, 0.5)) * t.girth;
    let mid = Vec3::Y * t.fork * 0.45 + bend;
    let fork = Vec3::Y * t.fork + bend * 0.5;
    b.limb(Vec3::NEG_Y * 0.4, mid, t.girth * 1.15, t.girth * 0.85, 8, t.bark, 0.0);
    b.limb(mid, fork, t.girth * 0.85, t.girth * 0.7, 7, t.bark, 0.0);
    let leader_top = t.surface(0.0, 0.0, 0.8);
    if t.leader {
        b.limb(fork, leader_top, t.girth * 0.7, t.girth * 0.08, 6, t.bark, 0.0);
    }

    let a0 = rng.range(0.0, std::f32::consts::TAU);
    let mut tips: Vec<(Vec3, f32)> = Vec::new();
    for i in 0..t.limbs {
        let order = (i as f32 + 0.5) / t.limbs as f32;
        let a = a0 + i as f32 * 2.399963 + rng.range(-0.25, 0.25);
        // Du sommet vers le bord du houppier, un peu sous l'equateur.
        let phi = 0.3 + 1.55 * order + rng.range(-0.1, 0.1);
        let tip = t.surface(phi, a, rng.range(0.72, 0.86));
        let start = if t.leader { leader_top.lerp(fork, (0.15 + 0.85 * order).min(1.0)) } else { fork };
        let d = tip - start;
        let flat = Vec3::new(d.x, 0.0, d.z);
        let elbow = start + flat * (0.62 - 0.22 * t.rise) + Vec3::Y * d.y * (0.18 + 0.35 * t.rise);
        let r0 = t.girth * t.limb * if t.leader { 0.75 + 0.5 * order } else { 1.0 };
        b.limb(start, elbow, r0, r0 * 0.62, 5, t.bark, 0.0);
        b.limb(elbow, tip, r0 * 0.62, r0 * 0.22, 4, t.bark, 0.0);
        tips.push((tip, 1.0));
        // Rameaux secondaires vers des points voisins de la surface.
        for s in [-1.0f32, 1.0] {
            let twig = t.surface(phi + rng.range(-0.3, 0.25), a + s * rng.range(0.35, 0.6), rng.range(0.8, 0.95));
            let from = elbow.lerp(tip, rng.range(0.1, 0.45));
            b.limb(from, twig, r0 * 0.4, r0 * 0.1, 3, t.bark, 0.0);
            tips.push((twig, 0.8));
        }
        // Brindilles en eventail au bout des branches : la silhouette d'hiver.
        let from = elbow.lerp(tip, 0.7);
        for _ in 0..4 {
            let spray = Vec3::new(rng.range(-1.0, 1.0), rng.range(-0.2, 1.0), rng.range(-1.0, 1.0)) * t.tuft * 1.1;
            b.limb(from, tip + spray, r0 * 0.12, r0 * 0.03, 3, t.bark, 0.0);
        }
    }
    for (tip, size) in tips {
        let count = if size < 1.0 { 1 } else { t.tufts };
        for k in 0..count {
            let jitter = Vec3::new(rng.range(-1.0, 1.0), rng.range(-0.5, 0.7), rng.range(-1.0, 1.0)) * t.tuft * 0.55;
            let mut p = tip + if k == 0 { Vec3::ZERO } else { jitter };
            p.y -= t.droop * t.tuft * 0.7;
            let s = t.tuft * size * rng.range(0.75, 1.15);
            let color = t.shade(p, &mut rng);
            b.blob(p, Vec3::new(s, s * t.flat, s), 1, 0.5, t.seed + 7 + b.verts.len() as u64, color, LEAF_SWAY);
        }
    }
    for k in 0..t.fill {
        let a = k as f32 * 2.1 + a0;
        let p = t.surface(0.6 + 0.4 * k as f32, a, 0.45);
        let s = t.tuft * 1.3;
        let color = t.shade(p, &mut rng).map(|v| v * 0.85);
        b.blob(p, Vec3::new(s, s * t.flat, s), 1, 0.45, t.seed + 900 + k as u64, color, LEAF_SWAY);
    }
}

/// Chene : un fut court et epais, de grosses branches etalees, un houppier large et bossele.
fn oak(b: &mut Builder, lod: u8) {
    broadleaf(
        b,
        lod,
        &Broadleaf {
            seed: 11,
            fork: 4.2,
            girth: 0.55,
            limb: 0.55,
            base: 4.5,
            height: 21.0,
            spread: 8.5,
            limbs: 6,
            rise: 0.25,
            leader: false,
            tufts: 2,
            tuft: 2.4,
            flat: 0.78,
            droop: 0.0,
            fill: 3,
            bark: BARK,
            leaf: LEAF,
        },
    );
}

/// Hetre : un fut gris et lisse, une fleche jusqu'en haut, un houppier dense en etages.
fn beech(b: &mut Builder, lod: u8) {
    broadleaf(
        b,
        lod,
        &Broadleaf {
            seed: 131,
            fork: 7.0,
            girth: 0.45,
            limb: 0.4,
            base: 8.0,
            height: 28.0,
            spread: 7.5,
            limbs: 8,
            rise: 0.55,
            leader: true,
            tufts: 2,
            tuft: 2.2,
            flat: 0.55,
            droop: 0.0,
            fill: 2,
            bark: GREY_BARK,
            leaf: [0.075, 0.125, 0.03],
        },
    );
}

/// Erable : un houppier ovale, des branches dressees.
fn maple(b: &mut Builder, lod: u8) {
    broadleaf(
        b,
        lod,
        &Broadleaf {
            seed: 141,
            fork: 3.4,
            girth: 0.42,
            limb: 0.5,
            base: 3.8,
            height: 19.0,
            spread: 6.6,
            limbs: 7,
            rise: 0.8,
            leader: false,
            tufts: 2,
            tuft: 2.1,
            flat: 0.85,
            droop: 0.0,
            fill: 3,
            bark: [0.15, 0.12, 0.095],
            leaf: [0.068, 0.118, 0.03],
        },
    );
}

/// Bouleau : un tronc blanc et fin jusqu'au sommet, des rameaux qui retombent, une cime claire et legere.
fn birch(b: &mut Builder, lod: u8) {
    broadleaf(
        b,
        lod,
        &Broadleaf {
            seed: 31,
            fork: 3.5,
            girth: 0.22,
            limb: 0.35,
            base: 5.0,
            height: 20.0,
            spread: 3.6,
            limbs: 8,
            rise: 0.75,
            leader: true,
            tufts: 2,
            tuft: 1.45,
            flat: 1.05,
            droop: 0.6,
            fill: 1,
            bark: [0.56, 0.54, 0.49],
            leaf: BIRCH_LEAF,
        },
    );
}

/// Olivier : un tronc court et tordu, un houppier gris-vert ebouriffe.
fn olive(b: &mut Builder, lod: u8) {
    broadleaf(
        b,
        lod,
        &Broadleaf {
            seed: 71,
            fork: 1.5,
            girth: 0.42,
            limb: 0.55,
            base: 2.3,
            height: 7.5,
            spread: 3.8,
            limbs: 5,
            rise: 0.45,
            leader: false,
            tufts: 3,
            tuft: 1.25,
            flat: 0.75,
            droop: 0.0,
            fill: 1,
            bark: [0.17, 0.15, 0.12],
            leaf: [0.085, 0.1, 0.062],
        },
    );
}

/// Acacia : un tronc qui fourche bas en vase, et un houppier plat comme une table.
fn acacia(b: &mut Builder, lod: u8) {
    broadleaf(
        b,
        lod,
        &Broadleaf {
            seed: 41,
            fork: 2.4,
            girth: 0.32,
            limb: 0.6,
            base: 7.2,
            height: 10.5,
            spread: 6.6,
            limbs: 5,
            rise: 0.9,
            leader: false,
            tufts: 3,
            tuft: 1.9,
            flat: 0.38,
            droop: 0.0,
            fill: 1,
            bark: BARK,
            leaf: [0.07, 0.105, 0.035],
        },
    );
}

/// Eucalyptus : un grand fut pale, peu de branches, des feuillages bleutes qui pendent, un houppier ouvert.
fn eucalyptus(b: &mut Builder, lod: u8) {
    broadleaf(
        b,
        lod,
        &Broadleaf {
            seed: 171,
            fork: 10.0,
            girth: 0.5,
            limb: 0.45,
            base: 13.0,
            height: 33.0,
            spread: 7.0,
            limbs: 5,
            rise: 0.85,
            leader: false,
            tufts: 2,
            tuft: 2.0,
            flat: 0.8,
            droop: 0.9,
            fill: 0,
            bark: [0.44, 0.42, 0.38],
            leaf: [0.058, 0.088, 0.062],
        },
    );
}

/// Pin parasol : un long fut nu, des branches en candelabre, une ombrelle plate tout en haut.
fn stone_pine(b: &mut Builder, lod: u8) {
    broadleaf(
        b,
        lod,
        &Broadleaf {
            seed: 61,
            fork: 8.5,
            girth: 0.4,
            limb: 0.5,
            base: 12.5,
            height: 17.0,
            spread: 6.8,
            limbs: 5,
            rise: 0.85,
            leader: false,
            tufts: 3,
            tuft: 2.0,
            flat: 0.45,
            droop: 0.0,
            fill: 2,
            bark: [0.2, 0.11, 0.07],
            leaf: [0.035, 0.068, 0.033],
        },
    );
}

/// Pin sylvestre : un long tronc orange, une cime claire et irreguliere tout en haut.
fn scots_pine(b: &mut Builder, lod: u8) {
    broadleaf(
        b,
        lod,
        &Broadleaf {
            seed: 111,
            fork: 16.0,
            girth: 0.36,
            limb: 0.35,
            base: 19.0,
            height: 28.0,
            spread: 4.2,
            limbs: 6,
            rise: 0.5,
            leader: true,
            tufts: 2,
            tuft: 1.6,
            flat: 0.55,
            droop: 0.0,
            fill: 1,
            bark: [0.24, 0.125, 0.07],
            leaf: [0.03, 0.06, 0.04],
        },
    );
}

/// Baobab : un enorme tronc en bouteille et des branches courtes et epaisses, peu feuillues.
fn baobab(b: &mut Builder, lod: u8) {
    let bark = [0.21, 0.18, 0.15];
    let t = Broadleaf {
        seed: 81,
        fork: 10.0,
        girth: 2.4,
        limb: 0.3,
        base: 11.0,
        height: 17.0,
        spread: 7.0,
        limbs: 8,
        rise: 0.6,
        leader: false,
        tufts: 1,
        tuft: 1.3,
        flat: 0.6,
        droop: 0.0,
        fill: 0,
        bark,
        leaf: LEAF.map(|v| v * 1.1),
    };
    if lod == 2 {
        b.cone(0.0, 10.0, 2.4, 1.4, 4, bark, 0.0);
        return;
    }
    broadleaf(b, lod, &t);
    // Le ventre de la bouteille.
    b.cone(0.0, 4.0, 2.5, 2.7, if lod == 0 { 10 } else { 4 }, bark, 0.0);
    b.cone(4.0, 10.0, 2.7, 1.6, if lod == 0 { 10 } else { 4 }, bark, 0.0);
}

/// Kapokier : l'arbre emergent des forets tropicales, un fut immense sur des contreforts, la cime etalee
/// au-dessus de la canopee.
fn kapok(b: &mut Builder, lod: u8) {
    let bark = [0.21, 0.19, 0.16];
    let t = Broadleaf {
        seed: 91,
        fork: 34.0,
        girth: 1.1,
        limb: 0.45,
        base: 40.0,
        height: 52.0,
        spread: 14.0,
        limbs: 7,
        rise: 0.3,
        leader: false,
        tufts: 2,
        tuft: 3.4,
        flat: 0.5,
        droop: 0.0,
        fill: 2,
        bark,
        leaf: [0.045, 0.09, 0.028],
    };
    broadleaf(b, lod, &t);
    if lod == 2 {
        return;
    }
    // Contreforts : des lames de bois qui partent du tronc.
    for k in 0..if lod == 0 { 5 } else { 3 } {
        let a = k as f32 * 1.256 + 0.3;
        let out = Vec3::new(a.cos(), 0.0, a.sin());
        let (foot, high) = (out * 4.2, Vec3::Y * 6.5 + out * 0.8);
        let side = Vec3::new(-a.sin(), 0.0, a.cos()) * 0.16;
        b.tri(out * 0.8 - side, foot - side, high - side, Vec3::Y * 2.0 - side * 10.0, bark, 0.0);
        b.tri(out * 0.8 + side, high + side, foot + side, Vec3::Y * 2.0 + side * 10.0, bark, 0.0);
    }
}

/// Saule pleureur des berges : des branches arquees, un dome, et des rameaux qui pendent en rideau.
fn willow(b: &mut Builder, lod: u8) {
    let color = [0.1, 0.14, 0.05];
    let t = Broadleaf {
        seed: 101,
        fork: 3.2,
        girth: 0.55,
        limb: 0.5,
        base: 6.0,
        height: 14.0,
        spread: 6.0,
        limbs: 5,
        rise: 0.6,
        leader: false,
        tufts: 1,
        tuft: 2.0,
        flat: 0.6,
        droop: 0.0,
        fill: 1,
        bark: BARK,
        leaf: color,
    };
    if lod == 2 {
        b.blob(Vec3::new(0.0, 7.5, 0.0), Vec3::new(6.2, 6.5, 6.2), 0, 0.25, 101, color, FAR_SWAY);
        return;
    }
    broadleaf(b, lod, &t);
    if lod == 1 {
        b.cone(2.2, 9.0, 6.4, 4.4, 5, color.map(|c| c * 0.85), LEAF_SWAY);
        return;
    }
    // Le rideau : des lanieres de longueurs inegales, accrochees partout sous le dome.
    let mut rng = Rng(102);
    for i in 0..56 {
        let a = i as f32 * 2.399963 + rng.range(-0.2, 0.2);
        let (dir, side) = (Vec3::new(a.cos(), 0.0, a.sin()), Vec3::new(-a.sin(), 0.0, a.cos()));
        let phi = rng.range(0.7, 1.75);
        let top = t.surface(phi, a, rng.range(0.85, 1.0));
        let length = rng.range(3.5, top.y - 1.0);
        let mid = top + dir * 0.4 - Vec3::Y * length * 0.5;
        let bottom = top + dir * rng.range(0.5, 1.1) - Vec3::Y * length;
        let (w0, w1, w2) = (side * 0.35, side * 0.28, side * 0.12);
        let shade = color.map(|c| c * rng.range(0.85, 1.25));
        let low = shade.map(|c| c * 0.8);
        for (p, q, wp, wq, cp, cq) in [(top, mid, w0, w1, shade, shade), (mid, bottom, w1, w2, shade, low)] {
            b.smooth_tri([p - wp, p + wp, q + wq], [dir; 3], [cp, cp, cq], 0.6);
            b.smooth_tri([p - wp, q + wq, q - wq], [dir; 3], [cp, cq, cq], 0.6);
        }
    }
}

/// Conifere en etages de branches qui retombent : sapin, epicea, meleze.
#[allow(clippy::too_many_arguments)]
fn conifer(b: &mut Builder, lod: u8, height: f32, base: f32, radius: f32, tiers: u32, branches: u32, color: [f32; 3]) {
    if lod == 2 {
        b.cone(base * 0.6, height, radius * 0.95, 0.0, 5, color, FAR_SWAY);
        return;
    }
    let far = lod > 0;
    b.limb(Vec3::NEG_Y * 0.4, Vec3::Y * height, height * 0.018, height * 0.002, if far { 3 } else { 7 }, BARK, 0.0);
    let (tiers, branches) = if far { (4, 5) } else { (tiers, branches) };
    let span = height - base;
    for i in 0..tiers {
        let k = i as f32 / (tiers - 1) as f32;
        let y0 = base + (span - 2.0) * k.powf(1.1);
        let tier = span / tiers as f32 * 2.6 * (1.0 - 0.5 * k);
        let r = radius * (1.0 - k).powf(0.85) + 0.3;
        let shade = 0.8 + 0.5 * k;
        b.fir_tier(y0, y0 + tier, r, branches, i as f32 * 1.7, color.map(|c| c * shade), LEAF_SWAY * (0.45 + 0.55 * k));
        if !far {
            // Les branches elles-memes : visibles quand le meleze a perdu ses aiguilles.
            for j in 0..3 {
                let a = i as f32 * 1.7 + j as f32 * 2.1;
                let tip = Vec3::new(a.cos() * r * 0.9, y0 - 0.3, a.sin() * r * 0.9);
                b.limb(Vec3::Y * (y0 + 0.5), tip, 0.06 + 0.05 * (1.0 - k), 0.02, 3, BARK, 0.0);
            }
        }
    }
}

/// Epicea ou sapin : une fleche sombre et dentelee.
fn pine(b: &mut Builder, lod: u8) {
    conifer(b, lod, 30.0, 3.5, 4.4, 16, 9, PINE);
}

/// Meleze : un conifere clair et aere, qui jaunit et perd ses aiguilles l'hiver.
fn larch(b: &mut Builder, lod: u8) {
    conifer(b, lod, 28.0, 4.5, 3.8, 11, 6, [0.07, 0.115, 0.04]);
}

/// Sequoia : un geant au tronc roux et cannele, dont la cime en touffes commence a mi-hauteur.
fn sequoia(b: &mut Builder, lod: u8) {
    let bark = [0.3, 0.13, 0.07];
    let color = [0.028, 0.058, 0.034];
    let (height, base, radius) = (68.0f32, 30.0f32, 7.5f32);
    if lod == 2 {
        b.cone(0.0, base, 2.0, 1.4, 3, bark, 0.0);
        b.cone(base, height, radius, 0.0, 5, color, FAR_SWAY);
        return;
    }
    let sides = if lod == 0 { 9 } else { 4 };
    b.limb(Vec3::NEG_Y * 0.5, Vec3::Y * 3.0, 3.1, 2.1, sides, bark, 0.0);
    b.limb(Vec3::Y * 3.0, Vec3::Y * height, 2.1, 0.2, sides, bark, 0.0);
    if lod == 1 {
        b.cone(base, height, radius, 0.4, 6, color, LEAF_SWAY);
        return;
    }
    let mut rng = Rng(181);
    let levels = 12;
    for i in 0..levels {
        let k = i as f32 / (levels - 1) as f32;
        let y = base + (height - base - 3.0) * k;
        let r = radius * (1.0 - k * 0.85) * rng.range(0.8, 1.05);
        for j in 0..3 {
            let a = i as f32 * 2.399963 + j as f32 * 2.094;
            let out = Vec3::new(a.cos(), 0.0, a.sin());
            let tip = Vec3::Y * (y - 0.6 * r * 0.3) + out * r;
            b.limb(Vec3::Y * y, tip, 0.25 * (1.0 - k) + 0.06, 0.05, 3, bark, 0.0);
            let s = rng.range(1.8, 2.6) * (1.0 - 0.4 * k);
            let p = Vec3::Y * y + out * r * 0.75;
            let tone = (0.7 + 0.5 * k) * rng.range(0.85, 1.15);
            b.blob(p, Vec3::new(s * 1.2, s * 0.65, s * 1.2), 1, 0.5, 182 + (i * 3 + j) as u64, color.map(|c| c * tone), LEAF_SWAY);
        }
    }
    b.blob(Vec3::Y * (height - 1.5), Vec3::new(1.3, 2.4, 1.3), 1, 0.4, 199, color.map(|c| c * 1.2), LEAF_SWAY);
}

/// Cedre : des etages de branches horizontales qui portent des plateaux de feuillage, le sommet aplati.
fn cedar(b: &mut Builder, lod: u8) {
    let color = [0.034, 0.06, 0.052];
    let levels: [(f32, f32); 5] = [(8.0, 8.5), (12.0, 7.4), (15.5, 5.8), (18.5, 4.2), (21.0, 2.6)];
    if lod == 2 {
        b.blob(Vec3::Y * 13.0, Vec3::new(7.5, 8.5, 7.5), 0, 0.3, 191, color, FAR_SWAY);
        return;
    }
    let sides = if lod == 0 { 7 } else { 3 };
    b.limb(Vec3::NEG_Y * 0.4, Vec3::Y * 22.5, 0.65, 0.12, sides, GREY_BARK.map(|c| c * 0.8), 0.0);
    if lod == 1 {
        for (i, (y, r)) in levels.iter().enumerate().step_by(2) {
            b.blob(Vec3::Y * (y + 0.6), Vec3::new(*r, 1.4, *r), 0, 0.35, 191 + i as u64, color, LEAF_SWAY);
        }
        return;
    }
    let mut rng = Rng(192);
    for (i, (y, r)) in levels.iter().enumerate() {
        let n = 4 - i / 2;
        for j in 0..n {
            let a = i as f32 * 1.3 + j as f32 * std::f32::consts::TAU / n as f32 + rng.range(-0.3, 0.3);
            let out = Vec3::new(a.cos(), 0.0, a.sin());
            let reach = r * rng.range(0.8, 1.0);
            let tip = Vec3::Y * (y + 0.6) + out * reach;
            b.limb(Vec3::Y * *y, tip, 0.2 - 0.025 * i as f32, 0.05, 3, BARK, 0.0);
            // Un plateau au bout, un autre a mi-longueur.
            for (k, s) in [(1.0f32, 0.55f32), (0.5, 0.45)] {
                let p = Vec3::Y * (y + 0.6 + 0.4 * k) + out * reach * k;
                let w = r * s * rng.range(0.85, 1.1);
                let tone = (0.75 + 0.1 * i as f32) * rng.range(0.88, 1.12);
                b.blob(p, Vec3::new(w, 0.55 + 0.1 * w, w), 1, 0.4, 193 + (i * 8 + j * 2) as u64, color.map(|c| c * tone), LEAF_SWAY);
            }
        }
    }
}

/// Feuillage en colonne : cypres de Provence (sombre, serre) ou peuplier d'Italie (clair, plus large).
#[allow(clippy::too_many_arguments)]
fn column(b: &mut Builder, lod: u8, seed: u64, height: f32, radius: f32, base: f32, color: [f32; 3], bark: [f32; 3], clumps: u32) {
    let mid = (base + height) * 0.5;
    let half = (height - base) * 0.5;
    if lod == 2 {
        b.blob(Vec3::Y * mid, Vec3::new(radius, half, radius), 0, 0.15, seed, color, FAR_SWAY);
        return;
    }
    b.limb(Vec3::NEG_Y * 0.4, Vec3::Y * (height - 1.0), radius * 0.16, 0.05, if lod == 0 { 6 } else { 3 }, bark, 0.0);
    if lod == 1 {
        b.blob(Vec3::Y * mid, Vec3::new(radius * 0.95, half, radius * 0.95), 0, 0.2, seed, color, LEAF_SWAY);
        return;
    }
    let mut rng = Rng(seed);
    for i in 0..clumps {
        let k = (i as f32 + 0.5) / clumps as f32;
        let y = base + (height - base) * k;
        // Fuseau : large au premier tiers, pointu en haut.
        let profile = (std::f32::consts::PI * k.powf(0.8)).sin().powf(0.6) * radius;
        let a = i as f32 * 2.399963;
        let off = Vec3::new(a.cos(), 0.0, a.sin()) * profile * rng.range(0.25, 0.5);
        let s = profile * rng.range(0.75, 0.95) + 0.3;
        let h = (height - base) / clumps as f32 * 1.9;
        let tone = (0.75 + 0.45 * k) * rng.range(0.88, 1.12);
        b.limb(Vec3::Y * (y - h * 0.3), Vec3::Y * y + off * 1.3, 0.05, 0.02, 3, bark, 0.0);
        b.blob(Vec3::Y * y + off, Vec3::new(s, h, s), 1, 0.35, seed + 1 + i as u64, color.map(|c| c * tone), LEAF_SWAY);
    }
}

fn cypress(b: &mut Builder, lod: u8) {
    column(b, lod, 51, 20.0, 1.7, 1.2, [0.026, 0.052, 0.03], BARK, 12);
}

fn poplar(b: &mut Builder, lod: u8) {
    column(b, lod, 151, 28.0, 2.8, 3.0, [0.075, 0.12, 0.035], GREY_BARK, 20);
}

fn palm(b: &mut Builder, lod: u8) {
    // Tronc courbe : chaque sommet est decale selon sa hauteur, donc les troncons restent soudes.
    let sides = if lod == 0 { 7 } else { 3 };
    let segments = if lod == 0 { 6 } else { 2 };
    let height = 16.0;
    let lean = |y: f32| 2.2 * (y / height) * (y / height);
    for i in 0..segments {
        let y0 = height * i as f32 / segments as f32;
        let y1 = height * (i + 1) as f32 / segments as f32;
        let r = |y: f32| 0.3 - 0.13 * y / height;
        let start = b.verts.len();
        b.cone(y0, y1, r(y0), r(y1), sides, [0.15, 0.12, 0.085], 0.0);
        for v in &mut b.verts[start..] {
            v.pos[0] += lean(v.pos[1]);
        }
    }
    let top = Vec3::new(lean(height), height, 0.0);
    if lod == 0 {
        // Les noix de coco, sous les palmes.
        b.blob(top - Vec3::Y * 0.5, Vec3::new(0.7, 0.5, 0.7), 0, 0.3, 8, [0.16, 0.11, 0.05], 0.0);
    }
    // Palmes pennees qui retombent : un rachis et des folioles de part et d'autre.
    let fronds = match lod {
        0 => 18,
        1 => 8,
        _ => 6,
    };
    let color = [0.06, 0.115, 0.03];
    for i in 0..fronds {
        let a = i as f32 * 2.399963;
        let (dir, side) = (Vec3::new(a.cos(), 0.0, a.sin()), Vec3::new(-a.sin(), 0.0, a.cos()));
        let droop = 0.4 + (i % 3) as f32 * 0.45;
        let length = 5.8 - (i % 4) as f32 * 0.4;
        let at = |k: f32| top + dir * length * k + Vec3::Y * (1.4 * k - droop * 2.8 * k * k);
        let segments = if lod == 0 { 6 } else { 2 };
        for s in 0..segments {
            let (k0, k1) = (s as f32 / segments as f32, (s + 1) as f32 / segments as f32);
            let (p0, p1) = (at(k0), at(k1));
            let shade = color.map(|c| c * (0.65 + 0.5 * k1));
            if lod == 0 {
                // Folioles : longues au milieu de la palme, pendantes.
                let leaflet = 1.25 * (k1 * (1.0 - k0) * 4.0).sqrt().clamp(0.25, 1.0);
                for side_sign in [-1.0f32, 1.0] {
                    let w = side * side_sign;
                    for (q0, q1) in [(p0, p0.lerp(p1, 0.5)), (p0.lerp(p1, 0.5), p1)] {
                        let tip = q1 + w * leaflet - Vec3::Y * leaflet * 0.45 + dir * 0.2;
                        let n = (Vec3::Y + w * 0.3).normalize();
                        b.smooth_tri([q0, q1, tip], [n; 3], [shade, shade, shade.map(|c| c * 1.15)], 0.5);
                    }
                }
            } else {
                let width = |k: f32| 0.9 * (k * (1.0 - k) * 4.0).sqrt().max(0.05);
                let (w0, w1) = (side * width(k0), side * width(k1));
                let n = (dir * 0.3 + Vec3::Y).normalize();
                b.smooth_tri([p0 - w0, p0 + w0, p1 + w1], [n; 3], [shade; 3], 0.5);
                b.smooth_tri([p0 - w0, p1 + w1, p1 - w1], [n; 3], [shade; 3], 0.5);
            }
        }
    }
}

/// Dragonnier : un tronc epais qui fourche et refourche en ombrelle, chaque rameau termine par une rosette
/// de feuilles raides.
fn dragon_tree(b: &mut Builder, lod: u8) {
    let bark = [0.26, 0.24, 0.2];
    let color = [0.07, 0.11, 0.05];
    if lod == 2 {
        b.cone(0.0, 3.5, 0.6, 0.5, 3, bark, 0.0);
        b.blob(Vec3::Y * 7.4, Vec3::new(4.6, 1.8, 4.6), 0, 0.2, 161, color, FAR_SWAY);
        return;
    }
    let sides = if lod == 0 { 7 } else { 3 };
    let fork = Vec3::Y * 3.5;
    b.limb(Vec3::NEG_Y * 0.3, fork, 0.65, 0.5, sides, bark, 0.0);
    if lod == 1 {
        b.blob(Vec3::Y * 7.2, Vec3::new(4.6, 1.9, 4.6), 0, 0.25, 161, color, LEAF_SWAY);
        for a in [0.4f32, 2.5, 4.6] {
            b.limb(fork, fork + Vec3::new(a.cos() * 1.8, 2.4, a.sin() * 1.8), 0.32, 0.2, 3, bark, 0.0);
        }
        return;
    }
    let mut rng = Rng(162);
    for i in 0..4 {
        let a = i as f32 * 1.571 + 0.3;
        let out = Vec3::new(a.cos(), 0.0, a.sin());
        let mid = fork + out * 1.6 + Vec3::Y * 1.9;
        b.limb(fork, mid, 0.33, 0.24, 5, bark, 0.0);
        for s in [-0.5f32, 0.5] {
            let b2 = a + s;
            let tip = mid + Vec3::new(b2.cos(), 0.0, b2.sin()) * rng.range(1.6, 2.2) + Vec3::Y * rng.range(1.3, 1.8);
            b.limb(mid, tip, 0.22, 0.13, 4, bark, 0.0);
            // Rosette : un coussin dense, herisse de lames raides qui partent en etoile.
            let tone = color.map(|v| v * rng.range(0.85, 1.1));
            b.blob(tip + Vec3::Y * 0.35, Vec3::new(1.25, 0.6, 1.25), 1, 0.35, 163 + (i * 2) as u64, tone, LEAF_SWAY);
            for k in 0..14 {
                let c = k as f32 * 2.399963;
                let blade = Vec3::new(c.cos() * 0.9, 0.35 + 0.5 * ((k % 3) as f32 / 2.0), c.sin() * 0.9).normalize();
                let side = Vec3::new(-c.sin(), 0.0, c.cos()) * 0.13;
                let end = tip + blade * rng.range(1.5, 2.0);
                let tone = color.map(|v| v * rng.range(0.85, 1.2));
                b.smooth_tri([tip - side, tip + side, end], [(blade + Vec3::Y).normalize(); 3], [tone; 3], 0.4);
            }
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
        [
            pine, oak, bush, cactus, rock, grass, flower, birch, palm, tall_grass, acacia, cypress, stone_pine, olive, baobab,
            kapok, willow, scots_pine, maple, beech, poplar, larch, sequoia, cedar, eucalyptus, dragon_tree,
        ];
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
        for (kind, [near, far, farthest]) in m.ranges.iter().enumerate() {
            assert!(!near.is_empty() && !far.is_empty() && !farthest.is_empty());
            assert!(far.len() <= near.len() && farthest.len() <= far.len());
            assert_eq!(near.len() % 3, 0);
            // Des dizaines de milliers d'arbres sont dessines en version lointaine : elle doit rester legere.
            let tris = |r: &Range<u32>| r.len() / 3;
            eprintln!("sorte {kind:2} : {} / {} / {} triangles", tris(near), tris(far), tris(farthest));
            assert!(tris(near) <= 4000 && tris(far) <= 130 && tris(farthest) <= 45, "sorte {kind}");
        }
        for v in &m.vertices {
            assert!(v.pos.iter().chain(&v.normal).all(|c| c.is_finite()));
        }
    }
}
