//! Bruit de gradient 3D a derivees analytiques, en f64 et sans periode : la planete
//! fait des milliers de longueurs d'onde de tour, une table de permutation s'y repeterait.

use glam::{DVec3, Vec3};

/// Petit generateur deterministe (splitmix64).
#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Valeur uniforme dans [0, 1[.
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.f32()
    }
}

/// Melange une graine et des coordonnees en une nouvelle graine.
pub fn hash_seed(seed: u64, a: i64, b: i64) -> u64 {
    let mut r = Rng(seed ^ (a as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let x = r.next_u64();
    let mut r = Rng(x ^ (b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    r.next_u64()
}

const GRADIENTS: [Vec3; 16] = [
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(-1.0, 1.0, 0.0),
    Vec3::new(1.0, -1.0, 0.0),
    Vec3::new(-1.0, -1.0, 0.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(-1.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, -1.0),
    Vec3::new(-1.0, 0.0, -1.0),
    Vec3::new(0.0, 1.0, 1.0),
    Vec3::new(0.0, -1.0, 1.0),
    Vec3::new(0.0, 1.0, -1.0),
    Vec3::new(0.0, -1.0, -1.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(-1.0, 1.0, 0.0),
    Vec3::new(0.0, -1.0, 1.0),
    Vec3::new(0.0, -1.0, -1.0),
];

/// Couche de bruit seedee.
pub struct Noise {
    seed: u64,
}

impl Noise {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Gradients des huit coins d'une maille. Les produits par coordonnee sont faits une fois
    /// (deux par axe au lieu de huit) : meme hachage, bit pour bit.
    #[inline(always)]
    fn corners(&self, ix: i64, iy: i64, iz: i64) -> [Vec3; 8] {
        const KX: u64 = 0x9E37_79B9_7F4A_7C15;
        const KY: u64 = 0xC2B2_AE3D_27D4_EB4F;
        const KZ: u64 = 0x1656_67B1_9E37_79F9;
        let x = [(ix as u64).wrapping_mul(KX), ((ix + 1) as u64).wrapping_mul(KX)];
        let y = [(iy as u64).wrapping_mul(KY), ((iy + 1) as u64).wrapping_mul(KY)];
        let z = [self.seed ^ (iz as u64).wrapping_mul(KZ), self.seed ^ ((iz + 1) as u64).wrapping_mul(KZ)];
        std::array::from_fn(|k| {
            let mut h = z[k >> 2] ^ x[k & 1] ^ y[(k >> 1) & 1];
            h = (h ^ (h >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h = (h ^ (h >> 32)).wrapping_mul(0x94D0_49BB_1331_11EB);
            GRADIENTS[(h >> 60) as usize]
        })
    }

    /// La valeur seule, sans le gradient : ce dont ont besoin fbm et ridged. Memes operations que `d3`.
    #[inline]
    pub fn value(&self, p: DVec3) -> f32 {
        let cell = p.floor();
        let f = (p - cell).as_vec3();
        let [ga, gb, gc, gd, ge, gf, gg, gh] = self.corners(cell.x as i64, cell.y as i64, cell.z as i64);
        let va = ga.dot(f);
        let vb = gb.dot(f - Vec3::new(1.0, 0.0, 0.0));
        let vc = gc.dot(f - Vec3::new(0.0, 1.0, 0.0));
        let vd = gd.dot(f - Vec3::new(1.0, 1.0, 0.0));
        let ve = ge.dot(f - Vec3::new(0.0, 0.0, 1.0));
        let vf = gf.dot(f - Vec3::new(1.0, 0.0, 1.0));
        let vg = gg.dot(f - Vec3::new(0.0, 1.0, 1.0));
        let vh = gh.dot(f - Vec3::new(1.0, 1.0, 1.0));
        let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let (u, v, w) = (fade(f.x), fade(f.y), fade(f.z));
        let (k1, k2, k3) = (vb - va, vc - va, ve - va);
        let k4 = va - vb - vc + vd;
        let k5 = va - vc - ve + vg;
        let k6 = va - vb - ve + vf;
        let k7 = -va + vb + vc - vd + ve - vf - vg + vh;
        va + k1 * u + k2 * v + k3 * w + k4 * u * v + k5 * v * w + k6 * w * u + k7 * u * v * w
    }

    /// Retourne (valeur, gradient), valeur dans environ [-1, 1].
    #[inline]
    pub fn d3(&self, p: DVec3) -> (f32, Vec3) {
        let cell = p.floor();
        let f = (p - cell).as_vec3();
        let [ga, gb, gc, gd, ge, gf, gg, gh] = self.corners(cell.x as i64, cell.y as i64, cell.z as i64);

        let va = ga.dot(f);
        let vb = gb.dot(f - Vec3::new(1.0, 0.0, 0.0));
        let vc = gc.dot(f - Vec3::new(0.0, 1.0, 0.0));
        let vd = gd.dot(f - Vec3::new(1.0, 1.0, 0.0));
        let ve = ge.dot(f - Vec3::new(0.0, 0.0, 1.0));
        let vf = gf.dot(f - Vec3::new(1.0, 0.0, 1.0));
        let vg = gg.dot(f - Vec3::new(0.0, 1.0, 1.0));
        let vh = gh.dot(f - Vec3::new(1.0, 1.0, 1.0));

        let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let dfade = |t: f32| 30.0 * t * t * (t * (t - 2.0) + 1.0);
        let (u, v, w) = (fade(f.x), fade(f.y), fade(f.z));
        let (du, dv, dw) = (dfade(f.x), dfade(f.y), dfade(f.z));

        let (k1, k2, k3) = (vb - va, vc - va, ve - va);
        let k4 = va - vb - vc + vd;
        let k5 = va - vc - ve + vg;
        let k6 = va - vb - ve + vf;
        let k7 = -va + vb + vc - vd + ve - vf - vg + vh;
        let value = va + k1 * u + k2 * v + k3 * w + k4 * u * v + k5 * v * w + k6 * w * u + k7 * u * v * w;

        let grad = ga
            + (gb - ga) * u
            + (gc - ga) * v
            + (ge - ga) * w
            + (ga - gb - gc + gd) * (u * v)
            + (ga - gc - ge + gg) * (v * w)
            + (ga - gb - ge + gf) * (w * u)
            + (-ga + gb + gc - gd + ge - gf - gg + gh) * (u * v * w)
            + Vec3::new(
                du * (k1 + k4 * v + k6 * w + k7 * v * w),
                dv * (k2 + k5 * w + k4 * u + k7 * w * u),
                dw * (k3 + k6 * u + k5 * v + k7 * u * v),
            );
        (value, grad)
    }

    /// Mouvement brownien fractionnaire classique, normalise dans environ [-1, 1].
    pub fn fbm(&self, mut p: DVec3, octaves: u32) -> f32 {
        let (mut sum, mut amp, mut norm) = (0.0, 1.0, 0.0);
        for _ in 0..octaves {
            sum += self.value(p) * amp;
            norm += amp;
            amp *= 0.5;
            p = rotate_double(p);
        }
        sum / norm
    }

    /// fBm amorti par la pente cumulee : donne des cretes nettes et des vallees lisses,
    /// proche du resultat d'une erosion hydraulique pour un cout negligeable.
    pub fn eroded(&self, mut p: DVec3, octaves: u32) -> f32 {
        let (mut sum, mut amp, mut norm) = (0.0, 1.0, 0.0);
        let mut slope = Vec3::ZERO;
        for _ in 0..octaves {
            let (n, d) = self.d3(p);
            slope += d;
            sum += amp * n / (1.0 + 0.9 * slope.length_squared());
            norm += amp;
            amp *= 0.52;
            p = rotate_double(p);
        }
        sum / norm * 1.8
    }

    /// Cretes vives dans [0, 1] : les zeros du bruit dessinent des aretes, et chaque octave ne
    /// se pose que sur les aretes de la precedente, comme les ravines sur un massif.
    /// `blunt` : largeur du maillage en periodes de la premiere octave. Une arete plus fine que le
    /// maillage y ferait des dents de scie : on l'arrondit a sa mesure.
    pub fn ridged(&self, mut p: DVec3, octaves: u32, blunt: f32) -> f32 {
        let (mut sum, mut amp, mut norm, mut weight) = (0.0, 1.0, 0.0, 1.0);
        let mut soft = blunt * 1.2;
        for _ in 0..octaves {
            let n = self.value(p);
            let crest = (1.0 - (n * n + soft * soft).sqrt() * 2.4).max(0.0);
            soft *= 2.0;
            let crest = crest * crest;
            sum += crest * amp * weight;
            norm += amp;
            weight = (crest * 1.7).clamp(0.0, 1.0);
            // Moins de la moitie par octave : la pente reste finie, meme aux plus petites echelles.
            amp *= 0.42;
            p = rotate_double(p);
        }
        sum / norm.max(1e-6) * 1.5
    }
}

/// Double la frequence avec une rotation, pour que les octaves ne s'alignent pas sur la grille.
#[inline]
fn rotate_double(p: DVec3) -> DVec3 {
    DVec3::new(
        1.6 * p.y + 1.2 * p.z + 37.31,
        -1.6 * p.x + 0.72 * p.y - 0.96 * p.z - 11.87,
        -1.2 * p.x - 0.96 * p.y + 1.28 * p.z + 5.43,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let n = Noise::new(7);
        let e = 1e-4;
        for i in 0..200 {
            let p = DVec3::new(i as f64 * 0.37 + 0.13, i as f64 * -0.73 + 0.29, i as f64 * 0.51 - 3.3);
            let (_, g) = n.d3(p);
            let diff = |d: DVec3| (n.d3(p + d).0 - n.d3(p - d).0) / (2.0 * e as f32);
            assert!((g.x - diff(DVec3::X * e)).abs() < 0.05);
            assert!((g.y - diff(DVec3::Y * e)).abs() < 0.05);
            assert!((g.z - diff(DVec3::Z * e)).abs() < 0.05);
        }
    }

    #[test]
    fn value_alone_is_the_value_of_d3() {
        let n = Noise::new(11);
        for i in 0..500 {
            let p = DVec3::new(i as f64 * 0.731 - 50.0, i as f64 * -1.37 + 3.1, i as f64 * 0.29 + 7.7);
            assert_eq!(n.value(p).to_bits(), n.d3(p).0.to_bits());
        }
    }

    #[test]
    fn stays_continuous_across_negative_coordinates() {
        let n = Noise::new(1);
        let a = n.d3(DVec3::new(-1.0001, -3.0, -0.5)).0;
        let b = n.d3(DVec3::new(-0.9999, -3.0, -0.5)).0;
        assert!((a - b).abs() < 0.01);
    }

    #[test]
    fn has_no_short_period() {
        let n = Noise::new(3);
        let mut same = 0;
        for i in 0..100 {
            let p = DVec3::new(i as f64 * 0.77 + 0.3, 0.41, 0.17);
            if (n.d3(p).0 - n.d3(p + DVec3::X * 256.0).0).abs() < 1e-5 {
                same += 1;
            }
        }
        assert!(same < 5, "{same} valeurs repetees");
    }
}
