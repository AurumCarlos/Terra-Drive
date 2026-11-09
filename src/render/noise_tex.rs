//! Texture de bruit 3D periodique calculee au demarrage, avec ses mips : un echantillon coute
//! bien moins cher qu'un bruit evalue dans le shader, et s'efface proprement avec la distance.
//!
//! r : Perlin-Worley (masses des nuages), g et b : Worley moyen et fin (erosion des bords),
//! a : Perlin fractal etale sur [0, 1] (usage general).

pub const NOISE_SIZE: usize = 96;

fn hash(x: i32, y: i32, z: i32, salt: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8DA6_B343)
        ^ (y as u32).wrapping_mul(0xD816_3841)
        ^ (z as u32).wrapping_mul(0xCB1A_B31F)
        ^ salt.wrapping_mul(0x1656_67B1);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5BD1_E995);
    h ^ (h >> 15)
}

fn unit(h: u32) -> f32 {
    (h & 0xFFFF) as f32 / 65535.0
}

fn wrap(v: i32, period: i32) -> i32 {
    v.rem_euclid(period)
}

fn gradient(x: i32, y: i32, z: i32, period: i32, salt: u32) -> [f32; 3] {
    let h = hash(wrap(x, period), wrap(y, period), wrap(z, period), salt);
    let (a, b) = (unit(h) * std::f32::consts::TAU, unit(h >> 16) * 2.0 - 1.0);
    let r = (1.0 - b * b).max(0.0).sqrt();
    [a.cos() * r, a.sin() * r, b]
}

/// Bruit de gradient periodique, a peu pres dans [-1, 1]. `p` va de 0 a 1 sur une periode.
fn perlin(p: [f32; 3], period: i32, salt: u32) -> f32 {
    let q = p.map(|c| c * period as f32);
    let i = q.map(|c| c.floor() as i32);
    let f = [q[0] - i[0] as f32, q[1] - i[1] as f32, q[2] - i[2] as f32];
    let u = f.map(|t| t * t * t * (t * (t * 6.0 - 15.0) + 10.0));
    let corner = |dx: i32, dy: i32, dz: i32| {
        let g = gradient(i[0] + dx, i[1] + dy, i[2] + dz, period, salt);
        g[0] * (f[0] - dx as f32) + g[1] * (f[1] - dy as f32) + g[2] * (f[2] - dz as f32)
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x0 = lerp(corner(0, 0, 0), corner(1, 0, 0), u[0]);
    let x1 = lerp(corner(0, 1, 0), corner(1, 1, 0), u[0]);
    let x2 = lerp(corner(0, 0, 1), corner(1, 0, 1), u[0]);
    let x3 = lerp(corner(0, 1, 1), corner(1, 1, 1), u[0]);
    lerp(lerp(x0, x1, u[1]), lerp(x2, x3, u[1]), u[2]) * 1.5
}

/// Bruit cellulaire periodique : 1 au centre d'une cellule, 0 loin de tout point.
fn worley(p: [f32; 3], period: i32, salt: u32) -> f32 {
    let q = p.map(|c| c * period as f32);
    let i = q.map(|c| c.floor() as i32);
    let mut best = 9.0f32;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (x, y, z) = (i[0] + dx, i[1] + dy, i[2] + dz);
                let h = hash(wrap(x, period), wrap(y, period), wrap(z, period), salt);
                let h2 = h.wrapping_mul(0x9E37_79B1);
                let point = [x as f32 + unit(h), y as f32 + unit(h >> 16), z as f32 + unit(h2)];
                let d = [point[0] - q[0], point[1] - q[1], point[2] - q[2]];
                best = best.min(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            }
        }
    }
    (1.0 - best.sqrt()).clamp(0.0, 1.0)
}

fn worley_fbm(p: [f32; 3], periods: [i32; 3], salt: u32) -> f32 {
    worley(p, periods[0], salt) * 0.625 + worley(p, periods[1], salt + 1) * 0.25 + worley(p, periods[2], salt + 2) * 0.125
}

fn perlin_fbm(p: [f32; 3], salt: u32) -> f32 {
    let mut sum = 0.0;
    let (mut amp, mut period) = (0.5, 4);
    for octave in 0..5 {
        sum += perlin(p, period, salt + octave) * amp;
        amp *= 0.5;
        period *= 2;
    }
    sum
}

fn voxel(p: [f32; 3]) -> [u8; 4] {
    let wf = worley_fbm(p, [4, 8, 16], 10);
    let pf = (perlin_fbm(p, 40) * 1.3 + 0.5).clamp(0.0, 1.0);
    // Le Perlin donne les grandes formes, le Worley les arrondit en choux-fleurs.
    let r = ((pf - (wf - 1.0)) / (2.0 - wf)).clamp(0.0, 1.0);
    let r = ((r - 0.25) / 0.6).clamp(0.0, 1.0);
    let g = worley_fbm(p, [8, 16, 32], 20);
    let b = worley_fbm(p, [12, 24, 48], 30);
    let a = (perlin_fbm(p, 60) * 1.7 + 0.5).clamp(0.0, 1.0);
    [r, g, b, a].map(|v| (v * 255.0 + 0.5) as u8)
}

/// Niveaux de la texture du plus grand au plus petit, en RGBA8.
pub fn build_noise() -> Vec<Vec<u8>> {
    let n = NOISE_SIZE;
    let mut base = vec![0u8; n * n * n * 4];
    let threads = std::thread::available_parallelism().map_or(4, |c| c.get());
    let per_chunk = (n / threads + 1) * n * n * 4;
    std::thread::scope(|s| {
        for (k, chunk) in base.chunks_mut(per_chunk).enumerate() {
            s.spawn(move || {
                let first = k * (n / threads + 1);
                for (index, px) in chunk.chunks_mut(4).enumerate() {
                    let (x, y, z) = (index % n, index / n % n, index / (n * n) + first);
                    let p = [x, y, z].map(|c| (c as f32 + 0.5) / n as f32);
                    px.copy_from_slice(&voxel(p));
                }
            });
        }
    });

    let mut levels = vec![base];
    let mut size = n;
    while size > 1 {
        let next = size / 2;
        let last = levels.last().expect("au moins un niveau");
        let mut mip = vec![0u8; next * next * next * 4];
        for z in 0..next {
            for y in 0..next {
                for x in 0..next {
                    for c in 0..4 {
                        let mut sum = 0u32;
                        for o in 0..8 {
                            let (a, b, d) = (x * 2 + (o & 1), y * 2 + ((o >> 1) & 1), z * 2 + ((o >> 2) & 1));
                            sum += last[((d % size) * size + b % size) * size * 4 + (a % size) * 4 + c] as u32;
                        }
                        mip[(z * next + y) * next * 4 + x * 4 + c] = (sum / 8) as u8;
                    }
                }
            }
        }
        levels.push(mip);
        size = next;
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noises_repeat_exactly_over_a_period() {
        let (a, b) = ([0.13, 0.57, 0.91], [1.13, 1.57, 1.91]);
        assert!((perlin(a, 8, 3) - perlin(b, 8, 3)).abs() < 1e-3);
        assert!((worley(a, 8, 3) - worley(b, 8, 3)).abs() < 1e-3);
    }

    #[test]
    fn every_channel_varies_and_mips_shrink_to_one_voxel() {
        let levels = build_noise();
        assert_eq!(levels[0].len(), NOISE_SIZE.pow(3) * 4);
        assert_eq!(levels.last().map(Vec::len), Some(4));
        for c in 0..4 {
            let values = levels[0].iter().skip(c).step_by(4);
            let (lo, hi) = values.fold((255u8, 0u8), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            assert!(hi - lo > 100, "canal {c} trop plat : {lo}..{hi}");
        }
    }
}
