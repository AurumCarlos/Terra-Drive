//! Grilles globales sur les six faces d'un cube projete sur la sphere, avec la meme projection
//! que le quadtree du terrain, et champs echantillonnables en tout point de la planete.

use glam::DVec3;
use std::f64::consts::FRAC_PI_4;

/// Axes (normale, tangente, bitangente) des six faces.
pub const FACES: [(DVec3, DVec3, DVec3); 6] = [
    (DVec3::X, DVec3::Z, DVec3::Y),
    (DVec3::NEG_X, DVec3::NEG_Z, DVec3::Y),
    (DVec3::Y, DVec3::X, DVec3::Z),
    (DVec3::NEG_Y, DVec3::NEG_X, DVec3::Z),
    (DVec3::Z, DVec3::NEG_X, DVec3::Y),
    (DVec3::NEG_Z, DVec3::X, DVec3::Y),
];

/// Direction d'un point d'une face. `u` et `v` vont de -1 a 1, et peuvent en deborder.
pub fn face_point(face: usize, u: f64, v: f64) -> DVec3 {
    let (n, a, b) = FACES[face];
    (n + a * (u * FRAC_PI_4).tan() + b * (v * FRAC_PI_4).tan()).normalize()
}

/// Face qui contient une direction, et coordonnees (u, v) sur cette face.
pub fn face_of(dir: DVec3) -> (usize, f64, f64) {
    let a = dir.abs();
    let face = if a.x >= a.y && a.x >= a.z {
        (dir.x < 0.0) as usize
    } else if a.y >= a.z {
        2 + (dir.y < 0.0) as usize
    } else {
        4 + (dir.z < 0.0) as usize
    };
    let (n, s, t) = FACES[face];
    let d = dir.dot(n);
    (face, (dir.dot(s) / d).atan() / FRAC_PI_4, (dir.dot(t) / d).atan() / FRAC_PI_4)
}

/// Grille de `n` x `n` cellules par face.
#[derive(Clone, Copy, Debug)]
pub struct CubeGrid {
    pub n: usize,
}

impl CubeGrid {
    /// Nombre total de cellules, sur les six faces.
    pub fn cells(&self) -> usize {
        6 * self.n * self.n
    }

    /// Face (0 a 5), colonne et ligne de la cellule `cell`.
    pub fn coords(&self, cell: usize) -> (usize, usize, usize) {
        let per_face = self.n * self.n;
        let (face, rest) = (cell / per_face, cell % per_face);
        (face, rest % self.n, rest / self.n)
    }

    fn to_uv(self, i: f64) -> f64 {
        (i + 0.5) / self.n as f64 * 2.0 - 1.0
    }

    /// Direction du centre de la cellule.
    pub fn center(&self, cell: usize) -> DVec3 {
        let (face, i, j) = self.coords(cell);
        face_point(face, self.to_uv(i as f64), self.to_uv(j as f64))
    }

    /// Cellule qui contient la direction `dir`.
    pub fn cell_of(&self, dir: DVec3) -> usize {
        let (face, u, v) = face_of(dir);
        let n = self.n as f64;
        let i = (((u + 1.0) * 0.5 * n) as usize).min(self.n - 1);
        let j = (((v + 1.0) * 0.5 * n) as usize).min(self.n - 1);
        (face * self.n + j) * self.n + i
    }

    /// Les huit voisines d'une cellule, y compris par-dessus les aretes du cube.
    /// Retourne leur nombre : un coin du cube n'en a que sept.
    pub fn neighbours(&self, cell: usize, out: &mut [u32; 8]) -> usize {
        let (face, i, j) = self.coords(cell);
        let n = self.n;
        if i > 0 && j > 0 && i + 1 < n && j + 1 < n {
            let (c, w) = (cell as u32, n as u32);
            *out = [c - w - 1, c - w, c - w + 1, c - 1, c + 1, c + w - 1, c + w, c + w + 1];
            return 8;
        }
        let mut count = 0;
        for dj in -1i32..=1 {
            for di in -1i32..=1 {
                if di == 0 && dj == 0 {
                    continue;
                }
                let dir = face_point(face, self.to_uv(i as f64 + di as f64), self.to_uv(j as f64 + dj as f64));
                let other = self.cell_of(dir) as u32;
                if other as usize != cell && !out[..count].contains(&other) {
                    out[count] = other;
                    count += 1;
                }
            }
        }
        count
    }
}

const PAD: usize = 4;

/// Valeurs d'une grille, avec une marge prise sur les faces voisines : l'interpolation
/// reste continue en passant d'une face a l'autre.
pub struct CubeField {
    n: usize,
    stride: usize,
    data: Vec<f32>,
}

impl CubeField {
    /// Champ a partir d'une valeur par cellule de `grid` (`cells` en compte `grid.cells()`). La marge
    /// de chaque face est echantillonnee sur les faces voisines, pour interpoler sans couture.
    pub fn from_cells(grid: CubeGrid, cells: &[f32]) -> Self {
        let n = grid.n;
        let stride = n + 2 * PAD;
        let mut data = vec![0.0; 6 * stride * stride];
        let raw = |dir: DVec3| {
            let (face, u, v) = face_of(dir);
            let x = ((u + 1.0) * 0.5 * n as f64 - 0.5).clamp(0.0, n as f64 - 1.0);
            let y = ((v + 1.0) * 0.5 * n as f64 - 0.5).clamp(0.0, n as f64 - 1.0);
            let (ix, iy) = ((x as usize).min(n - 2), (y as usize).min(n - 2));
            let (fx, fy) = ((x - ix as f64) as f32, (y - iy as f64) as f32);
            let at = |a: usize, b: usize| cells[(face * n + b) * n + a];
            let bottom = at(ix, iy) + (at(ix + 1, iy) - at(ix, iy)) * fx;
            let top = at(ix, iy + 1) + (at(ix + 1, iy + 1) - at(ix, iy + 1)) * fx;
            bottom + (top - bottom) * fy
        };
        for face in 0..6 {
            for j in 0..stride {
                for i in 0..stride {
                    let inside = (PAD..n + PAD).contains(&i) && (PAD..n + PAD).contains(&j);
                    let value = if inside {
                        cells[(face * n + j - PAD) * n + i - PAD]
                    } else {
                        let u = (i as f64 - PAD as f64 + 0.5) / n as f64 * 2.0 - 1.0;
                        let v = (j as f64 - PAD as f64 + 0.5) / n as f64 * 2.0 - 1.0;
                        raw(face_point(face, u, v))
                    };
                    data[(face * stride + j) * stride + i] = value;
                }
            }
        }
        Self { n, stride, data }
    }

    /// Interpolation bilineaire.
    #[inline]
    pub fn sample_linear(&self, dir: DVec3) -> f32 {
        self.joined(dir, Self::linear)
    }

    /// Pres d'une arete, melange les valeurs vues de chaque face : sur l'arete, et meme au coin ou
    /// trois faces se touchent, toutes calculent la meme moyenne et le raccord est parfait.
    #[inline]
    fn joined(&self, dir: DVec3, read: impl Fn(&Self, usize, f64, f64) -> f32) -> f32 {
        let (face, u, v) = face_of(dir);
        let value = read(self, face, u, v);
        let margin = 2.0 / self.n as f64;
        if u.abs().max(v.abs()) < 1.0 - margin {
            return value;
        }
        let (_, s, t) = FACES[face];
        let (mut sum, mut total) = (value, 1.0f32);
        for (coord, axis) in [(u, s), (v, t)] {
            let k = ((coord.abs() - (1.0 - margin)) / margin).clamp(0.0, 1.0) as f32;
            if k <= 0.0 {
                continue;
            }
            let across = axis * coord.signum();
            let other = FACES.iter().position(|f| f.0.dot(across) > 0.5).unwrap_or(face);
            let (n, os, ot) = FACES[other];
            let d = dir.dot(n);
            if d > 0.0 {
                let w = k * k * (3.0 - 2.0 * k);
                sum +=
                    w * read(self, other, (dir.dot(os) / d).atan() / FRAC_PI_4, (dir.dot(ot) / d).atan() / FRAC_PI_4);
                total += w;
            }
        }
        sum / total
    }

    /// `sample_linear` de plusieurs champs de meme taille au meme point : la face et ses coordonnees
    /// (deux arctangentes) ne se calculent qu'une fois. Memes valeurs, bit pour bit.
    pub fn sample_linear_many<const N: usize>(fields: [&CubeField; N], dir: DVec3) -> [f32; N] {
        let (face, u, v) = face_of(dir);
        let mut sum: [f32; N] = std::array::from_fn(|i| fields[i].linear(face, u, v));
        let margin = 2.0 / fields[0].n as f64;
        debug_assert!(fields.iter().all(|f| f.n == fields[0].n));
        if u.abs().max(v.abs()) < 1.0 - margin {
            return sum;
        }
        let (_, s, t) = FACES[face];
        let mut total = 1.0f32;
        for (coord, axis) in [(u, s), (v, t)] {
            let k = ((coord.abs() - (1.0 - margin)) / margin).clamp(0.0, 1.0) as f32;
            if k <= 0.0 {
                continue;
            }
            let across = axis * coord.signum();
            let other = FACES.iter().position(|f| f.0.dot(across) > 0.5).unwrap_or(face);
            let (n, os, ot) = FACES[other];
            let d = dir.dot(n);
            if d > 0.0 {
                let w = k * k * (3.0 - 2.0 * k);
                let (ou, ov) = ((dir.dot(os) / d).atan() / FRAC_PI_4, (dir.dot(ot) / d).atan() / FRAC_PI_4);
                for (i, f) in fields.iter().enumerate() {
                    sum[i] += w * f.linear(other, ou, ov);
                }
                total += w;
            }
        }
        sum.map(|v| v / total)
    }

    #[inline]
    fn linear(&self, face: usize, u: f64, v: f64) -> f32 {
        let n = self.n as f64;
        // Dans la marge, l'indice decale de PAD ne devient jamais negatif.
        let x = (u + 1.0) * 0.5 * n - 0.5 + PAD as f64;
        let y = (v + 1.0) * 0.5 * n - 0.5 + PAD as f64;
        let (ix, iy) = (x.floor(), y.floor());
        let (fx, fy) = ((x - ix) as f32, (y - iy) as f32);
        let base = (face * self.stride + iy as usize) * self.stride + ix as usize;
        let (a, b) = (self.data[base], self.data[base + 1]);
        let (c, d) = (self.data[base + self.stride], self.data[base + self.stride + 1]);
        let bottom = a + (b - a) * fx;
        bottom + (c + (d - c) * fx - bottom) * fy
    }

    /// Spline cubique : lisse, sans les facettes en losange d'une grille bilineaire.
    #[inline]
    pub fn sample(&self, dir: DVec3) -> f32 {
        self.joined(dir, Self::spline)
    }

    #[inline]
    fn spline(&self, face: usize, u: f64, v: f64) -> f32 {
        let n = self.n as f64;
        let x = (u + 1.0) * 0.5 * n - 0.5 + PAD as f64;
        let y = (v + 1.0) * 0.5 * n - 0.5 + PAD as f64;
        let (ix, iy) = (x.floor(), y.floor());
        let (fx, fy) = ((x - ix) as f32, (y - iy) as f32);
        let (x, y) = (ix as usize, iy as usize);
        let weights = |t: f32| {
            let (t2, t3) = (t * t, t * t * t);
            [
                (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0,
                (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
                (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
                t3 / 6.0,
            ]
        };
        let (wx, wy) = (weights(fx), weights(fy));
        let origin = (face * self.stride + y - 1) * self.stride + x - 1;
        let mut sum = 0.0;
        for (b, wb) in wy.iter().enumerate() {
            let row = &self.data[origin + b * self.stride..origin + b * self.stride + 4];
            sum += wb * (row[0] * wx[0] + row[1] * wx[1] + row[2] * wx[2] + row[3] * wx[3]);
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_many_fields_gives_the_same_values() {
        let grid = CubeGrid { n: 16 };
        let a = CubeField::from_cells(grid, &(0..grid.cells()).map(|i| (i as f32 * 0.37).sin()).collect::<Vec<_>>());
        let b = CubeField::from_cells(grid, &(0..grid.cells()).map(|i| (i as f32 * 0.11).cos()).collect::<Vec<_>>());
        for i in 0..2000 {
            let x = i as f64;
            let dir = DVec3::new((x * 0.7).sin(), (x * 1.3).cos(), (x * 0.29).sin() * 1.3).normalize();
            let [va, vb] = CubeField::sample_linear_many([&a, &b], dir);
            assert_eq!(va.to_bits(), a.sample_linear(dir).to_bits());
            assert_eq!(vb.to_bits(), b.sample_linear(dir).to_bits());
        }
    }

    #[test]
    fn cells_and_directions_round_trip() {
        let grid = CubeGrid { n: 32 };
        for cell in (0..grid.cells()).step_by(37) {
            assert_eq!(grid.cell_of(grid.center(cell)), cell);
        }
    }

    #[test]
    fn neighbours_are_close_and_mutual_even_across_cube_edges() {
        let grid = CubeGrid { n: 16 };
        let spacing = std::f64::consts::FRAC_PI_2 / 16.0;
        let mut out = [0; 8];
        let mut back = [0; 8];
        for cell in 0..grid.cells() {
            let count = grid.neighbours(cell, &mut out);
            assert!((7..=8).contains(&count), "cellule {cell} : {count} voisines");
            for &other in &out[..count] {
                let angle = grid.center(cell).dot(grid.center(other as usize)).clamp(-1.0, 1.0).acos();
                assert!(angle < spacing * 2.2, "voisine a {angle} rad");
                let m = grid.neighbours(other as usize, &mut back);
                assert!(back[..m].contains(&(cell as u32)), "{cell} et {other} non reciproques");
            }
        }
    }

    #[test]
    fn fields_are_smooth_across_the_faces() {
        // Un champ lineaire doit etre retrouve partout, meme a cheval sur deux faces.
        let grid = CubeGrid { n: 48 };
        let f = |d: DVec3| (d.x * 2.0 + d.y - d.z * 0.5) as f32;
        let cells: Vec<f32> = (0..grid.cells()).map(|c| f(grid.center(c))).collect();
        let field = CubeField::from_cells(grid, &cells);
        for i in 0..2000 {
            let a = i as f64 * 2.399963;
            let y = 1.0 - (i as f64 + 0.5) / 1000.0;
            let r = (1.0 - y * y).sqrt();
            let d = DVec3::new(a.cos() * r, y, a.sin() * r);
            assert!((field.sample_linear(d) - f(d)).abs() < 0.02);
            assert!((field.sample(d) - f(d)).abs() < 0.02);
        }
        // De part et d'autre d'une arete, l'ecart reste infime.
        let edge = DVec3::new(1.0, 0.2, 1.0).normalize();
        let e = DVec3::new(1.0, 0.0, -1.0).normalize() * 1e-7;
        assert!((field.sample(edge + e) - field.sample(edge - e)).abs() < 1e-3);
    }
}
