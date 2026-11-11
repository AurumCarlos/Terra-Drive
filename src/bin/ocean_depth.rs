//! Profondeur des mers : ocean_depth GRAINE
use terra::world::generator::{dir_to_lat_lon, planet_radius, Generator, PlanetParams};

fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let gen = Generator::new(PlanetParams { seed, ..Default::default() });
    let n = 60_000;
    let (mut base_d, mut full_d, mut lakes) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..n {
        let a = k as f64 * 2.399963229728653;
        let z = 1.0 - 2.0 * (k as f64 + 0.5) / n as f64;
        let r = (1.0 - z * z).sqrt();
        let dir = glam::DVec3::new(a.cos() * r, z, a.sin() * r);
        let s = gen.surface(dir);
        if s.height >= s.water_level {
            continue;
        }
        if s.fresh {
            lakes.push(s.water_level - s.height);
            continue;
        }
        full_d.push(s.water_level - s.height);
        base_d.push(-gen.water.elevation.sample(dir));
    }
    for (name, mut v) in [("grille", base_d), ("surface", full_d), ("douce", lakes)] {
        v.sort_by(f32::total_cmp);
        let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        println!(
            "{name:8} {} points : moyenne {mean:.0} m, mediane {:.0}, 10 % {:.0}, 90 % {:.0}, max {:.0}, <10 m {:.1} %, <200 m {:.1} %",
            v.len(),
            q(0.5),
            q(0.1),
            q(0.9),
            q(1.0),
            v.iter().filter(|d| **d < 10.0).count() as f32 / v.len() as f32 * 100.0,
            v.iter().filter(|d| **d < 200.0).count() as f32 / v.len() as f32 * 100.0
        );
    }

    // Profil depuis le rivage : on part d'un point de cote et on s'eloigne vers le large.
    let dists = [50.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 30000.0, 100000.0];
    let mut sums = vec![Vec::new(); dists.len()];
    let mut shown = 0;
    for k in 0..20_000 {
        let a = k as f64 * 2.399963229728653;
        let z = 1.0 - 2.0 * (k as f64 + 0.5) / 20_000.0;
        let r = (1.0 - z * z).sqrt();
        let dir = glam::DVec3::new(a.cos() * r, z, a.sin() * r);
        let s = gen.surface(dir);
        if s.height < 2.0 || s.height > 10.0 || s.fresh {
            continue;
        }
        // Direction du large : la ou la mer est la plus proche a 3 km.
        let side = dir.any_orthonormal_vector();
        let other = dir.cross(side);
        let mut best = None;
        for j in 0..16 {
            let t = j as f64 / 16.0 * std::f64::consts::TAU;
            let v = side * t.cos() + other * t.sin();
            let q = (dir + v * 3000.0 / planet_radius()).normalize();
            let sq = gen.surface(q);
            if !sq.fresh && sq.height < sq.water_level - 1.0 {
                best = Some(v);
                break;
            }
        }
        let Some(v) = best else { continue };
        // Le rivage : premier point de mer en marchant vers le large.
        let Some(shore) = (0..160).map(|k| k as f64 * 25.0).find(|d| {
            let sq = gen.surface((dir + v * *d / planet_radius()).normalize());
            !sq.fresh && sq.height < sq.water_level
        }) else {
            continue;
        };
        let dir = (dir + v * shore / planet_radius()).normalize();
        let row: Vec<f32> = dists
            .iter()
            .map(|d| {
                let q = (dir + v * *d / planet_radius()).normalize();
                let sq = gen.surface(q);
                sq.water_level - sq.height
            })
            .collect();
        if shown < 6 {
            let (lat, lon) = dir_to_lat_lon(dir);
            println!("cote {lat:.4},{lon:.4} cap {:.0} : {:?}", 0.0, row.iter().map(|d| *d as i32).collect::<Vec<_>>());
            shown += 1;
        }
        for (i, d) in row.iter().enumerate() {
            sums[i].push(*d);
        }
    }
    for (i, d) in dists.iter().enumerate() {
        let mut v = sums[i].clone();
        v.sort_by(f32::total_cmp);
        println!("a {:6.0} m du rivage : mediane {:6.1} m, 10 % {:6.1}, 90 % {:6.1} ({} cotes)", d, v[v.len() / 2], v[v.len() / 10], v[v.len() * 9 / 10], v.len());
    }
}
