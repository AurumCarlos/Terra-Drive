//! Lieux boises de chaque biome, pour les captures : biome_spots GRAINE
use terra::world::generator::{dir_to_lat_lon, Generator, PlanetParams};

fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let gen = Generator::new(PlanetParams { seed, ..Default::default() });
    let mut found: Vec<(String, f32, f64, f64)> = Vec::new();
    let n = 400_000;
    for k in 0..n {
        let a = k as f64 * 2.399963229728653;
        let z = 1.0 - 2.0 * (k as f64 + 0.5) / n as f64;
        let r = (1.0 - z * z).sqrt();
        let dir = glam::DVec3::new(a.cos() * r, z, a.sin() * r);
        let s = gen.surface(dir);
        if s.height < s.water_level + 5.0 || s.height > 1800.0 {
            continue;
        }
        let c = gen.climate(dir, s.height);
        let biome = gen.biome(c);
        let cover = gen.tree_cover(&c, gen.woods(dir)) * gen.forest_share(dir, s.ridge);
        let key = format!("{:?}/{:?}/{}", biome, c.koppen(), if dir.y > 0.0 { "N" } else { "S" });
        let score = cover + 0.3 * s.valley;
        if let Some(e) = found.iter_mut().find(|e| e.0 == key) {
            if score > e.1 {
                let (lat, lon) = dir_to_lat_lon(dir);
                *e = (key, score, lat, lon);
            }
        } else {
            let (lat, lon) = dir_to_lat_lon(dir);
            found.push((key, score, lat, lon));
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    for (k, s, lat, lon) in found {
        println!("{k:40} couvert {s:.2}  {lat:.4},{lon:.4}");
    }
}
