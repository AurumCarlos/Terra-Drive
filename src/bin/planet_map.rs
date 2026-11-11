//! Outil de verification : carte plate du relief, des plaques, des rivieres et des lacs.
//! planet_map GRAINE sortie.png [OCEAN] [VILLES]

use std::path::Path;
use std::time::Instant;
use terra::world::generator::{Generator, PlanetParams};

fn main() -> terra::Result<()> {
    let mut args = std::env::args().skip(1);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let path = args.next().unwrap_or_else(|| "map.png".to_string());
    let ocean: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.68);

    let t = Instant::now();
    let cities: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(PlanetParams::default().cities);
    let gen = Generator::new(PlanetParams { seed, ocean, cities, ..Default::default() });
    println!("planete generee en {:.2} s", t.elapsed().as_secs_f32());
    println!(
        "{} troncons, {:.0} km de rivieres, {} lacs",
        gen.water.edges.len(),
        gen.water.network_km(),
        gen.water.lakes.len()
    );
    let (w, h) = (2048, 1024);
    let image = gen.map_rgb(w, h, false);
    terra::write_png(Path::new(&path), w as u32, h as u32, png::ColorType::Rgb, &image)?;
    println!("carte : {path}");
    // Cartes du climat a cote, et bilan compare a la Terre (Peel et al. 2007).
    for mode in ["koppen", "temp", "pluie"] {
        let out = path.replace(".png", &format!("_{mode}.png"));
        let image = gen.climate_map_rgb(w, h, mode);
        terra::write_png(Path::new(&out), w as u32, h as u32, png::ColorType::Rgb, &image)?;
    }
    let (shares, [rain_land, temp_land, rain, temp], land) = gen.climate_stats();
    println!("terres : {:.1} % ; pluie {rain_land:.0} mm sur les terres (Terre : 800), {rain:.0} en tout (990)", land * 100.0);
    println!("temperature : {temp_land:.1} C sur les terres (Terre : 8,5 avec l'Antarctique), {temp:.1} en tout (14-15)");
    // Hypsometrie, comparee a la Terre (ETOPO1).
    let (mut land_h, mut sea_h): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
    let mut lakes = 0usize;
    for y in 0..720 {
        let lat = 90.0 - (y as f64 + 0.5) / 720.0 * 180.0;
        let count = (1440.0 * lat.to_radians().cos()).ceil() as usize;
        for x in 0..count {
            let dir = terra::world::generator::lat_lon_to_dir(lat, (x as f64 + 0.5) / count as f64 * 360.0 - 180.0);
            let e = gen.water.elevation.sample_linear(dir);
            if e > 0.0 {
                land_h.push(e);
                lakes += gen.water.is_lake(dir) as usize;
            } else {
                sea_h.push(e);
            }
        }
    }
    land_h.sort_by(f32::total_cmp);
    let share = |limit: f32| land_h.iter().filter(|h| **h > limit).count() as f32 / land_h.len() as f32 * 100.0;
    println!(
        "relief des terres : moyenne {:.0} m (Terre : 800), mediane {:.0} m (450), maximum {:.0} m (8 850)",
        land_h.iter().sum::<f32>() / land_h.len() as f32,
        land_h[land_h.len() / 2],
        land_h[land_h.len() - 1]
    );
    println!(
        "  au-dessus de 500 m : {:.0} % (Terre : 45), 1 000 m : {:.0} % (27), 2 000 m : {:.1} % (11), 4 000 m : {:.1} % (2)",
        share(500.0),
        share(1000.0),
        share(2000.0),
        share(4000.0)
    );
    let shelf = sea_h.iter().filter(|h| **h > -200.0).count() as f32 / sea_h.len() as f32 * 100.0;
    println!(
        "oceans : profondeur moyenne {:.0} m (Terre : 3 700), plateau continental {shelf:.1} % (7,5) ; lacs : {:.1} % des terres (3)",
        -sea_h.iter().sum::<f32>() / sea_h.len() as f32,
        lakes as f32 / land_h.len() as f32 * 100.0
    );
    // Part des terres en plaine : moins de 110 m de denivele dans un rayon de 3,5 km (le critere d'une ville).
    {
        use terra::world::generator::Detail;
        let coarse = Detail::for_spacing(150.0);
        let (mut flat, mut seen) = (0, 0);
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        for i in 0..6000 {
            let y = 1.0 - (i as f64 + 0.5) / 6000.0 * 2.0;
            let r = (1.0 - y * y).sqrt();
            let dir = glam::DVec3::new((i as f64 * golden).cos() * r, y, (i as f64 * golden).sin() * r);
            if gen.water.elevation.sample(dir) < 6.0 {
                continue;
            }
            seen += 1;
            let east = glam::DVec3::Y.cross(dir).normalize_or(glam::DVec3::X);
            let north = dir.cross(east);
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for k in 0..8 {
                let a = k as f64 / 8.0 * std::f64::consts::TAU;
                let p = (dir + (east * a.cos() + north * a.sin()) * (3500.0 / terra::world::generator::planet_radius())).normalize();
                let h = gen.surface_natural(p, &coarse).height;
                lo = lo.min(h);
                hi = hi.max(h);
            }
            flat += (hi - lo < 110.0) as usize;
        }
        println!("plaines : {:.0} % des terres (Terre : environ la moitie)", flat as f32 / seen.max(1) as f32 * 100.0);
    }
    let mut dry_lakes = 0;
    let mut big = Vec::new();
    for lake in &gen.water.lakes {
        let c = gen.climate(lake.outlet, lake.level);
        dry_lakes += (c.aridity() < 0.3) as usize;
        if lake.cells > 60 {
            let (lat, lon) = terra::world::generator::dir_to_lat_lon(lake.outlet);
            big.push(format!("{} cellules a {lat:.0},{lon:.0} ({}, aridite {:.2})", lake.cells, c.koppen().name(), c.aridity()));
        }
    }
    println!("lacs en pays aride : {dry_lakes} sur {} ; grands lacs : {}", gen.water.lakes.len(), big.len());
    for line in big.iter().take(12) {
        println!("  {line}");
    }
    let earth = [8.0, 11.0, 19.0, 11.2, 2.6, 10.8, 9.6, 15.0, 8.0, 4.8];
    for (k, kind) in terra::world::generator::Koppen::ALL.iter().enumerate() {
        println!("  {:<24} {:5.1} %   (Terre : {:.1} %)", kind.name(), shares[k] * 100.0, earth[k]);
    }
    Ok(())
}
