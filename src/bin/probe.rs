//! Sonde : provinces et relief autour d'un lieu. probe GRAINE LAT LON [PAS_M]
use terra::world::generator::{lat_lon_to_dir, planet_radius, Detail, Generator, PlanetParams};

fn main() {
    let parsed: Result<Vec<f64>, _> = std::env::args().skip(1).map(|s| s.parse()).collect();
    let (seed, lat, lon, step) = match parsed.as_deref() {
        Ok([seed, lat, lon]) => (*seed, *lat, *lon, 2000.0),
        Ok([seed, lat, lon, step]) if *step > 0.0 => (*seed, *lat, *lon, *step),
        _ => {
            eprintln!("probe GRAINE LAT LON [PAS_M] : trois ou quatre nombres, le pas strictement positif");
            std::process::exit(2);
        }
    };
    let gen = Generator::new(PlanetParams { seed: seed as u64, ..Default::default() });
    let dir = lat_lon_to_dir(lat, lon);
    println!(
        "{:?}
orogenie {:.2}, base {:.0} m",
        gen.provinces.at(dir),
        gen.orogeny.sample_linear(dir),
        gen.water.elevation.sample(dir)
    );
    let side = dir.any_orthonormal_vector();
    let other = dir.cross(side);
    let d = Detail::for_spacing((step / 40.0).max(1.5));
    for j in -5..=5 {
        let row: Vec<String> = (-5..=5)
            .map(|i| {
                let p = (dir + (side * i as f64 + other * j as f64) * step / planet_radius()).normalize();
                format!("{:5.0}", gen.surface_detail(p, &d).height)
            })
            .collect();
        println!("{}", row.join(""));
    }
}
