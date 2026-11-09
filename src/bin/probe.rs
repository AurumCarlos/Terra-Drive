//! Sonde : provinces et relief autour d'un lieu. probe GRAINE LAT LON [PAS_M]
use terra::world::generator::{lat_lon_to_dir, Detail, Generator, PlanetParams, planet_radius};

fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let gen = Generator::new(PlanetParams { seed: a[0] as u64, ..Default::default() });
    let dir = lat_lon_to_dir(a[1], a[2]);
    println!("{:?}
orogenie {:.2}, base {:.0} m", gen.provinces.at(dir), gen.orogeny.sample_linear(dir), gen.water.elevation.sample(dir));
    let side = dir.any_orthonormal_vector();
    let other = dir.cross(side);
    let step = a.get(3).copied().unwrap_or(2000.0);
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
