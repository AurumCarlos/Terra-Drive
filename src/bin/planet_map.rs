//! Outil de verification : carte plate du relief, des plaques, des rivieres et des lacs.
//! planet_map GRAINE sortie.png [OCEAN]

use std::path::Path;
use std::time::Instant;
use terra::world::generator::{Generator, PlanetParams};

fn main() -> terra::Result<()> {
    let mut args = std::env::args().skip(1);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let path = args.next().unwrap_or_else(|| "map.png".to_string());
    let ocean: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.68);

    let t = Instant::now();
    let gen = Generator::new(PlanetParams { seed, ocean, ..Default::default() });
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
    Ok(())
}
