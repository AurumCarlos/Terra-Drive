//! Plan d'une ville vu du ciel, sans lancer le jeu : city_plan GRAINE [RANG] sortie.png
//! Sans rang, liste les plus grandes villes de la planete.

use glam::DVec2;
use terra::world::city::{Ground, Zone};
use terra::world::generator::{dir_to_lat_lon, Generator, PlanetParams};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seed: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(3);
    let gen = Generator::new(PlanetParams { seed, ..Default::default() });
    let Some(rank) = args.get(1).and_then(|s| s.parse::<usize>().ok()) else {
        for (i, c) in gen.cities.list.iter().enumerate().take(40) {
            let (lat, lon) = dir_to_lat_lon(c.dir);
            println!("{i:3} {:<16} {:?} taille {:.2} rayon {:.1} km  {lat:.3},{lon:.3}", c.name, c.style, c.size, c.radius / 1000.0);
        }
        println!("{} villes", gen.cities.list.len());
        return;
    };
    let out = args.get(2).cloned().unwrap_or_else(|| "plan.png".into());
    let city = &gen.cities.list[rank];
    let t = std::time::Instant::now();
    let plan = city.plan(&gen);
    println!(
        "{} ({:?}) : {} rues, {} ilots, {} batiments en {:.2} s",
        city.name,
        city.style,
        plan.roads.len(),
        plan.blocks.len(),
        plan.buildings.len(),
        t.elapsed().as_secs_f32()
    );
    // Les premiers ponts : position du milieu et cap de la rue.
    let mut found = 0;
    for r in &plan.roads {
        if found >= 4 || r.length() < 60.0 {
            continue;
        }
        let deck = terra::world::city_mesh::deck_profile(city, &gen, r);
        if let Some((k, lift)) = deck.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            if *lift > 3.0 {
                let p = r.a + r.dir() * (k as f64 * terra::world::city_mesh::DECK_STEP);
                let (lat, lon) = dir_to_lat_lon(city.dir_of(p - r.dir() * 70.0));
                let yaw = 90.0 - r.dir().y.atan2(r.dir().x).to_degrees();
                println!("pont : {lat:.5},{lon:.5} cap {yaw:.0} surelevation {lift:.1} m largeur {}", r.width);
                found += 1;
            }
        }
    }
    let size = 1600usize;
    let half = city.radius * 1.15;
    let mut image = vec![0u8; size * size * 3];
    for y in 0..size {
        for x in 0..size {
            let p = DVec2::new((x as f64 / size as f64 * 2.0 - 1.0) * half, (1.0 - y as f64 / size as f64 * 2.0) * half);
            let c = match plan.ground(p) {
                Ground::Road => [40, 40, 44],
                Ground::Walk => [150, 145, 135],
                Ground::Building => match plan.block_zone(p) {
                    Some(Zone::Tower) => [60, 80, 120],
                    Some(Zone::Old) | Some(Zone::Plaza) => [170, 90, 60],
                    Some(Zone::Industry) => [120, 120, 125],
                    Some(Zone::Houses) => [190, 120, 90],
                    _ => [215, 195, 160],
                },
                Ground::Paved => [200, 190, 170],
                Ground::Yard => [175, 165, 150],
                Ground::Garden => [110, 160, 80],
                Ground::Park => [70, 140, 60],
                Ground::Wild => {
                    let s = gen.surface(city.dir_of(p));
                    if s.height < s.water_level { [40, 90, 160] } else { [140, 150, 110] }
                }
            };
            image[(y * size + x) * 3..][..3].copy_from_slice(&c);
        }
    }
    let file = std::fs::File::create(&out).expect("fichier de sortie");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), size as u32, size as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&image).unwrap();
    println!("plan : {out}");
}
