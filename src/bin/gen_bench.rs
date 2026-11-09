//! Temps de generation : creation de la planete, puis chunks, arbres et herbe autour d'un lieu
//! de depart et d'une ville, comme les demanderait le streaming ; cout de chaque etape d'un chunk ;
//! sondes du terrain sur un coeur et sur tous. gen_bench [GRAINE] (TERRA_PROFILE=1 : etapes de la creation)
use std::time::Instant;
use terra::world::chunk::{build_chunk, NodeId};
use terra::world::generator::{Detail, Generator, PlanetParams};
use terra::world::scatter::{build_grass, build_trees};

fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let t = Instant::now();
    let gen = Generator::new(PlanetParams { seed, ..Default::default() });
    println!("planete : {:.2} s", t.elapsed().as_secs_f64());

    let spawn = gen.find_spawn();
    let city = &gen.cities.list[0];
    let t = Instant::now();
    city.plan(&gen);
    println!("plan de {} : {:.2} s", city.name, t.elapsed().as_secs_f64());
    let town = city.dir_of(glam::DVec2::new(150.0, -80.0));
    {
        // Sondes eparses dans toute la ville, comme le plan.
        let d = Detail::for_spacing(25.0);
        let pts: Vec<_> = (0..20_000).map(|k| {
            let a = k as f64 * 2.399;
            let r = (k as f64 / 20_000.0).sqrt() * city.radius;
            city.dir_of(glam::DVec2::new(a.cos() * r, a.sin() * r))
        }).collect();
        for (name, f) in [
            ("surface", &(|p| gen.surface_detail(p, &d).height) as &dyn Fn(glam::DVec3) -> f32),
            ("naturel", &|p| gen.surface_natural(p, &d).height),
            ("aplanir", &|p| gen.cities.flatten(p)),
        ] {
            let t = Instant::now();
            let acc: f32 = pts.iter().map(|p| f(*p)).sum();
            std::hint::black_box(acc);
            println!("sonde ville {name} : {:.1} us", t.elapsed().as_secs_f64() * 1e6 / pts.len() as f64);
        }
        let t = Instant::now();
        let v = terra::world::generator::par_map(pts.len(), |k| gen.surface_detail(pts[k], &d).height);
        std::hint::black_box(v);
        // Un ecart faible entre 1 coeur et tous les coeurs trahit un verrou ou un cache partage.
        println!("sonde ville sur tous les coeurs ({}) : {:.2} us", std::thread::available_parallelism().map_or(0, |n| n.get()), t.elapsed().as_secs_f64() * 1e6 / pts.len() as f64);
    }

    let mut total_chunks = 0.0;
    let mut count = 0;
    for (name, dir) in [("depart", spawn), ("ville", town)] {
        let mut ids = Vec::new();
        for lod in 0..=12u8 {
            let c = NodeId::containing(dir, lod);
            ids.push(c);
            // Les voisins du meme niveau, comme autour d'une camera.
            for (dx, dz) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                let (x, z) = (c.x as i64 + dx, c.z as i64 + dz);
                let max = 1i64 << (17 - lod);
                if (0..max).contains(&x) && (0..max).contains(&z) {
                    ids.push(NodeId { x: x as u32, z: z as u32, ..c });
                }
            }
        }
        let t = Instant::now();
        let mut by_lod = [0.0f64; 13];
        for id in &ids {
            let s = Instant::now();
            std::hint::black_box(build_chunk(&gen, *id));
            by_lod[id.lod as usize] += s.elapsed().as_secs_f64() * 1000.0;
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        total_chunks += ms;
        count += ids.len();
        println!("{name} : {} chunks en {ms:.0} ms ({:.1} ms/chunk)", ids.len(), ms / ids.len() as f64);
        println!("  par LOD (ms, 5 chunks) : {}", by_lod.iter().map(|v| format!("{v:.0}")).collect::<Vec<_>>().join(" "));

        // Cellules de vegetation sous la camera.
        let t = Instant::now();
        let (mut trees, mut grass) = (0, 0);
        for lod in [4u8, 5, 6] {
            let c = NodeId::containing(dir, lod);
            trees += build_trees(&gen, c).objects.iter().map(Vec::len).sum::<usize>();
        }
        let tt = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        for lod in [1u8, 2] {
            let c = NodeId::containing(dir, lod);
            grass += build_grass(&gen, c).tufts.len();
        }
        println!("  arbres {trees} en {tt:.0} ms, herbe {grass} en {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    }
    // Detail d'un chunk : temps de chaque etape sur sa grille de 51 x 51 points.
    for (name, dir, lod) in [("depart L3", spawn, 3u8), ("depart L9", spawn, 9), ("ville L2", town, 2), ("ville L8", town, 8)] {
        let id = NodeId::containing(dir, lod);
        let d = Detail::for_spacing(id.spacing());
        let dirs: Vec<_> = (0..51 * 51).map(|k| id.dir_at((k % 51) as f64 / 48.0, (k / 51) as f64 / 48.0)).collect();
        let time = |f: &dyn Fn(glam::DVec3) -> f32| {
            let t = Instant::now();
            let mut acc = 0.0f32;
            for &dir in &dirs {
                acc += f(dir);
            }
            std::hint::black_box(acc);
            t.elapsed().as_secs_f64() * 1000.0
        };
        let parts = [
            ("surface", time(&|p| gen.surface_detail(p, &d).height)),
            ("naturel", time(&|p| gen.surface_natural(p, &d).height)),
            ("climat", time(&|p| gen.climate(p, 100.0).moisture)),
            ("sol", time(&|p| gen.ground_color(p, 10.0, 0.9, gen.climate(p, 100.0), 0.0, 0.0)[0])),
            ("canopee", time(&|p| gen.with_canopy([0.1; 3], p, 10.0, 0.9, gen.climate(p, 100.0), 0.0, id.spacing()).1)),
            ("villes", time(&|p| gen.cities.paint(p, id.spacing(), &gen).map_or(0.0, |c| c.1))),
            ("autoroutes", time(&|p| gen.cities.highways.paint(p, id.spacing()).map_or(0.0, |c| c.1))),
            ("circuits", time(&|p| gen.cities.circuits.paint(p, id.spacing()).map_or(0.0, |c| c.1))),
            ("roche", time(&|p| gen.rock_color(p).1)),
        ];
        let t = Instant::now();
        std::hint::black_box(build_chunk(&gen, id));
        let total = t.elapsed().as_secs_f64() * 1000.0;
        println!("{name} : chunk {total:.1} ms | {}", parts.iter().map(|(n, v)| format!("{n} {v:.1}")).collect::<Vec<_>>().join(", "));
    }
    println!("TOTAL chunks : {count} en {total_chunks:.0} ms ({:.1} ms/chunk)", total_chunks / count as f64);
}
