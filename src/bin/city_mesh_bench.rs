//! Temps de construction des chunks d'une ville, LOD par LOD : city_mesh_bench GRAINE RANG
use std::time::Instant;
use terra::world::chunk::{build_chunk, NodeId};
use terra::world::generator::{Generator, PlanetParams};


fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seed: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(3);
    let rank: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let gen = Generator::new(PlanetParams { seed, ..Default::default() });
    let city = &gen.cities.list[rank];
    let t = Instant::now();
    city.plan(&gen);
    println!("{} : plan en {:.2} s", city.name, t.elapsed().as_secs_f32());
    for lod in 0..=8u8 {
        for (k, off) in [(0.0, 0.0), (300.0, 200.0), (-900.0, 500.0)].iter().enumerate() {
            let dir = city.dir_of(glam::DVec2::new(off.0, off.1));
            let id = NodeId::containing(dir, lod);
            let t = Instant::now();
            let mesh = build_chunk(&gen, id);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let (v, tris) = mesh.city.as_ref().map_or((0, 0), |c| (c.vertices.len(), c.indices.len() / 3));
            println!("LOD {lod} #{k} : {ms:7.1} ms  {v:7} sommets  {tris:7} triangles");
        }
    }
}
