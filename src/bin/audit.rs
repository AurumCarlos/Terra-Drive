//! Outil de verification : ce qu'on voit depuis des lieux tires au hasard, sans lancer le jeu.
//! Chaque vignette est une vue a hauteur d'homme volant, calculee directement depuis la fonction de
//! planete. Affiche aussi la part de lieux ternes, et le temps de calcul d'un point.
//! audit GRAINE sortie.png [COLONNES] [LIGNES] [ALTITUDE]

use glam::{DVec3, Vec3};
use std::time::Instant;
use terra::world::generator::{dir_to_lat_lon, planet_radius, Detail, Generator, PlanetParams};
use terra::world::noise::{hash_seed, Rng};

const TILE_W: usize = 320;
const TILE_H: usize = 180;
const STEPS: usize = 900;
const NEAR: f64 = 30.0;
const FAR: f64 = 60_000.0;
const SKY_LOW: [f32; 3] = [0.74, 0.8, 0.86];
const SKY_HIGH: [f32; 3] = [0.34, 0.5, 0.74];

fn random_land(gen: &Generator, rng: &mut Rng) -> DVec3 {
    loop {
        let y = rng.f32() as f64 * 2.0 - 1.0;
        let a = rng.f32() as f64 * std::f64::consts::TAU;
        let r = (1.0 - y * y).sqrt();
        let dir = DVec3::new(a.cos() * r, y, a.sin() * r);
        let s = gen.surface_detail(dir, &Detail::for_spacing(50.0));
        if s.height > s.water_level {
            return dir;
        }
    }
}

/// Une vue : pour chaque colonne, on marche sur le relief et on dessine d'avant en arriere.
fn view(gen: &Generator, dir: DVec3, yaw: f64, altitude: f32, out: &mut [[f32; 3]]) {
    let axis = if dir.y.abs() > 0.99 { DVec3::Z } else { DVec3::Y };
    let north = (axis - dir * axis.dot(dir)).normalize();
    let east = north.cross(dir);
    let forward = north * yaw.cos() + east * yaw.sin();
    let right = forward.cross(dir).normalize();
    let eye = gen.surface_detail(dir, &Detail::for_spacing(4.0)).height.max(0.0) + altitude;
    let focal = TILE_W as f64 * 0.75;
    let horizon = TILE_H as f64 * 0.42;
    let sun_local = (forward * 0.55 - right * 0.75 + dir * 0.34).normalize();

    for y in 0..TILE_H {
        let t = (y as f32 / horizon as f32).clamp(0.0, 1.0);
        for x in 0..TILE_W {
            out[y * TILE_W + x] = std::array::from_fn(|c| SKY_HIGH[c] + (SKY_LOW[c] - SKY_HIGH[c]) * t);
        }
    }

    let ratio = (FAR / NEAR).powf(1.0 / STEPS as f64);
    let distances: Vec<f64> = (0..STEPS).map(|k| NEAR * ratio.powi(k as i32)).collect();
    let column_dir = |x: f64, z: f64| {
        let lateral = (x - TILE_W as f64 * 0.5) / focal * z;
        (dir + (forward * z + right * lateral) / planet_radius()).normalize()
    };
    // Le relief sur une grille polaire : la normale s'en deduit sans calcul de plus.
    let mut grid = vec![(0.0f32, 0.0f32, false, 0.0f32, 0.0f32); (TILE_W + 1) * STEPS];
    for k in 0..STEPS {
        let z = distances[k];
        let detail = Detail::for_spacing((z / focal).max(2.0));
        for x in 0..=TILE_W {
            let s = gen.surface_detail(column_dir(x as f64, z), &detail);
            grid[k * (TILE_W + 1) + x] = (s.height, s.water_level, s.fresh, s.bank, s.ridge);
        }
    }
    for x in 0..TILE_W {
        let mut lowest = TILE_H as i32;
        for k in 1..STEPS - 1 {
            let z = distances[k];
            let (height, level, fresh, bank, ridge) = grid[k * (TILE_W + 1) + x];
            let top = height.max(level);
            let drop = z * z / (2.0 * planet_radius());
            let screen = horizon - focal * ((top - eye) as f64 - drop) / z;
            let screen = screen.floor() as i32;
            if screen >= lowest {
                continue;
            }
            let p = column_dir(x as f64 + 0.5, z);
            let step_z = distances[k + 1] - distances[k - 1];
            let step_x = z / focal;
            let along = (grid[(k + 1) * (TILE_W + 1) + x].0 - grid[(k - 1) * (TILE_W + 1) + x].0) as f64 / step_z;
            let across = (grid[k * (TILE_W + 1) + x + 1].0 - height) as f64 / step_x;
            let normal = (dir - forward * along - right * across).normalize();
            let normal_y = normal.dot(dir) as f32;
            let color = if height < level {
                let depth = ((level - height) / 40.0).clamp(0.0, 1.0);
                [0.05 - 0.03 * depth, 0.14 - 0.06 * depth, 0.2 - 0.05 * depth]
            } else {
                let climate = gen.climate(p, height);
                let above = if fresh { (height - level) * 3.0 + (1.0 - bank) * 10.0 } else { height - level };
                let around = grid[(k - 1) * (TILE_W + 1) + x].0 + grid[(k + 1) * (TILE_W + 1) + x].0;
                let concavity = ((around - 2.0 * height) / step_z as f32 * 2.0).clamp(-1.0, 1.0);
                let ground = gen.ground_color(p, above, normal_y, climate, concavity, ridge);
                let lit = (normal.dot(sun_local) as f32).max(0.0) * 2.1 + 0.42;
                let (canopy, _) = gen.with_canopy(ground, p, above, normal_y, climate, ridge, z / focal);
                canopy.map(|v| v * lit)
            };
            let fog = 1.0 - (-(z as f32) / 42_000.0).exp();
            let color: [f32; 3] = std::array::from_fn(|c| color[c] + (SKY_LOW[c] - color[c]) * fog);
            let from = screen.max(0) as usize;
            for y in from..lowest as usize {
                out[y * TILE_W + x] = color;
            }
            lowest = screen.max(0);
        }
    }
}

/// Part des lieux dont le relief reste faible dans un rayon donne, sans eau visible.
fn stats(gen: &Generator, seed: u64) {
    let mut rng = Rng(hash_seed(seed, 0xA0D, 1));
    let detail = Detail::for_spacing(25.0);
    let (mut relief_2, mut relief_10, mut dull) = (Vec::new(), Vec::new(), 0);
    let count = 1500;
    for _ in 0..count {
        let dir = random_land(gen, &mut rng);
        let side = dir.any_orthonormal_vector();
        let other = dir.cross(side);
        let measure = |radius: f64| {
            let (mut lo, mut hi, mut water) = (f32::MAX, f32::MIN, false);
            for k in 0..24 {
                let a = k as f64 / 24.0 * std::f64::consts::TAU;
                let r = radius * (0.35 + 0.65 * ((k % 3) as f64 / 2.0));
                let s = gen.surface_detail(
                    (dir + (side * a.cos() + other * a.sin()) * (r / planet_radius())).normalize(),
                    &detail,
                );
                lo = lo.min(s.height);
                hi = hi.max(s.height);
                water |= s.height < s.water_level;
            }
            (hi - lo, water)
        };
        let (near, _) = measure(2000.0);
        let (far, water) = measure(10_000.0);
        relief_2.push(near);
        relief_10.push(far);
        if far < 150.0 && near < 50.0 && !water {
            dull += 1;
        }
    }
    relief_2.sort_by(f32::total_cmp);
    relief_10.sort_by(f32::total_cmp);
    let pct = |v: &[f32], p: usize| v[v.len() * p / 100];
    println!(
        "relief 2 km  : p10 {:.0} m, p50 {:.0} m, p90 {:.0} m",
        pct(&relief_2, 10),
        pct(&relief_2, 50),
        pct(&relief_2, 90)
    );
    println!(
        "relief 10 km : p10 {:.0} m, p50 {:.0} m, p90 {:.0} m",
        pct(&relief_10, 10),
        pct(&relief_10, 50),
        pct(&relief_10, 90)
    );
    println!("lieux ternes : {:.1} %", dull as f32 * 100.0 / count as f32);
}

fn main() -> terra::Result<()> {
    let mut args = std::env::args().skip(1);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let path = args.next().unwrap_or_else(|| "audit.png".to_string());
    let columns: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(6);
    let rows: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(6);
    let altitude: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(120.0);

    let t = Instant::now();
    let gen = Generator::new(PlanetParams { seed, ..Default::default() });
    println!("planete generee en {:.2} s", t.elapsed().as_secs_f32());

    let t = Instant::now();
    let mut rng = Rng(hash_seed(seed, 0xB0B, 2));
    let mut total = 0.0f32;
    for _ in 0..20_000 {
        let y = rng.f32() as f64 * 2.0 - 1.0;
        let a = rng.f32() as f64 * std::f64::consts::TAU;
        let r = (1.0 - y * y).sqrt();
        total += gen.surface(DVec3::new(a.cos() * r, y, a.sin() * r)).height;
    }
    println!("surface au plus fin : {:.1} us par point (somme {total:.0})", t.elapsed().as_secs_f32() * 1e6 / 20_000.0);

    stats(&gen, seed);
    let mut rng = Rng(hash_seed(seed, 0xA0D, 9));
    let mut share = std::collections::BTreeMap::new();
    for _ in 0..3000 {
        *share.entry(gen.landform_name(random_land(&gen, &mut rng))).or_insert(0) += 1;
    }
    println!("paysages : {share:?}");
    println!("{} volcans, dont :", gen.volcanoes.list.len());
    for v in gen.volcanoes.list.iter().step_by(gen.volcanoes.list.len() / 6 + 1) {
        let (lat, lon) = dir_to_lat_lon(v.dir);
        println!("  {lat:.2},{lon:.2} sommet {:.0} m, sol {:.0} m", v.summit, gen.water.elevation.sample(v.dir));
    }

    let places: Vec<(DVec3, f64)> = {
        let mut rng = Rng(hash_seed(seed, 0xA0D, 3));
        (0..columns * rows).map(|_| (random_land(&gen, &mut rng), rng.f32() as f64 * std::f64::consts::TAU)).collect()
    };
    for (i, (dir, yaw)) in places.iter().enumerate() {
        let (lat, lon) = dir_to_lat_lon(*dir);
        let s = gen.surface(*dir);
        let c = gen.climate(*dir, s.height);
        println!(
            "{:>2} (l{} c{}) {:7.2},{:8.2} cap {:3.0} {:6.0} m t{:.2} h{:.2} {:?} {}",
            i,
            i / columns,
            i % columns,
            lat,
            lon,
            yaw.to_degrees(),
            s.height,
            c.temperature,
            c.moisture,
            gen.biome(c),
            gen.landform_name(*dir)
        );
    }

    let t = Instant::now();
    let mut tiles = vec![vec![[0.0f32; 3]; TILE_W * TILE_H]; places.len()];
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<_> = tiles.iter_mut().map(std::sync::Mutex::new).collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= places.len() {
                    break;
                }
                let mut tile = terra::sync::lock(&slots[i]);
                view(&gen, places[i].0, places[i].1, altitude, &mut tile);
            });
        }
    });
    drop(slots);
    println!("vues en {:.1} s", t.elapsed().as_secs_f32());

    let (w, h) = (columns * (TILE_W + 2), rows * (TILE_H + 2));
    let mut image = vec![16u8; w * h * 3];
    for (i, tile) in tiles.iter().enumerate() {
        let (ox, oy) = ((i % columns) * (TILE_W + 2), (i / columns) * (TILE_H + 2));
        for y in 0..TILE_H {
            for x in 0..TILE_W {
                let c = Vec3::from_array(tile[y * TILE_W + x]);
                let k = ((oy + y) * w + ox + x) * 3;
                for (p, v) in image[k..k + 3].iter_mut().zip(c.to_array()) {
                    *p = (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
                }
            }
        }
    }
    terra::write_png(std::path::Path::new(&path), w as u32, h as u32, png::ColorType::Rgb, &image)?;
    println!("vues : {path}");
    Ok(())
}
