//! Statistiques d'une planete selon ses reglages : planet_stats GRAINE TAILLE VILLES RIVIERES(0/1)
use std::time::Instant;
use terra::world::generator::{dir_to_lat_lon, Generator, PlanetParams};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let get = |i: usize, d: f64| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let params = PlanetParams { seed: get(0, 3.0) as u64, size: get(1, 1.0) as f32, cities: get(2, 140.0) as u32, rivers: get(3, 1.0) > 0.5, ..Default::default() };
    terra::world::generator::set_planet_radius(params.radius());
    let t = Instant::now();
    let gen = Generator::new(params);
    println!("planete en {:.2} s, rayon {:.0} km", t.elapsed().as_secs_f32(), params.radius() / 1000.0);
    println!("{} villes, {} troncons de riviere", gen.cities.list.len(), gen.water.edges.len());
    let hw = &gen.cities.highways;
    let km: f64 = hw.list.iter().map(|h| h.length()).sum::<f64>() / 1000.0;
    println!("{} autoroutes, {km:.0} km", hw.list.len());
    let n = gen.cities.list.len();
    let mut comp: Vec<usize> = (0..n).collect();
    fn find(c: &mut Vec<usize>, x: usize) -> usize {
        if c[x] != x {
            let r = find(c, c[x]);
            c[x] = r;
        }
        c[x]
    }
    for h in &hw.list {
        let (a, b) = (find(&mut comp, h.from), find(&mut comp, h.to));
        comp[a] = b;
    }
    let mut roots: Vec<usize> = (0..n).map(|i| find(&mut comp, i)).collect();
    roots.sort_unstable();
    roots.dedup();
    let alone = (0..n).filter(|&i| !hw.list.iter().any(|h| h.from == i || h.to == i)).count();
    println!("{} reseaux separes, {alone} villes isolees", roots.len());
    // Chevauchements : points d'une autoroute sur la chaussee d'une autre.
    let mut overlap = 0.0;
    let mut cross = 0;
    for (i, h) in hw.list.iter().enumerate() {
        let mut run = 0.0;
        for p in h.points() {
            let mut on = false;
            hw.near(*p, 40.0, |hit| on |= hit.hw != i && hit.dist < 2.0 * h.half_width());
            if on {
                overlap += 100.0;
                run += 100.0;
            } else {
                if run > 0.0 && run < 400.0 {
                    cross += 1;
                }
                run = 0.0;
            }
        }
    }
    println!("chevauchements : {:.1} km, dont {cross} croisements", overlap / 1000.0);
    // Traversees de villes : points dans le bati d'une ville, hors des deux villes reliees.
    let mut through = 0;
    for h in &hw.list {
        for p in h.points().iter().step_by(5) {
            for (ci, c) in gen.cities.list.iter().enumerate() {
                if ci != h.from && ci != h.to && c.dir.dot(*p) > 0.999 && c.urban(c.local(*p)) > 0.5 {
                    through += 1;
                }
            }
        }
    }
    println!("traversees de villes : {:.1} km", through as f64 * 0.5);
    let names = |h: &terra::world::highway::Highway| (gen.cities.list[h.from].name.clone(), gen.cities.list[h.to].name.clone());
    for (i, h) in hw.list.iter().enumerate().take(6) {
        let s = h.length() * 0.35;
        let (d, d2) = (h.dir_at(s), h.dir_at(s + 50.0));
        let (lat, lon) = dir_to_lat_lon(d);
        let p = hw.profile(&gen, i, s);
        println!("{i} {:?} {:.0} km, {} voies : {lat:.4},{lon:.4} cap {:.0} route {:.0} m", names(h), h.length() / 1000.0, h.lanes, heading(d, d2), p.road);
    }
    // Un pont sur une riviere (eau sur moins de 400 m), un viaduc, et un depart de ville.
    let (mut river, mut viaduct) = (0, 0);
    for (i, h) in hw.list.iter().enumerate().take(40) {
        let mut s = 2000.0;
        let mut run = 0.0;
        while s < h.length() - 2000.0 {
            let p = hw.profile(&gen, i, s);
            if p.wet {
                run += 20.0;
            } else {
                if run > 20.0 && run < 400.0 && river < 2 {
                    let (d, d2) = (h.dir_at(s - run * 0.5 - 250.0), h.dir_at(s - run * 0.5));
                    let (lat, lon) = dir_to_lat_lon(d);
                    println!("pont sur {:?} : {lat:.4},{lon:.4} cap {:.0} route {:.0} m, {run} m d'eau", names(h), heading(d, d2), p.road);
                    // Vue de cote, a 120 m de l'axe.
                    let c = h.dir_at(s - run * 0.5);
                    let side = c.cross(h.tangent_at(s - run * 0.5));
                    let cam = (c + side * (120.0 / terra::world::generator::planet_radius())).normalize();
                    let (lat, lon) = dir_to_lat_lon(cam);
                    println!("  de cote : {lat:.5},{lon:.5} cap {:.0}", heading(cam, c));
                    river += 1;
                }
                run = 0.0;
                if p.road - p.ground > 18.0 && viaduct < 2 {
                    let (d, d2) = (h.dir_at(s - 250.0), h.dir_at(s));
                    let (lat, lon) = dir_to_lat_lon(d);
                    println!("viaduc sur {:?} : {lat:.4},{lon:.4} cap {:.0} route {:.0} m, sol {:.0} m", names(h), heading(d, d2), p.road, p.ground);
                    viaduct += 1;
                    s += 3000.0;
                }
            }
            s += 20.0;
        }
    }
    let h = &hw.list[0];
    let (d, d2) = (h.dir_at(0.0), h.dir_at(400.0));
    let (lat, lon) = dir_to_lat_lon(d);
    println!("depart de {} : {lat:.4},{lon:.4} cap {:.0}", gen.cities.list[h.from].name, heading(d, d2));

    // Reseau des rues : chaque ville d'un seul tenant, chaque autoroute branchee sur une rue.
    if std::env::var("TERRA_SKIP_PLANS").is_ok() {
        return;
    }
    let t = Instant::now();
    let count = gen.cities.list.len();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let r = params.radius();
    let results: Vec<(usize, usize, Vec<f64>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..std::thread::available_parallelism().map_or(4, |n| n.get()))
            .map(|_| {
                let (gen, next) = (&gen, &next);
                s.spawn(move || {
                    terra::world::generator::set_creating_radius(r);
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= count {
                            break;
                        }
                        let c = &gen.cities.list[i];
                        let plan = c.plan(gen);
                        let mut gaps = Vec::new();
                        for h in &gen.cities.highways.list {
                            if h.from == i {
                                gaps.push(plan.road_distance(c.local(h.points()[0])));
                            }
                            if h.to == i {
                                gaps.push(plan.road_distance(c.local(*h.points().last().unwrap())));
                            }
                        }
                        out.push((i, plan.road_networks(), gaps));
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let split: Vec<_> = results.iter().filter(|r| r.1 > 1).collect();
    let loose: Vec<f64> = results.iter().flat_map(|r| r.2.iter().copied()).filter(|g| *g > 2.0).collect();
    let ends: usize = results.iter().map(|r| r.2.len()).sum();
    println!("plans en {:.1} s : {} villes au reseau coupe, {}/{} bouts d'autoroute loin d'une rue", t.elapsed().as_secs_f32(), split.len(), loose.len(), ends);
    for (i, k, _) in split.iter().take(8) {
        println!("  {} : {k} morceaux", gen.cities.list[*i].name);
    }
    for (i, _, g) in results.iter().filter(|r| r.2.iter().any(|g| *g > 2.0)).take(8) {
        let c = &gen.cities.list[*i];
        println!("  {} ({:?}, rayon {:.0}) : {:?}", c.name, c.style, c.radius, g.iter().map(|x| x.round()).collect::<Vec<_>>());
        let plan = c.plan(&gen);
        for h in &gen.cities.highways.list {
            for (who, p) in [(h.from, h.points()[0]), (h.to, *h.points().last().unwrap())] {
                if who != *i {
                    continue;
                }
                let q = c.local(p);
                if plan.road_distance(q) <= 2.0 {
                    continue;
                }
                let a = q.y.atan2(q.x);
                let line = glam::DVec2::from_angle(a);
                let along: f64 = plan
                    .roads
                    .iter()
                    .filter(|r| line.perp().dot(r.a).abs() < 3.0 && line.perp().dot(r.b).abs() < 3.0)
                    .map(|r| r.a.dot(line).max(r.b.dot(line)))
                    .fold(f64::MIN, f64::max);
                let (lat, lon) = dir_to_lat_lon(p);
                println!("    bout a {:.0} m (contour {:.0} m), rue de sortie jusqu'a {:.0} m ; {lat:.5},{lon:.5}", q.length(), c.reach(a), along);
            }
        }
    }
}

/// Cap d'une direction vers une autre, en degres depuis le nord.
pub fn heading(at: glam::DVec3, to: glam::DVec3) -> f64 {
    let east = glam::DVec3::Y.cross(at).normalize();
    let north = at.cross(east);
    let t = to - at;
    t.dot(east).atan2(t.dot(north)).to_degrees()
}
