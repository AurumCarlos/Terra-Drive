//! Pentes, tunnels et routes enterrees des autoroutes : tunnels GRAINE [AUTOROUTES]
use terra::world::generator::{dir_to_lat_lon, Generator, PlanetParams};
use terra::world::highway::STEP;

fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let count: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(60);
    let t = std::time::Instant::now();
    let gen = Generator::new(PlanetParams { seed, ..Default::default() });
    println!("planete construite en {:.1} s", t.elapsed().as_secs_f64());
    let hw = &gen.cities.highways;
    let (mut total, mut steep6, mut steep10, mut buried, mut tunnel) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut max_grade: f64 = 0.0;
    let mut worst = (0, 0.0, 0.0, 0.0, false, false);
    let mut shown = 0;
    let mut tunnels = 0;
    let t = std::time::Instant::now();
    for (i, h) in hw.list.iter().enumerate().take(count) {
        let n = (h.length() / STEP) as usize;
        let mut prev = hw.profile(&gen, i, 0.0);
        let mut inside = false;
        for k in 1..=n {
            let s = k as f64 * STEP;
            let p = hw.profile(&gen, i, s);
            let grade = ((p.road - prev.road) / STEP).abs();
            total += STEP;
            if grade > 0.06 {
                steep6 += STEP;
            }
            if grade > 0.10 {
                steep10 += STEP;
            }
            if grade > max_grade {
                max_grade = grade;
                worst = (i, s, prev.road, p.road, p.tunnel, p.wet);
            }
            if p.tunnel {
                tunnel += STEP;
                if !inside {
                    tunnels += 1;
                    if shown < 8 {
                        let d = h.dir_at((s - 300.0).max(0.0));
                        let (lat, lon) = dir_to_lat_lon(d);
                        let mut len = 0.0;
                        while hw.profile(&gen, i, s + len).tunnel {
                            len += STEP;
                        }
                        let a = h.dir_at(s);
                        let east = glam::DVec3::Y.cross(d).normalize();
                        let north = d.cross(east);
                        let cap = (a - d).dot(east).atan2((a - d).dot(north)).to_degrees();
                        println!(
                            "tunnel {i} a {:.1} km : {lat:.5},{lon:.5} cap {cap:.0}, {len:.0} m, route {:.0} m, couverture {:.0} m",
                            s / 1000.0,
                            p.road,
                            p.ground - p.road
                        );
                        // Sur la voie de droite : 120 m avant la tete, et 150 m dedans.
                        for (name, at) in [("avant", s - 120.0), ("dedans", s + 150.0)] {
                            let c = h.dir_at(at);
                            let f = h.tangent_at(at);
                            let lane = (c - c.cross(f).normalize() * (h.median() + 5.0) / terra::world::generator::planet_radius()).normalize();
                            let (la, lo) = dir_to_lat_lon(lane);
                            let east = glam::DVec3::Y.cross(lane).normalize();
                            let north = lane.cross(east);
                            println!("    {name} : {la:.6},{lo:.6},{:.0}", f.dot(east).atan2(f.dot(north)).to_degrees());
                        }
                        shown += 1;
                    }
                }
            } else if p.ground - p.road > 15.0 && !p.wet {
                buried += STEP;
            }
            inside = p.tunnel;
            prev = p;
        }
    }
    println!(
        "{count} autoroutes, {:.0} km en {:.1} s : pente > 6 % sur {:.1} km, > 10 % sur {:.1} km, max {:.0} %",
        total / 1000.0,
        t.elapsed().as_secs_f64(),
        steep6 / 1000.0,
        steep10 / 1000.0,
        max_grade * 100.0
    );
    println!("pente max : autoroute {} a {:.2} km, {:.1} -> {:.1} m, tunnel {}, eau {}", worst.0, worst.1 / 1000.0, worst.2, worst.3, worst.4, worst.5);
    if let Ok(v) = std::env::var("TERRA_AT") {
        let (i, s0): (usize, f64) = {
            let mut it = v.split(',');
            (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap())
        };
        for k in -15..=15 {
            let s = s0 * 1000.0 + k as f64 * 100.0;
            let p = hw.profile(&gen, i, s);
            println!("  {:.1} km : route {:.1} sol {:.1} eau {} tunnel {}", s / 1000.0, p.road, p.ground, p.wet, p.tunnel);
        }
    }
    println!(
        "tunnels : {tunnels}, {:.1} km ; chaussee enterree hors tunnel : {:.2} km",
        tunnel / 1000.0,
        buried / 1000.0
    );
}
