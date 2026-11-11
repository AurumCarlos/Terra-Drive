//! Embouchures et confluences des grandes rivieres, pour les captures : river_spots GRAINE
use glam::DVec3;
use terra::world::generator::{dir_to_lat_lon, planet_radius, Generator, PlanetParams};
use terra::world::hydrology::RiverEnd;

/// Cap de a vers b, en degres a droite du nord.
fn heading(a: DVec3, b: DVec3) -> f64 {
    let east = DVec3::Y.cross(a).normalize();
    let north = a.cross(east);
    let t = b - a;
    t.dot(east).atan2(t.dot(north)).to_degrees()
}

fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let gen = Generator::new(PlanetParams { seed, ..Default::default() });
    let water = &gen.water;
    let r = planet_radius();

    // Embouchures : les plus larges, la ou le trace touche la mer.
    let mut mouths: Vec<usize> = (0..water.edges.len()).filter(|&e| water.edges[e].end == RiverEnd::Sea).collect();
    mouths.sort_by(|a, b| water.edges[*b].width[1].total_cmp(&water.edges[*a].width[1]));
    println!("{} embouchures", mouths.len());
    for &e in mouths.iter().take(if std::env::var("TERRA_PROFILE_MOUTH").is_ok() { 3 } else { 8 }) {
        let edge = &water.edges[e];
        let [a, m, b] = water.edge_points(edge);
        // Le long du troncon, jusqu'a la premiere eau salee.
        let along = |t: f64| gen.unbend(if t < 0.5 { a.lerp(m, t * 2.0) } else { m.lerp(b, t * 2.0 - 1.0) }.normalize());
        let mut coast = None;
        for k in 0..=400 {
            let t = k as f64 / 400.0;
            let s = gen.surface(along(t));
            if !s.fresh && s.height < s.water_level - 0.3 {
                coast = Some(t);
                break;
            }
        }
        if std::env::var("TERRA_PROFILE_MOUTH").is_ok() {
            let (lat, lon) = dir_to_lat_lon(along(0.0));
            println!("troncon {e} depuis {lat:.5},{lon:.5}, niveaux {:?}, largeur {:?}", edge.level, edge.width);
            for k in 0..=20 {
                let t = k as f64 / 20.0;
                let p = along(t);
                let s = gen.surface(p);
                let base = gen.water.elevation.sample(p);
                let (lat, lon) = dir_to_lat_lon(p);
                println!(
                    "  t {t:.2} {lat:.5},{lon:.5} : eau {:.1} sol {:.1} {} grille {:.0}",
                    s.water_level,
                    s.height,
                    if s.fresh { "douce" } else { "mer" },
                    base
                );
            }
        }
        let Some(t) = coast else { continue };
        let c = along(t);
        let (lat, lon) = dir_to_lat_lon(c);
        let back = along((t - 0.08).max(0.0));
        let levels: Vec<String> = [3000.0, 1500.0, 600.0, 200.0, 0.0]
            .iter()
            .map(|d| {
                let p = (c + (back - c).normalize() * (d / r)).normalize();
                let s = gen.surface(p);
                format!("{:.1}/{:.1}{}", s.water_level, s.height, if s.fresh { "d" } else { "s" })
            })
            .collect();
        println!(
            "embouchure {lat:.5},{lon:.5} cap {:.0} : lit {:.0} m, troncon {:.1} km, eau/sol a 3 km..0 : {}",
            heading(back, c),
            edge.width[1] * 2.0,
            (a - b).length() * r / 1000.0,
            levels.join(" ")
        );
    }

    // Confluences : deux rivieres larges qui se rejoignent.
    let mut inflows: std::collections::HashMap<u32, Vec<usize>> = std::collections::HashMap::new();
    for (i, e) in water.edges.iter().enumerate() {
        if e.end == RiverEnd::Continues {
            inflows.entry(e.next).or_default().push(i);
        }
    }
    let mut joins: Vec<(f32, usize, usize)> = Vec::new();
    for list in inflows.values() {
        if list.len() < 2 {
            continue;
        }
        let mut l = list.clone();
        l.sort_by(|a, b| water.edges[*b].width[1].total_cmp(&water.edges[*a].width[1]));
        let (main, side) = (l[0], l[1]);
        joins.push((water.edges[side].width[1], main, side));
    }
    joins.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("{} confluences", joins.len());
    if let Ok(k) = std::env::var("TERRA_PROFILE_JOIN") {
        let (_, main, side) = joins[k.parse::<usize>().unwrap_or(0)];
        let next = water.edges[main].next as usize;
        for (name, e) in [("principal", main), ("affluent", side), ("aval", next)] {
            let edge = &water.edges[e];
            let [a, m, b] = water.edge_points(edge);
            println!("{name} {e} : niveaux {:?}, largeurs {:?}", edge.level, edge.width);
            for k in 0..=12 {
                let t = k as f64 / 12.0;
                let p = gen.unbend(if t < 0.5 { a.lerp(m, t * 2.0) } else { m.lerp(b, t * 2.0 - 1.0) }.normalize());
                let s = gen.surface(p);
                let raw = gen.water.river_at(gen.meander(p), 2.0, 1.0, 0.0);
                let (lat, lon) = dir_to_lat_lon(p);
                println!(
                    "  t {t:.2} {lat:.5},{lon:.5} : eau {:.1} sol {:.1} {} | riviere niveau {:.1} profil {:.1} mouille {}",
                    s.water_level,
                    s.height,
                    if s.fresh { "douce" } else { "-" },
                    raw.map_or(f32::NAN, |r| r.level),
                    raw.map_or(f32::NAN, |r| r.profile),
                    raw.is_some_and(|r| r.wet)
                );
            }
        }
    }
    for &(w, main, side) in joins.iter().take(8) {
        let [_, sm, sb] = water.edge_points(&water.edges[side]).map(|p| gen.unbend(p));
        let [ma, _, mb] = water.edge_points(&water.edges[main]).map(|p| gen.unbend(p));
        let (lat, lon) = dir_to_lat_lon(sb);
        let s = gen.surface(sb);
        println!(
            "confluence {lat:.5},{lon:.5} : affluent {:.0} m (arrive cap {:.0}), principal {:.0} m (cap {:.0}), eau {:.1} sol {:.1}",
            w * 2.0,
            heading(sm, sb),
            water.edges[main].width[1] * 2.0,
            heading(ma, mb),
            s.water_level,
            s.height
        );
    }
}
