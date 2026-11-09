//! Tectonique des plaques simulee. Chaque plaque est un ensemble de particules de croute rigides
//! qui tournent autour d'un pole d'Euler. La ou deux plaques se recouvrent, l'une passe sous
//! l'autre et le relief monte. La ou elles s'ecartent, une dorsale cree de la croute oceanique.

use super::cube::CubeGrid;
use super::noise::{hash_seed, Noise, Rng};
use glam::{DVec3, Quat, Vec3};

/// Resolution de la simulation : environ 80 km par cellule.
pub const TECTONIC_N: usize = 128;
/// Nombre de pas : environ 15 millions d'annees chacun.
pub const STEPS: usize = 48;
const NONE: u32 = u32::MAX;

/// Type de frontiere de plaque a la fin de la simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    None = 0,
    Collision = 1,
    Subduction = 2,
    Ridge = 3,
}

pub struct Tectonics {
    pub grid: CubeGrid,
    pub plate: Vec<u16>,
    /// 0 pour la croute oceanique, 1 pour la croute continentale.
    pub crust: Vec<f32>,
    /// Soulevement accumule par les collisions, et lentement erode.
    pub orogeny: Vec<f32>,
    /// Age de la croute, en pas de simulation.
    pub age: Vec<f32>,
    /// Fosse de subduction recente.
    pub trench: Vec<f32>,
    pub boundary: Vec<Boundary>,
    /// Nombre de cellules ou se sont produites des collisions continentales.
    pub collisions: u64,
    pub subductions: u64,
}

#[derive(Clone, Copy)]
struct Particle {
    /// Position dans le repere de la plaque : fixe, donc aucune perte par reechantillonnage.
    home: Vec3,
    plate: u16,
    crust: f32,
    orogeny: f32,
    age: f32,
}

struct Plate {
    rotation: Quat,
    axis: Vec3,
    /// Radians par pas.
    rate: f32,
    base_rate: f32,
}

fn random_axis(rng: &mut Rng) -> Vec3 {
    let z = rng.range(-1.0, 1.0);
    let a = rng.range(0.0, std::f32::consts::TAU);
    let r = (1.0 - z * z).sqrt();
    Vec3::new(a.cos() * r, z, a.sin() * r)
}

/// Valeur au-dessus de laquelle se trouve la part `share` des valeurs.
pub fn threshold_for_share(values: &[f32], share: f32) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let index = (((1.0 - share) * sorted.len() as f32) as usize).min(sorted.len() - 1);
    sorted[index]
}

/// `land_share` : part de la surface couverte de croute continentale.
pub fn simulate(seed: u64, plate_count: usize, land_share: f32) -> Tectonics {
    let grid = CubeGrid { n: TECTONIC_N };
    let cells = grid.cells();
    let mut rng = Rng(hash_seed(seed, 0x7EC7, 1));
    let centers: Vec<DVec3> = (0..cells).map(|c| grid.center(c)).collect();
    let neighbours: Vec<([u32; 8], u8)> = (0..cells)
        .map(|c| {
            let mut out = [0; 8];
            let count = grid.neighbours(c, &mut out);
            (out, count as u8)
        })
        .collect();

    // Plaques de tailles inegales : des cellules de Voronoi deformees et ponderees.
    let plate_count = plate_count.clamp(2, 64);
    let seeds: Vec<(Vec3, f32)> = (0..plate_count)
        .map(|_| {
            let bias = rng.f32();
            (random_axis(&mut rng), bias * bias * 0.55)
        })
        .collect();
    let warp = [Noise::new(hash_seed(seed, 0x7EC7, 2)), Noise::new(hash_seed(seed, 0x7EC7, 3)), Noise::new(hash_seed(seed, 0x7EC7, 4))];
    let shape = Noise::new(hash_seed(seed, 0x7EC7, 5));
    let old_belts = Noise::new(hash_seed(seed, 0x7EC7, 6));

    // Les continents partent groupes, comme une Pangee que les plaques vont dechirer.
    let gather = random_axis(&mut rng).as_dvec3();
    let land_field: Vec<f32> = centers.iter().map(|d| shape.fbm(*d * 2.2, 5) + 0.28 * d.dot(gather) as f32).collect();
    let land_level = threshold_for_share(&land_field, land_share);

    let mut particles: Vec<Particle> = Vec::with_capacity(cells);
    for (c, dir) in centers.iter().enumerate() {
        let w = DVec3::new(warp[0].fbm(*dir * 3.0, 3) as f64, warp[1].fbm(*dir * 3.0, 3) as f64, warp[2].fbm(*dir * 3.0, 3) as f64);
        let warped = (*dir + w * 0.55).normalize().as_vec3();
        let mut best = (0u16, f32::MIN);
        for (p, (center, bias)) in seeds.iter().enumerate() {
            let score = warped.dot(*center) + bias;
            if score > best.1 {
                best = (p as u16, score);
            }
        }
        let crust = ((land_field[c] - land_level) / 0.05 + 0.5).clamp(0.0, 1.0);
        // De vieilles chaines deja usees, heritees d'un passe plus lointain.
        let ridge = 1.0 - old_belts.fbm(*dir * 3.5, 4).abs() * 5.0;
        particles.push(Particle {
            home: dir.as_vec3(),
            plate: best.0,
            crust,
            orogeny: ridge.max(0.0) * 0.14 * crust,
            age: rng.range(STEPS as f32 * 0.3, STEPS as f32),
        });
    }

    // Les plaques continentales, lourdes, derivent moins vite que les plaques oceaniques.
    let mut land_cells = vec![0u32; plate_count];
    let mut plate_cells = vec![1u32; plate_count];
    for p in &particles {
        plate_cells[p.plate as usize] += 1;
        land_cells[p.plate as usize] += (p.crust > 0.5) as u32;
    }
    let mut plates: Vec<Plate> = (0..plate_count)
        .map(|p| {
            let continental = land_cells[p] as f32 / plate_cells[p] as f32;
            let degrees = rng.range(0.9, 2.6) * (1.0 - 0.55 * continental);
            let rate = degrees.to_radians();
            Plate { rotation: Quat::IDENTITY, axis: random_axis(&mut rng), rate, base_rate: rate }
        })
        .collect();

    let mut occupant = vec![NONE; cells];
    let mut trench = vec![0.0f32; cells];
    let mut boundary = vec![Boundary::None; cells];
    let (mut collisions, mut subductions) = (0u64, 0u64);
    let mut blurred = vec![0.0f32; cells];

    for step in 0..STEPS {
        // Reorganisations : de temps en temps, une plaque change de direction.
        if step > 0 && step % 14 == 0 {
            for plate in &mut plates {
                if rng.f32() < 0.4 {
                    plate.axis = random_axis(&mut rng);
                }
            }
        }
        for plate in &mut plates {
            plate.rotation = (Quat::from_axis_angle(plate.axis, plate.rate) * plate.rotation).normalize();
        }

        occupant.fill(NONE);
        boundary.fill(Boundary::None);
        trench.iter_mut().for_each(|t| *t *= 0.55);
        let mut contacts = vec![0u32; plate_count];
        plate_cells.fill(1);
        for p in &particles {
            plate_cells[p.plate as usize] += 1;
        }

        for i in 0..particles.len() {
            let p = particles[i];
            let plate = &plates[p.plate as usize];
            let position = plate.rotation * p.home;
            let cell = grid.cell_of(position.as_dvec3());
            let q = occupant[cell];
            if q == NONE {
                occupant[cell] = i as u32;
                continue;
            }
            let other = particles[q as usize];
            if other.plate == p.plate {
                // Deux particules d'une meme plaque : un simple effet de grille.
                if p.crust > other.crust {
                    occupant[cell] = i as u32;
                }
                continue;
            }

            // Vitesse relative des deux plaques en ce point, en degres par pas.
            let speed_of = |pl: &Plate| (pl.axis * pl.rate).cross(position);
            let closing = (speed_of(plate) - speed_of(&plates[other.plate as usize])).length().to_degrees();
            contacts[p.plate as usize] += 1;
            contacts[other.plate as usize] += 1;

            let (p_land, q_land) = (p.crust > 0.5, other.crust > 0.5);
            let (winner, uplift, arc) = if p_land && q_land {
                // Collision de continents : la plus petite plaque passe dessous, la croute s'epaissit.
                collisions += 1;
                boundary[cell] = Boundary::Collision;
                let p_wins = plate_cells[p.plate as usize] >= plate_cells[other.plate as usize];
                (if p_wins { i as u32 } else { q }, 0.085 * closing, 0.0)
            } else if p_land != q_land {
                // L'ocean, dense, plonge sous le continent, qui se couvre de volcans (les Andes).
                subductions += 1;
                boundary[cell] = Boundary::Subduction;
                trench[cell] = 1.0;
                (if p_land { i as u32 } else { q }, 0.05 * closing, 0.0)
            } else {
                // Deux oceans : le plus vieux, plus froid, plonge. Un arc d'iles se forme.
                subductions += 1;
                boundary[cell] = Boundary::Subduction;
                trench[cell] = 1.0;
                (if p.age < other.age { i as u32 } else { q }, 0.022 * closing, 0.014 * closing)
            };
            let w = &mut particles[winner as usize];
            w.orogeny = (w.orogeny + uplift).min(1.7);
            w.crust = (w.crust + arc).min(1.0);
            occupant[cell] = winner;
        }

        // Rien que les particules placees survivent : les autres sont subductees.
        let mut next: Vec<Particle> = Vec::with_capacity(cells);
        for slot in occupant.iter_mut() {
            if *slot != NONE {
                let mut p = particles[*slot as usize];
                p.age += 1.0;
                p.orogeny *= 0.991;
                *slot = next.len() as u32;
                next.push(p);
            }
        }

        // Cellules vides. Entourees par une seule plaque : un trou de grille, qu'on rebouche.
        // Sinon, les plaques s'ecartent : une dorsale y cree de la croute oceanique neuve.
        let mut empty: Vec<usize> = (0..cells).filter(|c| occupant[*c] == NONE).collect();
        let mut fills: Vec<(usize, Particle)> = Vec::new();
        while !empty.is_empty() {
            fills.clear();
            empty.retain(|&cell| {
                let (list, count) = &neighbours[cell];
                let mut found: [(u16, u32); 8] = [(0, 0); 8];
                let mut kinds = 0;
                let (mut crust, mut orogeny, mut age, mut seen) = (0.0, 0.0, 0.0, 0u32);
                for &n in &list[..*count as usize] {
                    let o = occupant[n as usize];
                    if o == NONE {
                        continue;
                    }
                    let p = &next[o as usize];
                    crust += p.crust;
                    orogeny += p.orogeny;
                    age += p.age;
                    seen += 1;
                    match found[..kinds].iter_mut().find(|f| f.0 == p.plate) {
                        Some(f) => f.1 += 1,
                        None => {
                            found[kinds] = (p.plate, 1);
                            kinds += 1;
                        }
                    }
                }
                if seen == 0 {
                    return true;
                }
                let plate = found[..kinds].iter().max_by_key(|f| f.1).unwrap().0;
                let hole = kinds == 1 && seen >= 6;
                let n = seen as f32;
                let home = plates[plate as usize].rotation.inverse() * centers[cell].as_vec3();
                let particle = if hole {
                    Particle { home, plate, crust: crust / n, orogeny: orogeny / n, age: age / n }
                } else {
                    boundary[cell] = Boundary::Ridge;
                    Particle { home, plate, crust: 0.0, orogeny: 0.0, age: 0.0 }
                };
                fills.push((cell, particle));
                false
            });
            for (cell, particle) in fills.drain(..) {
                occupant[cell] = next.len() as u32;
                next.push(particle);
            }
            if fills.is_empty() && empty.len() == cells {
                break;
            }
        }
        particles = next;

        // Le soulevement s'etale sur la largeur d'une chaine, au lieu d'une ligne d'une cellule.
        if step % 3 == 0 {
            for (c, value) in blurred.iter_mut().enumerate() {
                let (list, count) = &neighbours[c];
                let mut sum = 0.0;
                for &n in &list[..*count as usize] {
                    sum += particles[occupant[n as usize] as usize].orogeny;
                }
                let own = particles[occupant[c] as usize].orogeny;
                *value = own + (sum / *count as f32 - own) * 0.11;
            }
            for (c, value) in blurred.iter().enumerate() {
                particles[occupant[c] as usize].orogeny = *value;
            }
        }

        // Les collisions freinent les plaques, qui ne reprennent de la vitesse que peu a peu.
        for (plate, contact) in plates.iter_mut().zip(&contacts) {
            let braking = (*contact as f32 / 120.0).min(1.0) * 0.35;
            plate.rate = (plate.rate * (1.0 - braking) + (plate.base_rate - plate.rate) * 0.04).max(plate.base_rate * 0.12);
        }
    }

    let read = |f: &dyn Fn(&Particle) -> f32| -> Vec<f32> {
        (0..cells).map(|c| f(&particles[occupant[c] as usize])).collect()
    };
    // Les plus hautes chaines valent environ 1, quelle que soit l'histoire de cette planete.
    let raw = read(&|p| p.orogeny);
    let top = threshold_for_share(&raw, 0.006).max(1e-3);
    Tectonics {
        grid,
        plate: (0..cells).map(|c| particles[occupant[c] as usize].plate).collect(),
        crust: read(&|p| p.crust),
        orogeny: raw.iter().map(|o| (o / top).min(1.4)).collect(),
        age: read(&|p| p.age),
        trench,
        boundary,
        collisions,
        subductions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(seed: u64) -> Tectonics {
        simulate(seed, 14, 0.32)
    }

    #[test]
    fn simulation_is_deterministic() {
        let (a, b) = (run(3), run(3));
        assert_eq!(a.plate, b.plate);
        assert_eq!(a.orogeny, b.orogeny);
        assert_ne!(run(4).plate, a.plate);
    }

    #[test]
    fn plates_collide_subduct_and_spread() {
        let t = run(1);
        assert!(t.collisions > 0, "aucune collision de continents");
        assert!(t.subductions > 500, "{} subductions", t.subductions);
        let ridges = t.boundary.iter().filter(|b| **b == Boundary::Ridge).count();
        assert!(ridges > 50, "{ridges} cellules de dorsale");
        // La simulation doit laisser plusieurs plaques vivantes.
        let alive: std::collections::HashSet<u16> = t.plate.iter().copied().collect();
        assert!(alive.len() >= 6, "{} plaques", alive.len());
    }

    #[test]
    fn young_crust_is_oceanic_and_continents_survive() {
        let t = run(2);
        let land = t.crust.iter().filter(|c| **c > 0.5).count() as f32 / t.crust.len() as f32;
        assert!((0.12..0.5).contains(&land), "{land} de croute continentale");
        let young = t.age.iter().zip(&t.crust).filter(|(a, c)| **a < 5.0 && **c > 0.5).count();
        assert!(young < t.age.len() / 50, "{young} cellules jeunes et continentales");
    }

    /// Les hautes chaines doivent se trouver sur la croute, pres des frontieres de plaques.
    #[test]
    fn mountains_form_along_plate_boundaries() {
        let t = run(5);
        let grid = t.grid;
        // Distance a la frontiere de plaque la plus proche, en cellules.
        let mut distance = vec![u32::MAX; grid.cells()];
        let mut front: Vec<usize> = Vec::new();
        let mut out = [0; 8];
        for c in 0..grid.cells() {
            let count = grid.neighbours(c, &mut out);
            if out[..count].iter().any(|n| t.plate[*n as usize] != t.plate[c]) {
                distance[c] = 0;
                front.push(c);
            }
        }
        let mut level = 0;
        while !front.is_empty() {
            level += 1;
            let mut next = Vec::new();
            for &c in &front {
                let count = grid.neighbours(c, &mut out);
                for &n in &out[..count] {
                    if distance[n as usize] == u32::MAX {
                        distance[n as usize] = level;
                        next.push(n as usize);
                    }
                }
            }
            front = next;
        }
        let mean = |pick: &dyn Fn(usize) -> bool| {
            let (mut sum, mut count) = (0u64, 0u64);
            for c in 0..grid.cells() {
                if pick(c) {
                    sum += distance[c] as u64;
                    count += 1;
                }
            }
            sum as f32 / count.max(1) as f32
        };
        let high = mean(&|c| t.orogeny[c] > 0.35);
        let all = mean(&|c| t.crust[c] > 0.5);
        assert!(high < all, "chaines a {high} cellules des frontieres, continents a {all}");
        assert!(t.orogeny.iter().any(|o| *o > 0.5), "aucune haute chaine");
    }
}
