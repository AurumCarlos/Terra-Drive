//! Tectonique des plaques simulee. Chaque plaque est un ensemble de particules de croute rigides
//! qui tournent autour d'un pole d'Euler. La ou deux plaques se recouvrent, l'une passe sous
//! l'autre et le relief monte. La ou elles s'ecartent, une dorsale cree de la croute oceanique.
//!
//! L'histoire suit le cycle de Wilson : les continents partent reunis en une Pangee, que des rifts dechirent
//! en plaques qui s'eloignent (l'Atlantique). Une plaque est surtout tiree par le poids de la croute qui
//! plonge dans ses fosses (Forsyth et Uyeda 1975) : les plaques bordees de subductions vont vite, comme le
//! Pacifique, les plaques continentales sans fosse derivent lentement. Deux continents qui se percutent
//! finissent soudes en un seul (l'Inde et l'Asie). Des points chauds fixes dans le manteau percent les
//! plaques qui passent au-dessus : des chapelets d'iles de plus en plus vieilles, qui s'enfoncent en
//! s'eloignant du panache (Hawaii et les monts Empereur).
//!
//! Le soulevement suit Cortial et al. 2019 (Â« Procedural Tectonic Planets Â») : il ne reste pas sur la
//! ligne de contact. Une subduction souleve la plaque chevauchante sur des centaines de kilometres, avec
//! un maximum en retrait de la fosse (les Andes et l'Altiplano) ; une collision de continents epaissit la
//! croute sur plus de mille kilometres (l'Himalaya et le Tibet).

use super::cube::CubeGrid;
use super::noise::{hash_seed, Noise, Rng};
use glam::{DVec3, Quat, Vec3};

/// Resolution de la simulation : environ 80 km par cellule.
pub const TECTONIC_N: usize = 128;
/// Nombre de pas : environ 15 millions d'annees chacun.
pub const STEPS: usize = 48;
const NONE: u32 = u32::MAX;
/// Distance, en km sur une planete de la taille de la Terre, ou le soulevement d'une subduction est le
/// plus fort, et distances ou ceux d'une subduction et d'une collision s'annulent.
const ARC_PEAK: f32 = 260.0;
const SUBDUCTION_REACH: f32 = 1000.0;
const COLLISION_REACH: f32 = 1300.0;

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
    /// Activite volcanique laissee par les points chauds, de 1 sur le panache a 0 loin en aval.
    pub hotspot: Vec<f32>,
    /// Nombre de cellules ou se sont produites des collisions continentales.
    pub collisions: u64,
    pub subductions: u64,
    /// Continents dechires par un rift, et paires de continents soudes par une collision.
    pub rifts: u32,
    pub sutures: u32,
}

#[derive(Clone, Copy)]
struct Particle {
    /// Position dans le repere de la plaque : fixe, donc aucune perte par reechantillonnage.
    home: Vec3,
    plate: u16,
    crust: f32,
    orogeny: f32,
    age: f32,
    /// Croute d'ile volcanique batie par un point chaud ; elle s'enfonce en refroidissant.
    island: f32,
    /// Pas ecoules depuis le passage au-dessus d'un point chaud.
    plume: f32,
}

impl Particle {
    fn land(&self) -> f32 {
        self.crust.max(self.island)
    }
}

/// Pas qu'une collision doit durer pour souder deux continents (environ 60 millions d'annees).
const SUTURE_STEPS: u32 = 4;
/// Cellules de collision a partir desquelles deux plaques sont dites en collision franche.
const SUTURE_CELLS: u32 = 16;
/// Plus de plaques que cela, et les rifts s'arretent.
const MAX_PLATES: usize = 40;

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
    let warp = [
        Noise::new(hash_seed(seed, 0x7EC7, 2)),
        Noise::new(hash_seed(seed, 0x7EC7, 3)),
        Noise::new(hash_seed(seed, 0x7EC7, 4)),
    ];
    let shape = Noise::new(hash_seed(seed, 0x7EC7, 5));
    let old_belts = Noise::new(hash_seed(seed, 0x7EC7, 6));
    let rift_noise = Noise::new(hash_seed(seed, 0x7EC7, 7));
    // Une quinzaine de panaches, fixes dans le manteau : la Terre en compte une quarantaine de grands.
    let hotspots: Vec<DVec3> =
        (0..10 + (rng.f32() * 8.0) as usize).map(|_| random_axis(&mut rng).as_dvec3()).collect();

    // Les continents partent groupes, comme une Pangee que les plaques vont dechirer.
    let gather = random_axis(&mut rng).as_dvec3();
    let land_field: Vec<f32> = centers.iter().map(|d| shape.fbm(*d * 2.2, 5) + 0.28 * d.dot(gather) as f32).collect();
    // Les collisions raccourcissent les continents : une croute qui passe sous une autre ne refait pas
    // surface. On part donc d'un peu plus de croute que la part demandee.
    let land_level = threshold_for_share(&land_field, (land_share * 1.1).min(0.95));

    let mut particles: Vec<Particle> = Vec::with_capacity(cells);
    for (c, dir) in centers.iter().enumerate() {
        let w = DVec3::new(
            warp[0].fbm(*dir * 3.0, 3) as f64,
            warp[1].fbm(*dir * 3.0, 3) as f64,
            warp[2].fbm(*dir * 3.0, 3) as f64,
        );
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
            island: 0.0,
            plume: 1e3,
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
    let (mut rifts, mut sutures) = (0u32, 0u32);
    let min_plates = (plate_count * 2 / 3).max(6);
    // Collisions en cours entre deux plaques : nombre de pas consecutifs.
    let mut clashing: std::collections::HashMap<(u16, u16), u32> = std::collections::HashMap::new();
    let mut blurred = vec![0.0f32; cells];
    // Fronts de soulevement du pas en cours : (cellule, collision ?, vitesse de rapprochement).
    let mut fronts: Vec<(u32, bool, f32)> = Vec::new();
    let mut reach: Vec<(u32, f32, bool)> = vec![(NONE, 0.0, false); cells];
    let mut queue = std::collections::VecDeque::new();

    for step in 0..STEPS {
        fronts.clear();
        // La Pangee se dechire tot ; d'autres continents se fendent ensuite de loin en loin.
        let rift_now = step == 3 || (step % 5 == 0 && step > 3 && step + 8 < STEPS && rng.f32() < 0.45);
        if rift_now && plates.len() < MAX_PLATES && split_continent(&mut particles, &mut plates, &mut rng, &rift_noise)
        {
            rifts += 1;
        }
        for plate in &mut plates {
            plate.rotation = (Quat::from_axis_angle(plate.axis, plate.rate) * plate.rotation).normalize();
        }

        occupant.fill(NONE);
        boundary.fill(Boundary::None);
        trench.iter_mut().for_each(|t| *t *= 0.55);
        let plate_count = plates.len();
        let mut contacts = vec![0u32; plate_count];
        let mut crashes = vec![0u32; plate_count];
        // Traction des fosses : nombre de cellules qui plongent, et axe de rotation qui y mene.
        let mut slab = vec![0u32; plate_count];
        let mut pull = vec![Vec3::ZERO; plate_count];
        let mut centroid = vec![Vec3::ZERO; plate_count];
        let mut pair_clash: std::collections::HashMap<(u16, u16), u32> = std::collections::HashMap::new();
        plate_cells = vec![1u32; plate_count];
        for p in &particles {
            plate_cells[p.plate as usize] += 1;
            centroid[p.plate as usize] += plates[p.plate as usize].rotation * p.home;
        }
        for c in &mut centroid {
            *c = c.normalize_or_zero();
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

            let (p_land, q_land) = (p.land() > 0.5, other.land() > 0.5);
            let (winner, uplift, arc) = if p_land && q_land {
                // Collision de continents : la plus petite plaque passe dessous, la croute s'epaissit.
                collisions += 1;
                boundary[cell] = Boundary::Collision;
                let p_wins = plate_cells[p.plate as usize] >= plate_cells[other.plate as usize];
                fronts.push((cell as u32, true, closing));
                crashes[p.plate as usize] += 1;
                crashes[other.plate as usize] += 1;
                let key = (p.plate.min(other.plate), p.plate.max(other.plate));
                *pair_clash.entry(key).or_insert(0) += 1;
                (if p_wins { i as u32 } else { q }, 0.03 * closing, 0.0)
            } else if p_land != q_land {
                // L'ocean, dense, plonge sous le continent, qui se couvre de volcans (les Andes).
                subductions += 1;
                boundary[cell] = Boundary::Subduction;
                trench[cell] = 1.0;
                fronts.push((cell as u32, false, closing));
                (if p_land { i as u32 } else { q }, 0.012 * closing, 0.0)
            } else {
                // Deux oceans : le plus vieux, plus froid, plonge. Un arc d'iles se forme.
                subductions += 1;
                boundary[cell] = Boundary::Subduction;
                trench[cell] = 1.0;
                (if p.age < other.age { i as u32 } else { q }, 0.022 * closing, 0.014 * closing)
            };
            // La croute oceanique qui plonge tire toute sa plaque vers la fosse.
            if !(p_land && q_land) {
                let sinking = if winner == i as u32 { other.plate } else { p.plate } as usize;
                slab[sinking] += 1;
                pull[sinking] += centroid[sinking].cross(position).normalize_or_zero();
            }
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
                p.plume += 1.0;
                // Loin du panache, l'ile refroidit et s'enfonce : apres quelques dizaines de millions
                // d'annees, il n'en reste qu'un mont sous-marin.
                p.island *= 0.8;
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
                let Some(&(plate, _)) = found[..kinds].iter().max_by_key(|f| f.1) else { return true };
                let hole = kinds == 1 && seen >= 6;
                let n = seen as f32;
                let home = plates[plate as usize].rotation.inverse() * centers[cell].as_vec3();
                let particle = if hole {
                    Particle { home, plate, crust: crust / n, orogeny: orogeny / n, age: age / n, island: 0.0, plume: 1e3 }
                } else {
                    boundary[cell] = Boundary::Ridge;
                    Particle { home, plate, crust: 0.0, orogeny: 0.0, age: 0.0, island: 0.0, plume: 1e3 }
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

        // Les points chauds percent la plaque qui passe : un volcan, et sur l'ocean une ile.
        for h in &hotspots {
            let cell = grid.cell_of(*h);
            let p = &mut particles[occupant[cell] as usize];
            p.plume = 0.0;
            if p.crust < 0.5 {
                p.island = (p.island + 0.75).min(1.0);
                p.orogeny = (p.orogeny + 0.03).min(1.7);
            } else {
                // Sous un continent, un plateau de laves et de grandes caldeiras (Yellowstone, le Deccan).
                p.orogeny = (p.orogeny + 0.012).min(1.7);
            }
        }

        // Le soulevement gagne l'interieur des continents depuis chaque front, selon la distance.
        reach.fill((NONE, 0.0, false));
        queue.clear();
        for &(cell, collision, closing) in &fronts {
            if reach[cell as usize].0 == NONE {
                reach[cell as usize] = (cell, closing, collision);
                queue.push_back(cell);
            }
        }
        while let Some(c) = queue.pop_front() {
            let (source, closing, collision) = reach[c as usize];
            let km = (centers[c as usize].distance(centers[source as usize]) * 6371.0) as f32;
            let particle = &mut particles[occupant[c as usize] as usize];
            if particle.land() > 0.5 {
                let gain = if collision {
                    let t = km / COLLISION_REACH;
                    0.05 * (1.0 - t * t).max(0.0).powi(2)
                } else {
                    let t = km / ARC_PEAK;
                    0.05 * t * (1.0 - t).exp() * (1.0 - km / SUBDUCTION_REACH).max(0.0)
                };
                particle.orogeny = (particle.orogeny + gain * closing).min(1.7);
            }
            let (list, count) = &neighbours[c as usize];
            for &n in &list[..*count as usize] {
                if reach[n as usize].0 == NONE {
                    let far = (centers[n as usize].distance(centers[source as usize]) * 6371.0) as f32;
                    if far <= if collision { COLLISION_REACH } else { SUBDUCTION_REACH } {
                        reach[n as usize] = (source, closing, collision);
                        queue.push_back(n);
                    }
                }
            }
        }

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

        // Moteur des plaques : la traction des fosses. Une plaque se tourne vers ses subductions et accelere
        // avec la part de son pourtour qui plonge ; les collisions la freinent.
        for (k, plate) in plates.iter_mut().enumerate() {
            if slab[k] > 0 {
                let turn = 0.25 * (slab[k] as f32 / 40.0).min(1.0);
                let toward = pull[k].normalize_or_zero();
                if toward != Vec3::ZERO {
                    plate.axis = (plate.axis * (1.0 - turn) + toward * turn).normalize();
                }
            }
            let share = slab[k] as f32 / (contacts[k] as f32 + 1.0);
            let target = plate.base_rate * (0.55 + 1.5 * share.min(1.0));
            // Un continent ne plonge pas : la collision bloque presque la convergence (l'Inde est passee de 15 a
            // 5 cm par an en touchant l'Asie).
            let braking = (crashes[k] as f32 / 40.0).min(1.0) * 0.7;
            plate.rate = (plate.rate + (target - plate.rate) * 0.2) * (1.0 - braking);
            plate.rate = plate.rate.max(plate.base_rate * 0.12);
        }

        // Une collision qui dure soude les deux continents : la plus petite plaque rejoint l'autre.
        clashing.retain(|key, _| pair_clash.get(key).is_some_and(|n| *n >= SUTURE_CELLS));
        let mut fresh: Vec<(u16, u16)> = pair_clash.iter().filter(|(_, n)| **n >= SUTURE_CELLS).map(|(k, _)| *k).collect();
        fresh.sort_unstable();
        for key in fresh {
            *clashing.entry(key).or_insert(0) += 1;
        }
        let mut ready: Vec<(u16, u16)> =
            clashing.iter().filter(|(_, n)| **n >= SUTURE_STEPS).map(|(k, _)| *k).collect();
        ready.sort_unstable();
        for (a, b) in ready {
            // Une planete garde toujours assez de plaques : les sutures s'arretent quand il en reste peu.
            let alive = plate_cells.iter().filter(|n| **n > 1).count();
            if !clashing.contains_key(&(a, b)) || alive <= min_plates {
                continue;
            }
            let (big, small) = if plate_cells[a as usize] >= plate_cells[b as usize] { (a, b) } else { (b, a) };
            let to_big = plates[big as usize].rotation.inverse() * plates[small as usize].rotation;
            for p in particles.iter_mut().filter(|p| p.plate == small) {
                p.home = to_big * p.home;
                p.plate = big;
            }
            plate_cells[big as usize] += plate_cells[small as usize];
            plate_cells[small as usize] = 0;
            sutures += 1;
            // Les autres collisions de la plaque absorbee deviennent celles de sa nouvelle plaque.
            clashing.retain(|k, _| k.0 != small && k.1 != small);
        }
    }

    let read = |f: &dyn Fn(&Particle) -> f32| -> Vec<f32> {
        (0..cells).map(|c| f(&particles[occupant[c] as usize])).collect()
    };
    // Les plus hautes chaines valent environ 1, quelle que soit l'histoire de cette planete : 4 % des
    // continents, comme les hauts massifs et plateaux de la Terre.
    let raw = read(&|p| p.orogeny);
    let on_land: Vec<f32> = particles_land(&raw, &read(&|p| p.land()));
    let top = threshold_for_share(if on_land.len() > 40 { &on_land } else { &raw }, 0.04).max(1e-3);
    Tectonics {
        grid,
        plate: (0..cells).map(|c| particles[occupant[c] as usize].plate).collect(),
        crust: read(&|p| p.land()),
        orogeny: raw.iter().map(|o| (o / top).min(1.4)).collect(),
        age: read(&|p| p.age),
        trench,
        boundary,
        hotspot: read(&|p| (-p.plume / 2.5).exp()),
        collisions,
        subductions,
        rifts,
        sutures,
    }
}

/// Fend en deux le plus grand continent le long d'une ligne irreguliere. Les deux moities s'eloignent
/// l'une de l'autre ; les bords du rift se soulevent en epaulements (les Ghats, le Grand Escarpement).
fn split_continent(particles: &mut [Particle], plates: &mut Vec<Plate>, rng: &mut Rng, noise: &Noise) -> bool {
    let mut land = vec![0u32; plates.len()];
    for p in particles.iter() {
        land[p.plate as usize] += (p.crust > 0.5) as u32;
    }
    let Some((target, &count)) = land.iter().enumerate().max_by_key(|(_, n)| **n) else { return false };
    if count < 900 {
        return false;
    }
    let rotation = plates[target].rotation;
    let world = |p: &Particle| rotation * p.home;
    let middle = particles
        .iter()
        .filter(|p| p.plate as usize == target && p.crust > 0.5)
        .fold(Vec3::ZERO, |sum, p| sum + world(p))
        .normalize_or_zero();
    if middle == Vec3::ZERO {
        return false;
    }
    // Un plan qui passe par le coeur du continent, et une ligne de rupture tordue par le bruit.
    let side_of = |x: Vec3, normal: Vec3| x.dot(normal) + 0.13 * noise.fbm(x.as_dvec3() * 3.2, 4);
    for _ in 0..6 {
        let r = random_axis(rng);
        let normal = (r - middle * r.dot(middle)).normalize_or_zero();
        if normal == Vec3::ZERO {
            continue;
        }
        let (mut a_land, mut b_land) = (0u32, 0u32);
        for p in particles.iter().filter(|p| p.plate as usize == target && p.crust > 0.5) {
            if side_of(world(p), normal) > 0.0 {
                b_land += 1;
            } else {
                a_land += 1;
            }
        }
        // Les deux morceaux doivent etre de vrais continents.
        if a_land.min(b_land) * 4 < count {
            continue;
        }
        let new = plates.len() as u16;
        let (mut ca, mut cb) = (Vec3::ZERO, Vec3::ZERO);
        for p in particles.iter_mut().filter(|p| p.plate as usize == target) {
            let x = rotation * p.home;
            let d = side_of(x, normal);
            if d > 0.0 {
                p.plate = new;
                cb += x;
            } else {
                ca += x;
            }
            // Epaulements du rift, sur 300 km de part et d'autre.
            if p.crust > 0.5 && d.abs() < 0.05 {
                p.orogeny = (p.orogeny + 0.12 * (1.0 - d.abs() / 0.05)).min(1.7);
            }
        }
        let (ca, cb) = (ca.normalize_or_zero(), cb.normalize_or_zero());
        let away = ca.cross(cb).normalize_or_zero();
        if away == Vec3::ZERO {
            return false;
        }
        let old = &mut plates[target];
        let rate = old.base_rate.max(1.3f32.to_radians());
        old.axis = -away;
        old.rate = rate;
        let base_rate = old.base_rate;
        plates.push(Plate { rotation, axis: away, rate, base_rate });
        return true;
    }
    false
}

/// Les valeurs des cellules de croute continentale.
fn particles_land(values: &[f32], crust: &[f32]) -> Vec<f32> {
    values.iter().zip(crust).filter(|(_, c)| **c > 0.5).map(|(v, _)| *v).collect()
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

    /// La Pangee se dechire, et sur plusieurs planetes au moins deux continents finissent soudes.
    #[test]
    fn continents_rift_apart_and_some_collide_into_one() {
        let (mut rifts, mut sutures) = (0, 0);
        for seed in 1..5 {
            let t = run(seed);
            assert!(t.rifts >= 1, "graine {seed} : aucun rift");
            rifts += t.rifts;
            sutures += t.sutures;
        }
        assert!(rifts >= 6, "{rifts} rifts");
        assert!(sutures >= 1, "{sutures} sutures");
    }

    /// Les points chauds laissent des iles, et derriere elles des traces de plus en plus vieilles.
    #[test]
    fn hotspots_leave_island_chains() {
        let t = run(6);
        let active = t.hotspot.iter().filter(|h| **h > 0.9).count();
        let trail = t.hotspot.iter().filter(|h| (0.05..0.9).contains(*h)).count();
        assert!(active >= 8, "{active} panaches actifs");
        assert!(trail > active, "{trail} cellules de trace pour {active} panaches");
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
        for (c, d) in distance.iter_mut().enumerate() {
            let count = grid.neighbours(c, &mut out);
            if out[..count].iter().any(|n| t.plate[*n as usize] != t.plate[c]) {
                *d = 0;
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
            for (c, d) in distance.iter().enumerate() {
                if pick(c) {
                    sum += *d as u64;
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
