//! Relief global et eau de la planete, calcules une fois pour toutes a sa creation. Le relief sort
//! de la tectonique, l'erosion (stream power) y creuse un reseau de drainage, puis le remplissage
//! des depressions donne les lacs et leur exutoire, et le cumul des pluies donne les rivieres.

use super::cube::{CubeField, CubeGrid};
use super::climate::{Climate, ClimateMap};
use super::generator::{planet_radius, smoothstep, PlanetParams};
use super::noise::{hash_seed, Noise, Rng};
use super::tectonics::{threshold_for_share, Tectonics};
use glam::{DVec3, Vec3};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Grille d'erosion grossiere (environ 39 km), puis fine (20 km), puis finale (13 km).
const COARSE_N: usize = 256;
const COARSE_STEPS: usize = 14;
const MEDIUM_N: usize = 512;
const MEDIUM_STEPS: usize = 4;
pub const FINAL_N: usize = 768;
/// Index spatial des rivieres : environ 39 km par case.
const INDEX_N: usize = 256;
const NONE: u32 = u32::MAX;
/// Segments droits qui approchent la courbe d'un troncon.
const CURVE_SEGMENTS: usize = 4;

/// Coefficient d'erosion, pour une aire en km2, une duree de simulation de 1.
const ERODIBILITY: f32 = 3.5;
/// Bassin versant equivalent, en km2 de pluie moyenne, a partir duquel coule une riviere.
/// Regle pour que la longueur du reseau approche celle de la Terre au meme seuil : HydroRIVERS compte
/// 36 millions de km pour les bassins de plus de 10 km2, et la longueur decroit comme la racine du seuil.
const MIN_RIVER_AREA: f32 = 1500.0;
/// Le meme seuil en montagne : les rivieres y naissent plus tot.
const MIN_MOUNTAIN_RIVER_AREA: f32 = 550.0;
const MIN_LAKE_DEPTH: f32 = 9.0;
/// Au-dela, la cuvette est comblee par les sediments : on evite les mers interieures a plat.
const MAX_LAKE_CELLS: usize = 160;

/// Valeur du masque de lac sur le rivage.
pub const SHORE: f32 = 0.3;
/// Ecart maximal du trace des rivieres par les meandres, en metres.
const MAX_WARP: f64 = 4_500.0;

/// Un troncon de riviere : une courbe de Bezier quadratique entre deux cellules de la grille.
#[derive(Clone, Copy)]
pub struct RiverEdge {
    a: [f32; 3],
    c: [f32; 3],
    b: [f32; 3],
    /// Niveau de l'eau a chaque bout, en metres.
    pub level: [f32; 2],
    /// Demi-largeur du lit a chaque bout, en metres.
    pub width: [f32; 2],
    reach: f32,
    /// Pente moyenne du troncon.
    pub slope: f32,
    /// Troncon suivant, ou `NONE` a l'arrivee.
    pub next: u32,
    pub end: RiverEnd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiverEnd {
    Continues,
    Sea,
    Lake,
}

pub struct Lake {
    pub level: f32,
    pub cells: u32,
    pub max_depth: f32,
    /// Direction de la cellule d'ou sort l'exutoire.
    pub outlet: DVec3,
    /// Troncon de la riviere qui s'en echappe.
    pub outflow: u32,
}

/// Ce qu'une riviere fait au sol et a l'eau en un point.
#[derive(Clone, Copy, Debug, Default)]
pub struct RiverSample {
    /// Fond de vallee, de 0 a 1 : le sol y est lisse et humide.
    pub valley: f32,
    /// Part du relief soumise au cone de la vallee, de 0 a 1 : nulle si le maillage est trop large.
    pub reach: f32,
    /// Part du sol relevee au niveau des berges, de 0 a 1.
    pub lift: f32,
    /// Niveau de l'eau.
    pub level: f32,
    /// Hauteur du sol par rapport a l'eau si la vallee etait seule.
    pub profile: f32,
    /// Vrai si l'eau de cette riviere doit etre dessinee ici.
    pub wet: bool,
    /// Couleur d'eau a mettre sur le sol quand la grille est trop large pour le lit.
    pub paint: f32,
    /// Sens du courant, de longueur la vitesse en m/s.
    pub flow: Vec3,
    /// Proximite du lit, de 0 a 1.
    pub bank: f32,
}

pub struct Hydrology {
    /// Relief a grande echelle, en metres.
    pub elevation: CubeField,
    lake_mask: CubeField,
    /// Masque multiplie par le niveau : leur rapport donne un niveau exact dans chaque lac,
    /// et continu entre deux lacs voisins.
    lake_level: CubeField,
    /// Largeur du lac de chaque cellule de la grille fine et de ses voisines, en km (0 : aucun lac).
    lake_span: Vec<u16>,
    pub edges: Vec<RiverEdge>,
    pub lakes: Vec<Lake>,
    index_grid: CubeGrid,
    index_start: Vec<u32>,
    index_edges: Vec<u32>,
}

/// Calcule une valeur par cellule, sur tous les coeurs.
pub fn par_cells(grid: CubeGrid, f: &(dyn Fn(usize) -> f32 + Sync)) -> Vec<f32> {
    let mut out = vec![0.0f32; grid.cells()];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = grid.cells() / threads + 1;
    let radius = crate::world::generator::planet_radius();
    std::thread::scope(|s| {
        for (i, slice) in out.chunks_mut(chunk).enumerate() {
            s.spawn(move || {
                crate::world::generator::worker_thread(radius);
                for (k, v) in slice.iter_mut().enumerate() {
                    *v = f(i * chunk + k);
                }
            });
        }
    });
    out
}

#[derive(PartialEq)]
struct Entry {
    height: f32,
    seq: u32,
    cell: u32,
}

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        // Le plus bas d'abord, puis dans un ordre tire au hasard : un fond plat se draine par un arbre
        // aleatoire, ramifie comme un delta, au lieu de lignes droites paralleles vers l'exutoire.
        o.height.total_cmp(&self.height).then(o.seq.cmp(&self.seq))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Rang pseudo-aleatoire, deterministe.
fn shuffle(k: u32) -> u32 {
    let mut x = k.wrapping_mul(0x9E37_79B9) ^ 0x85EB_CA6B;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^ (x >> 15)
}

/// Reseau de drainage : chaque cellule de terre coule vers `receiver`, par niveaux croissants.
struct Drainage {
    /// Cellules de terre, de l'aval vers l'amont.
    order: Vec<u32>,
    receiver: Vec<u32>,
    /// Relief dont les depressions sont remplies jusqu'a leur point de debordement.
    filled: Vec<f32>,
}

/// Sommets de la grille, chacun decale au hasard dans sa cellule. Sur une grille reguliere, l'eau suit les
/// huit directions des voisins et trace des lignes droites paralleles ; sur des points irreguliers, les
/// pentes different d'un voisin a l'autre et le reseau se ramifie comme un vrai bassin (Cordonnier et al.
/// 2016 erodent pour cette raison sur un semis de Poisson).
fn jittered_nodes(seed: u64, grid: CubeGrid) -> Vec<Vec3> {
    let spacing = std::f64::consts::FRAC_PI_2 / grid.n as f64;
    par_cells3(grid, &|c| node_at(seed, grid, c, spacing))
}

fn node_at(seed: u64, grid: CubeGrid, c: usize, spacing: f64) -> Vec3 {
    let dir = grid.center(c);
    let mut rng = Rng(hash_seed(seed, c as i64, 0x51DE));
    let side = dir.any_orthonormal_vector();
    let other = dir.cross(side);
    let (a, b) = (rng.range(-0.42, 0.42) as f64, rng.range(-0.42, 0.42) as f64);
    (dir + (side * a + other * b) * spacing).normalize().as_vec3()
}

/// Comme `par_cells`, pour des vecteurs.
fn par_cells3(grid: CubeGrid, f: &(dyn Fn(usize) -> Vec3 + Sync)) -> Vec<Vec3> {
    let mut out = vec![Vec3::ZERO; grid.cells()];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = grid.cells() / threads + 1;
    std::thread::scope(|s| {
        for (i, slice) in out.chunks_mut(chunk).enumerate() {
            s.spawn(move || {
                for (k, v) in slice.iter_mut().enumerate() {
                    *v = f(i * chunk + k);
                }
            });
        }
    });
    out
}

/// Remplissage par priorite : on remonte depuis les cotes en suivant toujours le point le plus bas. Une fois
/// les cuvettes remplies, chaque cellule qui domine un voisin coule vers celui de plus forte pente, mesuree
/// entre les sommets decales `nodes`.
fn route(grid: CubeGrid, h: &[f32], nodes: &[Vec3]) -> Drainage {
    let cells = grid.cells();
    let mut filled = h.to_vec();
    let mut receiver = vec![NONE; cells];
    let mut seen = vec![false; cells];
    let mut heap = BinaryHeap::new();
    let mut order = Vec::new();
    let mut around = [0u32; 8];

    for c in 0..cells {
        if h[c] > 0.0 {
            continue;
        }
        seen[c] = true;
        let count = grid.neighbours(c, &mut around);
        for &n in &around[..count] {
            let n = n as usize;
            if h[n] > 0.0 && !seen[n] {
                seen[n] = true;
                receiver[n] = c as u32;
                heap.push(Entry { height: h[n], seq: shuffle(n as u32), cell: n as u32 });
            }
        }
    }
    while let Some(Entry { height, cell, .. }) = heap.pop() {
        order.push(cell);
        let count = grid.neighbours(cell as usize, &mut around);
        for &n in &around[..count] {
            let n = n as usize;
            if !seen[n] {
                seen[n] = true;
                receiver[n] = cell;
                filled[n] = h[n].max(height);
                heap.push(Entry { height: filled[n], seq: shuffle(n as u32), cell: n as u32 });
            }
        }
    }
    // Plus forte pente, hors des cuvettes remplies ou l'ordre du remplissage fait foi. Un voisin plus bas a
    // ete atteint avant : l'ordre de l'aval vers l'amont reste valable.
    for &c in &order {
        let c = c as usize;
        let here = filled[c];
        let count = grid.neighbours(c, &mut around);
        let mut best = (0.0f32, NONE);
        for &n in &around[..count] {
            let drop = here - filled[n as usize];
            if drop > 0.0 {
                let slope = drop / nodes[c].distance(nodes[n as usize]).max(1e-9);
                if slope > best.0 {
                    best = (slope, n);
                }
            }
        }
        if best.1 != NONE {
            receiver[c] = best.1;
        }
    }
    Drainage { order, receiver, filled }
}

/// Eau recue par chaque cellule depuis l'amont, en km2 de pluie moyenne.
fn accumulate(flow: &Drainage, rain: &[f32], cell_area: f32) -> Vec<f32> {
    let mut area: Vec<f32> = rain.iter().map(|r| r * cell_area).collect();
    for &c in flow.order.iter().rev() {
        let r = flow.receiver[c as usize];
        if r != NONE {
            area[r as usize] += area[c as usize];
        }
    }
    area
}

/// Erosion fluviale implicite : l'incision depend de la pente et de la racine de l'aire drainee.
/// Les roches ne resistent pas toutes autant : les bassins sedimentaires, peu souleves, s'usent bien plus
/// vite que le socle des massifs, et l'eau y etale de grandes plaines.
#[allow(clippy::too_many_arguments)]
fn erode(
    grid: CubeGrid,
    nodes: &[Vec3],
    h: &mut Vec<f32>,
    uplift: &[f32],
    steps: usize,
    duration: f32,
    rain: &[f32],
) {
    let soft: Vec<f32> = uplift.iter().map(|u| 1.0 + 1.0 * (1.0 - smoothstep(250.0, 1300.0, *u))).collect();
    let spacing = (planet_radius() * std::f64::consts::FRAC_PI_2 / grid.n as f64) as f32;
    let cell_area = (spacing * 0.001).powi(2);
    let km = (planet_radius() * 0.001) as f32;
    let dt = duration / steps as f32;
    for _ in 0..steps {
        let flow = route(grid, h, nodes);
        let area = accumulate(&flow, rain, cell_area);
        let mut next = flow.filled;
        for &c in &flow.order {
            let c = c as usize;
            let r = flow.receiver[c] as usize;
            // Une riviere n'incise pas plus bas que la mer, quelle que soit la profondeur du large.
            let down = next[r].max(1.0);
            let length = (nodes[c].distance(nodes[r]) * km).max(spacing * 0.0003);
            let f = ERODIBILITY * soft[c] * dt * area[c].sqrt() / length;
            next[c] = ((next[c] + uplift[c] * dt + f * down) / (1.0 + f)).max(down);
        }
        *h = next;
    }
}

fn resample(grid: CubeGrid, from: CubeGrid, values: &[f32]) -> Vec<f32> {
    let field = CubeField::from_cells(from, values);
    par_cells(grid, &|c| field.sample(grid.center(c)))
}

/// Profondeur de l'ocean selon l'age de la croute : le plancher s'enfonce en refroidissant.
fn ocean_floor(shore: f32, age: f32) -> f32 {
    // Parsons et Sclater 1977 : 2 500 m a la dorsale, puis la profondeur croit comme la racine de l'age.
    let abyss = -2500.0 - 2300.0 * (age / 48.0).clamp(0.0, 1.0).sqrt();
    // Plateau continental jusqu'a 170 m, talus, puis plaine abyssale.
    if shore > -0.15 {
        -170.0 * (-shore / 0.15)
    } else {
        -170.0 + (abyss + 170.0) * smoothstep(-0.15, -0.4, shore)
    }
}

impl Hydrology {
    /// Simule l'eau de la planete : relief de base, erode sous le climat provisoire `climate_at` (direction,
    /// altitude), puis climat definitif sur ce relief, ecoulement, lacs et rivieres. Rend aussi ce climat.
    pub fn build(
        params: &PlanetParams,
        plates: &Tectonics,
        climate_at: &(dyn Fn(DVec3, f32) -> Climate + Sync),
    ) -> (Self, ClimateMap) {
        // Pluie, 1 pour 800 mm par an : la moyenne des terres.
        let rain_at = |dir: DVec3, height: f32| (climate_at(dir, height).rain / 800.0).max(0.02);
        let layer = |i: i64| Noise::new(hash_seed(params.seed, 0x11D0, i));
        let (coast, plateau, ranges, relief_noise, pits, chance) =
            (layer(1), layer(2), layer(3), layer(4), layer(5), layer(6));
        let crust = CubeField::from_cells(plates.grid, &plates.crust);
        let orogeny = CubeField::from_cells(plates.grid, &plates.orogeny);
        let age = CubeField::from_cells(plates.grid, &plates.age);
        let profile = std::env::var("TERRA_PROFILE").is_ok();
        let mut clock = std::time::Instant::now();
        let mut step = |name: &str| {
            if profile {
                println!("    eau : {name} {:.2} s", clock.elapsed().as_secs_f64());
            }
            clock = std::time::Instant::now();
        };
        let trench = CubeField::from_cells(plates.grid, &plates.trench);

        // Relief de depart, sans erosion. La cote suit la croute continentale, a un bruit pres.
        // Le soulevement total de chaque point, lui, est apporte peu a peu pendant l'erosion.
        let swells = layer(7);
        let base_and_uplift = |dir: DVec3| -> (f32, f32) {
            let shore = crust.sample(dir) - 0.5 + coast.fbm(dir * 11.0, 6) * 0.42;
            let o = orogeny.sample(dir).clamp(0.0, 1.4);
            let chain = 0.75 + 0.5 * (ranges.eroded(dir * 34.0, 5) * 0.5 + 0.5);
            // Les chaines : 2 km pour une chaine moyenne, plus de 5 pour les plus hautes, et un plafond
            // vers 7 km, la ou la croute epaissie s'etale en hauts plateaux au lieu de monter encore.
            let raw = o.powf(1.25) * 5200.0 * chain;
            let mountain = 7000.0 * (1.0 - (-raw / 7000.0).exp()) * params.relief;
            let bumps = plateau.fbm(dir * 9.0, 5);
            if shore > 0.0 {
                // Le socle : plaines cotieres basses, interieur plus haut, et de grands bombements de
                // milliers de kilometres (plateaux d'Afrique, du Bresil, de Mongolie) entre des bassins.
                let inland = smoothstep(0.0, 0.4, shore);
                let swell = smoothstep(0.02, 0.4, swells.fbm(dir * 2.4, 4));
                let land = 30.0
                    + 330.0 * inland
                    + 1150.0 * swell * inland * params.relief
                    + bumps * 260.0 * smoothstep(0.0, 0.2, shore);
                // Un soulevement inegal : sans lui, l'eau descendrait les pentes en stries paralleles.
                let uneven = 1.0 + relief_noise.fbm(dir * 26.0, 6) * 0.8;
                let total = land.max(4.0) + mountain;
                (land.max(4.0) + mountain * 0.3, (total * uneven).max(4.0))
            } else {
                let floor = ocean_floor(shore, age.sample(dir)) - trench.sample(dir).clamp(0.0, 1.0) * 2200.0;
                let h = (floor + mountain * 0.8).max(-6900.0);
                (h, h)
            }
        };

        // Premiere erosion, a l'echelle des grands bassins versants.
        let grid = CubeGrid { n: COARSE_N };
        let start = par_cells(grid, &|c| base_and_uplift(grid.center(c)).0);
        let uplift = par_cells(grid, &|c| base_and_uplift(grid.center(c)).1);
        let mut h = start;
        let rain = par_cells(grid, &|c| rain_at(grid.center(c), h[c]));
        step("relief grossier");
        ensure_a_sea(&mut h);
        erode(grid, &jittered_nodes(params.seed, grid), &mut h, &uplift, COARSE_STEPS, 1.0, &rain);
        step("erosion grossiere");

        // Seconde erosion : le meme relief, plus fin, avec un bruit que l'eau va inciser.
        let medium = CubeGrid { n: MEDIUM_N };
        let mut h2 = resample(medium, grid, &h);
        let bump = |dir: DVec3, value: f32| -> f32 {
            if value <= 0.0 {
                return value;
            }
            let strength = 18.0 + value.min(4000.0) * 0.1;
            (value + relief_noise.fbm(dir * 70.0, 5) * strength).max(1.0)
        };
        h2 = {
            let old = &h2;
            par_cells(medium, &|c| bump(medium.center(c), old[c]))
        };
        // Le soulevement continue, moins longtemps : ce pas d'erosion doit rester fidele au premier.
        let uplift2 = resample(medium, grid, &uplift);
        let rain2 = {
            let old = &h2;
            par_cells(medium, &|c| rain_at(medium.center(c), old[c]))
        };
        erode(medium, &jittered_nodes(params.seed, medium), &mut h2, &uplift2, MEDIUM_STEPS, 0.12, &rain2);
        step("erosion moyenne");

        // Grille finale. La ou le climat et l'histoire s'y pretent, des cuvettes : glaciers
        // disparus dans le froid, piemonts et fosses tectoniques. Elles deviendront des lacs.
        let fine = CubeGrid { n: FINAL_N };
        let mut h3 = resample(fine, medium, &h2);
        h3 = {
            let old = &h3;
            par_cells(fine, &|c| {
                let dir = fine.center(c);
                let value = old[c];
                if value < 12.0 {
                    return value;
                }
                let value = value + relief_noise.fbm(dir * 180.0, 3) * (2.0 + value.min(3000.0) * 0.006);
                // Les lacs de la Terre sont surtout l'oeuvre des glaciers : ils criblent les pays froids
                // et humides que les calottes ont rabotes (Canada, Scandinavie), et le pied des chaines.
                // Dans les pays arides, l'eau s'evapore avant de remplir une cuvette.
                let climate = climate_at(dir, value);
                let humid = smoothstep(0.3, 0.75, climate.aridity());
                let glaciated = smoothstep(6.0, -1.0, climate.mean) * smoothstep(-4.0, 6.0, climate.hot);
                let o = orogeny.sample(dir).clamp(0.0, 1.4);
                let piedmont = smoothstep(0.05, 0.2, o) * (1.0 - smoothstep(0.35, 0.7, o));
                let lucky = chance.fbm(dir * 8.0, 3) * 1.5;
                let likely = (glaciated * 0.85 + piedmont * 0.3 + lucky * 0.45).clamp(0.0, 1.0) * humid;
                if likely <= 0.05 {
                    return value;
                }
                // Des cuvettes de quelques dizaines de kilometres, aux contours dechiquetes.
                let bent = dir + DVec3::new(
                    pits.fbm(dir * 60.0 + DVec3::splat(3.0), 3) as f64,
                    pits.fbm(dir * 60.0 + DVec3::splat(11.0), 3) as f64,
                    pits.fbm(dir * 60.0 + DVec3::splat(19.0), 3) as f64,
                ) * 0.006;
                let spot = pits.fbm(bent * 330.0, 4);
                let hollow = smoothstep(0.5 - 0.3 * likely, 0.58 - 0.26 * likely, spot);
                (value - hollow * (22.0 + 110.0 * likely)).max(1.0)
            })
        };
        step("grille fine");
        // Le climat definitif, sur le relief erode : c'est lui qui remplit les lacs et les rivieres.
        let weather = {
            let relief = CubeField::from_cells(fine, &h3);
            ClimateMap::build(params, &|dir| relief.sample_linear(dir))
        };
        let climate_at = |dir: DVec3, height: f32| weather.at(dir, height, 1.0);
        let rain_at = |dir: DVec3, height: f32| (climate_at(dir, height).rain / 800.0).max(0.02);
        step("climat");
        let cell_area = ((planet_radius() * std::f64::consts::FRAC_PI_2 / FINAL_N as f64) * 0.001).powi(2) as f32;

        let fine_nodes = jittered_nodes(params.seed, fine);
        let flow = route(fine, &h3, &fine_nodes);
        step("routage");
        let (lake_of, lakes_found) =
            find_lakes(fine, &h3, &flow, &|c| {
                // Hors des pays rabotes par les glaces, une cuvette est presque toujours comblee ou videe
                // par sa riviere : seule une sur quatre garde un lac.
                let dir = fine.center(c);
                let climate = climate_at(dir, h3[c]);
                let glaciated = climate.mean < 4.0 && climate.hot > -2.0;
                let wet = climate.aridity();
                (glaciated && wet > 0.35) || (wet > 0.65 && chance.fbm(dir * 8.0, 3) > 0.12)
            });
        step("lacs");

        // Une fois les lacs connus, les autres cuvettes sont comblees de sediments.
        let surface: Vec<f32> =
            (0..fine.cells()).map(|c| if lake_of[c] == NONE { flow.filled[c] } else { h3[c] }).collect();
        let mask: Vec<f32> = lake_of.iter().map(|l| (*l != NONE) as u8 as f32).collect();

        let rain3 = {
            let filled = &flow.filled;
            par_cells(fine, &|c| rain_at(fine.center(c), filled[c]))
        };
        let area = accumulate(&flow, &rain3, cell_area);
        step("pluie et debits");

        let (mut edges, lakes) = build_rivers(&fine_nodes, fine, &flow, &area, &lake_of, &lakes_found);
        step("rivieres");
        // Planete sans rivieres : les lacs restent, leur trop-plein s'infiltre.
        if !params.rivers {
            edges.clear();
        }
        let (index_grid, index_start, index_edges) = build_index(&edges);
        step("index");

        let weighted: Vec<f32> =
            lake_of.iter().map(|l| if *l == NONE { 0.0 } else { lakes_found[*l as usize].level }).collect();

        // Largeur de chaque lac, pour ne pas le dessiner sur un maillage plus large que lui.
        let cell_km = (planet_radius() * std::f64::consts::FRAC_PI_2 / FINAL_N as f64 * 0.001) as f32;
        let mut lake_span = vec![0u16; fine.cells()];
        let mut ring = [0u32; 8];
        for (c, l) in lake_of.iter().enumerate() {
            if *l != NONE {
                let span = ((lakes_found[*l as usize].cells.len() as f32).sqrt() * cell_km).clamp(1.0, 60_000.0) as u16;
                lake_span[c] = lake_span[c].max(span);
                let count = fine.neighbours(c, &mut ring);
                for n in &ring[..count] {
                    lake_span[*n as usize] = lake_span[*n as usize].max(span);
                }
            }
        }

        let water = Self {
            elevation: CubeField::from_cells(fine, &surface),
            lake_mask: CubeField::from_cells(fine, &mask),
            lake_level: CubeField::from_cells(fine, &weighted),
            lake_span,
            edges,
            lakes,
            index_grid,
            index_start,
            index_edges,
        };
        (water, weather)
    }

    /// Lac en un point : (masque, niveau). Le masque vaut 0 loin de tout lac.
    #[inline]
    pub fn lake_at(&self, dir: DVec3) -> (f32, f32) {
        let m = self.lake_mask.sample(dir);
        if m < 0.015 {
            return (0.0, 0.0);
        }
        (m, self.lake_level.sample(dir) / m)
    }

    /// Largeur, en metres, du lac qui borde le lieu `dir` (0 s'il n'y en a pas).
    #[inline]
    pub fn lake_span(&self, dir: DVec3) -> f32 {
        self.lake_span[CubeGrid { n: FINAL_N }.cell_of(dir)] as f32 * 1000.0
    }

    /// `dir` : point deja deforme par les meandres. `spacing` : finesse du maillage demande.
    /// `widen` elargit la vallee, de 1 a `MAX_WIDEN` : en montagne, les versants sont plus hauts.
    /// `estuary`, de 0 a 1 : pres de la mer, le lit s'evase en estuaire.
    pub fn river_at(&self, dir: DVec3, spacing: f64, widen: f32, estuary: f32) -> Option<RiverSample> {
        let cell = self.index_grid.cell_of(dir);
        let (from, to) = (self.index_start[cell] as usize, self.index_start[cell + 1] as usize);
        if from == to {
            return None;
        }
        let spacing_m = spacing as f32;
        let paint_radius = (spacing * 0.75).min(10_000.0);

        let mut out = RiverSample::default();
        let flare = 1.0 + ESTUARY_FLARE * estuary;
        // Minimum doux des fonds de toutes les rivieres proches : chacune creuse son lit, et deux lits se
        // rejoignent sans seuil. Le niveau de l'eau est celui des lits les plus bas.
        let (mut low, mut sum, mut sum_level) = (f32::MAX, 0.0f32, 0.0f32);
        let mut best_flow = (f32::MAX, Vec3::ZERO);
        let mut found = false;
        for &e in &self.index_edges[from..to] {
            let edge = &self.edges[e as usize];
            let c = Vec3::from_array(edge.c).as_dvec3();
            let reach = edge.reach as f64 + paint_radius;
            if (dir - c).length_squared() * planet_radius() * planet_radius() > reach * reach {
                continue;
            }
            let (a, b) = (Vec3::from_array(edge.a).as_dvec3(), Vec3::from_array(edge.b).as_dvec3());
            let at = |t: f64| a * ((1.0 - t) * (1.0 - t)) + c * (2.0 * t * (1.0 - t)) + b * (t * t);
            let mut q0 = a;
            for segment in 0..CURVE_SEGMENTS {
                let q1 = at((segment + 1) as f64 / CURVE_SEGMENTS as f64);
                let seg = q1 - q0;
                let to = dir - q0;
                // Plan tangent : le trace ne suit pas exactement la courbure de la sphere.
                let (seg2, to2) = (seg - dir * seg.dot(dir), to - dir * to.dot(dir));
                let k = (to2.dot(seg2) / seg2.length_squared().max(1e-30)).clamp(0.0, 1.0);
                let distance = (to2 - seg2 * k).length() * planet_radius();
                let t = ((segment as f64 + k) / CURVE_SEGMENTS as f64) as f32;
                q0 = q1;

                let width = (edge.width[0] + (edge.width[1] - edge.width[0]) * t) * flare;
                let band = valley_band(width) * widen;
                let d = distance as f32;

                if distance < paint_radius {
                    // Une riviere etroite se devine, une grande riviere se voit de tres haut.
                    let strength = (0.4 + 0.2 * (width * 22.0 / spacing_m).max(1e-3).log2()).clamp(0.0, 1.0);
                    out.paint = out.paint.max(strength * (1.0 - smoothstep(0.4, 1.0, d / paint_radius as f32)));
                }
                if d >= band {
                    continue;
                }
                found = true;
                let fade = 1.0 - smoothstep(band * 0.04, band * 0.16, spacing_m);
                // Le cone s'efface au bord : un relief plus haut que lui ne doit pas y faire de marche.
                out.reach = out.reach.max(fade * (1.0 - smoothstep(0.82, 1.0, d / band)));
                let level = edge.level[0] + (edge.level[1] - edge.level[0]) * t;
                // Le lit, puis un versant dont la pente croit jusqu'a rejoindre n'importe quel relief.
                let u = d / width;
                let profile = if u < 1.0 {
                    -(0.8 + 0.2 * width.sqrt()).min(26.0) * (1.0 - u * u)
                } else {
                    let x = d / band;
                    (d - width) * VALLEY_SLOPE + band * 0.62 * x * x * x
                };
                let target = level + profile;
                if target < low {
                    let k = if sum > 0.0 { (-(low - target) / BED_BLEND).exp() } else { 0.0 };
                    sum = sum * k + 1.0;
                    sum_level = sum_level * k + level;
                    low = target;
                } else {
                    let k = (-(target - low) / BED_BLEND).exp();
                    sum += k;
                    sum_level += k * level;
                }
                if u < best_flow.0 {
                    let speed = (0.5 + edge.slope.max(0.0).sqrt() * 22.0).min(8.0);
                    best_flow = (u, seg2.normalize_or_zero().as_vec3() * speed);
                }
                out.valley = out.valley.max((1.0 - smoothstep(width, width * 3.0 + 120.0, d)) * fade);
                out.bank = out.bank.max(1.0 - smoothstep(width, width * 2.0 + 25.0, d));
                // L'eau deborde un peu du lit sous le sol, pour que le maillage coupe la berge.
                let wet_radius = width + spacing_m * 3.0 + 15.0;
                out.lift = out.lift.max((1.0 - smoothstep(wet_radius * 1.1, wet_radius * 2.2, d)) * fade);
                out.wet |= d < wet_radius && spacing_m < width * 2.5 + 6.0;
            }
        }
        if !found && out.paint <= 0.0 {
            return None;
        }
        if found {
            out.level = sum_level / sum;
            out.profile = low - BED_BLEND * sum.ln() - out.level;
            out.flow = best_flow.1;
        }
        Some(out)
    }

    /// Longueur totale du reseau, en km.
    pub fn network_km(&self) -> f64 {
        self.edges
            .iter()
            .map(|e| {
                let (a, b) = (Vec3::from_array(e.a).as_dvec3(), Vec3::from_array(e.b).as_dvec3());
                (a - b).length() * planet_radius() * 0.001
            })
            .sum()
    }

    /// Positions d'un troncon, pour les cartes et les tests : debut, milieu, fin.
    pub fn edge_points(&self, edge: &RiverEdge) -> [DVec3; 3] {
        let (a, c, b) = (
            Vec3::from_array(edge.a).as_dvec3(),
            Vec3::from_array(edge.c).as_dvec3(),
            Vec3::from_array(edge.b).as_dvec3(),
        );
        [a.normalize(), (a * 0.25 + c * 0.5 + b * 0.25).normalize(), b.normalize()]
    }
}

/// Demi-largeur d'une vallee autour d'un lit de demi-largeur donnee.
fn valley_band(width: f32) -> f32 {
    (2_600.0 + width * 10.0).min(9_000.0)
}

/// Pente des versants au pied de la vallee.
const VALLEY_SLOPE: f32 = 0.09;
/// Ou un affluent rejoint le troncon du cours principal, de 0 (debut) a 1 (fin).
const JUNCTION: f64 = 0.7;
/// Part de l'alignement de la fin d'un affluent sur le courant principal.
const ALIGN: f64 = 0.4;
/// Douceur du raccord entre deux lits, en metres.
const BED_BLEND: f32 = 0.6;
/// Elargissement du lit a l'embouchure : un estuaire.
const ESTUARY_FLARE: f32 = 1.8;
/// Elargissement maximal d'une vallee.
pub const MAX_WIDEN: f32 = 4.0;

/// Une planete sans mer n'aurait nulle part ou drainer ses eaux.
fn ensure_a_sea(h: &mut [f32]) {
    if h.iter().all(|v| *v > 0.0) {
        let lowest = threshold_for_share(h, 0.998);
        for v in h.iter_mut() {
            if *v <= lowest {
                *v = -50.0;
            }
        }
    }
}

struct FoundLake {
    level: f32,
    cells: Vec<u32>,
    max_depth: f32,
}

/// Les cuvettes assez profondes deviennent des lacs, la ou `keeps_water` dit que le climat les remplit :
/// dans un pays aride, l'eau s'evapore et la cuvette reste une plaine de sediments.
fn find_lakes(
    grid: CubeGrid,
    h: &[f32],
    flow: &Drainage,
    keeps_water: &dyn Fn(usize) -> bool,
) -> (Vec<u32>, Vec<FoundLake>) {
    let mut lake_of = vec![NONE; grid.cells()];
    let mut visited = vec![false; grid.cells()];
    let mut lakes = Vec::new();
    let mut around = [0u32; 8];
    let mut stack = Vec::new();
    for &start in &flow.order {
        let s = start as usize;
        if visited[s] || flow.filled[s] <= h[s] + 0.01 {
            continue;
        }
        let level = flow.filled[s];
        let mut cells = Vec::new();
        let mut max_depth = 0.0f32;
        stack.push(start);
        visited[s] = true;
        while let Some(c) = stack.pop() {
            cells.push(c);
            max_depth = max_depth.max(level - h[c as usize]);
            let count = grid.neighbours(c as usize, &mut around);
            for &n in &around[..count] {
                let k = n as usize;
                if !visited[k] && flow.filled[k] == level && h[k] < level {
                    visited[k] = true;
                    stack.push(n);
                }
            }
        }
        if max_depth >= MIN_LAKE_DEPTH && cells.len() <= MAX_LAKE_CELLS && keeps_water(s) {
            for &c in &cells {
                lake_of[c as usize] = lakes.len() as u32;
            }
            lakes.push(FoundLake { level, cells, max_depth });
        }
    }
    (lake_of, lakes)
}

fn half_width(area: f32) -> f32 {
    (0.42 * area.sqrt()).clamp(2.5, 3500.0)
}

fn build_rivers(
    jittered: &[Vec3],
    grid: CubeGrid,
    flow: &Drainage,
    area: &[f32],
    lake_of: &[u32],
    found: &[FoundLake],
) -> (Vec<RiverEdge>, Vec<Lake>) {
    let cells = grid.cells();
    let mut is_river = vec![false; cells];
    for &c in &flow.order {
        let c = c as usize;
        // Plus d'eau en altitude : la pluie y tombe plus, et les sources y sont plus proches.
        let min =
            MIN_RIVER_AREA + (MIN_MOUNTAIN_RIVER_AREA - MIN_RIVER_AREA) * smoothstep(300.0, 1400.0, flow.filled[c]);
        is_river[c] = area[c] >= min;
    }
    // Une riviere ne s'arrete pas en chemin, meme si le seuil est plus haut en aval.
    for &c in flow.order.iter().rev() {
        let r = flow.receiver[c as usize];
        if is_river[c as usize] && r != NONE {
            is_river[r as usize] = true;
        }
    }
    // Tout lac deborde : sa sortie est une riviere, meme si peu d'eau y arrive.
    let mut outlets = vec![NONE; found.len()];
    for (l, lake) in found.iter().enumerate() {
        // Le fond de l'arbre de drainage du lac : le premier atteint en remontant depuis l'aval.
        let mut outlet = NONE;
        for &c in &lake.cells {
            let r = flow.receiver[c as usize];
            if r != NONE
                && lake_of[r as usize] != l as u32
                && (outlet == NONE || area[c as usize] > area[outlet as usize])
            {
                outlet = c;
            }
        }
        outlets[l] = outlet;
        let mut c = outlet;
        while c != NONE && !is_river[c as usize] {
            is_river[c as usize] = true;
            c = flow.receiver[c as usize];
        }
        if outlet != NONE {
            let mut c = flow.receiver[outlet as usize];
            while c != NONE {
                is_river[c as usize] = true;
                c = flow.receiver[c as usize];
            }
        }
    }

    // Les sommets decales du routage : le trace suit exactement les pentes calculees.
    let node = |c: usize| -> DVec3 {
        if lake_of[c] != NONE {
            return grid.center(c);
        }
        jittered[c].as_dvec3()
    };

    let mut edge_of = vec![NONE; cells];
    let mut main_up = vec![NONE; cells];
    let mut river_cells = Vec::new();
    for &c in &flow.order {
        let c = c as usize;
        if !is_river[c] {
            continue;
        }
        let r = flow.receiver[c] as usize;
        let same_lake = lake_of[c] != NONE && lake_of[c] == lake_of[r];
        if same_lake {
            continue;
        }
        river_cells.push(c as u32);
        if main_up[r] == NONE || area[c] > area[main_up[r] as usize] {
            main_up[r] = c as u32;
        }
    }

    let level_of = |c: usize| -> f32 {
        let h = flow.filled[c];
        if h <= 0.0 || flow.receiver[c] == NONE {
            0.0
        } else {
            h
        }
    };
    let is_sea = |c: usize| flow.receiver[c] == NONE;
    let mut nodes: std::collections::HashMap<u32, DVec3> =
        std::collections::HashMap::with_capacity(river_cells.len() * 2);
    let mut pos = |c: usize| -> DVec3 { *nodes.entry(c as u32).or_insert_with(|| node(c)) };

    let mut edges = Vec::with_capacity(river_cells.len());
    for &c in &river_cells {
        let c = c as usize;
        let r = flow.receiver[c] as usize;
        let (pc, pr) = (pos(c), pos(r));
        let (zc, zr) = (level_of(c), level_of(r));

        let up = main_up[c];
        let (start, z0, w0) = if up != NONE && lake_of[c] == NONE {
            let u = up as usize;
            let pu = pos(u);
            ((pu + pc) * 0.5, (level_of(u) + zc) * 0.5, half_width((area[u] + area[c]) * 0.5))
        } else {
            (pc, zc, half_width(area[c]) * if up == NONE { 0.45 } else { 1.0 })
        };

        let joins_lake = lake_of[r] != NONE;
        let mut bend = pc;
        let (end, z1, w1) = if is_sea(r) || joins_lake {
            (pr, zr, half_width(area[c]) * if is_sea(r) { 1.5 } else { 1.0 })
        } else if main_up[r] == c as u32 {
            ((pc + pr) * 0.5, (zc + zr) * 0.5, half_width((area[c] + area[r]) * 0.5))
        } else {
            // Un affluent rejoint le cours principal exactement sur sa courbe, aux deux tiers du troncon
            // plutot qu'en son milieu : il arrive avec le courant, en Y, pas en T.
            let m = main_up[r] as usize;
            let rr = flow.receiver[r] as usize;
            let (pm, prr) = (pos(m), pos(rr));
            let (s, e) = ((pm + pr) * 0.5, (pr + prr) * 0.5);
            let t = JUNCTION;
            let j = s * ((1.0 - t) * (1.0 - t)) + pr * (2.0 * t * (1.0 - t)) + e * (t * t);
            let (zs, ze) = ((level_of(m) + zr) * 0.5, (zr + level_of(rr)) * 0.5);
            let z = zs + (ze - zs) * t as f32;
            // La fin de son trace s'incline dans le sens du courant principal.
            let along = ((pr - s) * (1.0 - t) + (e - pr) * t).normalize_or_zero();
            let aligned = j - along * (j - pc).length();
            bend = pc.lerp(aligned, ALIGN);
            (j, z.min(z0), half_width(area[c]))
        };
        let z1 = z1.min(z0);
        let length = ((start - end).length() * planet_radius()).max(1.0) as f32;
        let ext =
            (bend - start).length().max((bend - end).length()).max((start - end).length() * 0.5) * planet_radius();
        let reach = ext + (valley_band(w0.max(w1)) * MAX_WIDEN) as f64 + 200.0;
        edges.push(RiverEdge {
            a: start.as_vec3().to_array(),
            c: bend.as_vec3().to_array(),
            b: end.as_vec3().to_array(),
            level: [z0, z1],
            width: [w0, w1],
            reach: reach as f32,
            slope: (z0 - z1) / length,
            next: NONE,
            end: if is_sea(r) {
                RiverEnd::Sea
            } else if joins_lake {
                RiverEnd::Lake
            } else {
                RiverEnd::Continues
            },
        });
        edge_of[c] = edges.len() as u32 - 1;
    }
    for &c in &river_cells {
        let c = c as usize;
        let r = flow.receiver[c] as usize;
        let e = edge_of[c] as usize;
        if edges[e].end == RiverEnd::Continues {
            edges[e].next = edge_of[r];
        }
    }

    let lakes = found
        .iter()
        .enumerate()
        .map(|(l, lake)| {
            let outlet = outlets[l] as usize;
            Lake {
                level: lake.level,
                cells: lake.cells.len() as u32,
                max_depth: lake.max_depth,
                outlet: grid.center(outlet),
                outflow: edge_of[outlet],
            }
        })
        .collect();
    (edges, lakes)
}

fn build_index(edges: &[RiverEdge]) -> (CubeGrid, Vec<u32>, Vec<u32>) {
    let grid = CubeGrid { n: INDEX_N };
    // Chaque troncon se traite seul : les threads se partagent la liste, puis le tri remet tout en ordre.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per = edges.len().div_ceil(threads).max(1);
    let radius_of_planet = planet_radius();
    let mut pairs: Vec<(u32, u32)> = std::thread::scope(|scope| {
        let handles: Vec<_> = edges
            .chunks(per)
            .enumerate()
            .map(|(k, part)| {
                scope.spawn(move || {
                    crate::world::generator::worker_thread(radius_of_planet);
                    index_pairs(grid, part, (k * per) as u32)
                })
            })
            .collect();
        handles.into_iter().flat_map(crate::sync::join).collect()
    });
    pairs.sort_unstable();
    let mut start = vec![0u32; grid.cells() + 1];
    for (cell, _) in &pairs {
        start[*cell as usize + 1] += 1;
    }
    for i in 0..grid.cells() {
        start[i + 1] += start[i];
    }
    (grid, start, pairs.into_iter().map(|p| p.1).collect())
}

/// Les cases de l'index que touche chaque troncon d'une partie de la liste, numerotes a partir de `first`.
fn index_pairs(grid: CubeGrid, edges: &[RiverEdge], first: u32) -> Vec<(u32, u32)> {
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(edges.len() * 8);
    let mut cells: Vec<u32> = Vec::with_capacity(160);
    for (e, edge) in edges.iter().enumerate() {
        let e = first as usize + e;
        let (a, c, b) = (
            Vec3::from_array(edge.a).as_dvec3(),
            Vec3::from_array(edge.c).as_dvec3(),
            Vec3::from_array(edge.b).as_dvec3(),
        );
        let radius = (edge.reach as f64 + MAX_WARP + 10_000.0) / planet_radius();
        cells.clear();
        for t in [0.0, 0.5, 1.0] {
            let p = (a * ((1.0 - t) * (1.0 - t)) + c * (2.0 * t * (1.0 - t)) + b * (t * t)).normalize();
            let s = p.any_orthonormal_vector();
            let o = p.cross(s);
            cells.push(grid.cell_of(p) as u32);
            // Plusieurs cercles : la portee peut depasser la taille d'une case de l'index.
            for ring in [0.25, 0.5, 0.75, 1.0] {
                for k in 0..12 {
                    let angle = k as f64 * std::f64::consts::TAU / 12.0;
                    let q = (p + (s * angle.cos() + o * angle.sin()) * (radius * ring)).normalize();
                    cells.push(grid.cell_of(q) as u32);
                }
            }
        }
        cells.sort_unstable();
        cells.dedup();
        pairs.extend(cells.iter().map(|cell| (*cell, e as u32)));
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::generator::{Generator, PlanetParams, SEA_LEVEL};
    use std::sync::OnceLock;

    fn planet() -> &'static Generator {
        static PLANET: OnceLock<Generator> = OnceLock::new();
        PLANET.get_or_init(|| Generator::new(PlanetParams { seed: 11, ..Default::default() }))
    }

    #[test]
    fn every_river_flows_downhill_to_the_sea_or_a_lake() {
        let water = &planet().water;
        assert!(water.edges.len() > 5000, "{} troncons", water.edges.len());
        let (mut to_sea, mut to_lake) = (0, 0);
        for start in water.edges.iter().step_by(37) {
            let mut edge = start;
            let mut steps = 0;
            loop {
                assert!(edge.level[0] >= edge.level[1], "riviere qui remonte : {:?}", edge.level);
                match edge.end {
                    RiverEnd::Sea => {
                        to_sea += 1;
                        break;
                    }
                    RiverEnd::Lake => {
                        to_lake += 1;
                        break;
                    }
                    RiverEnd::Continues => {
                        let next = &water.edges[edge.next as usize];
                        // Un affluent rejoint le cours principal au milieu de son troncon.
                        assert!(next.level[1] <= edge.level[0] + 0.01, "{:?} puis {:?}", edge.level, next.level);
                        edge = next;
                    }
                }
                steps += 1;
                assert!(steps < 20_000, "riviere sans fin");
            }
        }
        assert!(to_sea > 100, "{to_sea} rivieres atteignent la mer");
        assert!(to_lake > 10, "{to_lake} rivieres finissent dans un lac");
    }

    #[test]
    fn rivers_are_born_in_the_heights_and_grow_downstream() {
        let water = &planet().water;
        let high = water.edges.iter().filter(|e| e.level[0] > 1200.0).count();
        assert!(high > 500, "{high} troncons au-dessus de 1200 m");
        // Les troncons qui arrivent a la mer sont bien plus larges que le plus etroit du reseau.
        let (mut mouth, mut count) = (0.0, 0);
        for e in water.edges.iter().filter(|e| e.end == RiverEnd::Sea) {
            mouth += e.width[1];
            count += 1;
        }
        let narrowest = water.edges.iter().map(|e| e.width[0]).fold(f32::MAX, f32::min);
        assert!(mouth / count as f32 > narrowest * 2.0);
        let widest = water.edges.iter().map(|e| e.width[1]).fold(0.0, f32::max);
        assert!(widest > 150.0, "plus grand fleuve : {widest} m de demi-largeur");
    }

    #[test]
    fn every_lake_overflows_into_a_river() {
        let water = &planet().water;
        assert!(water.lakes.len() > 50, "{} lacs", water.lakes.len());
        for lake in &water.lakes {
            assert!(lake.level > SEA_LEVEL);
            assert!(lake.max_depth >= MIN_LAKE_DEPTH);
            assert!(lake.outflow != NONE, "lac de {} cellules sans exutoire", lake.cells);
            let out = &water.edges[lake.outflow as usize];
            assert!(out.level[0] <= lake.level + 0.01, "l'exutoire part plus haut que le lac");
        }
    }

    /// Autour d'une riviere ou sur un rivage, le sol et l'eau restent continus et l'eau n'est
    /// jamais suspendue au-dessus du sol.
    #[test]
    fn ground_is_continuous_across_rivers_and_water_is_never_hanging() {
        let g = planet();
        let (mut wet, mut crossings) = (0, 0);
        for edge in g.water.edges.iter().filter(|e| e.width[0] > 15.0 && e.level[0] > 5.0).step_by(211).take(120) {
            let middle = g.water.edge_points(edge)[1];
            let side = middle.any_orthonormal_vector();
            let mut last: Option<f32> = None;
            // Une traversee de 6 km, assez large pour rencontrer le lit malgre les meandres.
            for i in -6000..6000 {
                let dir = (middle + side * (i as f64 / planet_radius())).normalize();
                let s = g.surface(dir);
                if let Some(h) = last {
                    if (s.height - h).abs() > 6.5 {
                        // Une paroi raide reste continue : a un centimetre, l'ecart disparait.
                        let close = (middle + side * ((i as f64 - 0.01) / planet_radius())).normalize();
                        let step = (g.surface(close).height - s.height).abs();
                        assert!(step < 0.5, "saut de {step} m en un centimetre (a {} m du fil de l'eau)", i);
                    }
                }
                last = Some(s.height);
                if s.fresh && s.flow.length() > 0.0 && s.height < s.water_level && s.water_level > 8.0 {
                    wet += 1;
                    assert!(s.water_level - s.height < 40.0, "{} m d'eau sur une riviere", s.water_level - s.height);
                }
            }
            crossings += 1;
        }
        assert!(crossings > 20, "{crossings} traversees");
        assert!(wet > 200, "{wet} points d'eau courante rencontres");
    }

    #[test]
    fn lakes_hold_water_at_their_level() {
        let g = planet();
        let (mut checked, mut under) = (0, 0);
        for lake in g.water.lakes.iter().step_by(13) {
            let s = g.surface(lake.outlet);
            checked += 1;
            if s.fresh && (s.water_level - lake.level).abs() < 200.0 && s.height < s.water_level {
                under += 1;
            }
        }
        assert!(under * 2 > checked, "{under} cellules de sortie en eau sur {checked}");
    }

    /// Les hauts reliefs de la planete doivent suivre l'orogenese simulee, pas un bruit.
    #[test]
    fn high_ground_follows_the_simulated_orogeny() {
        let g = planet();
        let grid = g.plates.grid;
        let (mut high, mut high_orogeny, mut low, mut low_orogeny) = (0.0, 0.0, 0.0, 0.0);
        for c in (0..grid.cells()).step_by(5) {
            if g.plates.crust[c] < 0.5 {
                continue;
            }
            let e = g.water.elevation.sample(grid.center(c));
            if e > 1500.0 {
                high += 1.0;
                high_orogeny += g.plates.orogeny[c];
            } else if e > 0.0 {
                low += 1.0;
                low_orogeny += g.plates.orogeny[c];
            }
        }
        assert!(high > 50.0, "{high} cellules hautes");
        assert!(high_orogeny / high > 1.5 * low_orogeny / low, "{} contre {}", high_orogeny / high, low_orogeny / low);
    }

    #[test]
    fn hydrology_is_deterministic() {
        let (a, b) = (
            Generator::new(PlanetParams { seed: 21, ..Default::default() }),
            Generator::new(PlanetParams { seed: 21, ..Default::default() }),
        );
        assert_eq!(a.water.edges.len(), b.water.edges.len());
        assert_eq!(a.water.lakes.len(), b.water.lakes.len());
        for (x, y) in a.water.edges.iter().zip(&b.water.edges).step_by(101) {
            assert_eq!(x.level, y.level);
            assert_eq!(x.a, y.a);
        }
    }
}
