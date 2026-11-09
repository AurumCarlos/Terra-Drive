//! Placement deterministe de la vegetation et des rochers. Une cellule est un noeud
//! du quadtree au niveau `CELL_LOD`, ce qui couvre la sphere sans raccord particulier.

use super::chunk::{encode_color, NodeId};
use super::city::{City, Ground};
use super::generator::{planet_radius, smoothstep, Biome, Generator};
use super::noise::{hash_seed, Rng};
use bytemuck::{Pod, Zeroable};
use glam::DVec3;

/// Niveau du quadtree qui sert de cellule : environ 150 m de cote.
pub const CELL_LOD: u8 = 1;
const TREE_ATTEMPTS: usize = 22;
const GRASS_ATTEMPTS: usize = 130;
/// Hauteurs echantillonnees par cote pour l'herbe : un calcul par touffe serait trop couteux.
/// Meme finesse que le terrain de LOD 1, donc les touffes restent posees sur le sol.
const GRASS_GRID: usize = 48;
const FLOWER_CHANCE: f32 = 0.018;
/// Dans les prairies en fleurs.
const MEADOW_FLOWER_CHANCE: f32 = 0.14;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Pine = 0,
    Oak = 1,
    Bush = 2,
    Cactus = 3,
    Rock = 4,
    Grass = 5,
    Flower = 6,
    Birch = 7,
    Palm = 8,
    TallGrass = 9,
    Acacia = 10,
}

pub const KIND_COUNT: usize = 11;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ObjectInstance {
    /// Relative au centre de la cellule dans le monde stocke, a la camera une fois envoyee au GPU.
    pub pos: [f32; 3],
    pub scale: f32,
    pub yaw: f32,
    pub tint: [u8; 4],
}

/// Instances d'une cellule, rangees par type d'objet.
pub struct CellObjects {
    pub center: DVec3,
    pub objects: [Vec<ObjectInstance>; KIND_COUNT],
}

pub struct GrassCell {
    pub center: DVec3,
    pub tufts: Vec<ObjectInstance>,
    pub flowers: Vec<ObjectInstance>,
    pub stalks: Vec<ObjectInstance>,
}

/// Teinte d'une herbe : alpha a 0 demande au shader l'eclairage des plantes minces.
fn herb_tint(color: [f32; 3]) -> [u8; 4] {
    let mut tint = encode_color(color);
    tint[3] = 0;
    tint
}

fn flatness(gen: &Generator, cell: NodeId, s: f64, t: f64, h: f32) -> f32 {
    let ds = 2.0 / cell.size();
    let (dx, dz) = (gen.height(cell.dir_at(s + ds, t)) - h, gen.height(cell.dir_at(s, t + ds)) - h);
    2.0 / (dx * dx + dz * dz + 4.0).sqrt()
}

fn cell_seed(gen: &Generator, salt: u64, cell: NodeId) -> u64 {
    let key = ((cell.face as i64) << 56) ^ cell.x as i64;
    hash_seed(gen.seed() ^ salt, key, cell.z as i64)
}

/// Centre de la cellule au niveau du sol : les distances a la camera restent justes en montagne.
fn cell_center(gen: &Generator, cell: NodeId) -> DVec3 {
    let dir = cell.center_dir();
    dir * (planet_radius() + gen.height(dir) as f64)
}

/// Arbres, buissons, cactus et rochers d'une cellule.
/// Villes qui couvrent une cellule.
fn cities_on(gen: &Generator, cell: NodeId) -> Vec<&City> {
    let dir = cell.center_dir();
    gen.cities
        .touching(dir, cell.size())
        .into_iter()
        .filter(|c| c.dir.dot(dir) > 0.5 && c.extent(c.local(dir)) < 1.3)
        .collect()
}

/// Arbres d'alignement des boulevards qui tombent dans la cellule.
fn street_trees(
    gen: &Generator,
    cell: NodeId,
    cities: &[&City],
    center: DVec3,
    out: &mut Vec<ObjectInstance>,
    rng: &mut Rng,
) {
    for city in cities {
        let corners: Vec<glam::DVec2> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .iter()
            .map(|&(s, t)| city.local(cell.dir_at(s, t)))
            .collect();
        let lo = corners.iter().fold(glam::DVec2::splat(f64::MAX), |a, b| a.min(*b)) - glam::DVec2::splat(4.0);
        let hi = corners.iter().fold(glam::DVec2::splat(f64::MIN), |a, b| a.max(*b)) + glam::DVec2::splat(4.0);
        for p in city.plan(gen).street_trees(lo, hi) {
            let dir = city.dir_of(p);
            let inside = cell.locate(dir).is_some_and(|(s, t)| (0.0..1.0).contains(&s) && (0.0..1.0).contains(&t));
            if !inside {
                continue;
            }
            let surface = gen.surface(dir);
            if surface.height < surface.water_level + 0.5 {
                continue;
            }
            let v = 0.85 + 0.3 * rng.f32();
            let green = [v * v * 0.95, v * v, v * v * 0.8];
            let scale = 0.7 + 0.25 * rng.f32();
            let pos = dir * (planet_radius() + surface.height as f64 + 0.1) - center;
            out.push(ObjectInstance {
                pos: pos.as_vec3().to_array(),
                scale,
                yaw: rng.range(0.0, std::f32::consts::TAU),
                tint: encode_color(green.map(|c| c * 0.25)),
            });
        }
    }
}

/// Arbres, buissons et rochers de la cellule `cell`, ranges par type. Deterministe pour une planete.
pub fn build_trees(gen: &Generator, cell: NodeId) -> CellObjects {
    let mut out: [Vec<ObjectInstance>; KIND_COUNT] = Default::default();
    let mut rng = Rng(cell_seed(gen, 0x7EE5, cell));
    let center = cell_center(gen, cell);
    let cities = cities_on(gen, cell);
    let spacing = 1.0 / TREE_ATTEMPTS as f64;

    for j in 0..TREE_ATTEMPTS {
        for i in 0..TREE_ATTEMPTS {
            let s = (i as f64 + rng.f32() as f64) * spacing;
            let t = (j as f64 + rng.f32() as f64) * spacing;
            let roll = rng.f32();
            let (scale_r, yaw, tint_r) = (rng.f32(), rng.range(0.0, std::f32::consts::TAU), rng.f32());
            let (species, hue) = (rng.f32(), rng.f32());
            let dir = cell.dir_at(s, t);

            let surface = gen.surface(dir);
            let h = surface.height;
            if h < surface.water_level + 1.5 {
                continue;
            }
            let climate = gen.climate(dir, h);
            if climate.temperature < 0.1 {
                continue;
            }
            let biome = gen.biome(climate);
            let mut density =
                gen.tree_cover(biome, gen.woods(dir), climate.temperature) * gen.forest_share(dir, surface.ridge);
            let mut kind = match biome {
                Biome::Jungle => {
                    if species < 0.22 {
                        Kind::Palm
                    } else {
                        Kind::Oak
                    }
                }
                Biome::Forest => {
                    let pines = 0.03 + 0.72 * gen.stand(dir);
                    if species < pines {
                        Kind::Pine
                    } else if species < pines + 0.22 {
                        Kind::Birch
                    } else {
                        Kind::Oak
                    }
                }
                Biome::Taiga => {
                    if species < 0.12 {
                        Kind::Birch
                    } else {
                        Kind::Pine
                    }
                }
                Biome::Grassland => {
                    if species < 0.5 {
                        Kind::Pine
                    } else if species < 0.65 {
                        Kind::Birch
                    } else {
                        Kind::Oak
                    }
                }
                Biome::Savanna => {
                    if species < 0.82 {
                        Kind::Acacia
                    } else {
                        Kind::Oak
                    }
                }
                Biome::Desert => {
                    density = 0.012;
                    Kind::Cactus
                }
                Biome::Tundra => {
                    density = 0.03;
                    Kind::Bush
                }
            };
            // Des palmiers sur les cotes chaudes.
            if climate.temperature > 0.72 && h < 18.0 && species < 0.5 {
                (kind, density) = (Kind::Palm, density.max(0.1));
            }

            // Quelques buissons et rochers en plus des especes dominantes, davantage dans les pays de roche.
            // En ville, les arbres ne poussent que dans les parcs et les jardins.
            let ground = if cities.is_empty() { Ground::Wild } else { gen.cities.ground(dir, gen) };
            // Rien ne pousse sur une autoroute ni sur ses accotements.
            if gen.cities.highways.clearance(dir).is_some_and(|d| d < 12.0) {
                continue;
            }
            // Ni sur les circuits ni dans leurs degagements.
            if gen.cities.circuits.clearance(dir).is_some_and(|d| d < super::circuit::RUNOFF + 6.0) {
                continue;
            }
            match ground {
                Ground::Wild => {}
                Ground::Park | Ground::Garden => {
                    density = if ground == Ground::Park { 0.11 } else { 0.03 };
                    if matches!(kind, Kind::Cactus | Kind::Bush | Kind::Pine) && biome != Biome::Taiga {
                        kind = if species < 0.3 { Kind::Birch } else { Kind::Oak };
                    }
                }
                _ => continue,
            }
            if roll > density {
                if ground != Ground::Wild {
                    continue;
                }
                let extra = roll - density;
                let province = gen.provinces.at(dir);
                let stony = province.moor + province.mesa + province.badlands + province.inselberg + province.glacial;
                if extra < 0.03 && biome != Biome::Desert {
                    kind = Kind::Bush;
                } else if extra < 0.042 + 0.07 * stony.min(1.0) {
                    kind = Kind::Rock;
                } else {
                    continue;
                }
            }

            if gen.provinces.at(dir).dunes > 0.5 {
                continue;
            }
            let limit = if kind == Kind::Rock { 0.55 } else { 0.78 };
            if flatness(gen, cell, s, t, h) < limit {
                continue;
            }

            // Age du peuplement : jeunes recrus, futaies adultes, vieux arbres isoles plus grands.
            let age = if matches!(kind, Kind::Pine | Kind::Oak | Kind::Birch | Kind::Acacia) {
                let tone = ((gen.canopy_tone(dir) - 0.55) / 0.8).clamp(0.0, 1.0);
                let lone = if density < 0.15 { 1.15 } else { 1.0 };
                (0.5 + 0.65 * tone) * lone
            } else {
                1.0
            };
            let scale = age
                * match kind {
                    Kind::Pine => 0.6 + 1.1 * scale_r * scale_r.sqrt(),
                    Kind::Oak => 0.6 + 0.95 * scale_r,
                    Kind::Acacia => 0.7 + 0.6 * scale_r,
                    Kind::Birch => 0.8 + 0.6 * scale_r,
                    Kind::Palm => 0.75 + 0.5 * scale_r,
                    Kind::Bush => 0.6 + 0.8 * scale_r,
                    Kind::Cactus => 0.7 + 0.6 * scale_r,
                    Kind::Rock => 0.4 + 1.6 * scale_r * scale_r,
                    Kind::Grass | Kind::Flower | Kind::TallGrass => 1.0,
                };
            // Chaque arbre a sa nuance, du vert bleute au vert jaune.
            let v = (0.72 + 0.46 * tint_r) * gen.canopy_tone(dir);
            let tint = if kind == Kind::Rock {
                let (color, _) = gen.rock_color(dir);
                // Le maillage du rocher est gris moyen : on lui rend la couleur de la roche du lieu.
                [color[0] * 7.0 * v * v, color[1] * 7.0 * v * v, color[2] * 7.0 * v * v]
            } else {
                let green = [v * v * (0.8 + 0.5 * hue), v * v, v * v * (1.05 - 0.4 * hue)];
                // Foret d'automne : chaque feuillu vire a sa facon, de l'or au rouge. Les resineux restent verts.
                let fall = if matches!(kind, Kind::Oak | Kind::Birch) { gen.autumn(dir, climate) } else { 0.0 };
                if species * 7.3 % 1.0 < fall * 0.9 {
                    let turned = [3.3, 1.75 - 1.25 * hue, 0.25];
                    if kind == Kind::Birch {
                        [turned[0] * 0.8, 1.55, 0.2]
                    } else {
                        turned
                    }
                } else {
                    green
                }
            };
            let pos = dir * (planet_radius() + h as f64 - 0.12 * scale as f64) - center;
            out[kind as usize].push(ObjectInstance {
                pos: pos.as_vec3().to_array(),
                scale,
                yaw,
                tint: {
                    let mut encoded = encode_color(tint.map(|c| c * 0.25));
                    if kind == Kind::Palm {
                        encoded[3] = 200;
                    }
                    encoded
                },
            });
        }
    }
    if !cities.is_empty() {
        street_trees(gen, cell, &cities, center, &mut out[Kind::Oak as usize], &mut rng);
    }
    CellObjects { center, objects: out }
}

const FLOWER_COLORS: [[f32; 3]; 6] =
    [[0.95, 0.72, 0.05], [0.9, 0.88, 0.8], [0.42, 0.14, 0.75], [0.85, 0.08, 0.1], [0.12, 0.25, 0.85], [0.9, 0.3, 0.55]];

/// Touffes d'herbe et fleurs d'une cellule, teintees comme le sol pour s'y fondre.
pub fn build_grass(gen: &Generator, cell: NodeId) -> GrassCell {
    let center_dir = cell.center_dir();
    let center = cell_center(gen, cell);
    let (mut tufts, mut flowers, mut stalks) = (Vec::new(), Vec::new(), Vec::new());

    // Le climat ne varie pas a l'echelle d'une cellule.
    let climate = gen.climate(center_dir, gen.height(center_dir));
    let biome = gen.biome(climate);
    let grows = matches!(biome, Biome::Grassland | Biome::Forest | Biome::Savanna | Biome::Taiga | Biome::Jungle);
    if !grows || climate.temperature < 0.15 {
        return GrassCell { center, tufts, flowers, stalks };
    }

    let n = GRASS_GRID;
    let side = n + 2;
    let mut heights = vec![0.0f32; side * side];
    let mut levels = vec![0.0f32; side * side];
    for j in 0..side {
        for i in 0..side {
            let surface = gen.surface(cell.dir_at(i as f64 / n as f64, j as f64 / n as f64));
            heights[j * side + i] = surface.height;
            levels[j * side + i] = surface.water_level;
        }
    }
    let step = (cell.size() / n as f64) as f32;
    let city = !cities_on(gen, cell).is_empty();

    let mut rng = Rng(cell_seed(gen, 0x6A55, cell));
    // Deux couleurs de fleurs par prairie, plus naturel qu'un melange de toutes.
    let palette = [FLOWER_COLORS[(rng.f32() * 6.0) as usize], FLOWER_COLORS[(rng.f32() * 6.0) as usize]];
    let spacing = 1.0 / GRASS_ATTEMPTS as f64;
    let flower_chance = FLOWER_CHANCE + (MEADOW_FLOWER_CHANCE - FLOWER_CHANCE) * gen.meadow(center_dir);
    for j in 0..GRASS_ATTEMPTS {
        for i in 0..GRASS_ATTEMPTS {
            let s = (i as f64 + rng.f32() as f64) * spacing;
            let t = (j as f64 + rng.f32() as f64) * spacing;
            let (scale_r, yaw, tint_r) = (rng.f32(), rng.range(0.0, std::f32::consts::TAU), rng.f32());
            let (flower, color) = (rng.f32(), rng.f32());

            let (fx, fz) = ((s * n as f64) as f32, (t * n as f64) as f32);
            let (ix, iz) = ((fx as usize).min(n - 1), (fz as usize).min(n - 1));
            let (kx, kz) = (fx - ix as f32, fz - iz as f32);
            let at = |a: usize, b: usize| heights[b * side + a];
            let (h00, h10, h01, h11) = (at(ix, iz), at(ix + 1, iz), at(ix, iz + 1), at(ix + 1, iz + 1));
            let (dx, dz) = (h10 - h00, h01 - h00);
            let normal_y = step / (dx * dx + dz * dz + step * step).sqrt();
            if normal_y < 0.8 {
                continue;
            }
            let h = (h00 + (h10 - h00) * kx) * (1.0 - kz) + (h01 + (h11 - h01) * kx) * kz;
            let above = h - levels[iz * side + ix].max(levels[(iz + 1) * side + ix + 1]);
            if above < 2.0 || h > 2600.0 {
                continue;
            }
            let dir = cell.dir_at(s, t);
            // Pas d'herbe sur le pave, les cours et sous les maisons ; les pelouses en gardent.
            if city && !matches!(gen.cities.ground(dir, gen), Ground::Wild | Ground::Park | Ground::Garden) {
                continue;
            }
            if gen.cities.highways.clearance(dir).is_some_and(|d| d < 1.5) {
                continue;
            }
            // Pelouse tondue autour des circuits.
            if gen.cities.circuits.clearance(dir).is_some_and(|d| d < super::circuit::RUNOFF) {
                continue;
            }
            let pos = dir * (planet_radius() + h as f64 - 0.12) - center;

            if flower < flower_chance && matches!(biome, Biome::Grassland | Biome::Forest | Biome::Taiga) {
                flowers.push(ObjectInstance {
                    pos: pos.as_vec3().to_array(),
                    scale: 0.7 + 0.7 * scale_r,
                    yaw,
                    tint: herb_tint(palette[(color < 0.5) as usize]),
                });
            }
            let ground = gen.ground_color(dir, above, normal_y, climate, 0.0, 0.0);
            let v = 0.9 + 0.5 * tint_r;
            tufts.push(ObjectInstance {
                pos: pos.as_vec3().to_array(),
                scale: 0.7 + 0.9 * scale_r,
                yaw,
                tint: herb_tint(ground.map(|c| c * v)),
            });
            // Les hautes herbes poussent par nappes : la position decide, pas le hasard seul.
            let wave = ((s * 7.3 + t * 4.1).sin() + (s * 3.1 - t * 8.7 + cell.x as f64).cos()) as f32 * 0.25 + 0.5;
            if color < 0.06 + 0.5 * smoothstep(0.42, 0.78, wave) {
                let shade = 0.82 + 0.36 * tint_r;
                stalks.push(ObjectInstance {
                    pos: pos.as_vec3().to_array(),
                    scale: 0.8 + 0.6 * scale_r,
                    yaw: yaw * 1.7,
                    tint: herb_tint([shade, shade * 0.96, shade * 0.86]),
                });
            }
        }
    }
    GrassCell { center, tufts, flowers, stalks }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::generator::PlanetParams;

    #[test]
    fn scatter_is_deterministic_and_above_water() {
        let g = Generator::new(PlanetParams { seed: 77, ..Default::default() });
        let mut total = 0;
        for face in 0..6u8 {
            for k in 0..40u32 {
                let cell = NodeId { face, lod: CELL_LOD, x: 3000 + k * 900, z: 20000 + k * 700 };
                let (a, b) = (build_trees(&g, cell), build_trees(&g, cell));
                for k in 0..KIND_COUNT {
                    assert_eq!(a.objects[k].len(), b.objects[k].len());
                    for inst in &a.objects[k] {
                        let p = a.center + glam::Vec3::from_array(inst.pos).as_dvec3();
                        assert!(p.length() > planet_radius(), "sous la mer");
                        assert!(glam::Vec3::from_array(inst.pos).length() < 3.0 * cell.size() as f32);
                    }
                    total += a.objects[k].len();
                }
            }
        }
        assert!(total > 0, "la planete ne peut pas etre entierement vide");
    }

    fn cell_under(dir: DVec3) -> NodeId {
        // Retrouve la cellule sous un point en descendant le quadtree.
        let mut best = NodeId::roots()[0];
        for root in NodeId::roots() {
            if root.center_dir().dot(dir) > best.center_dir().dot(dir) {
                best = root;
            }
        }
        while best.lod > CELL_LOD {
            let children = best.children();
            best = children
                .into_iter()
                .max_by(|a, b| a.center_dir().dot(dir).total_cmp(&b.center_dir().dot(dir)))
                .unwrap();
        }
        best
    }

    #[test]
    fn plants_grow_near_the_spawn() {
        for seed in 1..5 {
            let g = Generator::new(PlanetParams { seed, ..Default::default() });
            let cell = cell_under(g.find_spawn());
            let grass = build_grass(&g, cell);
            assert!(grass.tufts.len() > 1000, "graine {seed} : {} touffes", grass.tufts.len());
        }
    }

    #[test]
    fn grass_lies_on_the_ground() {
        let g = Generator::new(PlanetParams::default());
        let grass = build_grass(&g, cell_under(g.find_spawn()));
        for t in grass.tufts.iter().step_by(97) {
            let p = grass.center + glam::Vec3::from_array(t.pos).as_dvec3();
            let ground = planet_radius() + g.height(p.normalize()) as f64;
            assert!((p.length() - ground).abs() < 0.8, "{} m du sol", p.length() - ground);
        }
    }
}
