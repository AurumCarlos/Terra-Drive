//! Noeuds du quadtree de la planete (un par face d'un cube projete sur la sphere)
//! et construction de leurs maillages.

use super::city_mesh::{build_city_mesh, CityMesh, Terrain};
use super::generator::{planet_radius, smoothstep, Detail, Generator, Surface};
use bytemuck::{Pod, Zeroable};
use glam::DVec3;

/// Nombre de quads par cote d'un chunk, quel que soit son niveau de detail.
pub const GRID: usize = 48;
/// LOD le plus grossier : un noeud racine couvre une face entiere du cube.
pub const MAX_LOD: u8 = 17;
/// Longueur d'un quart de tour de planete, soit le cote d'une face.
#[allow(non_snake_case)]
pub fn FACE_ARC() -> f64 {
    planet_radius() * std::f64::consts::FRAC_PI_2
}

/// Axes (normale, tangente, bitangente) des six faces.
const FACES: [(DVec3, DVec3, DVec3); 6] = [
    (DVec3::X, DVec3::Z, DVec3::Y),
    (DVec3::NEG_X, DVec3::NEG_Z, DVec3::Y),
    (DVec3::Y, DVec3::X, DVec3::Z),
    (DVec3::NEG_Y, DVec3::NEG_X, DVec3::Z),
    (DVec3::Z, DVec3::NEG_X, DVec3::Y),
    (DVec3::NEG_Z, DVec3::X, DVec3::Y),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId {
    pub face: u8,
    pub lod: u8,
    pub x: u32,
    pub z: u32,
}

impl NodeId {
    /// Le noeud qui couvre toute la face `face` du cube (0 a 5).
    pub fn root(face: u8) -> Self {
        Self { face, lod: MAX_LOD, x: 0, z: 0 }
    }

    /// Les six noeuds racines : la planete entiere au plus grossier.
    pub fn roots() -> [Self; 6] {
        std::array::from_fn(|f| Self::root(f as u8))
    }

    /// Cote approximatif du noeud, en metres.
    pub fn size(&self) -> f64 {
        FACE_ARC() / (1u64 << (MAX_LOD - self.lod)) as f64
    }

    /// Distance entre deux sommets du maillage.
    pub fn spacing(&self) -> f64 {
        self.size() / GRID as f64
    }

    /// Les quatre noeuds deux fois plus fins qui couvrent celui-ci. Le noeud ne doit pas etre une feuille (`lod` > 0).
    pub fn children(&self) -> [NodeId; 4] {
        let (face, lod) = (self.face, self.lod - 1);
        let (x, z) = (self.x * 2, self.z * 2);
        [
            NodeId { face, lod, x, z },
            NodeId { face, lod, x: x + 1, z },
            NodeId { face, lod, x, z: z + 1 },
            NodeId { face, lod, x: x + 1, z: z + 1 },
        ]
    }

    /// Direction sur la sphere, s et t allant de 0 a 1 sur le noeud (les depasser reste valide).
    pub fn dir_at(&self, s: f64, t: f64) -> DVec3 {
        let span = 2.0 / (1u64 << (MAX_LOD - self.lod)) as f64;
        let u = -1.0 + (self.x as f64 + s) * span;
        let v = -1.0 + (self.z as f64 + t) * span;
        // La tangente egalise la taille des mailles entre le centre et les coins d'une face.
        let (n, a, b) = FACES[self.face as usize];
        (n + a * (u * std::f64::consts::FRAC_PI_4).tan() + b * (v * std::f64::consts::FRAC_PI_4).tan()).normalize()
    }

    /// Inverse de `dir_at` : coordonnees (s, t) d'une direction sur ce noeud, hors de [0, 1] si elle
    /// tombe a cote. None si elle est sur une autre face.
    pub fn locate(&self, dir: DVec3) -> Option<(f64, f64)> {
        let (n, a, b) = FACES[self.face as usize];
        let dn = dir.dot(n);
        if dn <= 1e-9 {
            return None;
        }
        let q = std::f64::consts::FRAC_PI_4;
        let u = (dir.dot(a) / dn).atan() / q;
        let v = (dir.dot(b) / dn).atan() / q;
        let span = 2.0 / (1u64 << (MAX_LOD - self.lod)) as f64;
        Some(((u + 1.0) / span - self.x as f64, (v + 1.0) / span - self.z as f64))
    }

    /// Le noeud d'un LOD donne qui contient une direction.
    pub fn containing(dir: DVec3, lod: u8) -> NodeId {
        let inside =
            |id: &NodeId| id.locate(dir).is_some_and(|(s, t)| (0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&t));
        let mut id = Self::roots().into_iter().find(inside).unwrap_or(Self::root(0));
        while id.lod > lod {
            let children = id.children();
            id = children.into_iter().find(inside).unwrap_or(children[0]);
        }
        id
    }

    /// Direction du centre du noeud, depuis le centre de la planete.
    pub fn center_dir(&self) -> DVec3 {
        self.dir_at(0.5, 0.5)
    }

    /// Direction du centre et rayon englobant (au niveau de la mer) du noeud.
    pub fn bounds(&self) -> (DVec3, f64) {
        let c = self.center_dir();
        let center = c * planet_radius();
        let mut r: f64 = 0.0;
        for (s, t) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
            r = r.max((self.dir_at(s, t) * planet_radius() - center).length());
        }
        (c, r)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TerrainVertex {
    /// Position relative au centre du chunk.
    pub pos: [f32; 3],
    /// Ecart vers la surface du LOD parent, applique progressivement par le vertex shader.
    pub morph: [f32; 3],
    pub normal: [f32; 3],
    /// Couleur encodee en racine carree (plus de precision dans les sombres).
    /// Pour l'eau : r = profondeur (WATER_DEPTH_SCALE), g = glace, b et a = courant vers l'est et le nord.
    pub color: [u8; 4],
    /// Roche du lieu, encodee comme la couleur ; a = force de ses strates.
    pub rock: [u8; 4],
    /// Altitude : les strates suivent des couches horizontales, au pixel.
    pub altitude: f32,
}

/// Profondeur d'eau representee par r = 1 dans les sommets d'ocean.
pub const WATER_DEPTH_SCALE: f32 = 60.0;

pub struct ChunkMesh {
    pub id: NodeId,
    /// Centre du chunk, au niveau de la mer, dans le repere de la planete.
    pub center: DVec3,
    pub vertices: Vec<TerrainVertex>,
    /// Surface de l'eau : mer, lacs et rivieres, seulement si le chunk en contient.
    pub water: Option<Vec<TerrainVertex>>,
    /// Rues et batiments des villes, s'il y en a sur ce chunk.
    pub city: Option<CityMesh>,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub min_radius: f64,
    pub max_radius: f64,
}

/// Vitesse du courant representee par la valeur maximale de `TerrainVertex::color`.
pub const FLOW_SCALE: f32 = 8.0;

/// Couleur d'une riviere trop etroite pour le maillage, vue de loin.
const FAR_RIVER: [f32; 3] = [0.012, 0.045, 0.085];

/// Pourtour du chunk : les jupes (skirts) y descendent pour masquer les fissures entre LOD.
fn perimeter() -> Vec<(usize, usize)> {
    let n = GRID;
    let mut p = Vec::with_capacity(4 * n);
    p.extend((0..n).map(|i| (i, 0)));
    p.extend((0..n).map(|j| (n, j)));
    p.extend((0..n).map(|i| (n - i, n)));
    p.extend((0..n).map(|j| (0, n - j)));
    p
}

/// Indices de la grille seule, sans les jupes : ce que dessine l'ocean.
pub const GRID_INDEX_COUNT: u32 = (GRID * GRID * 6) as u32;

/// Tous les chunks partagent la meme topologie, donc le meme index buffer.
pub fn build_indices() -> Vec<u16> {
    let n = GRID;
    let row = n + 1;
    let idx = |i: usize, j: usize| (j * row + i) as u16;
    let mut out = Vec::with_capacity(n * n * 6 + 4 * n * 6);
    for j in 0..n {
        for i in 0..n {
            let (a, b, c, d) = (idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1));
            // Diagonale fixe : celle du LOD parent, pour que le morphing tombe juste.
            out.extend_from_slice(&[a, c, d, a, d, b]);
        }
    }
    let per = perimeter();
    let base = row * row;
    for k in 0..per.len() {
        let k2 = (k + 1) % per.len();
        let (e0, e1) = (idx(per[k].0, per[k].1), idx(per[k2].0, per[k2].1));
        let (s0, s1) = ((base + k) as u16, (base + k2) as u16);
        out.extend_from_slice(&[e0, s0, s1, e0, s1, e1]);
    }
    out
}

/// Les memes chunks a un sommet sur deux : exactement la surface du LOD parent (ces sommets n'ont pas
/// d'ecart de morphing), avec ses jupes. Sert aux ombres lointaines, ou un texel couvre deja une maille.
pub fn build_coarse_indices() -> Vec<u16> {
    let n = GRID;
    let row = n + 1;
    let idx = |i: usize, j: usize| (j * row + i) as u16;
    let mut out = Vec::with_capacity(n * n * 3 / 2 + 4 * n * 3);
    for j in (0..n).step_by(2) {
        for i in (0..n).step_by(2) {
            let (a, b, c, d) = (idx(i, j), idx(i + 2, j), idx(i, j + 2), idx(i + 2, j + 2));
            out.extend_from_slice(&[a, c, d, a, d, b]);
        }
    }
    let per = perimeter();
    let base = row * row;
    // Les points pairs du pourtour sont des sommets pairs de la grille (GRID est pair).
    for k in (0..per.len()).step_by(2) {
        let k2 = (k + 2) % per.len();
        let (e0, e1) = (idx(per[k].0, per[k].1), idx(per[k2].0, per[k2].1));
        let (s0, s1) = ((base + k) as u16, (base + k2) as u16);
        out.extend_from_slice(&[e0, s0, s1, e0, s1, e1]);
    }
    out
}

/// Encode une couleur lineaire pour `TerrainVertex::color`.
pub fn encode_color(c: [f32; 3]) -> [u8; 4] {
    let e = |v: f32| (v.max(0.0).sqrt().min(1.0) * 255.0 + 0.5) as u8;
    [e(c[0]), e(c[1]), e(c[2]), 255]
}

/// Ecart vers le milieu du segment du LOD parent ; nul pour les sommets partages avec lui.
fn morph_of(points: &dyn Fn(i32, i32) -> DVec3, i: usize, j: usize) -> DVec3 {
    let (ii, jj) = (i as i32, j as i32);
    let p = points(ii, jj);
    let target = match (i % 2 == 1, j % 2 == 1) {
        (true, true) => (points(ii - 1, jj - 1) + points(ii + 1, jj + 1)) * 0.5,
        (true, false) => (points(ii - 1, jj) + points(ii + 1, jj)) * 0.5,
        (false, true) => (points(ii, jj - 1) + points(ii, jj + 1)) * 0.5,
        (false, false) => p,
    };
    target - p
}

/// Calcule le maillage du noeud `id` : relief, couleurs, eau, et la ville qui s'y trouve.
/// Long (quelques millisecondes) : a appeler depuis un thread de generation.
pub fn build_chunk(gen: &Generator, id: NodeId) -> ChunkMesh {
    let n = GRID;
    let detail = Detail::for_spacing(id.spacing());
    let center = id.center_dir() * planet_radius();

    // Une rangee de marge autour de la grille pour calculer les normales aux bords.
    let side = n + 3;
    let mut dirs = vec![DVec3::ZERO; side * side];
    let mut heights = vec![0.0f32; side * side];
    let mut surfaces: Vec<Surface> = Vec::with_capacity(side * side);
    for j in 0..side {
        for i in 0..side {
            let s = (i as f64 - 1.0) / n as f64;
            let t = (j as f64 - 1.0) / n as f64;
            let dir = id.dir_at(s, t);
            dirs[j * side + i] = dir;
            let surface = gen.surface_detail(dir, &detail);
            heights[j * side + i] = surface.height;
            surfaces.push(surface);
        }
    }
    let at = |i: i32, j: i32| ((j + 1) as usize) * side + (i + 1) as usize;
    let point = |i: i32, j: i32| {
        let k = at(i, j);
        dirs[k] * (planet_radius() + heights[k] as f64)
    };

    let row = n + 1;
    let mut vertices = Vec::with_capacity(row * row + 4 * n);
    let mut min = DVec3::splat(f64::MAX);
    let mut max = DVec3::splat(f64::MIN);
    let (mut min_h, mut max_h) = (f32::MAX, f32::MIN);

    for j in 0..row {
        for i in 0..row {
            let (ii, jj) = (i as i32, j as i32);
            let k = at(ii, jj);
            let (dir, y) = (dirs[k], heights[k]);
            let mut normal = (point(ii + 1, jj) - point(ii - 1, jj))
                .cross(point(ii, jj + 1) - point(ii, jj - 1))
                .normalize_or_zero();
            if normal.dot(dir) < 0.0 {
                normal = -normal;
            }
            let surface = &surfaces[k];
            let mut climate = gen.climate(dir, y);
            // L'eau reverdit le fond des vallees, jusque dans les pays secs.
            climate.moisture = (climate.moisture + surface.valley * 0.3).min(1.0);
            // Une berge de riviere ou de lac a bien moins de sable qu'une plage de mer.
            let above = if surface.fresh {
                // De loin, une berge est plus etroite que la maille : elle ne doit pas la colorer entiere.
                let far =
                    if y > surface.water_level { smoothstep(30.0, 200.0, id.spacing() as f32) * 10.0 } else { 0.0 };
                (y - surface.water_level) * 3.0 + (1.0 - surface.bank) * 10.0 + far
            } else {
                y - surface.water_level
            };
            // Courbure locale, sans unite : assombrit les creux et ramene du vert dans les vallees.
            let around =
                heights[at(ii - 1, jj)] + heights[at(ii + 1, jj)] + heights[at(ii, jj - 1)] + heights[at(ii, jj + 1)];
            let concavity = ((around - 4.0 * y) / id.spacing() as f32 * 0.7).clamp(-1.0, 1.0);
            let normal_y = normal.dot(dir) as f32;
            let site = gen.site(dir);
            let ground = gen.ground_color_at(dir, &site, above, normal_y, climate, concavity, surface.ridge);
            let (col, canopy) =
                gen.with_canopy_at(ground, dir, &site.w, above, normal_y, climate, surface.ridge, id.spacing());
            // Le sol des villes : chaussees, trottoirs, cours et jardins ; de loin, la teinte des quartiers.
            let (col, canopy) = match gen.cities.paint(dir, id.spacing(), gen) {
                Some((paint, k)) => (std::array::from_fn(|c| col[c] + (paint[c] - col[c]) * k), canopy * (1.0 - k)),
                None => (col, canopy),
            };
            // Les autoroutes : un trait gris vu de loin, un sol nu sous la chaussee de pres.
            let (col, canopy) = match gen.cities.highways.paint(dir, id.spacing()) {
                Some((paint, k)) => (std::array::from_fn(|c| col[c] + (paint[c] - col[c]) * k), canopy * (1.0 - k)),
                None => (col, canopy),
            };
            // Les circuits : piste, herbe rase et bacs a gravier.
            let (col, canopy) = match gen.cities.circuits.paint(dir, id.spacing()) {
                Some((paint, k)) => (std::array::from_fn(|c| col[c] + (paint[c] - col[c]) * k), canopy * (1.0 - k)),
                None => (col, canopy),
            };
            // Les routes de montagne.
            let (col, canopy) = match gen.cities.mountains.paint(dir, id.spacing()) {
                Some((paint, k)) => (std::array::from_fn(|c| col[c] + (paint[c] - col[c]) * k), canopy * (1.0 - k)),
                None => (col, canopy),
            };
            let mut color = encode_color(std::array::from_fn(|c| col[c] + (FAR_RIVER[c] - col[c]) * surface.paint));
            color[3] = (canopy * (1.0 - surface.paint) * 255.0) as u8;
            let (rock_color, strata) = (site.rock, site.strata);
            let mut rock = encode_color(rock_color);
            rock[3] = (strata * 255.0) as u8;

            let pos = point(ii, jj) - center;
            let morph = morph_of(&point, i, j);
            min = min.min(pos).min(pos + morph);
            max = max.max(pos).max(pos + morph);
            min_h = min_h.min(y);
            max_h = max_h.max(y);
            vertices.push(TerrainVertex {
                pos: pos.as_vec3().to_array(),
                morph: morph.as_vec3().to_array(),
                normal: normal.as_vec3().to_array(),
                color,
                rock,
                altitude: y,
            });
        }
    }

    let skirt = id.spacing() * 3.0 + 1.0;
    for (i, j) in perimeter() {
        let dir = dirs[at(i as i32, j as i32)];
        let mut v = vertices[j * row + i];
        let pos = DVec3::from_array(v.pos.map(|c| c as f64)) - dir * skirt;
        min = min.min(pos);
        max = max.max(pos);
        v.pos = pos.as_vec3().to_array();
        vertices.push(v);
    }

    // Ou le sol passe sous l'eau. Les sommets sans eau restent caches sous le sol.
    let row_cells = (0..row).flat_map(|j| (0..row).map(move |i| (i as i32, j as i32)));
    let has_water = row_cells.clone().any(|(i, j)| {
        let k = at(i, j);
        surfaces[k].height < surfaces[k].water_level
    });
    let has_fresh = row_cells.clone().any(|(i, j)| surfaces[at(i, j)].fresh);
    let mut max_level = 0.0f32;
    let level = |k: usize| {
        let s = &surfaces[k];
        if s.fresh {
            s.water_level
        } else if has_fresh {
            (s.height - 4.0).max(0.0)
        } else {
            0.0
        }
    };
    let water = has_water.then(|| {
        let at_level = |i: i32, j: i32| {
            let k = at(i, j);
            dirs[k] * (planet_radius() + level(k) as f64)
        };
        let mut out = Vec::with_capacity(row * row);
        for j in 0..row {
            for i in 0..row {
                let k = at(i as i32, j as i32);
                let dir = dirs[k];
                let pos = at_level(i as i32, j as i32) - center;
                let morph = morph_of(&at_level, i, j);
                min = min.min(pos).min(pos + morph);
                max = max.max(pos).max(pos + morph);
                let surface_level = level(k);
                max_level = max_level.max(surface_level);
                let climate = gen.climate(dir, surface_level);
                let mut depth = ((surface_level - heights[k]) / WATER_DEPTH_SCALE).clamp(0.0, 1.0);
                // De loin, les hauts-fonds d'un lac ne doivent pas laisser voir le fond en anneau clair.
                if surfaces[k].fresh {
                    depth = depth.max(smoothstep(30.0, 200.0, id.spacing() as f32) * 0.25);
                }
                // Le courant dans le plan local, comme le relit le shader de l'eau.
                let east = DVec3::Y.cross(dir).normalize_or_zero();
                let north = dir.cross(east);
                let flow = surfaces[k].flow.as_dvec3();
                let code = |v: f64| (128.0 + (v as f32 / FLOW_SCALE).clamp(-1.0, 1.0) * 127.0) as u8;
                out.push(TerrainVertex {
                    pos: pos.as_vec3().to_array(),
                    morph: morph.as_vec3().to_array(),
                    normal: dir.as_vec3().to_array(),
                    color: [
                        (depth * 255.0) as u8,
                        (gen.sea_ice(climate) * 255.0) as u8,
                        code(flow.dot(east)),
                        code(flow.dot(north)),
                    ],
                    rock: [0; 4],
                    altitude: surface_level,
                });
            }
        }
        out
    });

    let levels: Vec<f32> = surfaces.iter().map(|s| s.water_level).collect();
    let terrain = Terrain { id, heights: &heights, levels: &levels, gen, detail: &detail };
    let city = build_city_mesh(gen, id, center, &terrain);
    let mut max_radius = planet_radius() + max_h.max(max_level).max(0.0) as f64;
    if let Some(c) = &city {
        min = min.min(c.min);
        max = max.max(c.max);
        max_radius = max_radius.max(c.top);
    }

    ChunkMesh {
        id,
        center,
        vertices,
        water,
        city,
        aabb_min: min.as_vec3().to_array(),
        aabb_max: max.as_vec3().to_array(),
        min_radius: planet_radius() + (min_h.min(0.0) as f64) - skirt,
        max_radius,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::generator::PlanetParams;

    fn gen() -> Generator {
        Generator::new(PlanetParams::default())
    }

    #[test]
    fn indices_stay_in_bounds_of_the_vertices() {
        let mesh = build_chunk(&gen(), NodeId { face: 2, lod: 12, x: 3, z: 5 });
        let idx = build_indices();
        assert_eq!(idx.len() % 3, 0);
        assert!(idx.iter().all(|&i| (i as usize) < mesh.vertices.len()));
        assert_eq!(mesh.vertices.len(), (GRID + 1) * (GRID + 1) + 4 * GRID);
    }

    #[test]
    fn coarse_indices_only_use_even_vertices() {
        let idx = build_coarse_indices();
        let row = GRID + 1;
        assert_eq!(idx.len(), (GRID / 2) * (GRID / 2) * 6 + 2 * GRID * 6);
        for &i in &idx {
            let i = i as usize;
            if i < row * row {
                assert!((i % row).is_multiple_of(2) && (i / row).is_multiple_of(2), "sommet impair {i}");
            } else {
                assert!((i - row * row).is_multiple_of(2) && i - row * row < 4 * GRID);
            }
        }
    }

    #[test]
    fn locate_inverts_dir_at() {
        let id = NodeId { face: 3, lod: 6, x: 1234, z: 777 };
        for (s, t) in [(0.0, 0.0), (0.25, 0.8), (1.0, 0.5)] {
            let (s2, t2) = id.locate(id.dir_at(s, t)).unwrap();
            assert!((s - s2).abs() < 1e-6 && (t - t2).abs() < 1e-6);
        }
        let dir = id.dir_at(0.3, 0.6);
        assert_eq!(NodeId::containing(dir, 6), id);
    }

    #[test]
    fn finest_level_is_about_a_metre_and_a_half() {
        let lod0 = NodeId { face: 0, lod: 0, x: 0, z: 0 };
        assert!((1.2..2.0).contains(&lod0.spacing()), "{}", lod0.spacing());
    }

    #[test]
    fn children_exactly_cover_their_parent() {
        let p = NodeId { face: 4, lod: 9, x: 2, z: 7 };
        let c = p.children();
        assert!((p.dir_at(0.0, 0.0) - c[0].dir_at(0.0, 0.0)).length() < 1e-12);
        assert!((p.dir_at(1.0, 1.0) - c[3].dir_at(1.0, 1.0)).length() < 1e-12);
        assert!((p.dir_at(0.5, 0.5) - c[0].dir_at(1.0, 1.0)).length() < 1e-12);
    }

    #[test]
    fn neighbouring_chunks_share_their_border() {
        let g = gen();
        let a = build_chunk(&g, NodeId { face: 0, lod: 9, x: 40, z: 90 });
        let b = build_chunk(&g, NodeId { face: 0, lod: 9, x: 41, z: 90 });
        let row = GRID + 1;
        for j in 0..row {
            let (va, vb) = (a.vertices[j * row + GRID], b.vertices[j * row]);
            let pa = a.center + DVec3::from_array(va.pos.map(|c| c as f64));
            let pb = b.center + DVec3::from_array(vb.pos.map(|c| c as f64));
            assert!((pa - pb).length() < 0.01, "rangee {j} : {} m", (pa - pb).length());
        }
    }

    /// Les six faces doivent se recoller exactement : aucun point du bord d'une face
    /// ne reste sans point correspondant sur une autre face.
    #[test]
    fn cube_faces_join_without_gaps() {
        let g = gen();
        let meshes: Vec<_> = NodeId::roots().iter().map(|r| build_chunk(&g, *r)).collect();
        let row = GRID + 1;
        let world = |m: &ChunkMesh, i: usize| m.center + DVec3::from_array(m.vertices[i].pos.map(|c| c as f64));
        for (f, m) in meshes.iter().enumerate() {
            for i in 0..row {
                let p = world(m, i);
                let found = meshes
                    .iter()
                    .enumerate()
                    .any(|(o, other)| o != f && (0..row * row).any(|k| (world(other, k) - p).length() < 1.5));
                assert!(found, "face {f}, point {i} sans voisin");
            }
        }
    }

    #[test]
    fn full_morph_lands_on_the_parent_surface() {
        let m = build_chunk(&gen(), NodeId { face: 1, lod: 0, x: 9000, z: 9000 });
        let row = GRID + 1;
        let v = |i: usize, j: usize| m.vertices[j * row + i];
        let pos = |i: usize, j: usize| glam::Vec3::from_array(v(i, j).pos);
        for (i, j) in [(1, 2), (5, 8), (13, 20)] {
            let expected = (pos(i - 1, j) + pos(i + 1, j)) * 0.5;
            let morphed = pos(i, j) + glam::Vec3::from_array(v(i, j).morph);
            assert!((morphed - expected).length() < 1e-2);
        }
        assert_eq!(v(2, 2).morph, [0.0; 3]);
    }

    #[test]
    fn terrain_mesh_lies_on_the_sphere() {
        let m = build_chunk(&gen(), NodeId { face: 3, lod: 14, x: 3, z: 4 });
        for v in &m.vertices[..(GRID + 1) * (GRID + 1)] {
            let p = m.center + DVec3::from_array(v.pos.map(|c| c as f64));
            let altitude = p.length() - planet_radius();
            assert!(altitude.abs() < 20000.0, "altitude {altitude}");
            assert!(p.length() >= m.min_radius - 100.0 && p.length() <= m.max_radius + 100.0);
        }
    }
}
