//! Selection du quadtree de la planete et generation en arriere-plan, sans dependance au GPU.

use super::chunk::{build_chunk, ChunkMesh, NodeId};
use super::generator::{Generator, MAX_HEIGHT, MIN_HEIGHT, planet_radius};
use super::scatter::{build_grass, build_trees, CellObjects, GrassCell, Kind, CELL_LOD};
use glam::DVec3;
use super::fast_hash::{FastMap, FastSet};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

#[derive(Clone, Copy, Debug)]
pub struct StreamConfig {
    /// Distance de rendu, en metres : au-dela, le relief perd vite ses details.
    /// Se regle avec `set_render_distance`, qui ajuste aussi la vegetation.
    pub render_distance: f64,
    /// Un noeud se subdivise quand la distance est inferieure a `lod_factor * taille`.
    pub lod_factor: f64,
    pub tree_radius: f64,
    pub grass_radius: f64,
    /// Nombre de resultats integres par image, pour eviter les saccades.
    pub uploads_per_frame: usize,
    pub workers: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let mut config = Self {
            render_distance: 0.0,
            lod_factor: 2.2,
            tree_radius: 0.0,
            grass_radius: 0.0,
            uploads_per_frame: 24,
            workers: cores.saturating_sub(2).max(1),
        };
        config.set_render_distance(DEFAULT_RENDER_DISTANCE);
        config
    }
}

pub const DEFAULT_RENDER_DISTANCE: f64 = 80_000.0;
pub const MIN_RENDER_DISTANCE: f64 = 5_000.0;
pub const MAX_RENDER_DISTANCE: f64 = 1_000_000.0;

impl StreamConfig {
    pub fn set_render_distance(&mut self, metres: f64) {
        self.render_distance = metres.clamp(MIN_RENDER_DISTANCE, MAX_RENDER_DISTANCE);
        self.tree_radius = (self.render_distance * 0.03).clamp(500.0, 6_000.0);
        self.grass_radius = (self.render_distance * 0.003).clamp(90.0, 320.0);
    }
}

/// Objets qui bloquent le joueur, avec leur rayon a l'echelle 1.
const SOLID_KINDS: [(Kind, f64); 7] = [
    (Kind::Pine, 0.32),
    (Kind::Oak, 0.48),
    (Kind::Birch, 0.18),
    (Kind::Palm, 0.24),
    (Kind::Acacia, 0.26),
    (Kind::Cactus, 0.34),
    (Kind::Rock, 1.2),
];

/// Rayon de la sphere qui cache l'horizon. Plus bas que la mer : le relief sous-marin ne cache rien.
fn horizon_radius() -> f64 {
    planet_radius() - 200.0
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Job {
    Chunk(NodeId),
    Trees(NodeId),
    Grass(NodeId),
}

enum JobResult {
    Chunk(ChunkMesh),
    Trees(NodeId, CellObjects),
    Grass(NodeId, GrassCell),
}

#[derive(Default)]
struct Queue {
    /// Trie pour que la tache la plus prioritaire soit a la fin.
    jobs: Vec<Job>,
    /// Taches prises par un worker et dont le resultat n'a pas encore ete integre.
    active: FastSet<Job>,
    shutdown: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
}

/// Forme d'un noeud : direction du centre, rayon englobant et tranche d'altitude occupee.
#[derive(Clone, Copy)]
struct Shape {
    dir: DVec3,
    bound: f64,
    min_r: f64,
    max_r: f64,
}

impl Shape {
    fn new(id: NodeId, min_r: f64, max_r: f64) -> Self {
        let (dir, bound) = id.bounds();
        Self { dir, bound, min_r, max_r }
    }

    /// Distance au point le plus proche du noeud, en deca de sa vraie distance.
    fn distance(&self, cam: DVec3) -> f64 {
        let along = cam.dot(self.dir).clamp(self.min_r, self.max_r);
        ((cam - self.dir * along).length() - self.bound).max(0.0)
    }

    /// Vrai si le noeud est entierement cache derriere la courbure de la planete.
    fn below_horizon(&self, cam: DVec3) -> bool {
        let cam_r2 = cam.length_squared();
        let horizon2 = horizon_radius() * horizon_radius();
        if cam_r2 <= horizon2 {
            return false;
        }
        let top = self.dir * self.max_r;
        let reach = (cam_r2 - horizon2).sqrt() + (self.max_r * self.max_r - horizon2).max(0.0).sqrt();
        (cam - top).length() - self.bound > reach
    }
}

struct ChunkMeta {
    last_used: u64,
    shape: Shape,
}

/// Ce que le rendu doit synchroniser apres un `update`.
#[derive(Default)]
pub struct StreamUpdate {
    pub new_chunks: Vec<ChunkMesh>,
    pub evicted: Vec<NodeId>,
}

pub struct Streamer {
    pub config: StreamConfig,
    pub generator: Arc<Generator>,
    /// Noeuds a dessiner pour l'image courante.
    pub selected: Vec<NodeId>,
    pub trees: FastMap<NodeId, CellObjects>,
    pub grass: FastMap<NodeId, GrassCell>,
    /// Altitude de la camera au-dessus du sol ou de la mer.
    pub camera_altitude: f64,
    loaded: FastMap<NodeId, ChunkMeta>,
    shared: Arc<Shared>,
    results: Receiver<(Job, JobResult)>,
    workers: Vec<JoinHandle<()>>,
    frame: u64,
    idle: bool,
    /// Temps cumules des etapes de `update` (pour TERRA_PROFILE) : resultats, sol, quadtree, cellules, eviction, file.
    pub timings: [f64; 6],
    /// Cellules d'arbres et d'herbe autour de la camera, et la position ou elles ont ete calculees :
    /// le parcours du quadtree ne se refait qu'apres un deplacement de quelques metres.
    cells: (DVec3, Vec<(NodeId, f64)>, Vec<(NodeId, f64)>),
    /// Chunks recus des threads de generation depuis le debut.
    pub generated: usize,
    /// Position de la camera et image du dernier parcours du quadtree.
    last_select: (DVec3, u64),
}

impl Streamer {
    pub fn new(generator: Arc<Generator>, config: StreamConfig) -> Self {
        let shared = Arc::new(Shared { queue: Mutex::new(Queue::default()), wake: Condvar::new() });
        let (tx, results) = channel();
        let workers = (0..config.workers)
            .map(|i| {
                let (shared, gen, tx) = (shared.clone(), generator.clone(), tx.clone());
                std::thread::Builder::new()
                    .name(format!("terra-world-{i}"))
                    .spawn(move || {
                        crate::world::generator::worker_thread(gen.params.radius());
                        loop {
                        let job = {
                            let mut q = shared.queue.lock().unwrap();
                            loop {
                                if q.shutdown {
                                    return;
                                }
                                if let Some(job) = q.jobs.pop() {
                                    q.active.insert(job);
                                    break job;
                                }
                                q = shared.wake.wait(q).unwrap();
                            }
                        };
                        let result = match job {
                            Job::Chunk(id) => JobResult::Chunk(build_chunk(&gen, id)),
                            Job::Trees(c) => JobResult::Trees(c, build_trees(&gen, c)),
                            Job::Grass(c) => JobResult::Grass(c, build_grass(&gen, c)),
                        };
                        if tx.send((job, result)).is_err() {
                            return;
                        }
                        }
                    })
                    .expect("creation du thread de generation")
            })
            .collect();

        Self {
            config,
            generator,
            selected: Vec::new(),
            trees: FastMap::default(),
            grass: FastMap::default(),
            camera_altitude: 0.0,
            loaded: FastMap::default(),
            shared,
            results,
            workers,
            frame: 0,
            idle: false,
            timings: [0.0; 6],
            cells: (DVec3::ZERO, Vec::new(), Vec::new()),
            last_select: (DVec3::ZERO, 0),
            generated: 0,
        }
    }

    /// Vrai quand tout ce qui est demande autour de la camera est charge.
    pub fn is_idle(&self) -> bool {
        self.idle
    }

    /// Repousse un cercle vertical hors des troncs et rochers deja charges autour de lui.
    pub fn push_out_of_objects(&self, pos: &mut DVec3, radius: f64) {
        let reach = super::chunk::NodeId { face: 0, lod: CELL_LOD, x: 0, z: 0 }.size() + 20.0;
        for cell in self.trees.values() {
            if (cell.center - *pos).length_squared() > reach * reach * 4.0 {
                continue;
            }
            for (kind, trunk) in SOLID_KINDS {
                for it in &cell.objects[kind as usize] {
                    let up = pos.normalize();
                    let at = cell.center + glam::Vec3::from_array(it.pos).as_dvec3();
                    let mut d = *pos - at;
                    d -= up * d.dot(up);
                    let min = radius + trunk * it.scale as f64;
                    let dist = d.length();
                    if dist >= min {
                        continue;
                    }
                    let push = if dist > 1e-6 { d / dist } else { up.any_orthonormal_vector() };
                    *pos += push * (min - dist);
                }
            }
        }
    }

    pub fn loaded_chunks(&self) -> usize {
        self.loaded.len()
    }

    pub fn update(&mut self, cam: DVec3) -> StreamUpdate {
        self.frame += 1;
        let mut mark = std::time::Instant::now();
        let mut lap = |k: usize, t: &mut [f64; 6]| {
            let now = std::time::Instant::now();
            t[k] += (now - mark).as_secs_f64();
            mark = now;
        };
        let mut timings = self.timings;
        let mut update = StreamUpdate::default();
        self.integrate_results(&mut update);
        lap(0, &mut timings);

        let ground = self.generator.height(cam.normalize_or(DVec3::Y)).max(0.0) as f64;
        self.camera_altitude = cam.length() - planet_radius() - ground;
        lap(1, &mut timings);

        // Le quadtree ne change qu'avec un vrai deplacement ou l'arrivee de chunks : entre deux, la
        // selection et la file de travail de l'image precedente restent valables.
        let (last_cam, last_frame) = self.last_select;
        let since = self.frame - last_frame;
        let step = (self.camera_altitude * 0.02).max(1.0);
        let reselect = (last_cam - cam).length() > step
            || (!update.new_chunks.is_empty() && since >= 4)
            || since >= 30
            || self.selected.is_empty();
        if !reselect {
            lap(2, &mut timings);
            self.timings = timings;
            return update;
        }
        self.last_select = (cam, self.frame);
        let mut requests: Vec<(f64, Job)> = Vec::new();
        self.selected.clear();
        let range = (planet_radius() + MIN_HEIGHT as f64, planet_radius() + MAX_HEIGHT as f64);
        for root in NodeId::roots() {
            self.select(root, range, cam, &mut requests);
        }
        lap(2, &mut timings);

        if (self.cells.0 - cam).length() > 8.0 || self.frame % 120 == 0 {
            let (mut trees, mut grass) = (Vec::new(), Vec::new());
            let (tr, gr) = (self.config.tree_radius, self.config.grass_radius);
            self.for_cells_within(cam, tr.max(gr), &mut |cell, d| {
                if d < tr {
                    trees.push((cell, d));
                }
                if d < gr {
                    grass.push((cell, d));
                }
            });
            self.cells = (cam, trees, grass);
        }
        self.request_cells(&mut requests);
        lap(3, &mut timings);
        self.evict(cam, &mut update);
        lap(4, &mut timings);

        // Le grossier d'abord pres de la camera : la planete apparait vite puis s'affine.
        requests.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut q = self.shared.queue.lock().unwrap();
        q.jobs.clear();
        let active = std::mem::take(&mut q.active);
        q.jobs.extend(requests.iter().map(|r| r.1).filter(|j| !active.contains(j)));
        self.idle = requests.is_empty() && active.is_empty();
        q.active = active;
        let any = !q.jobs.is_empty();
        drop(q);
        if any {
            self.shared.wake.notify_all();
        }
        lap(5, &mut timings);
        self.timings = timings;
        update
    }

    fn integrate_results(&mut self, update: &mut StreamUpdate) {
        for _ in 0..self.config.uploads_per_frame {
            let Ok((job, result)) = self.results.try_recv() else { break };
            self.shared.queue.lock().unwrap().active.remove(&job);
            match result {
                JobResult::Chunk(mesh) => {
                    self.generated += 1;
                    let shape = Shape::new(mesh.id, mesh.min_radius, mesh.max_radius);
                    self.loaded.insert(mesh.id, ChunkMeta { last_used: self.frame, shape });
                    update.new_chunks.push(mesh);
                }
                JobResult::Trees(cell, objects) => {
                    self.trees.insert(cell, objects);
                }
                JobResult::Grass(cell, tufts) => {
                    self.grass.insert(cell, tufts);
                }
            }
        }
    }

    /// `range` : tranche d'altitude heritee du parent, tant que le noeud n'est pas calcule.
    fn select(&mut self, id: NodeId, range: (f64, f64), cam: DVec3, requests: &mut Vec<(f64, Job)>) {
        let frame = self.frame;
        let (shape, is_loaded) = match self.loaded.get_mut(&id) {
            Some(meta) => {
                meta.last_used = frame;
                (meta.shape, true)
            }
            None => (Shape::new(id, range.0, range.1), false),
        };
        if shape.below_horizon(cam) {
            return;
        }

        let distance = shape.distance(cam);
        // Au-dela de la distance de rendu, tout se passe comme si le noeud etait bien plus loin.
        // Depuis l'orbite la distance de rendu ne doit pas compter : la planete entiere reste nette.
        let range = self.config.render_distance.max(self.camera_altitude * 4.0);
        let apparent = if distance > range { distance * (distance / range).powi(2) } else { distance };
        if id.lod > 0 && apparent < self.config.lod_factor * id.size() {
            let children = id.children();
            let child_range = (shape.min_r, shape.max_r);
            // Un enfant derriere l'horizon n'a pas besoin d'exister pour que le parent se divise.
            let mut missing = [None; 4];
            let mut any = false;
            for (k, c) in children.iter().enumerate() {
                if self.loaded.contains_key(c) {
                    continue;
                }
                let child = Shape::new(*c, child_range.0, child_range.1);
                if !child.below_horizon(cam) {
                    missing[k] = Some(child.distance(cam));
                    any = true;
                }
            }
            if !any {
                for c in children {
                    self.select(c, child_range, cam, requests);
                }
                return;
            }
            // Le parent reste affiche tant que ses enfants visibles ne sont pas prets.
            for (c, d) in children.iter().zip(missing) {
                if let Some(d) = d {
                    requests.push((d / c.size(), Job::Chunk(*c)));
                }
            }
        }

        if is_loaded {
            self.selected.push(id);
        } else {
            requests.push((distance / id.size(), Job::Chunk(id)));
        }
    }

    /// Parcourt les cellules a moins de `radius` de la camera.
    fn for_cells_within(&self, cam: DVec3, radius: f64, f: &mut dyn FnMut(NodeId, f64)) {
        if self.camera_altitude > radius {
            return;
        }
        let ground = cam.length() - self.camera_altitude;
        let mut stack: Vec<NodeId> = NodeId::roots().to_vec();
        while let Some(id) = stack.pop() {
            let d = Shape::new(id, ground, ground).distance(cam);
            if d >= radius {
                continue;
            }
            if id.lod == CELL_LOD {
                f(id, d);
            } else {
                stack.extend(id.children());
            }
        }
    }

    fn request_cells(&self, requests: &mut Vec<(f64, Job)>) {
        let size = NodeId { face: 0, lod: CELL_LOD, x: 0, z: 0 }.size();
        for &(cell, d) in &self.cells.1 {
            if !self.trees.contains_key(&cell) {
                requests.push((d / size + 2.0, Job::Trees(cell)));
            }
        }
        for &(cell, d) in &self.cells.2 {
            if !self.grass.contains_key(&cell) {
                requests.push((d / size + 3.0, Job::Grass(cell)));
            }
        }
    }

    fn evict(&mut self, cam: DVec3, update: &mut StreamUpdate) {
        // Marge d'hysteresis : un aller-retour ne doit pas tout regenerer.
        const KEEP_FRAMES: u64 = 300;
        const CHUNK_BUDGET: usize = 2500;
        if self.loaded.len() > CHUNK_BUDGET {
            let frame = self.frame;
            let evicted = &mut update.evicted;
            self.loaded.retain(|id, meta| {
                let keep = frame - meta.last_used < KEEP_FRAMES;
                if !keep {
                    evicted.push(*id);
                }
                keep
            });
        }
        // A l'horizontale, comme les demandes : une cellule tout en bas d'une falaise serait
        // sinon jugee trop loin, oubliee aussitot que creee, puis redemandee sans fin.
        let radius = cam.length();
        let across = |center: DVec3| (center.normalize() * radius - cam).length();
        let tree_limit = self.config.tree_radius * 1.25 + 200.0;
        self.trees.retain(|_, c| across(c.center) < tree_limit);
        let grass_limit = self.config.grass_radius * 1.5 + 200.0;
        self.grass.retain(|_, c| across(c.center) < grass_limit);
    }
}

impl Drop for Streamer {
    fn drop(&mut self) {
        {
            let mut q = self.shared.queue.lock().unwrap();
            q.shutdown = true;
            q.jobs.clear();
        }
        self.shared.wake.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::generator::PlanetParams;
    use std::time::{Duration, Instant};

    fn settle(s: &mut Streamer, cam: DVec3) {
        let start = Instant::now();
        loop {
            s.update(cam);
            if s.is_idle() {
                return;
            }
            assert!(start.elapsed() < Duration::from_secs(120), "le streaming n'a pas converge");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn small_config() -> StreamConfig {
        StreamConfig { tree_radius: 200.0, grass_radius: 64.0, workers: 4, ..Default::default() }
    }

    fn generator(seed: u64) -> Arc<Generator> {
        Arc::new(Generator::new(PlanetParams { seed, ..Default::default() }))
    }

    fn surface_point(gen: &Generator, altitude: f64) -> DVec3 {
        let dir = gen.find_spawn();
        dir * (planet_radius() + gen.height(dir).max(0.0) as f64 + altitude)
    }

    fn is_ancestor(a: NodeId, b: NodeId) -> bool {
        if a.face != b.face || a.lod <= b.lod {
            return false;
        }
        let shift = a.lod - b.lod;
        (b.x >> shift) == a.x && (b.z >> shift) == a.z
    }

    fn assert_no_overlap(selected: &[NodeId]) {
        for a in selected {
            for b in selected {
                assert!(!is_ancestor(*a, *b), "{a:?} recouvre {b:?}");
            }
        }
    }

    /// Le noeud selectionne dont le centre est le plus proche d'une direction.
    fn node_under(selected: &[NodeId], dir: DVec3) -> NodeId {
        *selected
            .iter()
            .max_by(|a, b| {
                let ka = (a.center_dir() - dir).length() / a.size();
                let kb = (b.center_dir() - dir).length() / b.size();
                kb.total_cmp(&ka)
            })
            .unwrap()
    }

    #[test]
    fn selection_converges_to_full_detail_near_camera() {
        let gen = generator(9);
        let cam = surface_point(&gen, 2.0);
        let mut s = Streamer::new(gen, small_config());
        settle(&mut s, cam);

        assert_no_overlap(&s.selected);
        assert_eq!(node_under(&s.selected, cam.normalize()).lod, 0, "LOD 0 attendu sous la camera");
        assert!(!s.trees.is_empty() || !s.grass.is_empty());
    }

    #[test]
    fn from_space_the_whole_visible_face_is_covered_and_cheap() {
        let gen = generator(21);
        let cam = DVec3::new(0.3, 0.2, 1.0).normalize() * (planet_radius() * 3.0);
        let mut s = Streamer::new(gen, small_config());
        settle(&mut s, cam);

        assert_no_overlap(&s.selected);
        assert!(s.selected.len() < 200, "{} chunks vus de l'espace", s.selected.len());
        assert!(s.selected.iter().all(|n| n.lod > 8), "aucun detail fin depuis l'espace");
        assert!(s.trees.is_empty() && s.grass.is_empty(), "pas de vegetation depuis l'espace");

        // Chaque direction visible doit tomber dans un noeud selectionne.
        let up = cam.normalize();
        let side = up.any_orthonormal_vector();
        for i in 0..60 {
            let a = i as f64 * 0.21;
            let d = (up + (side * a.cos() + up.cross(side) * a.sin()) * (i as f64 / 60.0) * 0.8).normalize();
            let n = node_under(&s.selected, d);
            let (c, r) = n.bounds();
            assert!((c * planet_radius() - d * planet_radius()).length() <= r * 1.01, "trou vers {d:?}");
        }
    }

    #[test]
    fn far_side_of_the_planet_is_not_drawn() {
        let gen = generator(2);
        let cam = surface_point(&gen, 2.0);
        let mut s = Streamer::new(gen, small_config());
        settle(&mut s, cam);
        let up = cam.normalize();
        for n in &s.selected {
            let (c, r) = n.bounds();
            let angle = c.dot(up).clamp(-1.0, 1.0).acos() * planet_radius() - r;
            assert!(angle < 900_000.0, "{n:?} a {} km sous l'horizon", angle / 1000.0);
        }
    }

    #[test]
    fn selection_stays_valid_after_flying_around_the_planet() {
        let gen = generator(21);
        let mut s = Streamer::new(gen.clone(), small_config());
        settle(&mut s, surface_point(&gen, 50.0));
        // A l'oppose : l'ancienne zone doit etre oubliee, sans perdre en precision.
        let cam = -surface_point(&gen, 50.0).normalize() * (planet_radius() + 9000.0);
        settle(&mut s, cam);
        assert_no_overlap(&s.selected);
    }

    #[test]
    fn high_altitude_needs_fewer_chunks() {
        let gen = generator(4);
        let mut low = Streamer::new(gen.clone(), small_config());
        settle(&mut low, surface_point(&gen, 2.0));
        let mut high = Streamer::new(gen.clone(), small_config());
        settle(&mut high, surface_point(&gen, 2_000_000.0));
        assert!(high.selected.len() < low.selected.len());
    }

    #[test]
    fn render_distance_controls_detail_and_vegetation() {
        let gen = generator(4);
        let cam = surface_point(&gen, 300.0);
        let run = |metres: f64| {
            let mut config = small_config();
            config.set_render_distance(metres);
            let radius = config.tree_radius;
            let mut s = Streamer::new(gen.clone(), config);
            settle(&mut s, cam);
            assert_no_overlap(&s.selected);
            (s.selected.len(), radius)
        };
        let (near_chunks, near_trees) = run(MIN_RENDER_DISTANCE);
        let (far_chunks, far_trees) = run(MAX_RENDER_DISTANCE);
        assert!(near_chunks < far_chunks, "{near_chunks} contre {far_chunks} chunks");
        assert!(near_trees < far_trees);
    }

    #[test]
    fn trunks_push_the_player_out() {
        let mut found = false;
        for seed in 1..12 {
            let gen = generator(seed);
            let cam = surface_point(&gen, 1.75);
            let mut s = Streamer::new(gen.clone(), small_config());
            settle(&mut s, cam);
            let tree = s.trees.values().find_map(|c| {
                let list = &c.objects[Kind::Pine as usize];
                let list = if list.is_empty() { &c.objects[Kind::Oak as usize] } else { list };
                list.first().map(|t| c.center + glam::Vec3::from_array(t.pos).as_dvec3())
            });
            let Some(tree) = tree else { continue };
            found = true;
            let up = tree.normalize();
            let start = tree + up.any_orthonormal_vector() * 0.01 + up * 1.75;
            let mut pos = start;
            s.push_out_of_objects(&mut pos, 0.4);
            let mut d = pos - tree;
            d -= up * d.dot(up);
            assert!(d.length() >= 0.4, "toujours dans le tronc : {} m", d.length());
            break;
        }
        assert!(found, "aucun arbre autour des points de depart testes");
    }
}
