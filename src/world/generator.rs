//! Fonction de planete : (parametres, direction) -> hauteur, climat, biome. Pure et deterministe,
//! donc utilisable depuis n'importe quel thread, sans jamais stocker le monde.

use super::city::Cities;
pub use super::climate::{Climate, Koppen};
use super::climate::ClimateMap;
use super::cube::CubeField;
use super::hydrology::{Hydrology, MAX_WIDEN, SHORE};
use super::landforms::{terrace, Cells, Province, Provinces, Volcanoes};
use super::noise::{hash_seed, Noise};
use super::tectonics::{self, Boundary, Tectonics};
use glam::{DVec3, Vec3};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Rayon de la Terre : celui d'une planete de taille 1.
pub const EARTH_RADIUS: f64 = 6_371_000.0;

static RADIUS_BITS: AtomicU64 = AtomicU64::new(EARTH_RADIUS.to_bits());
/// Nombre de threads qui simulent une planete en ce moment. Tant qu'il est nul, `planet_radius`
/// ne lit pas la variable du thread : sous Windows (MinGW), elle coute plus cher que tout le reste
/// et ne passe pas a l'echelle quand tous les coeurs la lisent.
static CREATING_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Rayon de la planete en cours de creation sur ce thread (0 : aucune). Se retire du compte a la
/// fin du thread.
struct Creating(Cell<f64>);

impl Drop for Creating {
    fn drop(&mut self) {
        if self.0.get() > 0.0 {
            CREATING_THREADS.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

thread_local! {
    /// Rayon de la planete en cours de creation, sur le thread qui la simule.
    static CREATING: Creating = const { Creating(Cell::new(0.0)) };
}

/// Rayon de la planete, en metres. Celui de la planete affichee, sauf sur les threads qui en
/// simulent une nouvelle en arriere-plan.
#[inline]
pub fn planet_radius() -> f64 {
    if CREATING_THREADS.load(Ordering::Relaxed) > 0 {
        let local = CREATING.with(|c| c.0.get());
        if local > 0.0 {
            return local;
        }
    }
    f64::from_bits(RADIUS_BITS.load(Ordering::Relaxed))
}

/// Fixe le rayon de la planete affichee : a appeler quand elle remplace la precedente.
pub fn set_planet_radius(r: f64) {
    RADIUS_BITS.store(r.to_bits(), Ordering::Relaxed);
}

/// Le rayon d'une planete en cours de creation, pour ce thread (0 : celui de la planete affichee).
pub fn set_creating_radius(r: f64) {
    // Le rayon de la planete affichee n'a pas besoin de la variable du thread : on garde le chemin
    // rapide (le cas des threads de calcul parallele lances pendant le jeu).
    let r = if r.to_bits() == RADIUS_BITS.load(Ordering::Relaxed) { 0.0 } else { r };
    CREATING.with(|c| {
        let old = c.0.replace(r);
        if old <= 0.0 && r > 0.0 {
            CREATING_THREADS.fetch_add(1, Ordering::Relaxed);
        } else if old > 0.0 && r <= 0.0 {
            CREATING_THREADS.fetch_sub(1, Ordering::Relaxed);
        }
    });
}
/// Altitude de la surface de l'ocean.
pub const SEA_LEVEL: f32 = 0.0;
/// Profondeur gagnee par l'avant-cote, des plages aux cotes rocheuses (m), et la profondeur de la grille sur
/// laquelle elle s'ajoute (m) : quelques centaines de metres au large.
const SHOREFACE_DEPTH: (f32, f32) = (14.0, 36.0);
const SHOREFACE_SCALE: f32 = 2.0;
/// Bornes sures du relief, valables avant d'avoir calcule un chunk.
pub const MIN_HEIGHT: f32 = -7000.0;
pub const MAX_HEIGHT: f32 = 24000.0;

/// Ce qui definit une planete.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanetParams {
    pub seed: u64,
    /// Part approximative de la surface sous l'ocean, 0 a 1.
    pub ocean: f32,
    /// Multiplie la hauteur des montagnes.
    pub relief: f32,
    /// Decale le climat, de -0.5 (glaciaire) a +0.5 (torride).
    pub temperature: f32,
    /// Rayon, relatif a celui de la Terre.
    pub size: f32,
    /// Nombre de villes voulu ; moins s'il n'y a pas assez de bons sites.
    pub cities: u32,
    /// Rivieres sur la planete. Sans elles, les lacs restent et les vallees sont seches.
    pub rivers: bool,
}

/// Ramene `value` dans `[min, max]` ; une valeur qui n'est pas un nombre devient `default`.
fn bounded(value: f32, min: f32, max: f32, default: f32) -> f32 {
    if value.is_nan() {
        default
    } else {
        value.clamp(min, max)
    }
}

impl PlanetParams {
    /// Bornes de la part d'ocean.
    pub const OCEAN: (f32, f32) = (0.0, 1.0);
    /// Bornes du multiplicateur de relief.
    pub const RELIEF: (f32, f32) = (0.0, 4.0);
    /// Bornes du decalage de climat.
    pub const TEMPERATURE: (f32, f32) = (-1.0, 1.0);
    /// Bornes du rayon, relatif a celui de la Terre.
    pub const SIZE: (f32, f32) = (0.1, 4.0);
    /// Nombre maximal de villes : au-dela, la simulation des routes prend des minutes.
    pub const MAX_CITIES: u32 = 600;

    /// Rayon en metres.
    pub fn radius(&self) -> f64 {
        EARTH_RADIUS * bounded(self.size, Self::SIZE.0, Self::SIZE.1, 1.0) as f64
    }

    /// Les memes reglages, ramenes dans leurs bornes ; une valeur qui n'est pas un nombre reprend
    /// celle par defaut. Tout reglage venu de l'exterieur (ligne de commande, menu) passe par ici :
    /// un rayon nul ou infini, ou des milliards de villes, bloqueraient la generation.
    pub fn sanitized(self) -> Self {
        let d = Self::default();
        Self {
            seed: self.seed,
            ocean: bounded(self.ocean, Self::OCEAN.0, Self::OCEAN.1, d.ocean),
            relief: bounded(self.relief, Self::RELIEF.0, Self::RELIEF.1, d.relief),
            temperature: bounded(self.temperature, Self::TEMPERATURE.0, Self::TEMPERATURE.1, d.temperature),
            size: bounded(self.size, Self::SIZE.0, Self::SIZE.1, d.size),
            cities: self.cities.min(Self::MAX_CITIES),
            rivers: self.rivers,
        }
    }
}

impl Default for PlanetParams {
    fn default() -> Self {
        Self { seed: 1, ocean: 0.68, relief: 1.0, temperature: 0.0, size: 1.0, cities: 140, rivers: true }
    }
}

/// Les grands paysages vegetaux, tires du climat (Koppen-Geiger et indice d'aridite).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Biome {
    /// Desert : presque rien ne pousse.
    Desert,
    /// Savane et brousse a epineux des tropiques a saison seche.
    Savanna,
    /// Foret tropicale humide.
    Jungle,
    /// Prairie et steppe.
    Grassland,
    /// Foret temperee, de feuillus et de coniferes.
    Forest,
    /// Foret boreale de coniferes.
    Taiga,
    /// Toundra : mousses et arbustes nains, au-dela de la limite des arbres.
    Tundra,
    /// Maquis et garrigue des climats mediterraneens.
    Shrubland,
    /// Glace et neiges eternelles.
    Ice,
}

/// Nombre d'octaves utiles pour un espacement de sommets donne : calculer des details plus
/// fins que la grille serait du temps perdu, et creerait du crenelage.
#[derive(Clone, Copy, Debug)]
pub struct Detail {
    mountains: u32,
    /// Aretes et pics secondaires des montagnes.
    alps: u32,
    rolling: u32,
    /// Collines de quelques kilometres, entre les grandes ondulations et les mottes.
    knolls: u32,
    /// Ravines, de `GULLY_PERIOD` jusqu'a la dizaine de metres.
    gullies: u32,
    /// Plateaux et canyons.
    tables: u32,
    hills: u32,
    fine: u32,
    /// Meandres des rivieres : du plus large au plus fin.
    meanders: [u32; 3],
    spacing: f64,
}

/// Longueur d'onde et amplitude des meandres, en metres.
/// Altitude de la grille sous laquelle une riviere devient estuaire, en metres.
const ESTUARY_RISE: f32 = 8.0;

const MEANDERS: [(f64, f64); 3] = [(11_000.0, 3_500.0), (3_000.0, 1_000.0), (520.0, 130.0)];

/// Le sol et l'eau en un point de la planete.
#[derive(Clone, Copy, Debug)]
pub struct Surface {
    pub height: f32,
    /// Altitude de l'eau : celle d'une riviere ou d'un lac, ou le niveau de la mer.
    pub water_level: f32,
    /// Vrai si ce niveau est celui d'une riviere ou d'un lac dessinable ici.
    pub fresh: bool,
    /// Sens du courant, de longueur la vitesse en m/s.
    pub flow: Vec3,
    /// Couleur d'eau a mettre sur le sol quand le maillage est trop large pour le lit.
    pub paint: f32,
    /// Proximite d'une rive, de 0 a 1.
    pub bank: f32,
    /// Fond de vallee fluviale, de 0 a 1 : l'humidite y monte.
    pub valley: f32,
    /// Cretes et ravines, de -1 (creux) a 1 (crete) : la roche affleure en haut, les plantes poussent en bas.
    pub ridge: f32,
}

/// Distance minimale sur laquelle se mesure la pente que suivent les ravines.
const SLOPE_EPS: f64 = 700.0;
/// Grilles ou cette pente est calculee : (pas, finesse de maillage en dessous de laquelle on s'en sert).
/// Au-dela, les sommets sont plus espaces que les noeuds : aucun gain, on la calcule en chaque sommet.
const SLOPE_LATTICES: [(f64, f64); 1] = [(175.0, 60.0)];
const SLOPE_CACHE_SIZE: usize = 8192;

/// Une case du cache de pente : (planete, x, y, z) et la pente a ce noeud.
type SlopeNode = ((u64, i64, i64, i64), Vec3);

thread_local! {
    /// Noeuds de la grille de pente deja calcules sur ce thread.
    static SLOPE_CACHE: std::cell::RefCell<Vec<SlopeNode>> =
        std::cell::RefCell::new(vec![((u64::MAX, 0, 0, 0), Vec3::ZERO); SLOPE_CACHE_SIZE]);
}

/// Les memes noeuds, partages entre tous les threads : un thread neuf (calcul parallele d'un plan de
/// ville, par exemple) profite de ce que les autres ont deja calcule. Chaque morceau se vide quand il
/// est plein ; les valeurs ne dependent que de la position, donc rien ne change.
const SHARED_SHARDS: usize = 64;
const SHARED_SHARD_SIZE: usize = 4096;
type SlopeKey = (u64, i64, i64, i64);
static SHARED_SLOPES: std::sync::OnceLock<Vec<std::sync::Mutex<super::fast_hash::FastMap<SlopeKey, Vec3>>>> =
    std::sync::OnceLock::new();

fn shared_slopes() -> &'static [std::sync::Mutex<super::fast_hash::FastMap<SlopeKey, Vec3>>] {
    SHARED_SLOPES.get_or_init(|| (0..SHARED_SHARDS).map(|_| Default::default()).collect())
}

/// Numero unique de chaque planete creee, pour les caches par thread.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Periode des plus grandes ravines, en metres.
const GULLY_PERIOD: f64 = 6000.0;
/// Rayon d'un etang, en mailles de `POND_CELL`.
const POND_RADIUS: f32 = 0.25;
const POND_CELL: f64 = 2600.0;

fn octaves(wavelength: f64, spacing: f64, max: u32) -> u32 {
    let n = (wavelength / (1.5 * spacing)).log2().ceil();
    (n.max(0.0) as u32).clamp(0, max)
}

impl Detail {
    /// Le detail utile pour des sommets espaces de `spacing` metres.
    pub fn for_spacing(spacing: f64) -> Self {
        Self {
            mountains: octaves(2.6e4, spacing, 5),
            alps: octaves(6500.0, spacing, 5),
            rolling: octaves(9000.0, spacing, 5),
            knolls: octaves(2400.0, spacing, 4),
            gullies: octaves(GULLY_PERIOD, spacing, 9),
            tables: octaves(3.2e4, spacing, 8),
            hills: octaves(420.0, spacing, 5),
            fine: octaves(95.0, spacing, 5),
            meanders: [
                octaves(MEANDERS[0].0, spacing, 2),
                octaves(MEANDERS[1].0, spacing, 2),
                octaves(MEANDERS[2].0, spacing, 2),
            ],
            spacing,
        }
    }

    /// Tout le detail : pour la physique et le placement des objets.
    pub fn full() -> Self {
        Self::for_spacing(1.0)
    }

    /// Le meme relief, sans ce qui est plus fin que `spacing` : sert a mesurer une pente.
    fn coarse(&self, spacing: f64) -> Self {
        let other = Self::for_spacing(self.spacing.max(spacing));
        Self { mountains: other.mountains, alps: other.alps, rolling: other.rolling, knolls: other.knolls, ..*self }
    }
}

/// Ce qui ne depend pas de l'erosion en un point : les grandes masses du relief.
#[derive(Clone, Copy)]
struct Shape {
    height: f32,
    /// Crete de montagne, de 0 a 1.
    crest: f32,
    /// Energie du relief, de 0 (plaine) a 1 (pays de montagne).
    energy: f32,
}

pub struct Generator {
    pub params: PlanetParams,
    /// Numero unique : cle des caches par thread.
    id: u64,
    /// Plaques a la fin de la simulation.
    pub plates: Tectonics,
    /// Relief global, rivieres et lacs.
    pub water: Hydrology,
    /// Nature du paysage, region par region.
    pub provinces: Provinces,
    /// Villes : sites, puis plans calcules a l'approche.
    pub cities: Cities,
    pub volcanoes: Volcanoes,
    pub orogeny: CubeField,
    meander: [Noise; 6],
    warp: [Noise; 3],
    mountains: Noise,
    rolling: Noise,
    hills: Noise,
    fine: Noise,
    tables: Noise,
    canyons: Noise,
    badland: Noise,
    fjords: Noise,
    gullies: Cells,
    towers: Cells,
    sand: Cells,
    ponds: Cells,
    tint: Noise,
    /// Temperatures, pluies et vents simules.
    pub weather: ClimateMap,
    /// Variations locales de la pluie.
    moisture: Noise,
    patches: Noise,
    groves: Noise,
}

/// Provinces, teinte et roche d'un point : voir `Generator::site`.
pub struct Site {
    pub w: Province,
    pub hue: f32,
    pub rock: [f32; 3],
    pub strata: f32,
}

/// A appeler au debut d'un thread de generation : il passe sous la priorite normale, pour que le
/// thread du jeu (rendu, entrees) garde toujours son coeur quand tous les autres calculent ; et il
/// calcule au rayon `radius` (celui d'une planete en cours de creation, le cas echeant).
pub fn worker_thread(radius: f64) {
    set_creating_radius(radius);
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn SetThreadPriority(thread: isize, priority: i32) -> i32;
        }
        const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
        // SAFETY : la pseudo-poignee du thread courant est toujours valide.
        unsafe {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        }
    }
}

/// `f(0..n)` reparti sur tous les coeurs, dans l'ordre. Les threads calculent au rayon de la planete
/// du thread appelant (qui peut etre une planete en cours de creation).
pub fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).min(n.max(1));
    if threads <= 1 || n < 64 {
        return (0..n).map(f).collect();
    }
    // Petits paquets pris a tour de role : un paquet plus lourd que les autres ne retient pas tout le monde.
    let batch = (n / (threads * 16)).clamp(1, 256);
    let radius = planet_radius();
    let (f, next) = (&f, &std::sync::atomic::AtomicUsize::new(0));
    let mut parts: Vec<(usize, Vec<T>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(move || {
                    worker_thread(radius);
                    let mut out = Vec::new();
                    loop {
                        let from = next.fetch_add(batch, std::sync::atomic::Ordering::Relaxed);
                        if from >= n {
                            break out;
                        }
                        out.push((from, (from..(from + batch).min(n)).map(f).collect::<Vec<T>>()));
                    }
                })
            })
            .collect();
        handles.into_iter().flat_map(crate::sync::join).collect()
    });
    parts.sort_unstable_by_key(|p| p.0);
    parts.into_iter().flat_map(|p| p.1).collect()
}

/// Passage en douceur de 0 a 1 quand `x` va de `a` a `b`. `a` et `b` doivent etre differents.
pub fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
}

/// Plaques de la planete, de 10 a 19 selon la graine.
fn plate_count(seed: u64) -> usize {
    10 + (hash_seed(seed, 0x91A7, 0) % 10) as usize
}

impl Generator {
    /// Simule la planete decrite par `params` (ramenes dans leurs bornes) : plaques, relief, eau,
    /// climat, villes et routes. Quelques secondes de calcul sur tous les coeurs ; deterministe pour
    /// une graine donnee.
    pub fn new(params: PlanetParams) -> Self {
        let params = params.sanitized();
        // Toute la simulation se fait au rayon de la nouvelle planete, meme si une autre est affichee.
        let previous = CREATING.with(|c| c.0.get());
        set_creating_radius(params.radius());
        let gen = Self::create(params);
        set_creating_radius(previous);
        gen
    }

    fn create(params: PlanetParams) -> Self {
        let layer = |i: i64| Noise::new(hash_seed(params.seed, i, 0x5EED));
        let cells = |i: i64| Cells::new(hash_seed(params.seed, i, 0xCE11));
        let moisture = layer(9);
        // TERRA_PROFILE : temps de chaque etape de la creation.
        let profile = std::env::var("TERRA_PROFILE").is_ok();
        let mut clock = std::time::Instant::now();
        let mut step = |name: &str| {
            if profile {
                println!("  creation : {name} {:.2} s", clock.elapsed().as_secs_f64());
            }
            clock = std::time::Instant::now();
        };
        let plates = tectonics::simulate(params.seed, plate_count(params.seed), (1.0 - params.ocean).clamp(0.03, 0.92));
        step("plaques");
        let orogeny = CubeField::from_cells(plates.grid, &plates.orogeny);
        // Un premier climat sur le relief brut des plaques : l'erosion a besoin des pluies. Le climat
        // definitif est simule ensuite, sur le relief erode.
        let (water, weather) = {
            let crust = CubeField::from_cells(plates.grid, &plates.crust);
            let draft = ClimateMap::build(&params, &|dir| {
                if crust.sample_linear(dir) > 0.5 {
                    250.0 + orogeny.sample_linear(dir).clamp(0.0, 1.4).powf(1.25) * 3600.0 * params.relief
                } else {
                    -3000.0
                }
            });
            step("climat provisoire");
            Hydrology::build(&params, &plates, &|dir, height| draft.at(dir, height, 1.0))
        };
        step("eau et climat");
        let provinces = Provinces::build(params.seed, &water.elevation, &orogeny, &|dir, height| {
            weather.at(dir, height, 1.0)
        });
        step("provinces");
        let volcanoes = Volcanoes::build(params.seed, &plates, &water.elevation);
        step("volcans");
        let mut gen = Self {
            cities: Cities::empty(),
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            params,
            orogeny,
            plates,
            water,
            provinces,
            volcanoes,
            meander: [layer(1), layer(11), layer(12), layer(2), layer(3), layer(7)],
            warp: [layer(19), layer(20), layer(21)],
            mountains: layer(4),
            rolling: layer(13),
            hills: layer(5),
            fine: layer(6),
            tables: layer(22),
            canyons: layer(23),
            badland: layer(24),
            fjords: layer(25),
            gullies: cells(1),
            towers: cells(2),
            sand: cells(3),
            ponds: cells(4),
            tint: layer(18),
            weather,
            moisture,
            patches: layer(10),
            groves: layer(26),
        };
        gen.cities = Cities::place(&gen);
        step("villes");
        gen
    }

    /// Graine de la planete.
    pub fn seed(&self) -> u64 {
        self.params.seed
    }

    /// Hauteur exacte au-dessus du niveau de la mer (physique, placement des objets).
    pub fn height(&self, dir: DVec3) -> f32 {
        self.height_detail(dir, &Detail::full())
    }

    /// Hauteur au-dessus du niveau de la mer, avec le detail `d` seulement.
    pub fn height_detail(&self, dir: DVec3, d: &Detail) -> f32 {
        self.surface_detail(dir, d).height
    }

    /// Le sol et l'eau au plus fin.
    pub fn surface(&self, dir: DVec3) -> Surface {
        self.surface_detail(dir, &Detail::full())
    }

    /// Le trace des rivieres et le rivage des lacs serpentent autour de la grille.
    fn meander_point(&self, dir: DVec3, d: &Detail) -> DVec3 {
        let p = dir * planet_radius();
        let mut offset = DVec3::ZERO;
        for (i, (wavelength, amplitude)) in MEANDERS.iter().enumerate() {
            if d.meanders[i] > 0 {
                let q = p / *wavelength;
                let (a, b) = (self.meander[i * 2].fbm(q, d.meanders[i]), self.meander[i * 2 + 1].fbm(q, d.meanders[i]));
                // Dans le plan local : les trois composantes viennent de deux bruits.
                offset += DVec3::new(a as f64, b as f64, (a - b) as f64) * *amplitude;
            }
        }
        (dir + offset / planet_radius()).normalize()
    }

    /// Le point du trace des rivieres qui passe en `dir` : le lieu deforme par les meandres.
    pub fn meander(&self, dir: DVec3) -> DVec3 {
        self.meander_point(dir, &Detail::full())
    }

    /// Le lieu ou passe un point du trace des rivieres, meandres compris : l'inverse de `meander`.
    pub fn unbend(&self, target: DVec3) -> DVec3 {
        let d = Detail::full();
        let mut p = target;
        for _ in 0..12 {
            let moved = self.meander_point(p, &d) - p;
            p = (target - moved).normalize();
        }
        p
    }

    /// Les grandes masses du relief en un point : simulation, chaines de montagnes, collines, volcans.
    fn shape(&self, at: DVec3, d: &Detail, w: &Province) -> Shape {
        let q = at * planet_radius();
        let base = self.water.elevation.sample(at);
        let mut h = base;
        // Le relief tord tout ce qui se pose dessus : plus de lignes droites ni de formes de bruit.
        let bent = q + self.warp_at(q) * 7000.0;
        let o = self.orogeny.sample_linear(at).clamp(0.0, 1.4);
        let m = smoothstep(0.3, 0.85, o) * smoothstep(120.0, 900.0, base);
        let land = smoothstep(0.0, 25.0, base);
        let mut crest = 0.0;
        if m > 0.0 && d.mountains > 0 {
            crest = self.mountains.ridged(bent / 2.6e4, d.mountains, (d.spacing / 2.6e4) as f32).clamp(0.0, 1.0);
            h += m * crest * 2800.0 * self.params.relief;
            // Aretes secondaires et pics : plus vives en altitude, ou la glace a sculpte la roche
            // (multifractale hybride de Musgrave : le detail suit la hauteur deja acquise).
            if d.alps > 0 {
                let spur = self
                    .mountains
                    .ridged(bent / 6500.0 + DVec3::splat(71.0), d.alps, (d.spacing / 6500.0) as f32)
                    .clamp(0.0, 1.0);
                h += m * spur * (250.0 + 950.0 * crest) * self.params.relief;
            }
        }
        // L'energie du relief suit la geologie : forte dans les chaines et sur les hautes terres soulevees,
        // presque nulle dans les bassins bas, ou l'eau a tout aplani. Comme sur Terre, pres de la moitie des
        // terres sont des plaines ; le reste se partage entre collines, plateaux et montagnes.
        let rugged = (self.warp[2].fbm(q / 7.0e4 + DVec3::splat(11.0), 2) * 1.6 + 0.6).clamp(0.0, 1.0);
        let foothills = smoothstep(0.1, 0.5, o);
        let upland = smoothstep(200.0, 1200.0, base);
        let energy = (foothills.max(upland * 0.6) * (0.3 + 0.7 * rugged) + 0.1 * rugged * rugged * rugged).min(1.0);
        let swell = 14.0
            + energy
                * (w.hills * 620.0
                    + w.glacial * 300.0
                    + w.moor * 210.0
                    + w.badlands * 90.0
                    + w.mesa * 70.0
                    + w.karst * 40.0
                    + w.inselberg * 170.0
                    + w.dunes * 30.0
                    + foothills * 380.0
                    + 260.0);
        if d.rolling > 0 {
            // Bruit amorti par sa propre pente (Quilez) : fonds de vallee doux, cretes plus nettes, comme apres l'erosion.
            let r =
                self.rolling.fbm(bent / 9000.0, d.rolling) * 0.5 + self.rolling.eroded(bent / 9000.0, d.rolling) * 0.5;
            h += (r + 0.55) * swell * land * (1.0 - 0.6 * m) * self.params.relief;
        }
        if d.knolls > 0 {
            let k = self.hills.eroded(bent / 2400.0 + DVec3::splat(29.0), d.knolls);
            h += (k + 0.3)
                * (5.0 + (20.0 + 120.0 * rugged + 60.0 * w.hills + 50.0 * foothills) * (0.1 + 0.9 * energy))
                * land
                * (1.0 - 0.5 * m)
                * (1.0 - 0.8 * w.dunes)
                * self.params.relief;
        }
        if let Some((v, share)) = self.volcanoes.at(at) {
            h += (v.summit - h).max(0.0) * share;
        }
        Shape { height: h, crest, energy }
    }

    #[inline]
    fn warp_at(&self, q: DVec3) -> DVec3 {
        let s = q / 34_000.0;
        DVec3::new(self.warp[0].value(s) as f64, self.warp[1].value(s) as f64, self.warp[2].value(s) as f64)
    }

    /// Pente des grandes masses du relief (m/m, dans le plan tangent), mesuree sur `SLOPE_EPS` au moins :
    /// c'est elle qu'epousent les ravines. Differences centrees : sur une crete ou dans un creux, les deux
    /// cotes s'annulent et les ravines s'effacent.
    fn gully_slope(&self, dir: DVec3, d: &Detail, w: &Province) -> Vec3 {
        let eps = d.spacing.max(SLOPE_EPS);
        let side = dir.any_orthonormal_vector();
        let other = dir.cross(side);
        let coarse = d.coarse(SLOPE_EPS);
        let across = |a: DVec3| {
            let at = |k: f64| self.shape((dir + a * (k * eps / planet_radius())).normalize(), &coarse, w).height;
            (at(1.0) - at(-1.0)) as f64 / (2.0 * eps)
        };
        (side * across(side) + other * across(other)).as_vec3()
    }

    /// La meme pente, interpolee entre les noeuds d'une grille 3D fixe de `SLOPE_LATTICE` metres.
    /// Le champ ne varie qu'a l'echelle du kilometre : l'ecart est minime, et chaque noeud, calcule une
    /// fois par thread, sert a des centaines de sommets au lieu de quatre formes du relief par sommet.
    fn lattice_slope(&self, dir: DVec3, level: usize) -> Vec3 {
        let size = SLOPE_LATTICES[level].0;
        let q = dir * (planet_radius() / size);
        let cell = q.floor();
        let f = (q - cell).as_vec3();
        let (ix, iy, iz) = (cell.x as i64, cell.y as i64, cell.z as i64);
        let mut sum = Vec3::ZERO;
        SLOPE_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            for k in 0..8i64 {
                let (a, b, c) = (k & 1, (k >> 1) & 1, (k >> 2) & 1);
                let key = (self.id * 4 + level as u64, ix + a, iy + b, iz + c);
                let h = (key.1 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (key.2 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                    ^ (key.3 as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
                    ^ key.0.wrapping_mul(0x94D0_49BB_1331_11EB);
                let slot = ((h ^ (h >> 31)) as usize) & (SLOPE_CACHE_SIZE - 1);
                let value = match cache[slot] {
                    (k2, v) if k2 == key => v,
                    _ => {
                        let shard = &shared_slopes()[(h >> 58) as usize % SHARED_SHARDS];
                        let known = crate::sync::lock(shard).get(&key).copied();
                        let v = known.unwrap_or_else(|| {
                            let at = (DVec3::new(key.1 as f64, key.2 as f64, key.3 as f64) * size).normalize();
                            let v = self.gully_slope(at, &Detail::full(), &self.provinces.at(at));
                            let mut map = crate::sync::lock(shard);
                            if map.len() >= SHARED_SHARD_SIZE {
                                map.clear();
                            }
                            map.insert(key, v);
                            v
                        });
                        cache[slot] = (key, v);
                        v
                    }
                };
                let weight = (if a == 1 { f.x } else { 1.0 - f.x })
                    * (if b == 1 { f.y } else { 1.0 - f.y })
                    * (if c == 1 { f.z } else { 1.0 - f.z });
                sum += value * weight;
            }
        });
        // Les noeuds ont chacun leur plan tangent : on revient dans celui du point.
        let up = dir.as_vec3();
        sum - up * sum.dot(up)
    }

    /// Altitude d'un etang, qui ne depend que de son centre : la meme vue de tous les points de l'etang.
    fn pond_level(&self, center: DVec3, w: &Province) -> f32 {
        self.shape(center, &Detail::for_spacing(800.0), w).height
    }

    /// Le sol, l'eau, et les autoroutes qui le remodelent.
    pub fn surface_detail(&self, dir: DVec3, d: &Detail) -> Surface {
        let mut s = self.surface_natural(dir, d);
        self.cities.highways.shape_ground(self, dir, &mut s);
        self.cities.circuits.shape_ground(self, dir, &mut s);
        self.cities.mountains.shape_ground(self, dir, &mut s);
        s
    }

    /// Le sol et l'eau avant les autoroutes.
    pub fn surface_natural(&self, dir: DVec3, d: &Detail) -> Surface {
        let p = dir * planet_radius();
        let base = self.water.elevation.sample(dir);
        let mut out = Surface {
            height: base,
            water_level: SEA_LEVEL,
            fresh: false,
            flow: Vec3::ZERO,
            paint: 0.0,
            bank: 0.0,
            valley: 0.0,
            ridge: 0.0,
        };
        // Grand large : aucune couche suivante ne se verrait a travers l'eau. Sauf les iles volcaniques.
        let volcano = self.volcanoes.at(dir).map_or(0.0, |v| v.1);
        if base < -600.0 && volcano <= 0.0 {
            out.height = self.shoreface(dir, base);
            return out;
        }

        let w = self.provinces.at(dir);
        let land = smoothstep(0.0, 45.0, base);
        let shore = smoothstep(-220.0, -4.0, base);
        let o = self.orogeny.sample_linear(dir).clamp(0.0, 1.4);
        let m = smoothstep(0.3, 0.85, o) * smoothstep(120.0, 900.0, base);
        let plain = land * (1.0 - m);
        let relief = self.params.relief;
        let up = dir.as_vec3();

        let shape = self.shape(dir, d, &w);
        let mut h = shape.height;
        let q = p + self.warp_at(p) * 7000.0;
        let rises = land.max((volcano * 5.0).min(1.0));

        // Ravines : le relief se creuse selon sa pente, de la grande vallee a la rigole.
        let foothills = smoothstep(0.1, 0.5, o);
        let carve = m
            + (1.0 - m)
                * (w.hills * 0.72
                    + w.glacial * 0.9
                    + w.moor * 0.42
                    + w.badlands * 0.3
                    + w.mesa * 0.14
                    + w.inselberg * 0.4
                    + w.karst * 0.2
                    + w.dunes * 0.02)
                    .max(foothills * 0.45);
        // Les plaines ne sont pas ravinees : l'eau y coule sans creuser.
        let carve = carve * (0.06 + 0.94 * shape.energy.max(m));
        let carve = (carve + volcano.min(0.4) * 1.6).min(1.0) * rises * relief.sqrt();
        if d.gullies > 0 && carve > 1e-4 {
            // De pres, la pente vient d'une grille fixe mise en cache : le meme champ pour tous les chunks.
            let slope = match SLOPE_LATTICES.iter().position(|&(_, below)| d.spacing < below) {
                Some(level) => self.lattice_slope(dir, level),
                None => self.gully_slope(dir, d, &w),
            };
            let fade = (shape.crest * 2.0 - 1.0) * m * 0.5;
            let eroded = self.gullies.erosion(p, up, slope, GULLY_PERIOD, d.gullies, fade, carve);
            h += eroded.offset;
            out.ridge = eroded.ridge;
        }

        // Plateaux en gradins, entailles de canyons, avec buttes et falaises.
        let mesa = w.mesa * plain;
        if mesa > 0.01 && d.tables > 0 {
            let mut f = (self.tables.fbm(q / 3.2e4, d.tables) + 0.4) * 1500.0;
            let lines = self.canyons.fbm(q / 1.9e4, d.tables.min(6)).abs();
            f -= (1.0 - smoothstep(0.004, 0.05, lines)) * 1000.0;
            f += self.towers.towers(q / 3100.0, 0.45, 0.55, 0.22, 160.0, 420.0);
            let f = f.max(0.0) * relief;
            // Une falaise plus etroite que le maillage y ferait des dents de scie : elle s'elargit avec lui.
            let wide = d.spacing as f32 * 0.5;
            let stepped =
                terrace(f, 240.0, (wide / 240.0).max(0.16)) * 0.74 + terrace(f, 62.0, (wide / 62.0).max(0.28)) * 0.26;
            h += stepped * smoothstep(0.08, 0.5, mesa);
        }

        // Terres ravinees : de petites buttes striees, toutes en couches.
        let bad = w.badlands * plain;
        if bad > 0.01 && d.tables > 1 {
            let (value, grad) = self.badland.d3(q / 2400.0);
            let slope = (grad - up * grad.dot(up)) * (300.0 / 2400.0);
            let eroded =
                self.gullies.erosion(p + DVec3::splat(991.0), up, slope, 1100.0, d.gullies.saturating_sub(2), 0.2, 0.8);
            let f = ((value + 0.5) * 300.0 + eroded.offset).max(0.0);
            h += (terrace(f, 34.0, (d.spacing as f32 / 68.0).max(0.3)) * 0.7 + f * 0.3)
                * smoothstep(0.1, 0.5, bad)
                * relief;
            out.ridge = out.ridge * (1.0 - bad) + eroded.ridge * bad;
        }

        // Mer de dunes : des cretes vives, dont le vent regle l'orientation.
        let dunes = w.dunes * plain;
        if dunes > 0.01 {
            let east = DVec3::Y.cross(dir).normalize_or(DVec3::X);
            let north = dir.cross(east);
            let turn = self.warp[0].fbm(p / 42_000.0, 2) as f64 * 1.3;
            let wave = (east * turn.cos() + north * turn.sin()).as_vec3();
            let big = self.sand.dunes(q / 1500.0, wave);
            let mid = if d.spacing < 60.0 { self.sand.dunes(q / 260.0 + DVec3::splat(31.0), wave) } else { 0.5 };
            let swell = (self.warp[1].fbm(p / 9000.0, 2) * 1.5 + 0.65).clamp(0.15, 1.0);
            h += (big * 170.0 * swell + mid * 16.0) * smoothstep(0.1, 0.5, dunes);
        }

        // Tours calcaires, jusque dans les eaux peu profondes.
        let karst = w.karst * shore * (1.0 - m);
        if karst > 0.01 {
            let bump = if d.spacing < 200.0 { self.warp_at(p * 22.0) * 130.0 } else { DVec3::ZERO };
            let tower = self.towers.towers((p + bump) / 1250.0, 0.8, 0.66, 0.5, 120.0, 380.0)
                + self.towers.towers((p + bump) / 470.0 + DVec3::splat(7.0), 0.5, 0.6, 0.5, 30.0, 90.0);
            h += tower * smoothstep(0.2, 0.6, karst);
        }

        // Monolithes dans la plaine, chaos de granite dans les landes.
        let plainsman = w.inselberg * plain;
        if plainsman > 0.01 {
            // De grands monolithes, et une foule de buttes de granite a leur pied.
            // Une paroi de roche nue, puis un dome : les memes mailles, donc les memes rochers, pour les deux profils.
            // Les mailles sont en 3D et la surface n'en coupe qu'une tranche : il faut des rochers larges pour qu'ils affleurent.
            let (cell, small_cell) = (q / 8000.0, q / 2600.0 + DVec3::splat(19.0));
            let great = self.towers.towers(cell, 0.75, 0.44, 0.12, 200.0, 330.0)
                + self.towers.towers(cell, 0.75, 0.44, 0.95, 90.0, 190.0);
            let small = self.towers.towers(small_cell, 0.45, 0.4, 0.15, 20.0, 70.0)
                + self.towers.towers(small_cell, 0.45, 0.4, 0.95, 10.0, 50.0);
            h += (great + small) * smoothstep(0.04, 0.3, plainsman);
        }
        let open_land = plain * (1.0 - w.dunes) * (1.0 - w.karst) * (1.0 - w.inselberg);
        if open_land > 0.01 {
            let knoll = self.towers.towers(q / 7000.0 + DVec3::splat(43.0), 0.16, 0.45, 0.95, 50.0, 190.0);
            h += knoll * open_land;
        }
        let moor = (w.moor + w.hills * 0.15 + w.inselberg * 0.3) * plain;
        if moor > 0.01 && d.spacing < 120.0 {
            h += self.towers.towers((p + DVec3::splat(77.0)) / 340.0, 0.34, 0.6, 0.5, 7.0, 30.0)
                * smoothstep(0.05, 0.4, moor);
        }

        // Aiguilles marines le long des cotes rocheuses.
        if (-90.0..12.0).contains(&base) {
            let coast = 1.0 - smoothstep(-2.0, 12.0, base);
            h += self.towers.towers((p + DVec3::splat(13.0)) / 620.0, 0.3, 0.42, 0.4, 28.0, 80.0)
                * smoothstep(-90.0, -30.0, base)
                * coast
                * smoothstep(0.45, 0.75, self.provinces.palette.sample_linear(dir));
        }

        // Le detail de pres : mottes et rochers, moins presents sur les sols de sable.
        let plain_detail = 1.0 - w.dunes * 0.9;
        if d.hills > 0 {
            h += self.hills.eroded(p / 420.0, d.hills)
                * (15.0 + 15.0 * w.hills)
                * (0.25 + 0.75 * land)
                * shore
                * plain_detail;
        }

        // Vallees glaciaires : un fond plat, des flancs raides. Sous le niveau de la mer pres des cotes : fjords.
        if w.glacial > 0.02 && base > -220.0 && d.tables > 0 {
            let lines = self.fjords.fbm(q / 1.6e4, d.tables.min(3)).abs();
            let trough = 1.0 - smoothstep(0.035, 0.13, lines);
            if trough > 0.0 {
                let floor = -150.0 + base.max(0.0) * 0.42;
                let carved = h.min(floor);
                h += (carved - h) * trough * smoothstep(0.1, 0.5, w.glacial);
            }
        }

        // Une ville aplanit son sol : il ne reste que les grandes ondulations, et l'eau qui la traverse.
        let urban = self.cities.flatten(dir);
        if urban > 0.0 {
            let ground = self.shape(dir, &Detail::for_spacing(1500.0), &w).height;
            // La terre ferme le reste : une ville gagne parfois sur la mer, jamais l'inverse.
            let target = ground.max(h.min(1.5));
            h = lerp(h, target, urban);
        }

        let bent = self.meander_point(dir, d);
        let mut smooth = urban;

        // Etangs : le relief s'y creuse autour d'un niveau fixe, ni le maillage ni la position n'y changent rien.
        // Un etang plus petit que la maille n'est pas dessine : un seul sommet tombe dedans le changerait
        // en lac large comme la maille, visible depuis l'orbite.
        if d.spacing < 320.0 {
            let cells = p / POND_CELL;
            if let Some((center, t, roll)) = self.ponds.pivot(cells, POND_RADIUS) {
                let at = center.normalize();
                let w_at = self.provinces.at(at);
                let share = (w_at.moor * 0.95 + w_at.glacial * 0.6 + w_at.hills * 0.14 + w_at.karst * 0.2)
                    * (1.0 - smoothstep(0.1, 0.4, self.orogeny.sample_linear(at)) * 0.5);
                if roll < share && self.cities.flatten(at) <= 0.0 {
                    let level = self.pond_level(at, &w_at);
                    if level > 3.0 {
                        let target =
                            if t < 0.3 { level + (0.3 - t) * 34.0 } else { level - smoothstep(0.3, 0.55, t) * 14.0 };
                        let blend = smoothstep(0.0, 0.26, t);
                        h = lerp(h, target, blend);
                        smooth = blend;
                        if t > 0.2 {
                            out.water_level = level;
                            out.fresh = true;
                            out.bank = 1.0 - smoothstep(0.0, 0.12, (t - 0.3).abs());
                        }
                    }
                }
            }
        }

        // Lacs : le relief prend la forme d'une cuvette autour du niveau calcule par la simulation.
        let (mask, lake_level) = self.water.lake_at(bent);
        if mask > 0.0 && (d.spacing as f32) < self.water.lake_span(bent).max(13_000.0) * 0.7 {
            let floor = (lake_level - h).clamp(18.0, 130.0);
            let target = if mask < SHORE {
                lake_level + (SHORE - mask) / SHORE * 30.0
            } else {
                // La rive descend vite : un lac a de la profondeur, meme petit.
                lake_level - smoothstep(0.0, 0.12, mask - SHORE) * floor
            };
            let w = smoothstep(0.015, 0.12, mask);
            h = lerp(h, target, w);
            smooth = smooth.max(w);
            if mask > 0.12 {
                out.water_level = lake_level;
                out.fresh = true;
            }
            out.bank = out.bank.max(1.0 - smoothstep(0.0, 0.14, (mask - SHORE).abs()));
        }

        // Rivieres : un cone de pente autour du lit taille les versants, quelle que soit leur hauteur.
        // Pres de la mer, le lit s'evase en estuaire et son niveau rejoint celui de la mer au rivage : l'eau
        // de mer y remonte. Au large, la riviere s'efface, sans digues.
        let estuary = 1.0 - smoothstep(0.0, ESTUARY_RISE, base);
        let offshore = smoothstep(-1.0, -10.0, base);
        // Dans un estuaire, des berges basses de vase plutot que l'avant-cote abrupte de la mer.
        let mut mudflat = 0.0;
        if let Some(mut river) = self.water.river_at(
            bent,
            d.spacing,
            1.0 + (MAX_WIDEN - 1.0) * smoothstep(0.06, 0.5, o).max(mesa),
            estuary,
        ) {
            if estuary > 0.0 {
                // Le courant faiblit vers la mer : la teinte des rivieres passe a celle de la mer sans couture.
                let inland = smoothstep(0.0, ESTUARY_RISE * 0.75, base);
                river.level = SEA_LEVEL + (river.level - SEA_LEVEL) * inland;
                river.flow *= inland * inland;
                river.lift *= smoothstep(-2.0, 1.0, base);
                river.reach *= 1.0 - offshore;
                river.wet &= river.level > SEA_LEVEL + 0.25;
                mudflat = river.valley * estuary;
            }
            out.paint = river.paint;
            out.bank = out.bank.max(river.bank);
            out.valley = river.valley;
            if river.reach > 0.0 {
                let target = river.level + river.profile;
                // Minimum adouci : le fond de la vallee se raccorde sans arete au reste du relief.
                let k = 25.0;
                let blend = (0.5 + 0.5 * (target - h) / k).clamp(0.0, 1.0);
                let carved = lerp(target, h, blend) - k * blend * (1.0 - blend);
                h = lerp(h, carved, river.reach);
                // Jamais d'eau suspendue : pres du lit, le sol ne descend pas sous la riviere.
                let held = if river.profile > 0.0 { h.max(target) } else { target };
                h = lerp(h, held, river.lift);
                smooth = smooth.max(river.valley);
            }
            if river.wet && !out.fresh {
                out.water_level = river.level.max(SEA_LEVEL);
                out.fresh = true;
                out.flow = river.flow;
            }
        }

        if d.fine > 0 {
            h += self.fine.fbm(p / 95.0, d.fine) * 4.0 * shore * plain_detail * (1.0 - smooth * 0.92);
        }
        out.height = if out.fresh { h } else { lerp(self.shoreface(dir, h), h, mudflat) };
        out
    }

    /// Avant-cote : sous la mer, le fond plonge vite pres du rivage. La grille globale, a mailles de plusieurs
    /// kilometres, ne donne qu'une pente de 1 a 5 m par kilometre a la cote : on aurait de l'eau a hauteur de
    /// cheville sur des kilometres. Sur Terre, le fond atteint 10 a 20 m a quelques centaines de metres d'une
    /// plage, davantage le long des cotes rocheuses, avant la pente douce du plateau continental. Le profil
    /// ne depend que de la hauteur : le trait de cote ne bouge pas.
    fn shoreface(&self, dir: DVec3, h: f32) -> f32 {
        if h >= SEA_LEVEL {
            return h;
        }
        // Seulement en mer : les cuvettes d'une plaine cotiere, sous le niveau de la mer par endroits, restent
        // des lagunes peu profondes.
        let open_sea = smoothstep(1.5, -1.5, self.water.elevation.sample(dir));
        let steep = SHOREFACE_DEPTH.0 + (SHOREFACE_DEPTH.1 - SHOREFACE_DEPTH.0) * self.provinces.palette.sample_linear(dir);
        h - steep * open_sea * (1.0 - ((h - SEA_LEVEL) / SHOREFACE_SCALE).exp())
    }

    /// Temperature et humidite (0 a 1) a l'altitude `height` au-dessus du lieu `dir`.
    pub fn climate(&self, dir: DVec3, height: f32) -> Climate {
        // La pluie varie d'un pays a l'autre : les limites des biomes ne suivent pas la grille du climat.
        let at = dir * planet_radius();
        let local = 1.0 + 0.2 * self.moisture.fbm(at / 2.6e5, 3) + 0.08 * self.moisture.fbm(at / 3.1e4, 2);
        self.weather.at(dir, height, local)
    }

    /// Le paysage vegetal de ce climat. La foret demande un climat humide (indice d'aridite au-dessus de
    /// 0,65 environ) ; en dessous viennent la prairie, la steppe et la savane.
    pub fn biome(&self, c: Climate) -> Biome {
        let wet = c.aridity();
        match c.koppen() {
            Koppen::Ice => Biome::Ice,
            Koppen::Tundra => Biome::Tundra,
            Koppen::Desert => Biome::Desert,
            Koppen::Steppe => {
                if c.mean > 18.0 {
                    Biome::Savanna
                } else {
                    Biome::Grassland
                }
            }
            Koppen::Rainforest => Biome::Jungle,
            Koppen::Savanna => {
                if c.rain > 1600.0 && c.driest > 15.0 {
                    Biome::Jungle
                } else {
                    Biome::Savanna
                }
            }
            Koppen::Mediterranean => Biome::Shrubland,
            Koppen::Temperate | Koppen::Continental => {
                if wet > 0.62 {
                    Biome::Forest
                } else {
                    Biome::Grassland
                }
            }
            Koppen::Subarctic => Biome::Taiga,
        }
    }

    /// Nom du paysage dominant en un point.
    pub fn landform_name(&self, dir: DVec3) -> &'static str {
        let w = self.provinces.at(dir);
        let o = self.orogeny.sample_linear(dir);
        if self.volcanoes.at(dir).is_some_and(|v| v.1 > 0.08) {
            return "volcan";
        }
        if o > 0.58 {
            return "montagne";
        }
        let all = [
            (w.hills, "collines"),
            (w.mesa, "plateaux"),
            (w.badlands, "terres ravinees"),
            (w.dunes, "dunes"),
            (w.karst, "tours calcaires"),
            (w.moor, "landes et lacs"),
            (w.inselberg, "monolithes"),
            (w.glacial, "vallees glaciaires"),
        ];
        all.iter().fold((0.0, "?"), |a, b| if b.0 > a.0 { *b } else { a }).1
    }

    /// Bruit basse frequence pour regrouper la vegetation en massifs, dans [0, 1].
    pub fn patch(&self, dir: DVec3) -> f32 {
        let at = dir * planet_radius();
        // Des clairieres et des bosquets sur des massifs : jamais un tapis uniforme.
        let wide = self.groves.fbm(at / 5200.0, 2) * 1.5;
        (self.patches.fbm(at / 420.0, 3) * 1.3 + wide + 0.5).clamp(0.0, 1.0)
    }

    /// La roche du lieu, en lineaire, et la force de ses strates, de 0 a 1.
    pub fn rock_color(&self, dir: DVec3) -> ([f32; 3], f32) {
        let s = self.site(dir);
        (s.rock, s.strata)
    }

    /// Ce que les couleurs d'un point partagent : ses provinces, leur teinte et sa roche. A calculer une
    /// fois par sommet, puis a passer aux fonctions `*_at`.
    pub fn site(&self, dir: DVec3) -> Site {
        let w = self.provinces.at(dir);
        let hue = self.provinces.palette.sample_linear(dir);
        let (rock, strata) = self.rock_of(dir, &w, hue);
        Site { w, hue, rock, strata }
    }

    fn rock_of(&self, dir: DVec3, w: &Province, hue: f32) -> ([f32; 3], f32) {
        let mut rock = mix3([0.19, 0.19, 0.2], [0.27, 0.235, 0.2], hue);
        let mut strata = 0.1;
        let mut add = |share: f32, color: [f32; 3], layers: f32| {
            rock = mix3(rock, color, share);
            strata = lerp(strata, layers, share);
        };
        add(w.mesa, mix3([0.5, 0.2, 0.095], [0.6, 0.37, 0.17], hue), 1.0);
        add(w.badlands, mix3([0.48, 0.25, 0.2], [0.55, 0.42, 0.28], hue), 1.0);
        add(w.karst, [0.53, 0.5, 0.42], 0.4);
        add(w.moor, [0.2, 0.2, 0.19], 0.15);
        add(w.inselberg, [0.52, 0.24, 0.11], 0.3);
        add(w.dunes, [0.5, 0.3, 0.16], 0.0);
        if let Some((_, share)) = self.volcanoes.at(dir) {
            let dark = smoothstep(0.03, 0.2, share);
            rock = mix3(rock, [0.055, 0.05, 0.05], dark);
            strata *= 1.0 - dark * 0.8;
        }
        // En haute montagne, les couches sont plissees et brouillees.
        (rock, strata * (1.0 - 0.85 * smoothstep(0.3, 0.85, self.orogeny.sample_linear(dir))))
    }

    /// Couleur lineaire de la surface. `normal_y` vaut 1 sur le plat.
    /// `above` : hauteur au-dessus de l'eau la plus proche.
    pub fn surface_color(&self, dir: DVec3, above: f32, normal_y: f32, c: Climate, ridge: f32) -> [f32; 3] {
        self.surface_color_at(dir, &self.site(dir), above, normal_y, c, ridge)
    }

    /// Comme [`Generator::surface_color`], avec le `site` deja calcule par l'appelant.
    pub fn surface_color_at(
        &self,
        dir: DVec3,
        site: &Site,
        above: f32,
        normal_y: f32,
        c: Climate,
        ridge: f32,
    ) -> [f32; 3] {
        let (w, hue, rock) = (&site.w, site.hue, site.rock);
        // Le sol suit le climat. Sans eau, il est nu : sable clair ou rouge des deserts chauds, cailloux
        // gris des deserts froids. Puis viennent les herbes seches des steppes, et enfin l'herbe verte,
        // de plus en plus sombre et dense vers les tropiques humides.
        let wet = c.aridity();
        let sand = mix3([0.5, 0.39, 0.24], [0.46, 0.26, 0.13], hue);
        let bare = mix3([0.33, 0.29, 0.23], sand, smoothstep(4.0, 16.0, c.mean));
        let straw = mix3([0.3, 0.26, 0.12], [0.36, 0.27, 0.1], hue);
        let tundra = mix3([0.19, 0.16, 0.095], [0.15, 0.155, 0.075], hue);
        let mut green = mix3(tundra, [0.1, 0.15, 0.045], smoothstep(7.0, 12.0, c.hot));
        green = mix3(green, [0.078, 0.14, 0.036], smoothstep(12.0, 18.0, c.mean));
        green = mix3(green, [0.042, 0.105, 0.028], smoothstep(19.0, 24.0, c.mean));
        let mut col = mix3(bare, straw, smoothstep(0.05, 0.25, wet));
        let grown = smoothstep(0.3, 0.7, wet);
        col = mix3(col, green, grown);
        // Une longue saison seche jaunit l'herbe : savanes, pays mediterraneens.
        let parched = (1.0 - smoothstep(8.0, 45.0, c.driest)) * smoothstep(10.0, 18.0, c.mean) * grown;
        col = mix3(col, straw, parched * 0.45);

        // Chaque region a sa teinte : d'un vert emeraude a des herbes d'or.
        let shift = [lerp(0.84, 1.16, hue), lerp(1.05, 1.0, hue), lerp(0.95, 0.82, hue)];
        for i in 0..3 {
            col[i] *= shift[i];
        }

        // Les paysages qui ont leur propre sol.
        let sand = mix3([0.58, 0.43, 0.24], [0.56, 0.29, 0.14], hue);
        col = mix3(col, sand, w.dunes);
        let soil = mix3([0.38, 0.2, 0.1], [0.42, 0.28, 0.15], hue);
        let dry = 1.0 - smoothstep(0.28, 0.55, c.moisture);
        col = mix3(col, soil, w.mesa * (0.3 + 0.6 * dry));
        col = mix3(col, rock, w.badlands * 0.85);
        col = mix3(col, [0.4, 0.17, 0.07], w.inselberg * 0.5 * dry.max(0.4));
        // Bruyere et mousses d'or, selon les landes.
        let heath = mix3([0.15, 0.075, 0.13], [0.24, 0.2, 0.06], hue);
        col = mix3(col, heath, w.moor * 0.75);

        // Le sable des plages et des berges, et le fond, dominent pres de l'eau.
        const SAND: [f32; 3] = [0.5, 0.41, 0.26];
        const SEABED: [f32; 3] = [0.20, 0.20, 0.13];
        col = mix3(col, SAND, 1.0 - smoothstep(0.5, 1.8, above));
        col = mix3(col, SEABED, 1.0 - smoothstep(-9.0, -1.0, above));

        // Volcans : cendres noires pres du sommet.
        if let Some((_, share)) = self.volcanoes.at(dir) {
            col = mix3(col, [0.1, 0.085, 0.08], smoothstep(0.3, 0.62, share));
        }

        // Prairies d'altitude, puis roche : la vegetation monte dans les creux et s'arrete sur les cretes.
        // L'etage se lit a la temperature du mois le plus chaud : les arbres s'arretent vers 10 degres,
        // les pelouses vers 5, et la neige ne fond plus en dessous de 1 a 3 degres selon ce qu'il en tombe.
        let cold = 1.0 - smoothstep(2.5, 7.0, c.hot - ridge.min(0.0) * 2.0);
        let meadow = smoothstep(4.0, 7.0, c.hot) * (1.0 - smoothstep(9.0, 12.0, c.hot));
        col = mix3(col, [0.13, 0.17, 0.05], meadow * 0.6 * smoothstep(0.2, 0.5, wet));
        let bare = (1.0 - smoothstep(0.6, 0.76, normal_y - ridge.max(0.0) * 0.1)).max(cold * 0.85);
        col = mix3(col, rock, bare);

        // Neige la ou il fait froid : elle quitte les parois et les cretes soufflees.
        let line = 1.0 + (c.rain / 1200.0).min(2.0);
        let snow = smoothstep(line + 1.2, line - 0.8, c.hot + ridge.min(0.0) * 1.2) * smoothstep(0.55, 0.78, normal_y);
        mix3(col, [0.84, 0.88, 0.94], snow)
    }

    /// Part des emplacements occupes par un arbre, selon le biome, le massif (`woods`, de 0 a 1) et le froid.
    pub fn tree_cover(&self, c: &Climate, woods: f32) -> f32 {
        // Une vraie foret laisse passer le jour : couvert de 50 a 80 % au coeur des massifs,
        // bois clairs et bosquets sur leurs lisieres (FAO : une foret commence a 10 % de couvert).
        // Le couvert suit la pluie : les savanes se ferment en foret au-dela de 1 500 mm (Hirota et al. 2011).
        let wet = c.aridity();
        let cover = match self.biome(*c) {
            Biome::Jungle => 0.2 + 0.42 * woods,
            Biome::Forest => (0.04 + 0.42 * woods * woods.sqrt()) * smoothstep(0.5, 0.9, wet).max(0.35),
            Biome::Taiga => 0.04 + 0.4 * woods * woods.sqrt(),
            Biome::Grassland => (0.003 + 0.2 * woods * woods) * smoothstep(0.25, 0.62, wet),
            Biome::Savanna => 0.006 + (0.02 + 0.2 * smoothstep(500.0, 1500.0, c.rain)) * woods,
            Biome::Shrubland => 0.02 + 0.14 * woods,
            Biome::Desert | Biome::Tundra | Biome::Ice => 0.0,
        };
        // Limite des arbres : la ou le mois le plus chaud n'atteint pas 10 degres (Korner).
        cover * smoothstep(8.5, 11.5, c.hot)
    }

    /// Massif forestier en un point : le meme bruit que le placement des arbres.
    pub fn woods(&self, dir: DVec3) -> f32 {
        ((self.patch(dir) - 0.3) / 0.3).clamp(0.0, 1.0)
    }

    /// Part de terre boisee a l'echelle du pays, de 0 (grande clairiere) a 1 : assez large pour se voir de loin,
    /// et dessinee aussi sur les maillages lointains, ou le relief reste lisible entre les massifs.
    pub fn glade(&self, dir: DVec3) -> f32 {
        smoothstep(0.02, 0.16, self.groves.fbm(dir * planet_radius() / 1.7e4 + DVec3::splat(31.0), 3))
    }

    /// Part des arbres qui restent : les clairieres, et les cretes degarnies qui dessinent le relief sous la foret.
    pub fn forest_share(&self, dir: DVec3, ridge: f32) -> f32 {
        // Trouees de chablis et petites clairieres, de quelques dizaines de metres.
        let gaps = smoothstep(-0.32, -0.08, self.patches.fbm(dir * planet_radius() / 140.0 + DVec3::splat(5.0), 2));
        self.glade(dir) * (1.0 - 0.8 * smoothstep(0.12, 0.55, ridge)) * (0.25 + 0.75 * gaps)
    }

    /// Eclat de la canopee, de 0.55 a 1.35 : des taches claires et sombres, comme les ages d'une foret,
    /// assez larges pour se lire d'avion.
    pub fn canopy_tone(&self, dir: DVec3) -> f32 {
        let at = dir * planet_radius() + DVec3::splat(23.0);
        let n = self.patches.fbm(at / 2500.0, 3) * 1.4 + self.patches.fbm(at / 600.0, 2) * 0.5;
        0.95 + 0.4 * n.clamp(-1.0, 1.0)
    }

    /// Essence dominante du massif, de 0 (feuillus) a 1 (coniferes).
    pub fn stand(&self, dir: DVec3) -> f32 {
        smoothstep(-0.16, 0.16, self.groves.fbm(dir * planet_radius() / 8000.0 + DVec3::splat(57.0), 2))
    }

    /// Prairie en fleurs, de 0 a 1 : par grandes taches, dans les pays ouverts.
    pub fn meadow(&self, dir: DVec3) -> f32 {
        smoothstep(0.12, 0.34, self.groves.fbm(dir * planet_radius() / 2.4e4, 2))
    }

    /// Fonce le sol sous les forets. Vu de loin, ou les arbres ne sont plus dessines,
    /// le sol prend la couleur de la canopee : la foret reste visible jusqu'a l'horizon.
    /// Rend aussi la part de foret en ce point, dont le shader tire la texture des cimes.
    pub fn with_canopy(
        &self,
        col: [f32; 3],
        dir: DVec3,
        above: f32,
        normal_y: f32,
        c: Climate,
        ridge: f32,
        spacing: f64,
    ) -> ([f32; 3], f32) {
        self.with_canopy_at(col, dir, &self.provinces.at(dir), above, normal_y, c, ridge, spacing)
    }

    /// Comme [`Generator::with_canopy`], avec la province `w` deja calculee par l'appelant.
    #[allow(clippy::too_many_arguments)]
    pub fn with_canopy_at(
        &self,
        col: [f32; 3],
        dir: DVec3,
        w: &Province,
        above: f32,
        normal_y: f32,
        c: Climate,
        ridge: f32,
        spacing: f64,
    ) -> ([f32; 3], f32) {
        // Les massifs sont trop petits pour une grille large : on prend leur moyenne.
        let woods = if spacing < 100.0 { self.woods(dir) } else { 0.5 };
        let open = 1.0 - w.dunes.max(w.badlands) * 0.95;
        let cover = self.tree_cover(&c, woods)
            * self.forest_share(dir, ridge)
            * open
            * smoothstep(0.72, 0.82, normal_y)
            * smoothstep(1.5, 5.0, above);
        let far = smoothstep(4.0, 60.0, spacing as f32);
        // De tres loin, on ne voit plus le sol entre les arbres : une foret est une nappe sombre.
        let closed = (cover * 2.4).min(0.94);
        let strength = lerp(cover * lerp(0.3, 0.92, far), closed, smoothstep(80.0, 2500.0, spacing as f32));
        let green = mix3([0.042, 0.078, 0.024], [0.016, 0.042, 0.022], self.stand(dir));
        // L'automne et l'hiver des feuillus viennent du shader, selon le moment de l'annee.
        let canopy = green.map(|v| v * self.canopy_tone(dir));
        (mix3(col, canopy, strength), strength)
    }

    /// Couleur finale du sol : ajoute a `surface_color` les nuances d'herbe seche ou fraiche
    /// et l'ombre des creux. `concavity` va de -1 (crete) a 1 (fond de vallee).
    pub fn ground_color(
        &self,
        dir: DVec3,
        above: f32,
        normal_y: f32,
        c: Climate,
        concavity: f32,
        ridge: f32,
    ) -> [f32; 3] {
        self.ground_color_at(dir, &self.site(dir), above, normal_y, c, concavity, ridge)
    }

    /// Comme [`Generator::ground_color`], avec le `site` deja calcule par l'appelant.
    #[allow(clippy::too_many_arguments)]
    pub fn ground_color_at(
        &self,
        dir: DVec3,
        site: &Site,
        above: f32,
        normal_y: f32,
        c: Climate,
        concavity: f32,
        ridge: f32,
    ) -> [f32; 3] {
        let mut col = self.surface_color_at(dir, site, above, normal_y, c, ridge);
        let plants = smoothstep(4.0, 11.0, c.hot)
            * smoothstep(0.2, 0.5, c.aridity())
            * smoothstep(2.0, 6.0, above)
            * smoothstep(0.7, 0.86, normal_y);
        if plants > 0.0 {
            let wide = self.tint.fbm(dir * planet_radius() / 1.5e4 + DVec3::splat(11.0), 3) * 1.6;
            let dry = (self.tint.fbm(dir * planet_radius() / 2400.0, 4) * 1.5 + wide + 0.5).clamp(0.0, 1.0);
            // Les fonds de vallee et les ravines restent verts, les hauteurs sechent.
            let dry = (dry - concavity * 0.35 + ridge * 0.42).clamp(0.0, 1.0);
            let shift = [lerp(0.72, 1.4, dry), lerp(1.04, 1.0, dry), lerp(0.82, 0.72, dry)];
            for i in 0..3 {
                col[i] *= lerp(1.0, shift[i], plants);
            }
        }
        let shade = 1.0 - 0.42 * concavity.max(0.0) + 0.16 * (-concavity).max(0.0);
        col.map(|v| v * shade)
    }

    /// Part de banquise a la surface de l'ocean, dans [0, 1].
    pub fn sea_ice(&self, c: Climate) -> f32 {
        smoothstep(-1.0, -3.5, c.mean)
    }

    /// Un point de depart au bord d'une riviere, dans un pays tempere : la premiere image doit
    /// montrer ce qui fait cette planete. Retombe sur `find_inland_spawn` si rien ne convient.
    pub fn find_spawn(&self) -> DVec3 {
        let edges = &self.water.edges;
        let step = (edges.len() / 500).max(1);
        for edge in edges.iter().step_by(step) {
            if !(12.0..150.0).contains(&edge.width[0]) || !(50.0..1100.0).contains(&edge.level[0]) || edge.slope > 0.02
            {
                continue;
            }
            let middle = self.water.edge_points(edge)[1];
            let climate = self.climate(middle, edge.level[0]);
            // Un pays tempere et vert : entre 6 et 20 degres de moyenne, sans grand froid, humide.
            if !(6.0..20.0).contains(&climate.mean) || climate.cold < -8.0 || climate.aridity() < 0.65 {
                continue;
            }
            if let Some(dir) = self.bank_near(middle) {
                return dir;
            }
        }
        self.find_inland_spawn()
    }

    /// Un point de vue sur une rive, face a une grande etendue d'eau et au soleil que donne
    /// `sun_at`, si possible avec des montagnes de l'autre cote. Rend la direction et le cap.
    pub fn find_vista(&self, sun_at: impl Fn(DVec3) -> DVec3) -> Option<(DVec3, f32)> {
        const CANDIDATES: usize = 6000;
        const STEP: f64 = 2000.0;
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let is_water = |d: DVec3| self.water.elevation.sample(d) <= 0.0 || self.water.lake_at(d).0 > 0.5;
        let along = |o: DVec3, a: DVec3, m: f64| (o + a * (m / planet_radius())).normalize();

        // Un premier tri sur les grilles globales, tres peu couteuses.
        let mut ranked: Vec<(f32, DVec3, DVec3, f64)> = Vec::new();
        for i in 0..CANDIDATES {
            let y = 1.0 - (i as f64 + 0.5) / CANDIDATES as f64 * 2.0;
            let r = (1.0 - y * y).sqrt();
            let dir = DVec3::new((i as f64 * golden).cos() * r, y, (i as f64 * golden).sin() * r);
            let e = self.water.elevation.sample(dir);
            if !(5.0..800.0).contains(&e) || is_water(dir) {
                continue;
            }
            let climate = self.climate(dir, e);
            if !(4.0..22.0).contains(&climate.mean) || climate.aridity() < 0.3 {
                continue;
            }
            let sun = sun_at(dir);
            let flat = sun - dir * sun.dot(dir);
            if sun.dot(dir) < 0.0 || flat.length() < 0.01 {
                continue;
            }
            let toward = flat.normalize();
            let Some(first) = (1..60).find(|k| is_water(along(dir, toward, *k as f64 * STEP))) else { continue };
            let run = (first..first + 60).take_while(|k| is_water(along(dir, toward, *k as f64 * STEP))).count();
            if run < 2 {
                continue;
            }
            let shore = first as f64 * STEP;
            let beyond = shore + run as f64 * STEP;
            let peak = (0..24)
                .map(|k| self.water.elevation.sample(along(dir, toward, beyond + k as f64 * 6000.0)))
                .fold(0.0f32, f32::max);
            // L'ideal : assez d'eau pour le reflet du soleil, des montagnes derriere, un autre rivage visible.
            let score =
                (run as f32 * 2.0).min(14.0) + peak.min(2600.0) / 260.0 - (run as f32 * 2.0 - 40.0).max(0.0) * 0.2;
            ranked.push((score, dir, toward, shore));
        }
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

        // Puis on cherche la rive exacte et un sol degage, avec le relief detaille.
        let mut low = None;
        for (_, dir, toward, shore) in ranked.into_iter().take(250) {
            let mut d = (shore - 3000.0).max(0.0);
            let end = shore + 1500.0;
            while d < end && {
                let s = self.surface(along(dir, toward, d));
                s.height >= s.water_level - 0.2
            } {
                d += 25.0;
            }
            if d >= end {
                continue;
            }
            // Pas une mare : les grilles grossieres ne voient pas ce detail.
            let open = (1..16).all(|k| {
                let s = self.surface(along(dir, toward, d + k as f64 * 180.0));
                s.height < s.water_level
            });
            if !open {
                continue;
            }
            // Le plus haut possible au-dessus de l'eau : l'herbe cadre la vue sans la boucher.
            for back in [1500.0, 1000.0, 700.0, 450.0, 300.0, 200.0, 130.0, 80.0] {
                if back > d {
                    continue;
                }
                let spot = along(dir, toward, d - back);
                let s = self.surface(spot);
                let above = s.height - s.water_level;
                if !(2.0..110.0).contains(&above) {
                    continue;
                }
                // Un sol de prairie, et presque aucun arbre jusqu'a la rive.
                let sparse = |p: DVec3, height: f32| {
                    let c = self.climate(p, height);
                    self.tree_cover(&c, self.woods(p)) < 0.06
                };
                if !sparse(spot, s.height) {
                    continue;
                }
                let clear = (1..10).all(|k| {
                    let p = along(dir, toward, d - back + back * k as f64 / 10.0);
                    let h = self.height(p);
                    h < s.height + 0.8 && sparse(p, h)
                });
                let slope = (self.height(along(spot, toward, 4.0)) - s.height).abs();
                if !clear || slope > 2.5 {
                    continue;
                }
                let axis = if spot.y.abs() > 0.99 { DVec3::Z } else { DVec3::Y };
                let north = (axis - spot * axis.dot(spot)).normalize();
                let east = north.cross(spot);
                let vista = (spot, toward.dot(east).atan2(toward.dot(north)) as f32);
                if above >= 14.0 {
                    return Some(vista);
                }
                if low.is_none() {
                    low = Some(vista);
                }
                break;
            }
        }
        low
    }

    /// Cherche l'eau autour d'un point, puis un endroit sec et plat sur sa rive.
    fn bank_near(&self, center: DVec3) -> Option<DVec3> {
        let side = center.any_orthonormal_vector();
        let other = center.cross(side);
        let at = |a: f64, r: f64| (center + (side * a.cos() + other * a.sin()) * (r / planet_radius())).normalize();
        for k in 0..24 {
            let angle = k as f64 / 24.0 * std::f64::consts::TAU;
            let mut in_water = false;
            // Du centre vers le dehors : on quitte l'eau, puis on s'en ecarte un peu.
            for i in 0..80 {
                let radius = i as f64 * 60.0;
                let dir = at(angle, radius);
                let s = self.surface(dir);
                if s.fresh && s.height < s.water_level - 0.3 {
                    in_water = true;
                } else if in_water && s.height > s.water_level + 2.0 {
                    let spot = at(angle, radius + 35.0);
                    let e = self.surface(spot);
                    let near = at(angle, radius + 39.0);
                    let slope = (self.height(near) - e.height).abs();
                    if e.height > s.water_level + 1.5 && slope < 1.8 && self.woods(spot) < 0.3 {
                        return Some(spot);
                    }
                    break;
                }
            }
        }
        None
    }

    /// Cap qui regarde la riviere ou le lac le plus proche, en radians a droite du nord.
    pub fn yaw_towards_water(&self, dir: DVec3) -> f32 {
        let axis = if dir.y.abs() > 0.99 { DVec3::Z } else { DVec3::Y };
        let north = (axis - dir * axis.dot(dir)).normalize();
        let east = north.cross(dir);
        for r in [60.0, 120.0, 220.0, 400.0, 800.0] {
            for k in 0..32 {
                let yaw = k as f64 / 32.0 * std::f64::consts::TAU;
                let p = (dir + (north * yaw.cos() + east * yaw.sin()) * (r / planet_radius())).normalize();
                let s = self.surface(p);
                if s.height < s.water_level {
                    return yaw as f32;
                }
            }
        }
        0.0
    }

    /// Carte plate de la planete, en RVB : relief ombre, rivieres et lacs, et les frontieres de
    /// plaques en option (rouge : collision, orange : subduction, jaune : dorsale).
    pub fn map_rgb(&self, w: usize, h: usize, boundaries: bool) -> Vec<u8> {
        let mut image = vec![0u8; w * h * 3];
        let elevation = &self.water.elevation;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let rows = h / threads + 1;
        let radius = crate::world::generator::planet_radius();
        std::thread::scope(|s| {
            for (k, chunk) in image.chunks_mut(w * 3 * rows).enumerate() {
                s.spawn(move || {
                    crate::world::generator::worker_thread(radius);
                    for (i, px) in chunk.chunks_mut(3).enumerate() {
                        let (x, y) = (i % w, i / w + k * rows);
                        let lat = (0.5 - (y as f64 + 0.5) / h as f64) * 180.0;
                        let lon = ((x as f64 + 0.5) / w as f64 - 0.5) * 360.0;
                        let dir = lat_lon_to_dir(lat, lon);
                        let e = elevation.sample(dir);
                        let (mask, _) = self.water.lake_at(dir);
                        let lit = if e > 0.0 {
                            // Eclairage venu du nord-ouest : les chaines ressortent.
                            let d = 0.6 / w as f64 * std::f64::consts::TAU;
                            let nw = lat_lon_to_dir(lat + d.to_degrees(), lon - d.to_degrees());
                            (1.0 + (e - elevation.sample(nw)) * 0.0011).clamp(0.55, 1.45)
                        } else {
                            1.0
                        };
                        let c = if mask > SHORE { [46.0, 110.0, 190.0] } else { map_color(e).map(|v| v * lit) };
                        for (p, v) in px.iter_mut().zip(c) {
                            *p = v.clamp(0.0, 255.0) as u8;
                        }
                    }
                });
            }
        });
        let mut stamp = |dir: DVec3, c: [u8; 3]| {
            let (lat, lon) = dir_to_lat_lon(dir);
            let x = (((lon / 360.0 + 0.5) * w as f64) as usize).min(w - 1);
            let y = (((0.5 - lat / 180.0) * h as f64) as usize).min(h - 1);
            image[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&c);
        };
        // Seules les rivieres assez grandes pour cette echelle.
        let min_width = 90.0 * 2048.0 / w as f32 * 0.45;
        for edge in &self.water.edges {
            if edge.width[1] > min_width {
                let [a, m, b] = self.water.edge_points(edge);
                for (p, q) in [(a, m), (m, b)] {
                    for t in 0..3 {
                        stamp(p.lerp(q, t as f64 / 3.0).normalize(), [52, 118, 205]);
                    }
                }
            }
        }
        // Les autoroutes en rouge, les routes de montagne en orange, les villes en blanc.
        for hw in &self.cities.highways.list {
            for p in hw
                .points()
                .iter()
                .step_by(((10_000.0 * 2048.0 / w as f64) / super::highway::POINT_STEP).max(1.0) as usize)
            {
                stamp(*p, [215, 45, 30]);
            }
        }
        // Les routes de montagne en orange.
        for road in &self.cities.mountains.list {
            for p in road.paths[0].points.iter().step_by(((2_000.0 * 2048.0 / w as f64) / 8.0).max(1.0) as usize) {
                stamp(*p, [250, 160, 30]);
            }
        }
        for c in &self.cities.list {
            let (e, n) = c.axes();
            let px = 360.0 / w as f64 * std::f64::consts::PI / 180.0 * 0.7;
            for (i, j) in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] {
                stamp((c.dir + (e * i as f64 + n * j as f64) * px).normalize(), [255, 250, 235]);
            }
        }
        if boundaries {
            for c in 0..self.plates.grid.cells() {
                let color = match self.plates.boundary[c] {
                    Boundary::Collision => [255, 40, 40],
                    Boundary::Subduction => [255, 165, 30],
                    Boundary::Ridge => [255, 240, 90],
                    Boundary::None => continue,
                };
                stamp(self.plates.grid.center(c), color);
            }
        }
        image
    }

    /// Carte plate du climat. `mode` : "koppen" (types de climat), "temp" (temperature moyenne) ou "pluie".
    pub fn climate_map_rgb(&self, w: usize, h: usize, mode: &str) -> Vec<u8> {
        let mut image = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let lat = 90.0 - (y as f64 + 0.5) / h as f64 * 180.0;
                let lon = (x as f64 + 0.5) / w as f64 * 360.0 - 180.0;
                let dir = lat_lon_to_dir(lat, lon);
                let e = self.water.elevation.sample_linear(dir);
                let c = self.climate(dir, e);
                let land = e > 0.0;
                let ramp = |t: f32, stops: &[[f32; 3]]| -> [f32; 3] {
                    let t = t.clamp(0.0, 1.0) * (stops.len() - 1) as f32;
                    let k = (t as usize).min(stops.len() - 2);
                    mix3(stops[k], stops[k + 1], t - k as f32)
                };
                let col: [f32; 3] = match mode {
                    "temp" => ramp(
                        (c.mean + 30.0) / 60.0,
                        &[[60.0, 0.0, 120.0], [40.0, 90.0, 220.0], [150.0, 220.0, 240.0], [250.0, 240.0, 150.0], [240.0, 120.0, 40.0], [150.0, 0.0, 20.0]],
                    ),
                    "pluie" => ramp(
                        (c.rain / 3000.0).sqrt(),
                        &[[235.0, 215.0, 160.0], [200.0, 220.0, 130.0], [90.0, 190.0, 110.0], [40.0, 130.0, 170.0], [30.0, 50.0, 160.0], [60.0, 0.0, 110.0]],
                    ),
                    _ if !land => {
                        if c.mean < -2.0 {
                            [225.0, 235.0, 245.0]
                        } else {
                            [18.0, 40.0, 90.0]
                        }
                    }
                    _ => match c.koppen() {
                        Koppen::Rainforest => [0.0, 60.0, 255.0],
                        Koppen::Savanna => [70.0, 170.0, 250.0],
                        Koppen::Desert => [255.0, 60.0, 40.0],
                        Koppen::Steppe => [245.0, 165.0, 0.0],
                        Koppen::Mediterranean => [255.0, 255.0, 0.0],
                        Koppen::Temperate => [150.0, 230.0, 100.0],
                        Koppen::Continental => [0.0, 200.0, 220.0],
                        Koppen::Subarctic => [0.0, 110.0, 110.0],
                        Koppen::Tundra => [170.0, 170.0, 170.0],
                        Koppen::Ice => [245.0, 245.0, 250.0],
                    },
                };
                let col = if land || mode == "koppen" { col } else { col.map(|v| v * 0.55 + 30.0) };
                for k in 0..3 {
                    image[(y * w + x) * 3 + k] = col[k].clamp(0.0, 255.0) as u8;
                }
            }
        }
        image
    }

    /// Bilan du climat des terres : part de chaque type de Koppen, pluie et temperature moyennes
    /// (terres, puis planete entiere), part de terres.
    pub fn climate_stats(&self) -> ([f64; 10], [f64; 4], f64) {
        let (mut shares, mut land_area, mut area) = ([0.0f64; 10], 0.0f64, 0.0f64);
        let (mut rain_land, mut temp_land, mut rain_all, mut temp_all) = (0.0, 0.0, 0.0, 0.0);
        let (w, h) = (720, 360);
        for y in 0..h {
            let lat = 90.0 - (y as f64 + 0.5) / h as f64 * 180.0;
            let weight = lat.to_radians().cos();
            for x in 0..w {
                let dir = lat_lon_to_dir(lat, (x as f64 + 0.5) / w as f64 * 360.0 - 180.0);
                let e = self.water.elevation.sample_linear(dir);
                let c = self.climate(dir, e);
                area += weight;
                rain_all += c.rain as f64 * weight;
                temp_all += c.mean as f64 * weight;
                if e > 0.0 {
                    land_area += weight;
                    rain_land += c.rain as f64 * weight;
                    temp_land += c.mean as f64 * weight;
                    let k = Koppen::ALL.iter().position(|k| *k == c.koppen()).unwrap_or(0);
                    shares[k] += weight;
                }
            }
        }
        let land = land_area.max(1e-9);
        (shares.map(|s| s / land), [rain_land / land, temp_land / land, rain_all / area, temp_all / area], land_area / area)
    }

    /// Point de depart sans riviere : a terre, plat, tempere. Retombe sur le meilleur point sec.
    pub fn find_inland_spawn(&self) -> DVec3 {
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let count = 6000;
        // Le lieu qui remplit le plus de criteres, n'importe quel point sec a defaut.
        const PERFECT: u32 = 13;
        let mut best = (0, None);
        for i in 0..count {
            // Spirale de Fibonacci : couvre la sphere de facon reguliere.
            let y = 1.0 - (i as f64 + 0.5) / count as f64 * 2.0;
            // Les latitudes moyennes d'abord, plutot que le pole.
            let y = y * 0.75;
            let r = (1.0 - y * y).sqrt();
            let a = golden * i as f64;
            let dir = DVec3::new(a.cos() * r, y, a.sin() * r);

            let h = self.height(dir);
            if !(180.0..1500.0).contains(&h) {
                if h > SEA_LEVEL && best.1.is_none() {
                    best.1 = Some(dir);
                }
                continue;
            }
            let mut score = 4;
            let c = self.climate(dir, h);
            // Un paysage vert plutot qu'un desert pour commencer.
            let temperate = (5.0..19.0).contains(&c.mean) && c.cold > -8.0;
            score += temperate as u32 * 2 + (c.aridity() >= 0.65) as u32;
            if !temperate {
                if score > best.0 {
                    best = (score, Some(dir));
                }
                continue;
            }
            let east = dir.cross(DVec3::Y).normalize_or_zero();
            let north = east.cross(dir);
            let step = 4.0 / planet_radius();
            let slope = (self.height((dir + east * step).normalize()) - h).abs()
                + (self.height((dir + north * step).normalize()) - h).abs();
            // Une clairiere dans un pays vallonne : la premiere image doit montrer le paysage.
            let far = 1500.0 / planet_radius();
            let relief = (self.height((dir + east * far).normalize()) - h).abs()
                + (self.height((dir + north * far).normalize()) - h).abs();
            score += (relief > 70.0) as u32 * 3 + (self.woods(dir) < 0.35) as u32 * 2 + (slope < 2.5) as u32;
            if score == PERFECT {
                return dir;
            }
            if score > best.0 {
                best = (score, Some(dir));
            }
        }
        best.1.unwrap_or(DVec3::Y)
    }
}

fn map_color(e: f32) -> [f32; 3] {
    const STOPS: [(f32, [f32; 3]); 9] = [
        (-6000.0, [8.0, 16.0, 60.0]),
        (-2500.0, [18.0, 44.0, 110.0]),
        (-150.0, [46.0, 110.0, 170.0]),
        (0.0, [90.0, 165.0, 190.0]),
        (1.0, [64.0, 120.0, 60.0]),
        (500.0, [140.0, 160.0, 90.0]),
        (1500.0, [150.0, 115.0, 75.0]),
        (3500.0, [120.0, 90.0, 80.0]),
        (6500.0, [245.0, 245.0, 250.0]),
    ];
    for w in STOPS.windows(2) {
        if e <= w[1].0 {
            let t = ((e - w[0].0) / (w[1].0 - w[0].0)).clamp(0.0, 1.0);
            return mix3(w[0].1, w[1].1, t);
        }
    }
    [255.0; 3]
}

/// Direction sur la sphere depuis une latitude et une longitude en degres.
pub fn lat_lon_to_dir(lat: f64, lon: f64) -> DVec3 {
    let (la, lo) = (lat.to_radians(), lon.to_radians());
    DVec3::new(la.cos() * lo.cos(), la.sin(), la.cos() * lo.sin())
}

/// Latitude et longitude en degres.
pub fn dir_to_lat_lon(dir: DVec3) -> (f64, f64) {
    (dir.y.clamp(-1.0, 1.0).asin().to_degrees(), dir.z.atan2(dir.x).to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitized_params_stay_in_bounds_and_drop_nan() {
        let wild = PlanetParams {
            seed: 9,
            ocean: f32::NAN,
            relief: f32::INFINITY,
            temperature: -40.0,
            size: 0.0,
            cities: u32::MAX,
            rivers: false,
        };
        let p = wild.sanitized();
        let d = PlanetParams::default();
        assert_eq!((p.seed, p.rivers), (9, false));
        assert_eq!(p.ocean, d.ocean, "NaN reprend la valeur par defaut");
        assert_eq!(p.relief, PlanetParams::RELIEF.1);
        assert_eq!(p.temperature, PlanetParams::TEMPERATURE.0);
        assert_eq!(p.size, PlanetParams::SIZE.0);
        assert_eq!(p.cities, PlanetParams::MAX_CITIES);
        assert!(p.radius().is_finite() && p.radius() > 0.0);
        // Un rayon qui n'est pas un nombre ne se propage pas.
        assert!(PlanetParams { size: f32::NAN, ..d }.radius().is_finite());
        // Des reglages valides ne changent pas.
        assert_eq!(d.sanitized(), d);
    }

    /// Un thread de calcul au rayon de la planete affichee ne passe pas par sa variable de thread :
    /// sous Windows, elle coutait plus que le reste et bloquait le calcul parallele.
    #[test]
    fn worker_threads_keep_the_fast_radius_path() {
        let r = planet_radius();
        std::thread::spawn(move || {
            worker_thread(r);
            assert_eq!(CREATING.with(|c| c.0.get()), 0.0);
            worker_thread(r * 0.5);
            assert_eq!(CREATING.with(|c| c.0.get()), r * 0.5);
            assert_eq!(planet_radius(), r * 0.5);
        })
        .join()
        .unwrap();
        let v = par_map(1000, |_| planet_radius());
        assert!(v.iter().all(|x| *x == r));
    }

    /// La pente interpolee sur la grille reste proche de la pente exacte : les ravines gardent leur cap.
    #[test]
    fn lattice_slope_follows_the_exact_slope() {
        for level in 0..SLOPE_LATTICES.len() {
            lattice_slope_at_level(level);
        }
    }

    fn lattice_slope_at_level(level: usize) {
        let gen = Generator::new(PlanetParams { seed: 3, ..Default::default() });
        let full = Detail::full();
        let (mut worst, mut sum_angle, mut n) = (0.0f32, 0.0f32, 0);
        let mut rel = 0.0f32;
        let mut rng = crate::world::noise::Rng(5);
        while n < 3000 {
            let dir = DVec3::new(rng.f32() as f64 - 0.5, rng.f32() as f64 - 0.5, rng.f32() as f64 - 0.5).normalize();
            if gen.water.elevation.sample(dir) < 50.0 {
                continue;
            }
            let exact = gen.gully_slope(dir, &full, &gen.provinces.at(dir));
            let approx = gen.lattice_slope(dir, level);
            // Le cap ne compte que la ou il y a une pente a suivre.
            if exact.length() > 0.02 {
                let angle = exact.normalize().dot(approx.normalize()).clamp(-1.0, 1.0).acos().to_degrees();
                worst = worst.max(angle);
                sum_angle += angle;
                rel += (approx - exact).length() / exact.length();
                n += 1;
            }
        }
        let mean = sum_angle / n as f32;
        println!(
            "grille {} m : ecart de cap moyen {mean:.2} deg, pire {worst:.1} deg, ecart relatif moyen {:.3}",
            SLOPE_LATTICES[level].0,
            rel / n as f32
        );
        assert!(mean < 2.0 && rel / (n as f32) < 0.05, "cap {mean} deg");
    }

    /// Cout des etapes de `surface_natural` sur la grille d'un chunk (TERRA_BENCH=1, avec --nocapture).
    #[test]
    fn surface_cost_breakdown() {
        if std::env::var("TERRA_BENCH").is_err() {
            return;
        }
        use crate::world::chunk::NodeId;
        let gen = Generator::new(PlanetParams { seed: 3, ..Default::default() });
        let spawn = gen.find_spawn();
        for lod in [3u8, 9] {
            let id = NodeId::containing(spawn, lod);
            let d = Detail::for_spacing(id.spacing());
            let dirs: Vec<_> =
                (0..51 * 51).map(|k| id.dir_at((k % 51) as f64 / 48.0, (k / 51) as f64 / 48.0)).collect();
            let time = |name: &str, f: &dyn Fn(DVec3) -> f32| {
                let t = std::time::Instant::now();
                let mut acc = 0.0f32;
                for &dir in &dirs {
                    acc += f(dir);
                }
                std::hint::black_box(acc);
                print!("{name} {:.2}, ", t.elapsed().as_secs_f64() * 1000.0);
            };
            print!("LOD {lod} : ");
            time("total", &|p| gen.surface_natural(p, &d).height);
            time("provinces", &|p| gen.provinces.at(p).hills);
            time("elevation", &|p| gen.water.elevation.sample(p));
            time("orogenie", &|p| gen.orogeny.sample_linear(p));
            time("volcans", &|p| gen.volcanoes.at(p).map_or(0.0, |v| v.1));
            time("warp", &|p| gen.warp_at(p * planet_radius()).x as f32);
            let w = gen.provinces.at(spawn);
            time("shape", &|p| gen.shape(p, &d, &w).height);
            let coarse = d.coarse(700.0);
            time("shape grossier x4", &|p| {
                (0..4).map(|k| gen.shape(p + DVec3::splat(k as f64 * 1e-5), &coarse, &w).height).sum()
            });
            time("ravines", &|p| {
                gen.gullies
                    .erosion(p * planet_radius(), p.as_vec3(), Vec3::X * 0.1, GULLY_PERIOD, d.gullies, 0.0, 1.0)
                    .offset
            });
            time("meandres", &|p| gen.meander_point(p, &d).x as f32);
            time("lacs", &|p| gen.water.lake_at(p).0);
            time("rivieres", &|p| gen.water.river_at(p, d.spacing, 1.0, 0.0).map_or(0.0, |r| r.level));
            time("etangs", &|p| gen.ponds.pivot(p * planet_radius() / POND_CELL, POND_RADIUS).map_or(0.0, |v| v.1));
            time("villes", &|p| gen.cities.flatten(p));
            time("collines", &|p| gen.hills.eroded(p * planet_radius() / 420.0, d.hills));
            time("fin", &|p| gen.fine.fbm(p * planet_radius() / 95.0, d.fine));
            time("tours", &|p| gen.towers.towers(p * planet_radius() / 7000.0, 0.16, 0.45, 0.95, 50.0, 190.0));
            println!();
            println!("  detail {:?}", (d.mountains, d.alps, d.rolling, d.knolls, d.gullies, d.tables, d.hills, d.fine));
            println!(
                "  provinces au depart : collines {:.2} glaciaire {:.2} landes {:.2} mesa {:.2}",
                w.hills, w.glacial, w.moor, w.mesa
            );
        }
    }

    fn gen(seed: u64) -> Generator {
        Generator::new(PlanetParams { seed, ..Default::default() })
    }

    /// Points repartis sur toute la sphere.
    fn sphere_points(count: usize) -> impl Iterator<Item = DVec3> {
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        (0..count).map(move |i| {
            let y = 1.0 - (i as f64 + 0.5) / count as f64 * 2.0;
            let r = (1.0 - y * y).sqrt();
            let a = golden * i as f64;
            DVec3::new(a.cos() * r, y, a.sin() * r)
        })
    }

    #[test]
    fn planet_is_deterministic_for_a_seed() {
        let (a, b, c) = (gen(42), gen(42), gen(43));
        let mut differs = false;
        for dir in sphere_points(200) {
            assert_eq!(a.height(dir), b.height(dir));
            differs |= a.height(dir) != c.height(dir);
        }
        assert!(differs, "deux graines doivent donner deux planetes");
    }

    #[test]
    fn terrain_is_continuous_at_metre_scale() {
        let g = gen(1);
        let dir = g.find_spawn();
        let east = dir.cross(DVec3::Y).normalize();
        let mut last = g.height(dir);
        for i in 1..500 {
            let h = g.height((dir + east * (i as f64 * 0.5 / planet_radius())).normalize());
            assert!(h.is_finite());
            assert!((h - last).abs() < 6.0, "saut de {} m", (h - last).abs());
            last = h;
        }
    }

    #[test]
    fn looks_like_an_earth() {
        for seed in [1, 2, 3, 4] {
            let g = gen(seed);
            let (mut sea, mut land, mut high, mut max) = (0, 0, 0, f32::MIN);
            let (mut min, mut biomes) = (f32::MAX, std::collections::HashSet::new());
            for dir in sphere_points(6000) {
                let h = g.height(dir);
                max = max.max(h);
                min = min.min(h);
                if h < SEA_LEVEL {
                    sea += 1;
                } else {
                    land += 1;
                    biomes.insert(format!("{:?}", g.biome(g.climate(dir, h))));
                }
                if h > 2000.0 {
                    high += 1;
                }
            }
            let ocean = sea as f32 / (sea + land) as f32;
            assert!((0.45..0.88).contains(&ocean), "graine {seed} : {ocean} d'ocean");
            assert!(high > 20, "graine {seed} : {high} points au-dessus de 2000 m");
            assert!(max < MAX_HEIGHT && min > MIN_HEIGHT, "graine {seed} : {min}..{max}");
            assert!(biomes.len() >= 4, "graine {seed} : {biomes:?}");
        }
    }

    /// La terre doit etre vallonnee a toutes les echelles, pas seulement dans les montagnes.
    #[test]
    fn land_has_relief_at_every_scale() {
        for seed in [1, 2, 3] {
            let g = gen(seed);
            let (mut small, mut medium, mut count) = (0.0, 0.0, 0);
            for dir in sphere_points(1500) {
                let h = g.height(dir);
                if h < 50.0 {
                    continue;
                }
                let side = dir.any_orthonormal_vector();
                let at = |d: f64| g.height((dir + side * (d / planet_radius())).normalize());
                small += (at(150.0) - h).abs();
                medium += (at(1500.0) - h).abs();
                count += 1;
            }
            let (small, medium) = (small / count as f32, medium / count as f32);
            assert!(small > 6.0, "graine {seed} : {small} m d'ecart moyen sur 150 m");
            assert!(medium > 30.0, "graine {seed} : {medium} m d'ecart moyen sur 1500 m");
        }
    }

    #[test]
    fn ocean_slider_changes_the_sea() {
        let share = |ocean: f32| {
            let g = Generator::new(PlanetParams { seed: 5, ocean, ..Default::default() });
            sphere_points(3000).filter(|d| g.height(*d) < SEA_LEVEL).count()
        };
        assert!(share(0.9) > share(0.5));
        assert!(share(0.5) > share(0.1));
    }

    #[test]
    fn poles_are_cold_and_equator_is_warm() {
        let g = gen(9);
        assert!(g.climate(DVec3::Y, 0.0).temperature < 0.2);
        assert!(g.climate(DVec3::X, 0.0).temperature > 0.6);
    }

    #[test]
    fn spawn_is_on_dry_land() {
        for seed in 0..6 {
            let g = gen(seed);
            assert!(g.height(g.find_spawn()) > SEA_LEVEL, "graine {seed}");
            // Un soleil fixe suffit ici : la vue doit se trouver a terre, tournee vers ce soleil.
            let sun = DVec3::new(0.7, 0.1, 0.7).normalize();
            let start = std::time::Instant::now();
            if let Some((dir, _)) = g.find_vista(|_| sun) {
                let s = g.surface(dir);
                assert!(s.height > s.water_level, "graine {seed} : vue sous l'eau");
                assert!(start.elapsed().as_secs_f32() < 3.0, "graine {seed} : recherche trop longue");
            }
        }
    }

    /// Comme sur Terre, de grandes plaines (un tiers a la moitie des terres), mais pas partout : le reste
    /// a du relief ou de l'eau en vue, et chaque planete a des paysages varies.
    #[test]
    fn almost_no_place_is_dull_and_every_planet_is_varied() {
        for seed in [1, 2, 3, 4] {
            let g = gen(seed);
            let detail = Detail::for_spacing(25.0);
            let (mut land, mut dull) = (0, 0);
            let mut kinds = std::collections::HashSet::new();
            for dir in sphere_points(4000) {
                if g.surface_detail(dir, &detail).height <= SEA_LEVEL {
                    continue;
                }
                land += 1;
                kinds.insert(g.landform_name(dir));
                let (side, other) = (dir.any_orthonormal_vector(), dir.cross(dir.any_orthonormal_vector()));
                let (mut lo, mut hi, mut water) = (f32::MAX, f32::MIN, false);
                for k in 0..24 {
                    let a = k as f64 / 24.0 * std::f64::consts::TAU;
                    let r = 10_000.0 * (0.35 + 0.65 * ((k % 3) as f64 / 2.0)) / planet_radius();
                    let s = g.surface_detail((dir + (side * a.cos() + other * a.sin()) * r).normalize(), &detail);
                    lo = lo.min(s.height);
                    hi = hi.max(s.height);
                    water |= s.height < s.water_level;
                }
                if hi - lo < 150.0 && !water {
                    dull += 1;
                }
            }
            assert!(dull * 100 < land * 55, "graine {seed} : {dull} lieux plats sur {land}");
            assert!(dull * 100 > land * 5, "graine {seed} : {dull} lieux plats sur {land}");
            assert!(kinds.len() >= 7, "graine {seed} : {kinds:?}");
        }
    }

    /// Une foret dense ne doit pas couvrir tout un pays : le relief et les clairieres doivent se voir.
    #[test]
    fn climate_gives_both_wet_and_dry_lands() {
        for seed in [1, 2, 3] {
            let g = gen(seed);
            let (mut dry, mut wet, mut land) = (0, 0, 0);
            for dir in sphere_points(3000) {
                let e = g.water.elevation.sample(dir);
                if e > 0.0 {
                    let c = g.climate(dir, e);
                    land += 1;
                    // Indice d'aridite de l'UNEP : aride en dessous de 0,2, humide au-dessus de 0,65. Sur
                    // Terre, environ 20 % et 45 % des terres.
                    dry += (c.aridity() < 0.2) as i32;
                    wet += (c.aridity() > 0.65) as i32;
                }
            }
            assert!(dry * 14 > land && wet * 5 > land, "graine {seed} : {dry} secs, {wet} humides sur {land}");
            assert!(dry * 2 < land && wet * 4 < land * 3, "graine {seed} : {dry} secs, {wet} humides sur {land}");
        }
    }

    #[test]
    fn lat_lon_round_trip() {
        let (la, lo) = dir_to_lat_lon(lat_lon_to_dir(48.85, 2.35));
        assert!((la - 48.85).abs() < 1e-9 && (lo - 2.35).abs() < 1e-9);
    }
}
