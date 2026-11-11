//! Rendu wgpu : ombres en cascade, scene HDR, ocean, nuages, puis tone mapping, FXAA et interface.

mod gpu_timer;
pub use gpu_timer::SECTIONS as GPU_SECTIONS;
pub mod meshes;
pub mod noise_tex;

use crate::camera::Camera;
use crate::error::Error;
use crate::sky::Sky;
use crate::world::chunk::{build_indices, ChunkMesh, NodeId, TerrainVertex, GRID_INDEX_COUNT};
use crate::world::city_mesh::CityVertex;
use crate::world::generator::planet_radius;
use crate::world::scatter::{Kind, ObjectInstance, KIND_COUNT};
use crate::world::streaming::{StreamUpdate, Streamer};
use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Mat4, Vec3, Vec4};
use meshes::{build_object_meshes, ObjectVertex};
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::window::Window;

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SHADOW_SIZE: u32 = 2048;
const CASCADES: usize = 3;
const CASCADE_RADII: [f32; CASCADES] = [40.0, 220.0, 1500.0];
/// Marge le long de la lumiere : un sommet peut ombrer de loin.
const SHADOW_DEPTH_MARGIN: f32 = 4000.0;
const MAX_CHUNK_INSTANCES: usize = 8192;
const MAX_OBJECT_INSTANCES: usize = 1_200_000;
/// Threads au plus pour preparer les instances d'objets d'une image.
const PREPARE_THREADS: usize = 6;

/// Listes d'instances d'un thread : par sorte d'objet, par niveau de detail.
/// Listes d'instances par sorte, puis par niveau de detail et par bande d'ombre (`lod * BANDS + band`).
type Scratch = [[Vec<ObjectInstance>; 3 * BANDS]; KIND_COUNT];
/// Bande d'ombre d'un objet : la premiere cascade dont la zone peut recevoir son ombre (les zones sont
/// emboitees, la cascade i dessine les bandes 0 a i) ; la derniere bande ne porte d'ombre dans aucune.
const BANDS: usize = CASCADES + 1;
/// Rayon englobant d'un arbre, pour savoir si son ombre peut tomber dans une cascade.
const SHADOW_OBJECT_RADIUS: f32 = 45.0;

/// Ajoute les objets d'une cellule, relatifs a la camera, dans la liste de leur niveau de detail.
#[allow(clippy::too_many_arguments)]
fn add_instances(
    out: &mut [Vec<ObjectInstance>; 3 * BANDS],
    items: &[ObjectInstance],
    center: DVec3,
    cam: DVec3,
    radius: f32,
    fade: f32,
    lods: [f32; 2],
    cascades: Option<&[Mat4; CASCADES]>,
) {
    let offset = (center - cam).as_vec3();
    let (r2, inv_fade) = (radius * radius, 1.0 / fade);
    // Bandes possibles pour toute la cellule : avant `lo`, elle est entierement hors du carre de la
    // cascade ; a partir de `hi`, entierement dedans. Le test par objet ne sert qu'entre les deux.
    let (lo, hi) = match cascades {
        Some(m) => {
            let reach = |c: usize| {
                let p = m[c].project_point3(offset);
                let (cell, margin) =
                    (CELL_CULL_RADIUS / CASCADE_RADII[c], 1.0 + SHADOW_OBJECT_RADIUS / CASCADE_RADII[c]);
                (p.x.abs().max(p.y.abs()) - cell <= margin, p.x.abs().max(p.y.abs()) + cell <= margin)
            };
            let tests: [(bool, bool); CASCADES] = std::array::from_fn(reach);
            (tests.iter().position(|t| t.0).unwrap_or(CASCADES), tests.iter().position(|t| t.1).unwrap_or(CASCADES))
        }
        None => (CASCADES, CASCADES),
    };
    for it in items {
        let pos = Vec3::from_array(it.pos) + offset;
        let d2 = pos.length_squared();
        if d2 >= r2 {
            continue;
        }
        let d = d2.sqrt();
        // Retrecit a la limite de la zone : pas d'apparition brutale.
        let k = ((radius - d) * inv_fade).min(1.0);
        let lod = (d > lods[0]) as usize + (d > lods[1]) as usize;
        let band = match cascades {
            _ if lo >= hi => hi,
            // Vu du soleil : la premiere cascade dont le carre contient l'objet, marge comprise.
            Some(m) => (lo..hi)
                .find(|&c| {
                    let p = m[c].project_point3(pos);
                    let margin = 1.0 + SHADOW_OBJECT_RADIUS / CASCADE_RADII[c];
                    p.x.abs() <= margin && p.y.abs() <= margin
                })
                .unwrap_or(hi),
            None => CASCADES,
        };
        out[lod * BANDS + band].push(ObjectInstance { pos: pos.to_array(), scale: it.scale * k, ..*it });
    }
}
const WRAP: f64 = 4096.0;
/// Au-dela de cette distance, un objet utilise sa version lointaine.
const NEAR_MESH_DISTANCE: f32 = 160.0;
/// Au-dela, version la plus legere, sans ombre portee.
const FAR_MESH_DISTANCE: f32 = 900.0;
/// Les herbes sont bien plus nombreuses : elles passent plus tot a leurs versions legeres.
const HERB_MESH_DISTANCES: [f32; 2] = [32.0, 120.0];
/// En deca, un objet hors champ reste prepare : son ombre peut tomber dans l'image.
const SHADOW_CELL_DISTANCE: f32 = 1800.0;
/// Au-dessus de cette altitude, les ombres de proximite ne se voient plus.
const SHADOW_MAX_ALTITUDE: f64 = 15_000.0;
/// Reductions successives de l'image pour le halo lumineux.
const BLOOM_LEVELS: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: Mat4,
    inv_view_proj: Mat4,
    cascades: [Mat4; CASCADES],
    sun_dir: Vec4,
    up: Vec4,
    tangent: Vec4,
    center: Vec4,
    camera_wrap: Vec4,
    misc: Vec4,
    camera_km: Vec4,
    /// Moment de l'annee, precipitations, part de neige, sol mouille.
    weather: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChunkInstance {
    offset: [f32; 3],
    size: f32,
    wrap: [f32; 3],
}

struct GpuCity {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    count: u32,
}

struct GpuChunk {
    vertices: wgpu::Buffer,
    water: Option<wgpu::Buffer>,
    /// Rues et batiments.
    city: Option<GpuCity>,
    center: DVec3,
    aabb_min: Vec3,
    aabb_max: Vec3,
}

struct DrawnChunk {
    id: NodeId,
    min: Vec3,
    max: Vec3,
}

struct ObjectDraw {
    mesh: Range<u32>,
    instances: Range<u32>,
    casts_shadow: bool,
    /// Bande d'ombre : dessine dans les cascades `band` et suivantes.
    band: usize,
}

struct SizedTargets {
    hdr_texture: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    /// Copie de la scene et des nuages : ce que l'eau reflete.
    scene_copy_texture: wgpu::Texture,
    scene_copy_view: wgpu::TextureView,
    clouds_view: wgpu::TextureView,
    clouds_group: wgpu::BindGroup,
    bloom_views: Vec<wgpu::TextureView>,
    /// `bloom_down[i]` lit le niveau au-dessus du niveau i, `bloom_up[i]` le niveau en dessous.
    bloom_down: Vec<wgpu::BindGroup>,
    bloom_up: Vec<wgpu::BindGroup>,
    rays_view: wgpu::TextureView,
    rays_group: wgpu::BindGroup,
    post_group: wgpu::BindGroup,
    depth_group: wgpu::BindGroup,
    water_group: wgpu::BindGroup,
}

/// Mises en page des ressources qui dependent de la taille de la fenetre.
struct SizedLayouts {
    texture: wgpu::BindGroupLayout,
    post: wgpu::BindGroupLayout,
    depth: wgpu::BindGroupLayout,
    water: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

pub struct FrameStats {
    pub chunks_drawn: usize,
    pub triangles: u64,
    pub objects: usize,
}

/// Interface deja tessellee, a dessiner par-dessus l'image.
pub struct UiFrame {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

/// Cote d'une face de la carte de nebulosite.
const WEATHER_SIZE: u32 = 128;

pub struct Renderer {
    /// Temps cumules (TERRA_PROFILE) : image d'affichage, preparation, enregistrement, envoi et presentation.
    pub timings: [f64; 4],
    /// Temps GPU par groupe de passes, si TERRA_PROFILE est defini et la carte le permet.
    pub gpu_timer: Option<gpu_timer::GpuTimer>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pub adapter_name: String,

    terrain_pipeline: wgpu::RenderPipeline,
    object_pipeline: wgpu::RenderPipeline,
    city_pipeline: wgpu::RenderPipeline,
    shadow_city_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    clouds_march_pipeline: wgpu::RenderPipeline,
    clouds_composite_pipeline: wgpu::RenderPipeline,
    bloom_first_pipeline: wgpu::RenderPipeline,
    bloom_down_pipeline: wgpu::RenderPipeline,
    bloom_up_pipeline: wgpu::RenderPipeline,
    rays_pipeline: wgpu::RenderPipeline,
    post_pipeline: wgpu::RenderPipeline,
    shadow_terrain_pipeline: wgpu::RenderPipeline,
    shadow_object_pipeline: wgpu::RenderPipeline,

    globals_buffer: wgpu::Buffer,
    globals_group: wgpu::BindGroup,
    weather_texture: wgpu::Texture,
    shadow_read_group: wgpu::BindGroup,
    empty_group: wgpu::BindGroup,
    cascade_buffers: Vec<wgpu::Buffer>,
    cascade_groups: Vec<wgpu::BindGroup>,
    cascade_views: Vec<wgpu::TextureView>,

    sized_layouts: SizedLayouts,
    targets: SizedTargets,

    index_buffer: wgpu::Buffer,
    index_count: u32,
    /// Un sommet sur deux : pour la cascade d'ombre lointaine.
    coarse_index_buffer: wgpu::Buffer,
    coarse_index_count: u32,
    chunk_instances: wgpu::Buffer,
    chunks: crate::world::fast_hash::FastMap<NodeId, GpuChunk>,

    object_vertices: wgpu::Buffer,
    object_ranges: [[Range<u32>; 3]; KIND_COUNT],
    object_instances: wgpu::Buffer,
    /// Instances de l'image, une liste par thread de preparation, par sorte d'objet et par niveau de detail.
    scratch_instances: Vec<Scratch>,

    /// La voiture : modele, sommets poses a chaque image.
    car_model: crate::vehicle::CarModel,
    car_scratch: Vec<CityVertex>,
    car_vertices: wgpu::Buffer,
    car_indices: wgpu::Buffer,

    ui_renderer: egui_wgpu::Renderer,
}

fn wrap(v: f64) -> f32 {
    v.rem_euclid(WRAP) as f32
}

fn wrap_vec(v: DVec3) -> [f32; 3] {
    [wrap(v.x), wrap(v.y), wrap(v.z)]
}

/// Plans de clipping (hors plan lointain, inexistant avec Z inverse infini).
fn frustum_planes(m: Mat4) -> [Vec4; 5] {
    let (r0, r1, r2, r3) = (m.row(0), m.row(1), m.row(2), m.row(3));
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2]
}

fn aabb_in_frustum(planes: &[Vec4; 5], min: Vec3, max: Vec3) -> bool {
    planes.iter().all(|p| {
        let v = Vec3::new(
            if p.x >= 0.0 { max.x } else { min.x },
            if p.y >= 0.0 { max.y } else { min.y },
            if p.z >= 0.0 { max.z } else { min.z },
        );
        p.truncate().dot(v) + p.w >= 0.0
    })
}

/// Rayon englobant une cellule de vegetation et ses plus grands arbres.
const CELL_CULL_RADIUS: f32 = 200.0;

fn sphere_in_frustum(planes: &[Vec4; 5], center: Vec3, radius: f32) -> bool {
    planes.iter().all(|p| p.truncate().dot(center) + p.w >= -radius * p.truncate().length())
}

fn aabb_in_light_box(m: Mat4, min: Vec3, max: Vec3) -> bool {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for i in 0..8 {
        let c = Vec3::new(
            if i & 1 == 0 { min.x } else { max.x },
            if i & 2 == 0 { min.y } else { max.y },
            if i & 4 == 0 { min.z } else { max.z },
        );
        let p = m.project_point3(c);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    hi.x >= -1.0 && lo.x <= 1.0 && hi.y >= -1.0 && lo.y <= 1.0 && hi.z >= 0.0 && lo.z <= 1.0
}

/// Tangente stable aux pieds de la camera : ne depend pas du regard, sinon les arbres tourneraient avec lui.
fn stable_tangent(up: Vec3) -> Vec3 {
    let axis = if up.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
    axis.cross(up).normalize()
}

/// Matrices des cascades, calees sur les texels du monde pour que les ombres ne scintillent pas.
fn cascade_matrices(cam: &Camera, light_dir: Vec3) -> [Mat4; CASCADES] {
    let fwd = cam.forward();
    let up_hint = if light_dir.y.abs() > 0.95 { Vec3::X } else { Vec3::Y };
    let right = light_dir.cross(up_hint).normalize();
    let up = right.cross(light_dir).normalize();

    std::array::from_fn(|i| {
        let r = CASCADE_RADII[i];
        let ideal = cam.position + (fwd * r * 0.6).as_dvec3();
        let texel = (2.0 * r as f64) / SHADOW_SIZE as f64;
        let snap = |axis: Vec3| {
            let a = axis.as_dvec3();
            a * ((ideal.dot(a) / texel).round() * texel)
        };
        let along = light_dir.as_dvec3();
        let snapped: DVec3 = snap(right) + snap(up) + along * ideal.dot(along);
        let center = (snapped - cam.position).as_vec3();

        let dist = r + SHADOW_DEPTH_MARGIN;
        let view = Mat4::look_to_rh(center + light_dir * dist, -light_dir, up);
        Mat4::orthographic_rh(-r, r, -r, r, 0.0, 2.0 * dist) * view
    })
}

impl Renderer {
    /// Ouvre la carte graphique la plus rapide capable de dessiner dans `window` et prepare tous les
    /// pipelines. `vsync` cale l'affichage sur l'ecran.
    ///
    /// # Errors
    ///
    /// Si aucune carte ne convient, ou si elle refuse la surface ou le peripherique.
    pub async fn new(window: Arc<Window>, vsync: bool) -> crate::Result<Self> {
        // Vulkan compile les shaders bien plus vite que le chemin HLSL/FXC de DX12.
        let (surface, adapter) = 'found: {
            for backends in [wgpu::Backends::VULKAN, wgpu::Backends::PRIMARY] {
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends, ..Default::default() });
                let surface = instance.create_surface(window.clone())?;
                let adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: Some(&surface),
                        force_fallback_adapter: false,
                    })
                    .await;
                if let Some(adapter) = adapter {
                    break 'found (surface, adapter);
                }
            }
            return Err(Error::NoAdapter);
        };
        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?})", info.name, info.backend);
        let gpu_profile =
            std::env::var("TERRA_PROFILE").is_ok() && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("terra"),
                    required_features: if gpu_profile {
                        wgpu::Features::TIMESTAMP_QUERY
                    } else {
                        wgpu::Features::empty()
                    },
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;
        let gpu_timer = gpu_profile.then(|| gpu_timer::GpuTimer::new(&device, &queue));

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or(Error::NoSurfaceFormat)?;
        let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        if caps.usages.contains(wgpu::TextureUsages::COPY_SRC) {
            usage |= wgpu::TextureUsages::COPY_SRC;
        }
        let config = wgpu::SurfaceConfiguration {
            usage,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: if vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync },
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // Groupes de ressources partages.
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bruit 3D partage par les nuages, le terrain et l'eau.
        let noise_levels = noise_tex::build_noise();
        let side = noise_tex::NOISE_SIZE as u32;
        let noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("noise"),
            size: wgpu::Extent3d { width: side, height: side, depth_or_array_layers: side },
            mip_level_count: noise_levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (level, data) in noise_levels.iter().enumerate() {
            let size = (side >> level).max(1);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &noise_texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size * 4), rows_per_image: Some(size) },
                wgpu::Extent3d { width: size, height: size, depth_or_array_layers: size },
            );
        }
        let noise_view = noise_texture.create_view(&Default::default());
        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("noise"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // Climat de la planete, sur les six faces d'un cube : nebulosite (r), temperature moyenne (g), ecart
        // entre l'hiver et l'ete (b) et pluie de l'annee (a), au niveau de la mer.
        let weather_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("weather"),
            size: wgpu::Extent3d { width: WEATHER_SIZE, height: WEATHER_SIZE, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let weather_view = weather_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let globals_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &globals_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&noise_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&noise_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&weather_view) },
            ],
        });

        let shadow_read_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow read"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow map"),
            size: wgpu::Extent3d { width: SHADOW_SIZE, height: SHADOW_SIZE, depth_or_array_layers: CASCADES as u32 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_array_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let cascade_views: Vec<_> = (0..CASCADES as u32)
            .map(|layer| {
                shadow_tex.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let shadow_read_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow read"),
            layout: &shadow_read_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&shadow_array_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shadow_sampler) },
            ],
        });

        let cascade_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cascade"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let cascade_buffers: Vec<_> = (0..CASCADES)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cascade"),
                    size: 64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let cascade_groups: Vec<_> = cascade_buffers
            .iter()
            .map(|b| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("cascade"),
                    layout: &cascade_layout,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: b.as_entire_binding() }],
                })
            })
            .collect();

        let color_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let depth_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sized_layouts = SizedLayouts {
            texture: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("texture"),
                entries: &[color_entry(0), sampler_entry(1)],
            }),
            post: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("post"),
                entries: &[color_entry(0), sampler_entry(1), color_entry(2), color_entry(3)],
            }),
            depth: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene depth"),
                entries: &[depth_entry],
            }),
            water: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("water scene"),
                entries: &[depth_entry, color_entry(1), sampler_entry(2)],
            }),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("hdr"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        };

        // Pipelines.
        let module = |label: &str, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        const COMMON: &str = include_str!("shaders/common.wgsl");
        const LIT: &str = include_str!("shaders/lit.wgsl");
        let terrain_sm = module("terrain", &[COMMON, LIT, include_str!("shaders/terrain.wgsl")].concat());
        let object_sm = module("object", &[COMMON, LIT, include_str!("shaders/object.wgsl")].concat());
        let city_sm = module("city", &[COMMON, LIT, include_str!("shaders/city.wgsl")].concat());
        let shadow_sm = module("shadow", &[COMMON, include_str!("shaders/shadow.wgsl")].concat());
        let sky_sm = module("sky", &[COMMON, include_str!("shaders/sky.wgsl")].concat());
        let water_sm = module("water", &[COMMON, LIT, include_str!("shaders/water.wgsl")].concat());
        let clouds_sm = module("clouds", &[COMMON, include_str!("shaders/clouds.wgsl")].concat());
        let bloom_sm = module("bloom", &[COMMON, include_str!("shaders/bloom.wgsl")].concat());
        let post_sm = module("post", &[COMMON, include_str!("shaders/post.wgsl")].concat());

        let layout = |bgs: &[&wgpu::BindGroupLayout]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: bgs,
                push_constant_ranges: &[],
            })
        };
        let empty_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("empty"), entries: &[] });
        let empty_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("empty"),
            layout: &empty_layout,
            entries: &[],
        });
        let lit_layout = layout(&[&globals_layout, &shadow_read_layout]);
        let shadow_layout = layout(&[&globals_layout, &cascade_layout]);
        let sky_layout = layout(&[&globals_layout]);
        let water_pipeline_layout = layout(&[&globals_layout, &shadow_read_layout, &sized_layouts.water]);
        let march_layout = layout(&[&globals_layout, &empty_layout, &sized_layouts.depth]);
        let texture_layout = layout(&[&globals_layout, &sized_layouts.texture]);
        let rays_layout = layout(&[&globals_layout, &sized_layouts.texture, &sized_layouts.depth]);
        let post_pipeline_layout = layout(&[&globals_layout, &sized_layouts.post, &sized_layouts.depth]);

        let terrain_attrs = wgpu::vertex_attr_array![
            0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Unorm8x4, 7 => Unorm8x4, 8 => Float32
        ];
        let chunk_attrs = wgpu::vertex_attr_array![4 => Float32x3, 5 => Float32, 6 => Float32x3];
        let terrain_buffers = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<TerrainVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &terrain_attrs,
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<ChunkInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &chunk_attrs,
            },
        ];
        let city_attrs =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Unorm8x4, 7 => Uint8x4];
        let city_buffers = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<CityVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &city_attrs,
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<ChunkInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &chunk_attrs,
            },
        ];
        let object_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Unorm8x4];
        let instance_attrs = wgpu::vertex_attr_array![3 => Float32x3, 4 => Float32, 5 => Float32, 6 => Unorm8x4];
        let object_buffers = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<ObjectVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &object_attrs,
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<ObjectInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &instance_attrs,
            },
        ];

        let depth_state = |write: bool, compare| wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: write,
            depth_compare: compare,
            stencil: Default::default(),
            bias: Default::default(),
        };
        let shadow_depth = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
            bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.5, clamp: 0.0 },
        };

        struct Desc<'a> {
            label: &'a str,
            layout: &'a wgpu::PipelineLayout,
            sm: &'a wgpu::ShaderModule,
            vs: &'a str,
            fs: Option<&'a str>,
            buffers: &'a [wgpu::VertexBufferLayout<'a>],
            depth: Option<wgpu::DepthStencilState>,
            target: Option<wgpu::ColorTargetState>,
        }
        let make = |d: Desc| {
            let targets = [d.target];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(d.label),
                layout: Some(d.layout),
                vertex: wgpu::VertexState {
                    module: d.sm,
                    entry_point: Some(d.vs),
                    compilation_options: Default::default(),
                    buffers: d.buffers,
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: d.depth,
                multisample: Default::default(),
                fragment: d.fs.map(|fs| wgpu::FragmentState {
                    module: d.sm,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview: None,
                cache: None,
            })
        };
        let hdr_target =
            |blend| Some(wgpu::ColorTargetState { format: HDR_FORMAT, blend, write_mask: wgpu::ColorWrites::ALL });

        let terrain_pipeline = make(Desc {
            label: "terrain",
            layout: &lit_layout,
            sm: &terrain_sm,
            vs: "vs",
            fs: Some("fs"),
            buffers: &terrain_buffers,
            depth: Some(depth_state(true, wgpu::CompareFunction::Greater)),
            target: hdr_target(None),
        });
        let object_pipeline = make(Desc {
            label: "object",
            layout: &lit_layout,
            sm: &object_sm,
            vs: "vs",
            fs: Some("fs"),
            buffers: &object_buffers,
            depth: Some(depth_state(true, wgpu::CompareFunction::Greater)),
            target: hdr_target(None),
        });
        let city_pipeline = make(Desc {
            label: "city",
            layout: &lit_layout,
            sm: &city_sm,
            vs: "vs",
            fs: Some("fs"),
            buffers: &city_buffers,
            depth: Some(depth_state(true, wgpu::CompareFunction::Greater)),
            target: hdr_target(None),
        });
        let sky_pipeline = make(Desc {
            label: "sky",
            layout: &sky_layout,
            sm: &sky_sm,
            vs: "vs",
            fs: Some("fs"),
            buffers: &[],
            depth: Some(depth_state(false, wgpu::CompareFunction::Always)),
            target: hdr_target(None),
        });
        let water_pipeline = make(Desc {
            label: "water",
            layout: &water_pipeline_layout,
            sm: &water_sm,
            vs: "vs",
            fs: Some("fs"),
            buffers: &terrain_buffers,
            depth: Some(depth_state(false, wgpu::CompareFunction::Greater)),
            target: hdr_target(Some(wgpu::BlendState::ALPHA_BLENDING)),
        });
        let clouds_march_pipeline = make(Desc {
            label: "clouds march",
            layout: &march_layout,
            sm: &clouds_sm,
            vs: "vs",
            fs: Some("fs_march"),
            buffers: &[],
            depth: None,
            target: hdr_target(None),
        });
        // Lumiere emise par le nuage, plus la scene attenuee par sa transparence (dans alpha).
        let clouds_composite_pipeline = make(Desc {
            label: "clouds composite",
            layout: &texture_layout,
            sm: &clouds_sm,
            vs: "vs",
            fs: Some("fs_composite"),
            buffers: &[],
            depth: Some(depth_state(false, wgpu::CompareFunction::Always)),
            target: hdr_target(Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::SrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            })),
        });
        let bloom_pass = |label, entry, blend| {
            make(Desc {
                label,
                layout: &texture_layout,
                sm: &bloom_sm,
                vs: "vs",
                fs: Some(entry),
                buffers: &[],
                depth: None,
                target: hdr_target(blend),
            })
        };
        let bloom_first_pipeline = bloom_pass("bloom first", "fs_down_first", None);
        let bloom_down_pipeline = bloom_pass("bloom down", "fs_down", None);
        let bloom_up_pipeline = bloom_pass(
            "bloom up",
            "fs_up",
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            }),
        );
        let rays_pipeline = make(Desc {
            label: "rays",
            layout: &rays_layout,
            sm: &bloom_sm,
            vs: "vs",
            fs: Some("fs_rays"),
            buffers: &[],
            depth: None,
            target: hdr_target(None),
        });
        let post_pipeline = make(Desc {
            label: "post",
            layout: &post_pipeline_layout,
            sm: &post_sm,
            vs: "vs",
            fs: Some("fs"),
            buffers: &[],
            depth: None,
            target: Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL }),
        });
        let shadow_terrain_pipeline = make(Desc {
            label: "shadow terrain",
            layout: &shadow_layout,
            sm: &shadow_sm,
            vs: "vs_terrain",
            fs: None,
            buffers: &terrain_buffers,
            depth: Some(shadow_depth.clone()),
            target: None,
        });
        let shadow_city_pipeline = make(Desc {
            label: "shadow city",
            layout: &shadow_layout,
            sm: &shadow_sm,
            vs: "vs_city",
            fs: None,
            buffers: &city_buffers,
            depth: Some(shadow_depth.clone()),
            target: None,
        });
        let shadow_object_pipeline = make(Desc {
            label: "shadow object",
            layout: &shadow_layout,
            sm: &shadow_sm,
            vs: "vs_object",
            fs: None,
            buffers: &object_buffers,
            depth: Some(shadow_depth),
            target: None,
        });

        // Geometrie partagee.
        let indices = build_indices();
        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("chunk indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let coarse_indices = crate::world::chunk::build_coarse_indices();
        let coarse_index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("coarse chunk indices"),
            contents: bytemuck::cast_slice(&coarse_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let chunk_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk instances"),
            size: (MAX_CHUNK_INSTANCES * std::mem::size_of::<ChunkInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let meshes = build_object_meshes();
        let object_vertices = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("object meshes"),
            contents: bytemuck::cast_slice(&meshes.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let object_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object instances"),
            size: (MAX_OBJECT_INSTANCES * std::mem::size_of::<ObjectInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let car_model = crate::vehicle::CarModel::new(0);
        let car_vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("car vertices"),
            size: (car_model.vertices.len() * std::mem::size_of::<CityVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let car_indices = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("car indices"),
            contents: bytemuck::cast_slice(&car_model.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let targets = Self::make_targets(&device, &config, &sized_layouts);
        let ui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

        Ok(Self {
            device,
            queue,
            surface,
            config,
            adapter_name,
            terrain_pipeline,
            object_pipeline,
            city_pipeline,
            shadow_city_pipeline,
            sky_pipeline,
            water_pipeline,
            clouds_march_pipeline,
            clouds_composite_pipeline,
            bloom_first_pipeline,
            bloom_down_pipeline,
            bloom_up_pipeline,
            rays_pipeline,
            post_pipeline,
            shadow_terrain_pipeline,
            shadow_object_pipeline,
            globals_buffer,
            globals_group,
            weather_texture,
            shadow_read_group,
            empty_group,
            cascade_buffers,
            cascade_groups,
            cascade_views,
            sized_layouts,
            targets,
            index_buffer,
            index_count: indices.len() as u32,
            coarse_index_buffer,
            coarse_index_count: coarse_indices.len() as u32,
            chunk_instances,
            chunks: Default::default(),
            object_vertices,
            object_ranges: meshes.ranges,
            object_instances,
            scratch_instances: Vec::new(),

            car_model,
            car_scratch: Vec::new(),
            timings: [0.0; 4],
            gpu_timer,
            car_vertices,
            car_indices,
            ui_renderer,
        })
    }

    fn make_targets(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        layouts: &SizedLayouts,
    ) -> SizedTargets {
        let (w, h) = (config.width, config.height);
        let texture = |label, format, usage, w: u32, h: u32, mips: u32| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
                mip_level_count: mips,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let hdr_texture = texture("hdr", HDR_FORMAT, usage | wgpu::TextureUsages::COPY_SRC, w, h, 1);
        let hdr_view = hdr_texture.create_view(&Default::default());
        let depth_view = texture("depth", DEPTH_FORMAT, usage, w, h, 1).create_view(&Default::default());
        let scene_copy_texture = texture("scene copy", HDR_FORMAT, usage | wgpu::TextureUsages::COPY_DST, w, h, 1);
        let scene_copy_view = scene_copy_texture.create_view(&Default::default());
        let clouds_view = texture("clouds", HDR_FORMAT, usage, w / 2, h / 2, 1).create_view(&Default::default());
        let rays_view = texture("rays", HDR_FORMAT, usage, w / 4, h / 4, 1).create_view(&Default::default());

        let levels = BLOOM_LEVELS.min(32 - (w.min(h) / 2).max(1).leading_zeros());
        let bloom_texture = texture("bloom", HDR_FORMAT, usage, w / 2, h / 2, levels);
        let bloom_views: Vec<_> = (0..levels)
            .map(|level| {
                bloom_texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        let sampled = |view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layouts.texture,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&layouts.sampler) },
                ],
            })
        };
        let clouds_group = sampled(&clouds_view);
        let bloom_down: Vec<_> =
            (0..levels as usize).map(|i| sampled(if i == 0 { &hdr_view } else { &bloom_views[i - 1] })).collect();
        let bloom_up: Vec<_> = (0..levels as usize - 1).map(|i| sampled(&bloom_views[i + 1])).collect();
        let rays_group = sampled(&bloom_views[0]);

        let post_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post"),
            layout: &layouts.post,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&layouts.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&bloom_views[0]) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&rays_view) },
            ],
        });
        let depth_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene depth"),
            layout: &layouts.depth,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&depth_view) }],
        });
        let water_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water scene"),
            layout: &layouts.water,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&depth_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&scene_copy_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&layouts.sampler) },
            ],
        });
        SizedTargets {
            hdr_texture,
            hdr_view,
            depth_view,
            scene_copy_texture,
            scene_copy_view,
            clouds_view,
            clouds_group,
            bloom_views,
            bloom_down,
            bloom_up,
            rays_view,
            rays_group,
            post_group,
            depth_group,
            water_group,
        }
    }

    /// Adapte la surface et les cibles de rendu a la nouvelle taille de la fenetre, en pixels.
    /// Une taille nulle (fenetre reduite) est ignoree.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.targets = Self::make_targets(&self.device, &self.config, &self.sized_layouts);
    }

    /// Rapport largeur / hauteur de l'image.
    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height as f32
    }

    /// Oublie tous les chunks : la planete a change.
    pub fn clear_world(&mut self) {
        self.chunks.clear();
    }

    /// La nebulosite de la planete : `at` rend la part de ciel couverte (0 a 1) dans une direction.
    pub fn set_weather(&mut self, at: impl Fn(glam::DVec3) -> [f32; 4]) {
        let n = WEATHER_SIZE as usize;
        for face in 0..6u32 {
            let mut data = vec![0u8; n * n * 4];
            for y in 0..n {
                for x in 0..n {
                    // Convention des textures cubiques : u vers la droite, v vers le bas.
                    let (u, v) = ((x as f64 + 0.5) / n as f64 * 2.0 - 1.0, (y as f64 + 0.5) / n as f64 * 2.0 - 1.0);
                    let dir = match face {
                        0 => glam::DVec3::new(1.0, -v, -u),
                        1 => glam::DVec3::new(-1.0, -v, u),
                        2 => glam::DVec3::new(u, 1.0, v),
                        3 => glam::DVec3::new(u, -1.0, -v),
                        4 => glam::DVec3::new(u, -v, 1.0),
                        _ => glam::DVec3::new(-u, -v, -1.0),
                    };
                    let v = at(dir.normalize());
                    for c in 0..4 {
                        data[(y * n + x) * 4 + c] = (v[c].clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                }
            }
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.weather_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: face },
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(WEATHER_SIZE * 4),
                    rows_per_image: Some(WEATHER_SIZE),
                },
                wgpu::Extent3d { width: WEATHER_SIZE, height: WEATHER_SIZE, depth_or_array_layers: 1 },
            );
        }
    }

    /// Envoie au GPU les nouveaux chunks et libere ceux qui sont sortis du cache.
    pub fn sync_chunks(&mut self, update: StreamUpdate) {
        for mesh in update.new_chunks {
            self.upload_chunk(mesh);
        }
        for id in update.evicted {
            self.chunks.remove(&id);
        }
    }

    fn upload_chunk(&mut self, mesh: ChunkMesh) {
        let vertices = self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let water = mesh.water.as_ref().map(|w| {
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(w),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let city = mesh.city.as_ref().map(|c| GpuCity {
            vertices: self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("city vertices"),
                contents: bytemuck::cast_slice(&c.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            indices: self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("city indices"),
                contents: bytemuck::cast_slice(&c.indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            count: c.indices.len() as u32,
        });
        self.chunks.insert(
            mesh.id,
            GpuChunk {
                vertices,
                water,
                city,
                center: mesh.center,
                aabb_min: Vec3::from_array(mesh.aabb_min),
                aabb_max: Vec3::from_array(mesh.aabb_max),
            },
        );
    }

    /// Prepare les instances d'objets de l'image, relatives a la camera. Les cellules visibles sont
    /// reparties entre plusieurs threads ; chaque thread remplit ses propres listes.
    fn prepare_objects(
        &mut self,
        world: &Streamer,
        cam: DVec3,
        planes: &[Vec4; 5],
        cascades: &[Mat4; CASCADES],
    ) -> (Vec<ObjectDraw>, usize) {
        let (grass, flower, stalk) = (Kind::Grass as usize, Kind::Flower as usize, Kind::TallGrass as usize);
        let trees: Vec<&crate::world::scatter::CellObjects> = world
            .trees
            .values()
            .filter(|cell| {
                let offset = (cell.center - cam).as_vec3();
                offset.length() <= SHADOW_CELL_DISTANCE || sphere_in_frustum(planes, offset, CELL_CULL_RADIUS)
            })
            .collect();
        let herbs: Vec<&crate::world::scatter::GrassCell> = world
            .grass
            .values()
            .filter(|cell| sphere_in_frustum(planes, (cell.center - cam).as_vec3(), CELL_CULL_RADIUS))
            .collect();
        let tree_radius = world.config.tree_radius as f32;
        let grass_radius = world.config.grass_radius as f32;

        // Le travail se mesure en objets : on coupe les listes de cellules en parts a peu pres egales.
        let work: usize = trees.iter().map(|c| c.objects.iter().map(Vec::len).sum::<usize>()).sum::<usize>()
            + herbs.iter().map(|c| c.tufts.len() + c.flowers.len() + c.stalks.len()).sum::<usize>();
        let threads = if work < 20_000 { 1 } else { PREPARE_THREADS.min(work / 10_000).max(1) };
        if self.scratch_instances.len() < threads {
            self.scratch_instances.resize_with(threads, Scratch::default);
        }
        let scratch = &mut self.scratch_instances[..threads];
        let fill = |out: &mut Scratch,
                    trees: &[&crate::world::scatter::CellObjects],
                    herbs: &[&crate::world::scatter::GrassCell]| {
            for per_kind in out.iter_mut() {
                for list in per_kind {
                    list.clear();
                }
            }
            for cell in trees {
                for (kind, items) in cell.objects.iter().enumerate() {
                    add_instances(
                        &mut out[kind],
                        items,
                        cell.center,
                        cam,
                        tree_radius,
                        200.0,
                        [NEAR_MESH_DISTANCE, FAR_MESH_DISTANCE],
                        Some(cascades),
                    );
                }
            }
            for cell in herbs {
                add_instances(
                    &mut out[grass],
                    &cell.tufts,
                    cell.center,
                    cam,
                    grass_radius,
                    50.0,
                    HERB_MESH_DISTANCES,
                    None,
                );
                add_instances(
                    &mut out[flower],
                    &cell.flowers,
                    cell.center,
                    cam,
                    grass_radius,
                    50.0,
                    HERB_MESH_DISTANCES,
                    None,
                );
                add_instances(
                    &mut out[stalk],
                    &cell.stalks,
                    cell.center,
                    cam,
                    grass_radius,
                    50.0,
                    HERB_MESH_DISTANCES,
                    None,
                );
            }
        };
        if threads == 1 {
            fill(&mut scratch[0], &trees, &herbs);
        } else {
            let (tc, hc) = (trees.len().div_ceil(threads), herbs.len().div_ceil(threads));
            std::thread::scope(|scope| {
                for (t, out) in scratch.iter_mut().enumerate() {
                    let tr = &trees[(t * tc).min(trees.len())..((t + 1) * tc).min(trees.len())];
                    let hr = &herbs[(t * hc).min(herbs.len())..((t + 1) * hc).min(herbs.len())];
                    let fill = &fill;
                    scope.spawn(move || fill(out, tr, hr));
                }
            });
        }

        // Une plage contigue par sorte et niveau : les listes des threads se suivent, envoyees telles quelles
        // a leur place dans le tampon du GPU (pas de copie intermediaire).
        let size = std::mem::size_of::<ObjectInstance>() as u64;
        let mut count = 0usize;
        let mut draws = Vec::new();
        for kind in 0..KIND_COUNT {
            for slot in 0..3 * BANDS {
                let (lod, band) = (slot / BANDS, slot % BANDS);
                let total: usize = scratch.iter().map(|s| s[kind][slot].len()).sum();
                if total == 0 || count + total > MAX_OBJECT_INSTANCES {
                    continue;
                }
                let start = count;
                for s in scratch.iter() {
                    let list = &s[kind][slot];
                    if !list.is_empty() {
                        self.queue.write_buffer(
                            &self.object_instances,
                            count as u64 * size,
                            bytemuck::cast_slice(list),
                        );
                        count += list.len();
                    }
                }
                draws.push(ObjectDraw {
                    mesh: self.object_ranges[kind][lod].clone(),
                    instances: start as u32..count as u32,
                    casts_shadow: kind != grass && kind != flower && kind != stalk && lod < 2 && band < CASCADES,
                    band,
                });
            }
        }
        (draws, count)
    }

    /// Dessine une image ; si `screenshot` est fourni, enregistre l'image finale en PNG.
    pub fn render(
        &mut self,
        camera: &Camera,
        sky: &Sky,
        world: &Streamer,
        ui: UiFrame,
        screenshot: Option<&Path>,
        vehicle: Option<&crate::vehicle::VehicleFrame>,
    ) -> Result<FrameStats, wgpu::SurfaceError> {
        let t0 = std::time::Instant::now();
        if let Some(t) = &mut self.gpu_timer {
            t.begin_frame();
        }
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&Default::default());
        let t1 = std::time::Instant::now();

        let cam = camera.position;
        let up = camera.up().as_vec3();
        let view_proj = camera.view_proj(self.aspect());
        let sun = sky.sun_dir();
        let cascades = cascade_matrices(camera, sun);
        let wrapped = wrap_vec(cam);
        let altitude = cam.length() - planet_radius();

        let globals = Globals {
            view_proj,
            inv_view_proj: view_proj.inverse(),
            cascades,
            sun_dir: sun.extend(0.0),
            up: up.extend(altitude as f32),
            tangent: stable_tangent(up).extend(0.0),
            center: (-cam).as_vec3().extend(planet_radius() as f32),
            camera_wrap: Vec3::from_array(wrapped).extend(0.0),
            misc: Vec4::new(
                sky.elapsed,
                world.config.lod_factor as f32,
                self.config.width as f32,
                self.config.height as f32,
            ),
            camera_km: (cam * 0.001).as_vec3().extend((world.generator.params.seed % 1000) as f32 * 37.1),
            weather: Vec4::new(
                sky.weather.season,
                if sky.weather.sheltered { 0.0 } else { sky.weather.rain },
                sky.weather.snow,
                sky.weather.wet,
            ),
        };
        self.queue.write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));
        for (buffer, m) in self.cascade_buffers.iter().zip(&cascades) {
            self.queue.write_buffer(buffer, 0, bytemuck::bytes_of(m));
        }

        // Instances de chunks : l'index dans la liste sert d'instance de dessin.
        let mut drawn = Vec::with_capacity(world.selected.len());
        let mut instances = Vec::with_capacity(world.selected.len());
        for id in world.selected.iter().take(MAX_CHUNK_INSTANCES - 1) {
            let Some(chunk) = self.chunks.get(id) else { continue };
            let offset = (chunk.center - cam).as_vec3();
            instances.push(ChunkInstance {
                offset: offset.to_array(),
                size: id.size() as f32,
                wrap: wrap_vec(chunk.center),
            });
            drawn.push(DrawnChunk { id: *id, min: chunk.aabb_min + offset, max: chunk.aabb_max + offset });
        }
        // La voiture, posee a sa place : une instance de plus, apres les chunks.
        let car = match vehicle {
            Some(v) if (v.position - cam).length() < 5000.0 => {
                crate::vehicle::pose(&self.car_model, v, &mut self.car_scratch);
                self.queue.write_buffer(&self.car_vertices, 0, bytemuck::cast_slice(&self.car_scratch));
                instances.push(ChunkInstance {
                    offset: (v.position - cam).as_vec3().to_array(),
                    size: 1.0,
                    wrap: wrap_vec(v.position),
                });
                Some((instances.len() - 1) as u32)
            }
            _ => None,
        };
        let car_count = self.car_model.indices.len() as u32;
        if !instances.is_empty() {
            self.queue.write_buffer(&self.chunk_instances, 0, bytemuck::cast_slice(&instances));
        }
        let planes = frustum_planes(view_proj);
        let (object_draws, object_count) = self.prepare_objects(world, cam, &planes, &cascades);
        let t2 = std::time::Instant::now();

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Ombres.
        let cast_shadows = altitude < SHADOW_MAX_ALTITUDE;
        let timer = self.gpu_timer.as_ref();
        let ts = |section: usize, begin: bool, end: bool| timer.and_then(|t| t.writes(section, begin, end));
        for (c, cascade) in cascades.iter().enumerate() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.cascade_views[c],
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: ts(0, c == 0, c + 1 == cascades.len()),
                occlusion_query_set: None,
            });
            if !cast_shadows {
                continue;
            }
            pass.set_bind_group(0, &self.globals_group, &[]);
            pass.set_bind_group(1, &self.cascade_groups[c], &[]);
            pass.set_pipeline(&self.shadow_terrain_pipeline);
            // La cascade lointaine a des texels de pres de 1,5 m : la surface du LOD parent lui suffit.
            let (indices, index_count) = if c + 1 == CASCADES {
                (&self.coarse_index_buffer, self.coarse_index_count)
            } else {
                (&self.index_buffer, self.index_count)
            };
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(1, self.chunk_instances.slice(..));
            for (i, d) in drawn.iter().enumerate() {
                if aabb_in_light_box(*cascade, d.min, d.max) {
                    pass.set_vertex_buffer(0, self.chunks[&d.id].vertices.slice(..));
                    let i = i as u32;
                    pass.draw_indexed(0..index_count, 0, i..i + 1);
                }
            }
            pass.set_pipeline(&self.shadow_city_pipeline);
            for (i, d) in drawn.iter().enumerate() {
                if let Some(city) = &self.chunks[&d.id].city {
                    if aabb_in_light_box(*cascade, d.min, d.max) {
                        pass.set_vertex_buffer(0, city.vertices.slice(..));
                        pass.set_index_buffer(city.indices.slice(..), wgpu::IndexFormat::Uint32);
                        let i = i as u32;
                        pass.draw_indexed(0..city.count, 0, i..i + 1);
                    }
                }
            }
            if let Some(i) = car {
                pass.set_vertex_buffer(0, self.car_vertices.slice(..));
                pass.set_index_buffer(self.car_indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..car_count, 0, i..i + 1);
            }
            pass.set_pipeline(&self.shadow_object_pipeline);
            pass.set_vertex_buffer(0, self.object_vertices.slice(..));
            pass.set_vertex_buffer(1, self.object_instances.slice(..));
            for d in object_draws.iter().filter(|d| d.casts_shadow && d.band <= c) {
                pass.draw(d.mesh.clone(), d.instances.clone());
            }
        }

        // Scene opaque.
        let mut chunks_drawn = 0;
        let mut triangles = 0u64;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: ts(1, true, true),
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.globals_group, &[]);

            pass.set_pipeline(&self.sky_pipeline);
            pass.draw(0..3, 0..1);

            pass.set_bind_group(1, &self.shadow_read_group, &[]);
            pass.set_pipeline(&self.terrain_pipeline);
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(1, self.chunk_instances.slice(..));
            for (i, d) in drawn.iter().enumerate() {
                if aabb_in_frustum(&planes, d.min, d.max) {
                    pass.set_vertex_buffer(0, self.chunks[&d.id].vertices.slice(..));
                    let i = i as u32;
                    pass.draw_indexed(0..self.index_count, 0, i..i + 1);
                    chunks_drawn += 1;
                    triangles += self.index_count as u64 / 3;
                }
            }

            pass.set_pipeline(&self.city_pipeline);
            for (i, d) in drawn.iter().enumerate() {
                if let Some(city) = &self.chunks[&d.id].city {
                    if aabb_in_frustum(&planes, d.min, d.max) {
                        pass.set_vertex_buffer(0, city.vertices.slice(..));
                        pass.set_index_buffer(city.indices.slice(..), wgpu::IndexFormat::Uint32);
                        let i = i as u32;
                        pass.draw_indexed(0..city.count, 0, i..i + 1);
                        triangles += city.count as u64 / 3;
                    }
                }
            }
            if let Some(i) = car {
                pass.set_vertex_buffer(0, self.car_vertices.slice(..));
                pass.set_index_buffer(self.car_indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..car_count, 0, i..i + 1);
                triangles += car_count as u64 / 3;
            }

            pass.set_pipeline(&self.object_pipeline);
            pass.set_vertex_buffer(0, self.object_vertices.slice(..));
            pass.set_vertex_buffer(1, self.object_instances.slice(..));
            for d in &object_draws {
                pass.draw(d.mesh.clone(), d.instances.clone());
                triangles += (d.mesh.len() / 3 * d.instances.len()) as u64;
            }
        }

        // Nuages en volume, dans une image a demi-resolution : alpha garde leur transparence.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clouds"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.clouds_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: ts(2, true, true),
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.clouds_march_pipeline);
            pass.set_bind_group(0, &self.globals_group, &[]);
            pass.set_bind_group(1, &self.empty_group, &[]);
            pass.set_bind_group(2, &self.targets.depth_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // L'eau reflete la scene et les nuages : on les compose dans une copie.
        encoder.copy_texture_to_texture(
            self.targets.hdr_texture.as_image_copy(),
            self.targets.scene_copy_texture.as_image_copy(),
            wgpu::Extent3d { width: self.config.width, height: self.config.height, depth_or_array_layers: 1 },
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clouds in reflection"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.scene_copy_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth_view,
                    depth_ops: None,
                    stencil_ops: None,
                }),
                timestamp_writes: ts(3, true, true),
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.clouds_composite_pipeline);
            pass.set_bind_group(0, &self.globals_group, &[]);
            pass.set_bind_group(1, &self.targets.clouds_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Ocean et nuages : la profondeur est lue, mais pas ecrite.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("transparent"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth_view,
                    depth_ops: None,
                    stencil_ops: None,
                }),
                timestamp_writes: ts(4, true, true),
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.globals_group, &[]);
            pass.set_bind_group(1, &self.shadow_read_group, &[]);
            pass.set_bind_group(2, &self.targets.water_group, &[]);

            pass.set_pipeline(&self.water_pipeline);
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(1, self.chunk_instances.slice(..));
            for (i, d) in drawn.iter().enumerate() {
                let Some(water) = &self.chunks[&d.id].water else { continue };
                if aabb_in_frustum(&planes, d.min, d.max) {
                    pass.set_vertex_buffer(0, water.slice(..));
                    let i = i as u32;
                    pass.draw_indexed(0..GRID_INDEX_COUNT, 0, i..i + 1);
                }
            }

            pass.set_pipeline(&self.clouds_composite_pipeline);
            pass.set_bind_group(1, &self.targets.clouds_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Halo : reductions, rayons du soleil, puis agrandissements additionnes.
        fn fullscreen(
            encoder: &mut wgpu::CommandEncoder,
            label: &str,
            target: &wgpu::TextureView,
            load: wgpu::LoadOp<wgpu::Color>,
            timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
        ) -> wgpu::RenderPass<'static> {
            encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes,
                    occlusion_query_set: None,
                })
                .forget_lifetime()
        }
        let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
        for (i, source) in self.targets.bloom_down.iter().enumerate() {
            let mut pass =
                fullscreen(&mut encoder, "bloom down", &self.targets.bloom_views[i], clear, ts(5, i == 0, false));
            pass.set_pipeline(if i == 0 { &self.bloom_first_pipeline } else { &self.bloom_down_pipeline });
            pass.set_bind_group(0, &self.globals_group, &[]);
            pass.set_bind_group(1, source, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = fullscreen(&mut encoder, "rays", &self.targets.rays_view, clear, None);
            pass.set_pipeline(&self.rays_pipeline);
            pass.set_bind_group(0, &self.globals_group, &[]);
            pass.set_bind_group(1, &self.targets.rays_group, &[]);
            pass.set_bind_group(2, &self.targets.depth_group, &[]);
            pass.draw(0..3, 0..1);
        }
        for (i, source) in self.targets.bloom_up.iter().enumerate().rev() {
            let mut pass = fullscreen(
                &mut encoder,
                "bloom up",
                &self.targets.bloom_views[i],
                wgpu::LoadOp::Load,
                ts(5, false, i == 0),
            );
            pass.set_pipeline(&self.bloom_up_pipeline);
            pass.set_bind_group(0, &self.globals_group, &[]);
            pass.set_bind_group(1, source, &[]);
            pass.draw(0..3, 0..1);
        }

        // Sortie.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: ts(6, true, false),
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.post_pipeline);
            pass.set_bind_group(0, &self.globals_group, &[]);
            pass.set_bind_group(1, &self.targets.post_group, &[]);
            pass.set_bind_group(2, &self.targets.depth_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Interface, par-dessus tout le reste.
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: ui.pixels_per_point,
        };
        for (id, delta) in &ui.textures.set {
            self.ui_renderer.update_texture(&self.device, &self.queue, *id, delta);
        }
        let ui_commands =
            self.ui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &ui.primitives, &screen);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: ts(6, false, true),
                occlusion_query_set: None,
            });
            self.ui_renderer.render(&mut pass.forget_lifetime(), &ui.primitives, &screen);
        }
        for id in &ui.textures.free {
            self.ui_renderer.free_texture(id);
        }

        let capture = screenshot.and_then(|path| self.encode_capture(&mut encoder, &frame.texture).map(|b| (path, b)));
        if let Some(t) = &self.gpu_timer {
            t.resolve(&mut encoder);
        }
        let t3 = std::time::Instant::now();
        self.queue.submit(ui_commands.into_iter().chain([encoder.finish()]));
        if let Some(t) = &mut self.gpu_timer {
            t.after_submit();
            self.device.poll(wgpu::Maintain::Poll);
        }
        if let Some((path, buffer)) = capture {
            match self.save_capture(&buffer, path) {
                Ok(()) => println!("capture : {}", path.display()),
                Err(e) => eprintln!("capture impossible : {e}"),
            }
        }
        frame.present();
        let t4 = std::time::Instant::now();
        for (acc, (a, b)) in self.timings.iter_mut().zip([(t0, t1), (t1, t2), (t2, t3), (t3, t4)]) {
            *acc += (b - a).as_secs_f64();
        }

        Ok(FrameStats { chunks_drawn, triangles, objects: object_count })
    }

    fn encode_capture(&self, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) -> Option<wgpu::Buffer> {
        if !self.config.usage.contains(wgpu::TextureUsages::COPY_SRC) {
            eprintln!("cette surface ne permet pas les captures d'ecran");
            return None;
        }
        let bytes_per_row = (self.config.width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture"),
            size: bytes_per_row as u64 * self.config.height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d { width: self.config.width, height: self.config.height, depth_or_array_layers: 1 },
        );
        Some(buffer)
    }

    /// Relit l'image copiee dans `buffer` et l'ecrit en PNG.
    fn save_capture(&self, buffer: &wgpu::Buffer, path: &Path) -> crate::Result<()> {
        let slice = buffer.slice(..);
        // Le resultat de la projection en memoire arrive pendant `poll` : sans lui, lire la plage
        // d'un tampon dont la projection a echoue ferait paniquer.
        let (sent, mapped) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sent.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        mapped.recv().unwrap_or(Err(wgpu::BufferAsyncError))?;

        let (w, h) = (self.config.width as usize, self.config.height as usize);
        let stride = (w * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        let bgra = matches!(self.config.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
        let mapped = slice.get_mapped_range();
        let mut rgba = Vec::with_capacity(w * h * 4);
        for row in mapped.chunks(stride).take(h) {
            for px in row[..w * 4].as_chunks::<4>().0 {
                if bgra {
                    rgba.extend_from_slice(&[px[2], px[1], px[0], 255]);
                } else {
                    rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
                }
            }
        }
        drop(mapped);
        buffer.unmap();

        crate::write_png(path, self.config.width, self.config.height, png::ColorType::Rgba, &rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globals_size_matches_the_wgsl_struct() {
        // 5 mat4 (2 + 3 cascades) + 8 vec4 : a garder aligne avec common.wgsl.
        assert_eq!(std::mem::size_of::<Globals>(), 5 * 64 + 8 * 16);
    }

    #[test]
    fn frustum_keeps_what_is_ahead_and_rejects_what_is_behind() {
        let cam = Camera::new(DVec3::new(6.4e6, 0.0, 0.0), 0.0, 0.0);
        let f = cam.forward();
        let planes = frustum_planes(cam.view_proj(1.5));
        let box_at = |d: f32| (f * d - Vec3::ONE, f * d + Vec3::ONE);
        let (a, b) = box_at(50.0);
        assert!(aabb_in_frustum(&planes, a, b));
        let (a, b) = box_at(500_000.0);
        assert!(aabb_in_frustum(&planes, a, b), "pas de plan lointain");
        let (a, b) = box_at(-50.0);
        assert!(!aabb_in_frustum(&planes, a, b));
    }

    #[test]
    fn cascades_contain_the_camera_and_stay_snapped_to_texels() {
        let light = Vec3::new(0.4, 0.8, 0.3).normalize();
        let mut cam = Camera::new(DVec3::new(6.0e6, 40.0, -2.0e6), 0.0, 0.0);
        let a = cascade_matrices(&cam, light);
        for m in &a {
            let p = m.project_point3(Vec3::ZERO);
            assert!(p.x.abs() <= 1.0 && p.y.abs() <= 1.0 && (0.0..=1.0).contains(&p.z));
        }
        // Un deplacement infime ne doit pas decaler la grille de la cascade la plus large.
        cam.position.x += 0.001;
        let b = cascade_matrices(&cam, light);
        let texel = 2.0 / SHADOW_SIZE as f32;
        let (pa, pb) = (a[2].project_point3(Vec3::ZERO), b[2].project_point3(Vec3::ZERO));
        assert!((pa.x - pb.x).abs() < texel && (pa.y - pb.y).abs() < texel);
    }

    #[test]
    fn tangent_is_stable_when_only_the_gaze_changes() {
        let up = Vec3::new(0.3, 0.5, 0.8).normalize();
        let t = stable_tangent(up);
        assert!(t.dot(up).abs() < 1e-5);
        assert!((stable_tangent(Vec3::Y).length() - 1.0).abs() < 1e-5, "defini au pole");
    }
}
