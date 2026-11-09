//! Temps passe par la carte graphique dans chaque groupe de passes (TERRA_PROFILE=1), par horodatages.
//! La lecture est asynchrone : on n'ecrit de nouveaux horodatages que lorsque les precedents sont lus,
//! donc jamais d'attente du GPU.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// Groupes de passes mesures, dans l'ordre de l'image.
pub const SECTIONS: [&str; 7] = ["ombres", "scene", "nuages", "reflets", "eau", "halo", "sortie"];
const QUERIES: u32 = SECTIONS.len() as u32 * 2;

/// Etats du tampon de lecture.
const FREE: u8 = 0;
const PENDING: u8 = 1;
const MAPPED: u8 = 2;

pub struct GpuTimer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: Arc<wgpu::Buffer>,
    state: Arc<AtomicU8>,
    /// Nanosecondes par tic.
    period: f32,
    /// Vrai si les passes de cette image ecrivent leurs horodatages.
    active: bool,
    /// Millisecondes cumulees par section, et nombre d'images lues.
    pub totals: [f64; SECTIONS.len()],
    pub frames: u32,
}

impl GpuTimer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let size = QUERIES as u64 * 8;
        Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("gpu timer"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERIES,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu timer resolve"),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu timer readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            state: Arc::new(AtomicU8::new(FREE)),
            period: queue.get_timestamp_period(),
            active: false,
            totals: [0.0; SECTIONS.len()],
            frames: 0,
        }
    }

    /// Debut d'image : recupere les mesures lues, et decide si cette image en ecrit de nouvelles.
    pub fn begin_frame(&mut self) {
        if self.state.load(Ordering::Acquire) == MAPPED {
            {
                let data = self.readback.slice(..).get_mapped_range();
                let ticks: &[u64] = bytemuck::cast_slice(&data);
                // Une section sans passe ce frame-ci (ou un compteur remis a zero) se lit 0 et ne compte pas.
                for (i, total) in self.totals.iter_mut().enumerate() {
                    let (a, b) = (ticks[i * 2], ticks[i * 2 + 1]);
                    if b > a {
                        *total += (b - a) as f64 * self.period as f64 * 1e-6;
                    }
                }
                self.frames += 1;
            }
            self.readback.unmap();
            self.state.store(FREE, Ordering::Release);
        }
        self.active = self.state.load(Ordering::Acquire) == FREE;
    }

    /// Horodatages d'une passe : debut et/ou fin de la section donnee.
    pub fn writes(&self, section: usize, begin: bool, end: bool) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        (self.active && (begin || end)).then(|| wgpu::RenderPassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: begin.then_some(section as u32 * 2),
            end_of_pass_write_index: end.then_some(section as u32 * 2 + 1),
        })
    }

    /// Fin d'enregistrement : copie les horodatages vers le tampon de lecture.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.active {
            encoder.resolve_query_set(&self.queries, 0..QUERIES, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, QUERIES as u64 * 8);
        }
    }

    /// Apres l'envoi : demande la lecture, qui arrivera quelques images plus tard.
    pub fn after_submit(&mut self) {
        if !self.active {
            return;
        }
        self.state.store(PENDING, Ordering::Release);
        let state = self.state.clone();
        self.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            state.store(if r.is_ok() { MAPPED } else { FREE }, Ordering::Release);
        });
        self.active = false;
    }

    /// Millisecondes moyennes par section depuis la derniere remise a zero, puis remise a zero.
    pub fn take(&mut self) -> Option<[f64; SECTIONS.len()]> {
        if self.frames == 0 {
            return None;
        }
        let n = self.frames as f64;
        let out = self.totals.map(|t| t / n);
        self.totals = [0.0; SECTIONS.len()];
        self.frames = 0;
        Some(out)
    }
}
