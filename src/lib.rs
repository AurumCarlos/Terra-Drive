//! Terra : moteur 3D pour planetes entieres generees proceduralement.
//!
//! - `world` : generation deterministe sur la sphere (relief, biomes, vegetation) et streaming.
//! - `render` : rendu wgpu (ombres en cascade, HDR, atmosphere, ocean, nuages).
//! - `ui` : menu et affichage tete haute.
//! - `app` : boucle de jeu et trait [`app::Game`] a implementer par le jeu.

pub mod app;
pub mod audio;
pub mod camera;
pub mod input;
pub mod player;
pub mod render;
pub mod sky;
pub mod ui;
pub mod vehicle;
pub mod world;

pub use app::{run, Context, EngineConfig, Game, GameEvent};
pub use camera::Camera;
pub use input::Input;
pub use player::PlayerController;
pub use sky::Sky;
pub use world::generator::{Generator, PlanetParams, planet_radius};
pub use world::streaming::Streamer;
