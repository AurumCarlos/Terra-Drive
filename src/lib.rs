//! Terra : moteur 3D pour planetes entieres generees proceduralement.
//!
//! - `world` : generation deterministe sur la sphere (relief, biomes, vegetation) et streaming.
//! - `render` : rendu wgpu (ombres en cascade, HDR, atmosphere, ocean, nuages).
//! - `ui` : menu et affichage tete haute.
//! - `app` : boucle de jeu et trait [`app::Game`] a implementer par le jeu.

pub mod app;
pub mod audio;
pub mod camera;
pub mod error;
pub mod input;
pub mod player;
pub mod render;
pub mod sky;
pub mod sync;
pub mod ui;
pub mod vehicle;
/// Generation de la planete : relief, climat, eau, villes, routes, et streaming autour de la camera.
pub mod world;

pub use app::{run, Context, EngineConfig, Game, GameEvent};
pub use camera::Camera;
pub use error::{write_png, Error, Result};
pub use input::Input;
pub use player::PlayerController;
pub use sky::Sky;
pub use world::generator::{planet_radius, Generator, PlanetParams};
pub use world::streaming::Streamer;
