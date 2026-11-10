# Terra

A 3D engine in Rust that generates entire Earth-sized planets, with a demo: walk, fly, climb all the way
into space, come back down without a loading screen, and create a new planet from a menu.

## Running

```powershell
.\cargo.ps1 run --release
```

`cargo.ps1` adds Rust and MinGW to the PATH and builds into `%LOCALAPPDATA%\terra-target`.
It expects Rust (toolchain `stable-x86_64-pc-windows-gnu`) and WinLibs MinGW, installed via winget.

## Controls

Keys are physical positions on a QWERTY keyboard: on AZERTY, W A S D is Z Q S D and M is the comma key.

| Key | Action |
| --- | --- |
| Left click | Capture the mouse |
| Escape | Open / close the menu |
| W A S D | Move |
| F | Toggle walking / flying |
| Mouse wheel, Page Up / Down | Set flying speed, from 2 m/s to 30,000 km/s |
| Shift | Run (x8 speed when flying) |
| Space | Jump, or go up when flying |
| Ctrl | Go down when flying |
| T / G | Move time forward / backward |
| F12 | PNG screenshot |
| M | Mute / unmute sound |
| E | Get in a car (your own if it is within 12 m, otherwise a new one parked in the nearest street or highway); get out when stopped |

While driving:

| Key | Action |
| --- | --- |
| W / S | Accelerate / brake, then reverse |
| A / D | Steer |
| Space | Handbrake (makes the car drift) |
| C | Chase view or driver view |
| L | Headlights |
| R | Put the car back on its wheels (when stopped) |
| Mouse | Look around |

A gauge in the bottom right shows speed, RPM and gear.

To go to space: F, a few notches of the mouse wheel, look at the sky and move forward.
The "Go to space" button in the menu takes you there directly.

## Menu

- **New random planet**, or **Generate with this seed**: the seed can be a number or text.
- Settings for the next planet: ocean share, relief height, temperature, **size** (0.25 to 2 times the Earth),
  **number of cities** (0 to 400) and **rivers** (uncheck for a planet without rivers).
- **Render distance**, from 5 to 1000 km (80 by default), applied immediately and kept from one planet to the next.
  Beyond it, the terrain quickly loses detail. It also sets the vegetation range: up to 6 km for trees,
  320 m for grass. At startup: `--render-distance 200`.
- **Go to space** and **Back to the surface**.
- **Go to the nearest city**: arrive flying, 2 km south of the center. The HUD shows the name, the map and the direction of the nearest city.
- Planet **map**: terrain, rivers and lakes, plate boundaries as an option. Clicking on it sends the player flying over that spot.

Simulating a new planet takes a few seconds in the background: the game keeps running on the old one in the meantime.

From space, the camera stays where it is and the new planet appears below.
Near the ground, it is moved to a starting point on the new planet.

## Options

`terra-demo --help` lists everything. Values containing a comma must be quoted in PowerShell. Examples:

```powershell
# Planet 3, seen from space at noon
.\cargo.ps1 run --release -- --seed 3 --altitude 13000000 --pitch -89 --time 0.5

# Above a specific spot, at sunrise
.\cargo.ps1 run --release -- --pos "48.85,2.35" --altitude 2000 --time 0.28

# Automatic screenshot once the world has loaded
.\cargo.ps1 run --release -- --seed 7 --screenshot view.png

# A small planet, 60 cities, no rivers
.\cargo.ps1 run --release -- --planet-size 0.5 --cities 60 --no-rivers

# Driving, on the highway leaving Pallac (planet 3); --cockpit for the driver view
.\cargo.ps1 run --release -- --seed 3 --drive --pos "-48.1126,48.5451" --yaw 93

# On the starting grid of the circuit closest to the arrival point (Sochi Autodrom near Venolport, planet 3)
.\cargo.ps1 run --release -- --seed 3 --circuit

# At the foot of the nearest mountain road: here the one at Toranbourg, 13 hairpins (planet 3)
.\cargo.ps1 run --release -- --seed 3 --pos "-53.9113,81.1318" --montagne

# Descent from orbit down to the ground, prints frame times
.\cargo.ps1 run --release -- --no-vsync --altitude 13000000 --pitch -89 --autopilot 440000
```

## Writing a game

Implement `terra::Game`, then call `terra::run`:

```rust
use terra::{planet_radius, Camera, Context, EngineConfig, Game, PlanetParams, PlayerController};

struct MyGame { player: PlayerController }

impl Game for MyGame {
    fn start(&mut self, ctx: &mut Context) {
        let dir = ctx.world.find_spawn();
        let radius = planet_radius() + ctx.world.height(dir) as f64 + 2.0;
        *ctx.camera = Camera::new(dir * radius, 0.0, 0.0);
    }
    fn update(&mut self, ctx: &mut Context, dt: f32) {
        self.player.update(ctx.input, ctx.camera, ctx.streamer, dt);
    }
}

fn main() {
    let planet = PlanetParams { seed: 42, ..Default::default() };
    let game = MyGame { player: PlayerController::default() };
    terra::run(EngineConfig { planet, ..Default::default() }, game).unwrap();
}
```

`Context` gives access to input, the camera, the sky, the planet function
(height, climate, biome for a direction), loaded objects and regeneration (`ctx.regenerate`).

## Architecture

| File | Role |
| --- | --- |
| `world/noise.rs` | 3D gradient noise with analytic derivatives, in f64 |
| `world/cube.rs` | Global grids on the cube-sphere, seamlessly interpolated fields |
| `world/tectonics.rs` | Plate simulation: drift, collisions, subduction, mid-ocean ridges |
| `world/landforms.rs` | Erosion filter, landscape provinces, volcanoes, winds and moisture |
| `world/hydrology.rs` | Erosion, drainage, lakes and river network |
| `world/generator.rs` | Function from (parameters, direction) to terrain, water, climate, biome, color |
| `world/chunk.rs` | Quadtree on the 6 faces of a cube-sphere, meshes (morphing, skirts), water surface |
| `world/city.rs` | Cities: placement on the planet, street plan (grid, blocks, star, old town), blocks, lots, buildings, collisions |
| `world/highway.rs` | Highways between cities: neighborhood graph, A* routing that avoids the sea and other highways, bridges, viaducts and overpasses, longitudinal profile, ground reshaping |
| `world/mountain.rs` | Mountain roads: target summit, switchback routing by slope-limited A* (32 headings, hairpins), smoothing, longitudinal profile, ground reshaping, spatial index |
| `world/circuit.rs`, `world/circuit_data.rs` | Circuits: 40 real F1 layouts, placed near each city on flat dry land, access road, longitudinal profile, run-off areas, spatial index |
| `world/city_mesh.rs` | Meshes for buildings, streets, bridges and street furniture, per level of detail |
| `world/scatter.rs` | Deterministic placement of vegetation and rocks |
| `world/streaming.rs` | LOD selection, horizon, generation threads, cache, collisions |
| `render/mod.rs` | wgpu rendering: shadows, HDR scene, ocean, clouds, output, UI |
| `render/meshes.rs` | Procedurally generated trees, grass and rocks |
| `render/noise_tex.rs` | Periodic 3D noise texture, for clouds, terrain and water |
| `render/shaders/` | WGSL shaders, including `city.wgsl`: facades, windows, roofs, roadway, night lighting, and the car drawn per pixel (panel gaps, handles, lights, grille, plates, tires, gauges) |
| `vehicle/model.rs` | Procedurally generated car (over 20,000 vertices): body smoothed over 161 sections, wheel arches, tires, 10-spoke rims, discs and calipers, mirrors, wipers, exhausts, interior (seats, dashboard, gauges, screen, steering wheel) |
| `vehicle/dynamics.rs` | Vehicle dynamics: rigid 3D body, suspension (springs, dampers, bump stops, anti-roll bars), Pacejka tires with relaxation, engine, 7-speed automatic gearbox, differential, ABS, traction control, aerodynamics, body contact with the ground |
| `vehicle/mod.rs` | Driving: controls, ground plane under each wheel (terrain, highway decks and city bridges, water, surface type), impacts against buildings and trees, body dents, flipping back onto the wheels, cameras |
| `audio.rs` | Sound synthesized live (cpal): engine, tires, rolling, wind, impacts; wind, foliage, birds, crickets, waves, river, city |
| `world/fast_hash.rs` | Fast hash table (FxHash) for node and cell identifiers |
| `ui.rs` | Menu and HUD, with egui |
| `app.rs`, `camera.rs`, `input.rs`, `player.rs`, `sky.rs` | Game loop, API and controller |
| `bin/audit.rs`, `bin/probe.rs`, `tour.ps1` | Visual quality checks: statistics, location probe (`probe SEED LAT LON`), screenshot contact sheets (build before `tour.ps1`) |

Main design choices:

- **One simulation, then a pure function.** Radius of 6,371 km. Creation computes the global grids once; after that, everything is derived from these grids and from noise, and the same direction always gives the same result.
- **Tectonics.** 10 to 19 rigid plates rotate around their Euler pole for 48 steps. Continental crust collides and thickens, oceanic crust sinks under continents, mid-ocean ridges create new crust. Mountain ranges come from this uplift.
- **Erosion.** A stream power model carves the drainage basins at 39 then 20 km resolution.
- **Lakes.** Depression filling on a 13 km grid gives their level and outlet. Extra basins are added in cold regions and at the foot of mountain ranges. Every lake overflows into a river.
- **Rivers.** Accumulated rainfall determines the rivers, which start earlier at high altitude and flow down to the sea or a lake. Discharge sets their width. Their course meanders around the grid and a slope cone carves their valley.
- **Seamless.** Noise is evaluated in 3D on the sphere and grids are stitched along the edges of the cube.
- **Erosion without simulation.** Rune Skovbo Johansen's gully filter, adapted to the sphere: stripes aligned with the slope, in several branching octaves, draw ridges and valleys at every point, without neighbors or passes.
- **Landscape provinces.** Each region gets a signature based on its climate: hills, layered plateaus and canyons, badlands, dunes, karst towers, lake-dotted moors, monoliths, glacial valleys and fjords. On top of that come about 150 volcanoes, along subduction zones and in island chains.
- **Wind-driven climate.** Moisture leaves the oceans, follows the prevailing winds and falls on high ground: deserts form behind mountain ranges.
- **Terrain at every scale.** The terrain spectrum roughly follows the Earth's (amplitude proportional to wavelength): broad 9 km undulations, 2.4 km hills, 420 m mounds, 95 m detail. Undulations and hills use noise damped by its own slope (Inigo Quilez): soft valley floors, sharp ridges, as after erosion. A 70 km regional noise alternates rare true plains with bumpy country.
- **Alpine mountains.** On top of the large 26 km ridges come secondary 6.5 km arêtes and peaks, stronger at altitude (Musgrave's hybrid multifractal).
- **Readable forests.** Large clearings, stands of broadleaf or conifer trees, light and dark patches in the canopy, and bare ridges that reveal the terrain under the trees. Distant meshes and trees follow the same rules.
- **Living plains.** A ground swell over all land, isolated buttes as landmarks, large swathes of golden and green grass, and monoliths with bare rock walls.
- **Forests to the horizon.** Beyond tree range, the ground takes the canopy's color, following the same rules as tree placement.
- **CDLOD.** 18 levels, from 1.5 m to over 200 km between two vertices. Nodes hidden by the horizon are neither generated nor drawn.
- **f64 coordinates.** The camera is in f64 and the GPU works relative to it, so there is no jitter at the surface.
- **Flight.** The view follows the curvature near the ground, then keeps a straight path above 300 km altitude.
- **Background generation.** A parent stays displayed until its visible children are ready, so there are no holes.
- **Rendering.** Single-scattering atmosphere computed per pixel, valid from the ground to orbit, sun based on longitude, 3 shadow cascades near the ground, HDR, bloom, sun rays, FXAA. Tone mapping blends ACES with a luminance curve so sunsets stay orange.
- **Volumetric clouds.** Half-resolution ray marching through a 3D noise computed at startup, with high-altitude veils. Flat bases, silver linings facing the sun, shadows cast on terrain and water.
- **Water.** Waves at four scales, screen-space reflections (mountains and clouds) then of the computed sky, sun glitter path, foam along shores.
- **Terrain.** Color and relief added per pixel, from pebbles to massifs: the distance keeps its texture, and forests their treetops.
- **Vegetation.** Curved grass, tall seed-headed grass that glows against the light, jagged firs, foliage cut out per pixel.
- **Cities.** About 140 cities on planet 3, set on dry, gently sloping land, often by the water. Four plans inspired by real cities: grid with wide avenues (Manhattan, 1811 plan), chamfered blocks (Cerdà's Eixample in Barcelona), star of radiating avenues with a business district at the end of the axis (Paris and La Défense), old town with organic streets (segment-based growth, after Parish and Müller 2001). Blocks are subdivided into lots, in the manner of Vanegas et al. 2012. Buildings thin out toward the edge: lobes, indentations, hamlets and houses along the outgoing roads.
- **Highways.** Neighboring cities are connected by a relative neighborhood graph (Toussaint 1980), completed to reach isolated pieces. The route avoids relief, lakes and cities, crosses shallow sea on long bridges, and keeps away from other highways; crossings become overpasses. Smooth longitudinal profile, cuts and fills, viaducts on piers, concrete barriers, gantries, 2 or 3 lanes per direction depending on city size.
- **Intersections.** A street meeting another at an angle stops along the edge of the crossing roadway, with no triangle of dirt between them; a test checks there is asphalt on both sides of every street end.
- **Connected streets.** Each city plan is checked: all its streets form a single network (contacts are computed geometrically), and every highway end arrives on a street. An isolated street is reconnected by the shortest path among the discarded streets, or removed with its block. `planet_stats` checks every plan on a planet.
- **Car physics.** In the spirit of simulators like BeamNG, nothing is animated: everything comes out of the computation. The body is a rigid 3D body (1,550 kg, pitch, yaw and roll inertia) carried by four independent suspensions: spring of about 1.5 Hz, damper stiffer in rebound than in compression, bump stop, anti-roll bars. Each tire's load is the force of its suspension, so dive under braking, squat, body roll in corners, wheels lifting off, jumps and landings all emerge on their own. The tires follow the Pacejka magic formula for combined slip, with a relaxation length: the tire deforms before it slides, which also holds the car still on a slope. The ground is a plane probed under each wheel (terrain, streets, bridges, highways, circuits, water), with its grip and rolling resistance. If the body touches the ground (rollover, big jump), its panels scrape and bounce; R puts the car back on its wheels. Impacts against buildings and trees are impulses at the contact point (a hit on a corner makes the car spin) and leave dents in the bodywork. 470 Nm engine, 7-speed automatic gearbox, limited-slip differential, ABS, traction control, handbrake, drag and downforce. Integrated at 240 Hz. Measured by tests: 0 to 100 km/h in 4.9 s, braking from 100 to 0 in 35 m at most, about 1.2 g in corners (very grippy tires), one oscillation then rest after a drop.
- **F1 circuits.** Near each city with enough room, a circuit with a real Formula 1 layout (Monza, Spa, Suzuka, Silverstone, Interlagos, Baku...), at exact scale, set on dry, gently sloping land, away from highways. An access road leaves from one of the city's avenues. Track with white lines and rubber marks, red and white kerbs in corners, gravel traps outside fast bends, grass run-off areas, barriers, checkered finish line, starting grid, covered grandstand and pit wall. Gradients capped at 12%. Layouts taken from [bacinger/f1-circuits](https://github.com/bacinger/f1-circuits) (MIT license).
- **Mountain roads.** Each city at the foot of high ground (about thirty on planet 3) has its road climbing to the nearest summit at least 350 m higher, ending in a turning loop around a patch of grass. The route is found by A* on a 40 m grid where each state is a cell and one of 32 headings, in the spirit of Galin et al. 2010: the road only turns one notch per step and never climbs more than 10%; where the slope is too steep, the only way up is a hairpin, a U-turn offset by one or two cells, followed by at least 120 m of straight road. Switchbacks therefore stack up by themselves on steep slopes, and the road stays straight elsewhere. A road that crosses itself is searched again avoiding that spot. Longitudinal profile capped at 12%, cuts and fills, two narrow lanes with a dashed center line, steel guardrails or stone parapets on the drop side.
- **Car.** A sedan modeled in code and drawn per pixel, posed every frame: wheels that spin and steer, steering wheel, suspension, brake lights, reverse lights, turn signals and headlights.
- **Open forests.** 50 to 80% cover in the heart of stands, open woods at the edges, gaps of a few tens of meters, large clearings over a third of the land. Stand age varies tree size; isolated trees are taller.
- **Distinct cities.** Each city has its own character: low and sprawling or tall and dense, and a dominant palette (white render, ochre, brick, pastels; red tiles or slate). L-shaped houses, bungalows and villas, small apartment blocks among the houses; towers on podiums, stepped or twin; an irregular skyline.
- **Buildings.** Houses with garage, pool and hedge; townhouses with shutters; Haussmann-style buildings with balconies and mansard roofs; slab blocks and residences; offices; four families of skyscrapers (art deco, glass, octagon, spindle); warehouses, tanks, chimneys, churches, squares and fountains. Windows, frames, shop windows and signs are drawn per pixel; at night, some of the windows light up.
- **Streets.** Roadway with lanes and markings, sidewalks and curbs, crosswalks, street lights, street trees, parked cars, bridges with parapets and piers over water. From afar, only the silhouette of the towers remains.
- **Arrival.** The player appears on an open shore, facing the setting sun: it stays fixed 5 degrees above the horizon until you change the time with T and G, or launch with `--day-length 1800`.

## Sound

No audio files: everything is computed live by `audio.rs`, on the default sound card (`--mute` or M to turn it off).

- **Car.** The engine follows the RPM and load from the dynamics: ignition harmonics of a V8 (one cycle every two revolutions, order 8 for ignition, order 4 for the rumble), exhaust noise on each firing, a filter that opens with load, intake whoosh, rev limiter, backfires on lift-off. On top of that come the clunk of gear shifts, tire squeal when they exceed their peak grip (on asphalt only), rolling noise and gravel off the road, wind rising with speed, and impacts. In the driver view, the cabin muffles the highs.
- **Ambience.** Computed around the camera four times per second: gusty wind (stronger at altitude), foliage and birds near trees during the day, crickets on warm nights, waves on a shore, running water near a river or lake, distant hum and horns in the city, rush of speed when flying. Ground sounds fade as you gain height.
- **Cost.** Under 4% of one core with all sources at once (measured by a test). The audio thread never waits for the game.
- **Listen without launching the game.** A 48 s demo (standing start with the real dynamics, drift, crash, woods, night, shore, city):

```powershell
$env:TERRA_WAV = "$PWD\sound_demo.wav"; .\cargo.ps1 test --release --lib write_a_listening_demo
```

## Performance

`TERRA_PROFILE=1` prints every 5 s the time of each frame phase (game, streaming, chunk upload, UI, rendering), streaming details (chunks selected, loaded, generated), the GPU time of each pass (shadows, scene, clouds, reflections, water, bloom, output) and, at startup, each step of planet creation (water, cities, highways, city plans). `--no-vsync --autopilot SPEED` runs a 30 s measurement flight and reports frames longer than 12 ms.

`gen_bench [SEED]` measures generation alone: planet creation, city plans, chunks at the start and in a city per level of detail, trees and grass, per-step detail of a chunk, and terrain probes on 1 core then on all of them.

```powershell
.\cargo.ps1 run --release --bin gen_bench -- 3
```

Measurements (planet 3, Ryzen 5 9600X, RX 9070 XT, no vsync):

| Measurement | Start | Before this pass | After |
| --- | --- | --- | --- |
| City flyover at 40 m/s, median frame | 4.49 ms | 2.59 ms | 2.18 ms |
| City flyover, 99th percentile | | 3.45 ms | 2.73 ms |
| Fast flight at 250 m/s, median frame | 5.41 ms | 2.65 ms | 1.95 ms |
| Fast flight, 99th percentile | | 5.3 ms | 3.0 ms |
| Average chunk (1 core) | | 32.4 ms | 19.4 ms |
| Speedup on 12 threads | | x2.4 | x7.6 |
| Chunks generated in 5 s of play | | 920 | 1526 |
| Planet creation | | 8.0 s | 3.6 s |
| City plans | | 2.2 s | 0.36 s |
| GPU shadows (1080p, 60 m) | | 0.93 ms | 0.5 ms |

- The planet radius was read from a thread-local variable on every noise sample; on Windows (MinGW) this read cost 14 ns and kept the computation from scaling across cores. An atomic counter avoids it when no planet is being created.
- The noise computes its 8 corners only once and has a variant without derivatives; province fields share the cube projection; a vertex's colors share a single site computation.
- Gully slope is interpolated on a fixed 175 m grid (mean deviation of 0.8° from the exact slope) and cached across chunks.
- River index, city plans, flooded roads and highways are computed in parallel (`par_map`); generation threads run at low priority so they don't steal frames from rendering.
- The quadtree is only reselected if the camera moved or new chunks arrived.
- Objects are only drawn in the shadow cascades they touch; the far cascade uses the parent level-of-detail geometry; water only computes the sky when the screen-space reflection is missing.
- Tree and grass instances (over 100,000 per frame) are prepared on several threads and written straight into the GPU buffer; chunk tables use FxHash.

## Tests

```powershell
.\cargo.ps1 test --lib
```

## Current limitations

- Parked cars and street lights are simple volumes; only the driven car is detailed.
- Headlights don't cast light on the road. Windows are opaque from the outside.
- The car is not deformable like in BeamNG (no beams or nodes): the body stays rigid, only dents are shown, and damage doesn't affect handling. No tire wear or heating. Keyboard steering is progressive but less precise than a wheel. There is no traffic.
- Mountain roads: only one per city, as a dead end (no pass road going down the other side); no tunnels or bridges, so a road stops before a cliff or river, and a road that would cross another is abandoned. The planet's relief stays moderate: 250 to 1,000 m of elevation gain. The computation adds about 0.75 s to planet creation.
- Circuits: no opponents or timer. The access road is a straight line. A city with no free flat land around it has no circuit, and the same layout can appear more than once. Facade detail fades with distance.
- Bridge ramps can leave a step at intersections, and buildings sink a little into slopes.
- Highways start from the edge of cities, in line with an avenue: no interchanges or ramps.
  Two crossing highways pass over each other without connecting.
- Islands and continents separated by deep sea (over 70 m) stay unconnected: there can be several networks.
- With highways, a planet generates more slowly: about 10 s for 140 cities, 18 s for 400.
- Highway signs have no text. Some suburban hamlets have no access street.
- No traffic or inhabitants. Seen from afar at night, the city looks a bit pale.
- A street extending over water is removed, but a few footbridges remain in lagoons.
- No caves or overhangs: the terrain is a height field. Canyon and tower cliffs are very steep slopes.
- Gullies remain a local illusion: they don't always join the simulated rivers.
- No waterfalls or deltas: a steep drop gives rapids, and rivers reach the sea through a single channel.
- All lakes overflow: no endorheic salt lakes in deserts.
- The river network stops at basins of about 300 to 800 km²: below that, there are no streams, only dry valleys.
- No audio, NPCs, saving, or 3D model loading.
- A game cannot yet add its own objects to the scene.
- Physics is limited to the ground, trunks and rocks. There is no orbital gravity: flight is free.
- Clouds have no temporal filtering: they keep a slight grain up close.
- Reflections only show what is on screen; the rest comes from the computed sky, without clouds.
- Nearby trees remain simple volumes dressed per pixel, without real leaves.
