// Prelude commun : uniformes globaux, bruit, atmosphere, deformation du terrain et des objets.

struct Globals {
    view_proj: mat4x4f,
    inv_view_proj: mat4x4f,
    cascades: array<mat4x4f, 3>,
    sun_dir: vec4f,     // xyz vers le soleil
    up: vec4f,          // xyz verticale de la camera, w = altitude au-dessus de la mer (m)
    tangent: vec4f,     // xyz tangente a la verticale, repere des objets
    center: vec4f,      // xyz centre de la planete relatif a la camera (m), w = rayon (m)
    camera_wrap: vec4f, // xyz position de la camera enroulee sur WRAP
    misc: vec4f,        // temps, lod_factor, largeur, hauteur
    camera_km: vec4f,   // xyz position de la camera en km, w = decalage des nuages
};

@group(0) @binding(0) var<uniform> g: Globals;
// r : Perlin-Worley, g et b : Worley, a : Perlin fractal. Periodique, avec mips.
@group(0) @binding(1) var noise3: texture_3d<f32>;
@group(0) @binding(2) var noise_sampler: sampler;

const WRAP: f32 = 4096.0;

fn hash3(p: vec3f) -> f32 {
    var q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

// Bruit de valeur 3D periodique : indispensable car les coordonnees du shader sont enroulees.
fn vnoise(p: vec3f, period: f32) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let w = i - period * floor(i / period);
    let n = w + 1.0 - period * floor((w + 1.0) / period);
    return mix(
        mix(mix(hash3(w), hash3(vec3f(n.x, w.y, w.z)), u.x),
            mix(hash3(vec3f(w.x, n.y, w.z)), hash3(vec3f(n.x, n.y, w.z)), u.x), u.y),
        mix(mix(hash3(vec3f(w.x, w.y, n.z)), hash3(vec3f(n.x, w.y, n.z)), u.x),
            mix(hash3(vec3f(w.x, n.y, n.z)), hash3(n), u.x), u.y),
        u.z);
}

// Atmosphere physique a diffusion simple, en kilometres.
const ATM_THICKNESS: f32 = 80.0;
const BETA_R: vec3f = vec3f(5.8e-3, 13.5e-3, 33.1e-3);
const BETA_M: f32 = 7e-3;
const H_R: f32 = 8.0;
const H_M: f32 = 1.4;
const SUN_INTENSITY: f32 = 15.0;

fn planet_km() -> f32 {
    return g.center.w * 0.001;
}

fn ray_sphere(ro: vec3f, rd: vec3f, r: f32) -> vec2f {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - r * r;
    let d = b * b - c;
    if d < 0.0 {
        return vec2f(1.0, -1.0);
    }
    let s = sqrt(d);
    return vec2f(-b - s, -b + s);
}

// Masse d'air relative (Kasten et Young), bornee a l'horizon.
fn airmass(mu: f32) -> f32 {
    let c = clamp(mu, 0.0, 1.0);
    let deg = acos(c) * 57.29578;
    return 1.0 / (c + 0.15 * pow(max(93.885 - deg, 0.1), -1.253));
}

// Part de la lumiere du soleil qui atteint l'altitude h (km), soleil a mu de la verticale.
fn sun_transmittance(h: f32, mu: f32) -> vec3f {
    let od = BETA_R * H_R * exp(-h / H_R) + vec3f(BETA_M * 1.1 * H_M * exp(-h / H_M));
    return exp(-od * airmass(mu));
}

struct Scatter {
    inscatter: vec3f,
    transmittance: vec3f,
};

fn atmosphere(ro: vec3f, rd: vec3f, tmax: f32, steps: i32) -> Scatter {
    var out = Scatter(vec3f(0.0), vec3f(1.0));
    let rp = planet_km();
    let hit = ray_sphere(ro, rd, rp + ATM_THICKNESS);
    if hit.y < 0.0 {
        return out;
    }
    let t0 = max(hit.x, 0.0);
    let t1 = min(hit.y, tmax);
    if t1 <= t0 {
        return out;
    }
    let sun = g.sun_dir.xyz;
    let cos_v = dot(rd, sun);
    let phase_r = 0.0597 * (1.0 + cos_v * cos_v);
    let mg = 0.8;
    let phase_m = 0.1194 * (1.0 - mg * mg) * (1.0 + cos_v * cos_v)
        / ((2.0 + mg * mg) * pow(max(1.0 + mg * mg - 2.0 * mg * cos_v, 0.001), 1.5));

    var sum_r = vec3f(0.0);
    var sum_m = vec3f(0.0);
    var od_r = 0.0;
    var od_m = 0.0;
    let span = t1 - t0;
    let n = f32(steps);
    for (var i = 0; i < steps; i++) {
        // Pas quadratiques : l'air dense est pres de la camera quand on est au sol.
        let f = (f32(i) + 0.5) / n;
        let t = t0 + span * f * f;
        let dt = span * 2.0 * f / n;
        let p = ro + rd * t;
        let r = length(p);
        let h = max(r - rp, 0.0);
        let dr = exp(-h / H_R) * dt;
        let dm = exp(-h / H_M) * dt;
        od_r += dr;
        od_m += dm;
        let mu = dot(p / r, sun);
        // Ombre de la planete sur son atmosphere.
        let k = rp / r;
        let horizon = -sqrt(max(1.0 - k * k, 0.0));
        let lit = smoothstep(horizon - 0.015, horizon + 0.015, mu);
        if lit > 0.0 {
            let view = exp(-(BETA_R * od_r + vec3f(BETA_M * 1.1 * od_m)));
            let light = sun_transmittance(h, mu) * view * lit;
            sum_r += light * dr;
            sum_m += light * dm;
        }
    }
    out.inscatter = (sum_r * BETA_R * phase_r + sum_m * BETA_M * phase_m) * SUN_INTENSITY;
    out.transmittance = exp(-(BETA_R * od_r + vec3f(BETA_M * 1.1 * od_m)));
    return out;
}

// Ajoute a une couleur l'air traverse jusqu'a un point relatif a la camera.
fn apply_atmosphere(col: vec3f, rel: vec3f, steps: i32) -> vec3f {
    let d = length(rel);
    // A quelques centaines de metres l'air ne se voit pas : autant ne rien calculer.
    if d < 150.0 {
        return col;
    }
    let s = atmosphere(g.camera_km.xyz, rel / d, d * 0.001, steps);
    return col * s.transmittance + s.inscatter;
}

// Lumiere du soleil a l'altitude h (km), soleil a mu de la verticale.
// En altitude, le soleil se couche plus tard : les nuages restent eclaires apres le sol.
fn sun_light_at(h: f32, mu: f32) -> vec3f {
    let rp = planet_km();
    let k = rp / (rp + h);
    let horizon = -sqrt(max(1.0 - k * k, 0.0));
    let visible = smoothstep(horizon - 0.015, horizon + 0.03, mu);
    return vec3f(2.9, 2.75, 2.5) * sun_transmittance(h, max(mu - horizon, 0.0)) * visible;
}

// Lumiere du soleil recue au sol sous la verticale `up`.
fn sun_light(up: vec3f) -> vec3f {
    return sun_light_at(0.0, dot(up, g.sun_dir.xyz));
}

// Eclairement diffus du ciel, de la nuit (0) au plein jour (1).
fn day_factor(up: vec3f) -> f32 {
    return smoothstep(-0.12, 0.28, dot(up, g.sun_dir.xyz));
}

// Lumiere diffuse du ciel : bleue le jour, mauve et chaude quand le soleil est bas.
fn sky_ambient(up: vec3f) -> vec3f {
    let mu = dot(up, g.sun_dir.xyz);
    let day = smoothstep(-0.12, 0.28, mu);
    let low = smoothstep(-0.1, 0.03, mu) * (1.0 - smoothstep(0.1, 0.42, mu));
    let base = mix(vec3f(0.016, 0.022, 0.05), vec3f(0.33, 0.43, 0.62), day);
    return base + vec3f(0.17, 0.095, 0.075) * low;
}

// --- Nuages, partages par leur rendu et par les ombres qu'ils portent. ---

const CLOUD_BASE: f32 = 1.6;
const CLOUD_TOP: f32 = 7.5;
// Le bruit de forme fait 4 periodes par tuile : des masses de 4 a 5 km.
const CLOUD_TILE: f32 = 18.0;

fn cloud_offset() -> vec3f {
    return vec3f(g.misc.x * 0.012 + g.camera_km.w, g.camera_km.w * 0.37, -g.misc.x * 0.006);
}

// Part de ciel couverte : des fronts a l'echelle du continent, des bancs a l'echelle de la region.
fn cloud_cover(q: vec3f) -> f32 {
    let a = textureSampleLevel(noise3, noise_sampler, q / 6100.0, 0.0).a;
    let b = textureSampleLevel(noise3, noise_sampler, q / 1450.0, 0.0).a;
    let c = textureSampleLevel(noise3, noise_sampler, q / 340.0, 0.0).a;
    return smoothstep(0.42, 0.74, a * 0.46 + b * 0.32 + c * 0.22);
}

// Masse du nuage en p : bases plates et sommets arrondis, plus hauts sous les grands bancs.
fn cloud_shape(p: vec3f, cover: f32, lod: f32) -> f32 {
    let h = (length(p) - planet_km() - CLOUD_BASE) / (CLOUD_TOP - CLOUD_BASE);
    if h <= 0.0 || h >= 1.0 {
        return 0.0;
    }
    let top = 0.22 + 0.78 * cover * cover;
    let profile = smoothstep(0.0, 0.05, h) * (1.0 - smoothstep(top * 0.45, top, h));
    if profile <= 0.0 {
        return 0.0;
    }
    let q = p + cloud_offset() + vec3f(h * 2.5, 0.0, h * 1.2);
    let s = textureSampleLevel(noise3, noise_sampler, q / CLOUD_TILE, lod).r;
    // Meme sous un grand banc, il reste des trouees : des cumulus distincts plutot qu'une nappe.
    let threshold = 1.0 - min(cover, 0.8) * 0.62 * profile;
    return clamp((s - threshold) / max(1.0 - threshold, 0.05) * 1.8, 0.0, 1.0);
}

// Les bords se rongent en volutes, inutile a grande distance.
fn cloud_detail(p: vec3f, base: f32, lod: f32, t: f32) -> f32 {
    if lod > 3.2 {
        return base;
    }
    let q = p + cloud_offset() * 1.3;
    let a = textureSampleLevel(noise3, noise_sampler, q / 3.4, max(lod - 2.0, 0.0)).g;
    var erode = (1.0 - a) * 0.5;
    if t < 40.0 {
        let b = textureSampleLevel(noise3, noise_sampler, q / 0.8, 0.0).b;
        erode += (1.0 - b) * 0.18 * (1.0 - smoothstep(16.0, 40.0, t));
    }
    return clamp((base - erode) / max(1.0 - erode, 0.05), 0.0, 1.0);
}

// Ombre des nuages au sol : le rayon vers le soleil traverse la couche a mi-hauteur.
fn cloud_shadow(rel: vec3f, up: vec3f) -> f32 {
    let sun = g.sun_dir.xyz;
    let mu = dot(up, sun);
    if mu <= 0.02 {
        return 1.0;
    }
    let p = g.camera_km.xyz + rel * 0.001;
    let t = 3.4 / max(mu, 0.05);
    let q = p + sun * t;
    let cover = cloud_cover(q + cloud_offset());
    if cover <= 0.0 {
        return 1.0;
    }
    let s = textureSampleLevel(noise3, noise_sampler, (q + cloud_offset()) / CLOUD_TILE, 1.0).r;
    let dark = smoothstep(0.0, 0.2, s - (1.0 - min(cover, 0.8) * 0.62));
    return 1.0 - dark * 0.82;
}

// Couleur approchee du ciel local, pour les reflets sur l'eau.
fn local_sky(dir: vec3f, up: vec3f) -> vec3f {
    let day = day_factor(up);
    let h = max(dot(dir, up), 0.0);
    let zenith = mix(vec3f(0.002, 0.003, 0.009), vec3f(0.06, 0.18, 0.50), day);
    let horizon = mix(vec3f(0.006, 0.010, 0.022), vec3f(0.50, 0.65, 0.88), day);
    var col = mix(horizon, zenith, 1.0 - exp(-h * 3.5));
    let sunset = smoothstep(0.42, 0.0, abs(dot(up, g.sun_dir.xyz)));
    col += vec3f(1.15, 0.40, 0.12) * sunset * pow(max(dot(dir, g.sun_dir.xyz) * 0.5 + 0.5, 0.0), 5.0) * exp(-h * 5.0);
    return col;
}

// Morphing CDLOD : glisse vers la surface du LOD parent avant la bascule de niveau.
fn morphed_position(pos: vec3f, morph: vec3f, offset: vec3f, size: f32) -> vec3f {
    let rel = pos + offset;
    let k = g.misc.y;
    let t = smoothstep(1.4 * k * size, 1.95 * k * size, length(rel));
    return rel + morph * t;
}

// Les objets sont trop proches pour que la courbure se voie : tous partagent le repere de la camera.
fn object_axes() -> mat3x3f {
    let t = g.tangent.xyz;
    let u = g.up.xyz;
    return mat3x3f(t, u, cross(t, u));
}

fn yaw_rotate(p: vec3f, yaw: f32) -> vec3f {
    let c = cos(yaw);
    let s = sin(yaw);
    return vec3f(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}

fn object_position(pos: vec3f, sway: f32, inst_pos: vec3f, scale: f32, yaw: f32) -> vec3f {
    var p = yaw_rotate(pos, yaw) * scale;
    // Vent : la phase depend de la position pour que tout ne bouge pas d'un bloc.
    let t = g.misc.x;
    let phase = dot(inst_pos + g.camera_wrap.xyz, vec3f(0.13, 0.11, 0.17));
    p.x += sway * sin(t * 1.3 + phase) * 0.12 * scale;
    p.z += sway * sin(t * 1.7 + phase * 1.3) * 0.09 * scale;
    return object_axes() * p + inst_pos;
}
