// Nuages en volume : marche de rayons a demi-resolution, puis composition par-dessus la scene.

@group(2) @binding(0) var scene_depth: texture_depth_2d;

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) ndc: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var o: VsOut;
    o.clip = vec4f(p, 0.0, 1.0);
    o.ndc = p;
    return o;
}

const SIGMA: f32 = 11.0;
const CIRRUS_ALTITUDE: f32 = 10.0;
const NEAR_STEPS: f32 = 44.0;
const GRAZING_STEPS: f32 = 110.0;
const LIGHT_STEPS: i32 = 4;

fn hg(c: f32, k: f32) -> f32 {
    let d = 1.0 + k * k - 2.0 * k * c;
    return (1.0 - k * k) / pow(max(d, 1e-3), 1.5);
}

// Epaisseur de nuage rencontree en remontant vers le soleil.
fn sun_depth(p: vec3f, sun: vec3f, cover: f32, lod: f32) -> f32 {
    var od = 0.0;
    var stride = 0.22;
    var t = stride * 0.6;
    for (var i = 0; i < LIGHT_STEPS; i++) {
        od += cloud_shape(p + sun * t, cover, lod + f32(i) * 0.7) * stride;
        t += stride * 1.5;
        stride *= 2.0;
    }
    return od * SIGMA;
}

fn cirrus(p: vec3f) -> f32 {
    let q = p + cloud_offset() * 1.7;
    let big = textureSampleLevel(noise3, noise_sampler, q / 2300.0, 0.0).a;
    // Des filaments allonges par le vent d'altitude.
    let warp = textureSampleLevel(noise3, noise_sampler, q / 340.0, 0.0).a - 0.5;
    let a = textureSampleLevel(noise3, noise_sampler, (q + warp * 16.0) * vec3f(1.0 / 900.0, 1.0 / 70.0, 1.0 / 100.0), 0.0).a;
    let b = textureSampleLevel(noise3, noise_sampler, (q + warp * 10.0) * vec3f(1.0 / 320.0, 1.0 / 17.0, 1.0 / 26.0), 0.0).a;
    let c = textureSampleLevel(noise3, noise_sampler, (q + warp * 6.0) * vec3f(1.0 / 110.0, 1.0 / 5.0, 1.0 / 8.0), 0.0).a;
    let n = a * 0.55 + b * 0.3 + c * 0.15;
    return smoothstep(0.54, 0.78, n) * smoothstep(0.38, 0.62, big);
}

@fragment
fn fs_march(i: VsOut) -> @location(0) vec4f {
    let h = g.inv_view_proj * vec4f(i.ndc, 1.0, 1.0);
    let dir = normalize(h.xyz / h.w);
    let ro = g.camera_km.xyz;
    let rp = planet_km();
    let inner = rp + CLOUD_BASE;
    let outer = rp + CLOUD_TOP;
    let r0 = length(ro);
    let camera_up = ro / r0;

    // Le sol et le relief deja dessines arretent le rayon.
    var tmax = 1.0e9;
    let ground = ray_sphere(ro, dir, rp);
    if ground.x > 0.0 && ground.y > 0.0 {
        tmax = ground.x;
    }
    let pixel = vec2i(clamp((i.ndc * vec2f(0.5, -0.5) + 0.5) * g.misc.zw, vec2f(0.0), g.misc.zw - 1.0));
    let raw = textureLoad(scene_depth, pixel, 0);
    if raw > 0.0 {
        let q = g.inv_view_proj * vec4f(i.ndc, raw, 1.0);
        tmax = min(tmax, length(q.xyz / q.w) * 0.001);
    }

    var light = vec3f(0.0);
    var through = 1.0;
    var weighted = 0.0;
    var weight = 0.0;
    let sun = g.sun_dir.xyz;
    let cos_sun = dot(dir, sun);
    // Liseres d'argent face au soleil et lueur douce de dos.
    let phase = (hg(cos_sun, 0.72) * 0.55 + hg(cos_sun, -0.2) * 0.45) * 0.72;

    let hit_outer = ray_sphere(ro, dir, outer);
    let hit_inner = ray_sphere(ro, dir, inner);
    var t0 = 0.0;
    var t1 = -1.0;
    if hit_outer.y > 0.0 {
        if r0 > outer {
            t0 = hit_outer.x;
            t1 = select(hit_outer.y, hit_inner.x, hit_inner.y > 0.0 && hit_inner.x > 0.0);
        } else if r0 > inner {
            t1 = select(hit_outer.y, hit_inner.x, hit_inner.y > 0.0 && hit_inner.x > 0.0);
        } else {
            t0 = hit_inner.y;
            t1 = hit_outer.y;
        }
    }
    t1 = min(min(t1, tmax), t0 + 320.0);

    if t1 > t0 {
        let span = t1 - t0;
        let n = mix(GRAZING_STEPS, NEAR_STEPS, smoothstep(0.0, 0.5, abs(dot(dir, camera_up))));
        // Bruit a gradient entrelace (bruit bleu), decale a chaque image : la resolution temporelle
        // moyenne les pas de la marche au lieu de figer un grain a l'ecran.
        let ign = fract(52.9829189 * fract(dot(i.clip.xy, vec2f(0.06711056, 0.00583715))));
        let jitter = fract(ign + f32(u32(g.frame.x) % 64u) * 0.618034);
        let pixel_size = 2.0 / g.misc.w;
        var k = 0.0;
        loop {
            if k >= n || through < 0.015 {
                break;
            }
            let f0 = (k + jitter) / n;
            let f = f0 * f0 * 0.7 + f0 * 0.3;
            let t = t0 + span * f;
            let dt = span * (1.4 * f0 + 0.3) / n;
            let p = ro + dir * t;
            k += 1.0;

            let cover = cloud_cover(p + cloud_offset());
            if cover <= 0.0 {
                continue;
            }
            let lod = max(log2(t * pixel_size / (CLOUD_TILE / 96.0)), 0.0);
            let base = cloud_shape(p, cover, lod);
            if base <= 0.001 {
                continue;
            }
            let d = cloud_detail(p, base, lod, t);
            if d <= 0.001 {
                continue;
            }

            let r = length(p);
            let up = p / r;
            let alt = r - rp;
            let mu = dot(up, sun);
            let od = sun_depth(p, sun, cover, lod + 1.0);
            // Plusieurs diffusions : la lumiere finit par traverser, en perdant de sa direction.
            let scatter = exp(-od) + 0.22 * exp(-od * 0.3) + 0.07 * exp(-od * 0.09);
            let powder = 1.0 - 0.55 * exp(-d * dt * SIGMA * 3.0 - od * 0.4);
            let direct = sun_light_at(alt, mu) * scatter * mix(phase, 1.0, 0.25) * powder;
            let height = clamp((alt - CLOUD_BASE) / (CLOUD_TOP - CLOUD_BASE), 0.0, 1.0);
            let ambient = sky_ambient(up) * (0.3 + 0.8 * height);
            let step_t = exp(-d * SIGMA * dt);
            let energy = through * (1.0 - step_t);
            light += (direct + ambient) * 0.92 * energy;
            weighted += t * energy;
            weight += energy;
            through *= step_t;
        }
    }

    // Voile d'altitude : derriere les cumulus vu du sol, devant vu de l'orbite.
    let shell = ray_sphere(ro, dir, rp + CIRRUS_ALTITUDE);
    if shell.y > 0.0 {
        let above = r0 > rp + CIRRUS_ALTITUDE;
        let t = select(shell.y, shell.x, above);
        if t > 0.0 && t < tmax {
            let p = ro + dir * t;
            let up = normalize(p);
            let density = cirrus(p) * 0.3;
            if density > 0.002 {
                let mu = dot(up, sun);
                let c = (sun_light_at(CIRRUS_ALTITUDE, mu) * (0.3 + phase * 0.42) + sky_ambient(up) * 0.9) * density;
                if above {
                    light = c + light * (1.0 - density);
                } else {
                    light += c * through;
                }
                let energy = through * density;
                weighted += t * energy;
                weight += energy;
                through *= 1.0 - density;
            }
        }
    }

    if weight <= 0.0 {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    // L'air entre la camera et le nuage : le voile des lointains.
    let s = atmosphere(ro, dir, weighted / weight, 5);
    return vec4f(light * s.transmittance + s.inscatter * (1.0 - through), through);
}

@group(1) @binding(0) var clouds_tex: texture_2d<f32>;
@group(1) @binding(1) var clouds_sampler: sampler;

@group(2) @binding(0) var history_tex: texture_2d<f32>;
@group(2) @binding(1) var history_sampler: sampler;

// Resolution temporelle : l'image precedente, reprojetee par direction (les nuages sont loin), se melange a
// la marche de cette image. Elle est bornee par le voisinage actuel pour ne pas laisser de trainees.
@fragment
fn fs_resolve(i: VsOut) -> @location(0) vec4f {
    let size = vec2i(textureDimensions(clouds_tex));
    let pixel = clamp(vec2i(i.clip.xy), vec2i(0), size - 1);
    let current = textureLoad(clouds_tex, pixel, 0);
    var mean = vec4f(0.0);
    var square = vec4f(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let c = textureLoad(clouds_tex, clamp(pixel + vec2i(dx, dy), vec2i(0), size - 1), 0);
            mean += c;
            square += c * c;
        }
    }
    mean /= 9.0;
    let sigma = sqrt(max(square / 9.0 - mean * mean, vec4f(0.0)));
    let lo = min(current, mean - sigma * 1.25);
    let hi = max(current, mean + sigma * 1.25);

    let h = g.inv_view_proj * vec4f(i.ndc, 1.0, 1.0);
    let dir = normalize(h.xyz / h.w);
    let prev = g.prev_view_proj * vec4f(dir, 0.0);
    var fresh = 1.0;
    var history = current;
    if prev.w > 1e-6 && g.frame.y < 0.5 {
        let uv = (prev.xy / prev.w) * vec2f(0.5, -0.5) + 0.5;
        if all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0)) {
            history = clamp(textureSampleLevel(history_tex, history_sampler, uv, 0.0), lo, hi);
            fresh = 0.1;
        }
    }
    return mix(history, current, fresh);
}

// Composition : un petit filtre gomme le grain de la marche de rayons.
@fragment
fn fs_composite(i: VsOut) -> @location(0) vec4f {
    let uv = i.ndc * vec2f(0.5, -0.5) + 0.5;
    let px = 1.0 / vec2f(textureDimensions(clouds_tex));
    var sum = textureSampleLevel(clouds_tex, clouds_sampler, uv, 0.0) * 0.28;
    sum += textureSampleLevel(clouds_tex, clouds_sampler, uv + px * vec2f(0.9, 0.35), 0.0) * 0.18;
    sum += textureSampleLevel(clouds_tex, clouds_sampler, uv + px * vec2f(-0.35, 0.9), 0.0) * 0.18;
    sum += textureSampleLevel(clouds_tex, clouds_sampler, uv + px * vec2f(-0.9, -0.35), 0.0) * 0.18;
    sum += textureSampleLevel(clouds_tex, clouds_sampler, uv + px * vec2f(0.35, -0.9), 0.0) * 0.18;
    return sum;
}
