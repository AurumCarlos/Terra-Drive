// Eclairage : ombres en cascade, soleil filtre par l'atmosphere, lumiere du ciel.

@group(1) @binding(0) var shadow_map: texture_depth_2d_array;
@group(1) @binding(1) var shadow_sampler: sampler_comparison;

const SHADOW_TEXEL: f32 = 1.0 / 2048.0;

fn shadow_factor(rel: vec3f, n: vec3f) -> f32 {
    // Decalage le long de la normale, proportionnel a la distance, contre l'acne d'ombre.
    let p = rel + n * (0.04 + length(rel) * 0.0035);
    for (var i = 0; i < 3; i++) {
        let clip = g.cascades[i] * vec4f(p, 1.0);
        let uv = clip.xy * vec2f(0.5, -0.5) + 0.5;
        if any(uv < vec2f(0.01)) || any(uv > vec2f(0.99)) || clip.z < 0.0 || clip.z > 1.0 {
            continue;
        }
        var sum = 0.0;
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let o = vec2f(f32(x), f32(y)) * SHADOW_TEXEL;
                sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + o, i, clip.z);
            }
        }
        let shadow = sum / 9.0;
        if i == 2 {
            // Disparition progressive au bord de la derniere cascade.
            let e = max(abs(uv.x - 0.5), abs(uv.y - 0.5)) * 2.0;
            return mix(shadow, 1.0, smoothstep(0.8, 0.97, e));
        }
        return shadow;
    }
    return 1.0;
}

// Soleil sur le point, ombres du relief et des nuages comprises, sans le cosinus de la surface.
fn sun_at_point(n: vec3f, rel: vec3f, up: vec3f) -> vec3f {
    let light = sun_light(up);
    if dot(light, light) <= 0.0 {
        return vec3f(0.0);
    }
    return light * shadow_factor(rel, n) * cloud_shadow(rel, up);
}

fn ambient_light(n: vec3f, up: vec3f) -> vec3f {
    let day = day_factor(up);
    let sun = g.sun_dir.xyz;
    let sky = sky_ambient(up);
    let bounce = mix(vec3f(0.006, 0.006, 0.01), vec3f(0.17, 0.155, 0.11), day);
    let side = dot(n, up) * 0.5 + 0.5;
    var ambient = mix(bounce, sky, side);
    // Le ciel est plus clair et plus chaud du cote du soleil : les ombres gardent une direction.
    let flat_sun = normalize(sun - up * dot(sun, up) + vec3f(1e-4));
    let toward = dot(n, flat_sun) * 0.5 + 0.5;
    let low = smoothstep(-0.08, 0.04, dot(up, sun)) * (1.0 - smoothstep(0.15, 0.5, dot(up, sun)));
    ambient *= mix(1.0, mix(0.78, 1.2, toward), low);
    ambient += vec3f(0.14, 0.075, 0.04) * toward * toward * low;
    // Clair de lune : une lueur froide sans ombres, sur la face de nuit.
    return ambient + vec3f(0.05, 0.07, 0.13) * (1.0 - day) * max(side, 0.2);
}

fn shade(albedo: vec3f, n: vec3f, rel: vec3f, up: vec3f) -> vec3f {
    let ndl = max(dot(n, g.sun_dir.xyz), 0.0);
    var direct = vec3f(0.0);
    if ndl > 0.0 {
        direct = sun_at_point(n, rel, up) * ndl;
    }
    return albedo * (direct + ambient_light(n, up));
}
