// Mer, lacs et rivieres : une surface par chunk, a l'altitude de l'eau en chaque sommet.

@group(2) @binding(0) var scene_depth: texture_depth_2d;
@group(2) @binding(1) var scene_color: texture_2d<f32>;
@group(2) @binding(2) var scene_sampler: sampler;

struct VsIn {
    @location(0) pos: vec3f,
    @location(1) morph: vec3f,
    @location(2) up: vec3f,
    @location(3) data: vec4f, // r = profondeur, g = glace, b et a = courant vers l'est et le nord
    @location(4) offset: vec3f,
    @location(5) size: f32,
    @location(6) wrap: vec3f,
};

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) rel: vec3f,
    @location(1) up: vec3f,
    @location(2) data: vec2f,
    @location(3) wrapped: vec3f,
    @location(4) flow: vec2f,
};

const WATER_DEPTH_SCALE: f32 = 60.0;
const FLOW_SCALE: f32 = 8.0;

@vertex
fn vs(v: VsIn) -> VsOut {
    let rel = morphed_position(v.pos, v.morph, v.offset, v.size);
    var o: VsOut;
    o.clip = g.view_proj * vec4f(rel, 1.0);
    o.rel = rel;
    o.up = v.up;
    o.data = v.data.rg;
    o.flow = (v.data.ba * 255.0 - 128.0) / 127.0 * FLOW_SCALE;
    o.wrapped = v.wrap + v.pos;
    return o;
}

// Une couche de vagues : vecteur de pente pris dans trois tranches du bruit, filtre par ses mips.
fn wave_layer(p: vec3f, tile: f32, drift: vec3f, t: f32) -> vec3f {
    let q = p / tile;
    let a = textureSample(noise3, noise_sampler, q + drift * t).a;
    let b = textureSample(noise3, noise_sampler, q + vec3f(0.37, 0.11, 0.73) + drift.zxy * t).a;
    let c = textureSample(noise3, noise_sampler, q + vec3f(0.61, 0.83, 0.29) + drift.yzx * t).a;
    return vec3f(a, b, c) - 0.5;
}

// Longueurs d'onde de 40 cm a 30 m. Le bruit fait 4 periodes par tuile, d'ou ces tuiles entieres.
fn ripple(p: vec3f, t: f32) -> vec3f {
    let long = wave_layer(p, 128.0, vec3f(0.006, 0.0, 0.004), t);
    let mid = wave_layer(p, 32.0, vec3f(-0.012, 0.0, 0.009), t);
    let small = wave_layer(p, 8.0, vec3f(0.03, 0.0, -0.022), t);
    let fine = wave_layer(p, 2.0, vec3f(-0.06, 0.0, 0.05), t);
    return long * 0.28 + mid * 0.34 + small * 0.46 + fine * 0.5;
}

// Remonte le reflet dans l'image deja dessinee. Rend alpha = 0 si le rayon sort de l'ecran.
fn screen_reflection(rel: vec3f, r: vec3f, dist: f32) -> vec4f {
    let size = g.misc.zw;
    var travel = max(dist * 0.015, 1.5);
    var last = vec2f(-1.0);
    for (var k = 0; k < 26; k++) {
        let p = rel + r * travel;
        let clip = g.view_proj * vec4f(p, 1.0);
        if clip.w <= 0.0 {
            break;
        }
        let ndc = clip.xyz / clip.w;
        let uv = ndc.xy * vec2f(0.5, -0.5) + 0.5;
        if any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) {
            break;
        }
        last = uv;
        let scene = textureLoad(scene_depth, vec2i(uv * size), 0);
        // Z inverse : la scene est devant le rayon si sa profondeur est plus grande.
        if scene > ndc.z {
            let q = g.inv_view_proj * vec4f(ndc.xy, scene, 1.0);
            let gap = length(q.xyz / q.w) - length(p);
            if gap > -travel * 0.6 {
                let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
                let c = textureSampleLevel(scene_color, scene_sampler, uv, 0.0).rgb;
                return vec4f(c, smoothstep(0.0, 0.06, edge));
            }
        }
        travel *= 1.28;
    }
    // Rien sur le trajet : c'est le ciel, tel qu'il est dessine a cet endroit s'il est a l'ecran.
    if last.x >= 0.0 {
        let far_clip = g.view_proj * vec4f(r * 1.0e7, 1.0);
        if far_clip.w > 0.0 {
            let uv = far_clip.xy / far_clip.w * vec2f(0.5, -0.5) + 0.5;
            if all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0)) && textureLoad(scene_depth, vec2i(uv * size), 0) == 0.0 {
                let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
                return vec4f(textureSampleLevel(scene_color, scene_sampler, uv, 0.0).rgb, smoothstep(0.0, 0.06, edge));
            }
        }
    }
    return vec4f(0.0);
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4f {
    let t = g.misc.x;
    let dist = length(i.rel);
    let view = i.rel / dist;
    let up = normalize(i.up);
    // Banquise et lacs geles : la glace suit la saison, avec un mois de retard sur le froid de l'air ; la
    // glace de toute l'annee (i.data.y) reste en ete.
    let lagged = season_temperature(up, length(i.rel - g.center.xyz) - g.center.w)
        + (textureSampleLevel(weather_map, noise_sampler, up, 0.0).b * 60.0 - 30.0)
        * (cos(6.2831853 * g.weather.x) - cos(6.2831853 * (g.weather.x - 0.08)));
    let ice = max(smoothstep(0.35, 0.65, i.data.y) * smoothstep(3.0, -2.0, lagged), smoothstep(-1.5, -6.0, lagged));
    let sun = g.sun_dir.xyz;

    // Le courant dans le monde. Sur les rivieres, les rides descendent vers l'aval : deux phases
    // decalees se relaient pour que le motif ne s'etire jamais trop.
    let east = normalize(cross(vec3f(0.0, 1.0, 0.0), up));
    let north = cross(up, east);
    let flow = east * i.flow.x + north * i.flow.y;
    let speed = length(i.flow);
    let river = smoothstep(0.1, 0.5, speed);

    // Les mips effacent ce que le pixel ne resout plus : cette pente perdue devient de la rugosite.
    var waves = ripple(i.wrapped, t);
    if river > 0.0 {
        let phase = fract(t * 0.3);
        let a = ripple(i.wrapped - flow * phase * 3.3, t * 0.4);
        let b = ripple(i.wrapped - flow * fract(phase + 0.5) * 3.3 + vec3f(13.0), t * 0.4);
        waves = mix(waves, mix(a, b, abs(phase * 2.0 - 1.0)) * (1.0 + speed * 0.3), river);
    }
    let lake_calm = mix(0.5, 1.0, river);
    var n = normalize(up + (waves - up * dot(waves, up)) * 0.42 * lake_calm);
    if g.up.w < 0.0 {
        n = -n;
    }

    // Epaisseur d'eau : la scene pres de la camera, le relief stocke dans les sommets au loin,
    // car la precision de la profondeur ne suit plus depuis l'orbite.
    var thickness = i.data.x * WATER_DEPTH_SCALE;
    if dist < 8000.0 {
        var scene = 1000.0;
        let raw = textureLoad(scene_depth, vec2i(i.clip.xy), 0);
        if raw > 0.0 {
            let size = g.misc.zw;
            let ndc = vec2f(i.clip.x / size.x, i.clip.y / size.y) * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0);
            let q = g.inv_view_proj * vec4f(ndc, raw, 1.0);
            scene = max(length(q.xyz / q.w) - dist, 0.0);
        }
        thickness = mix(scene, thickness, smoothstep(3000.0, 8000.0, dist));
    }

    let day = day_factor(up);
    let sun_in_water = sun_light(up) * max(dot(up, sun), 0.0) * 0.35 + sky_ambient(up) * 0.35;
    // Les rivieres charrient des sediments : moins turquoises que la mer.
    let tint_near = mix(vec3f(0.10, 0.55, 0.55), vec3f(0.17, 0.36, 0.30), river);
    let shallow = tint_near * sun_in_water;
    let deep = vec3f(0.012, 0.09, 0.2) * sun_in_water;
    let body = mix(shallow, deep, 1.0 - exp(-thickness * 0.09));

    let cos_view = max(dot(-view, n), 0.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - cos_view, 5.0);

    // Le reflet ne descend jamais sous l'horizon, meme sur une vague raide.
    var r = reflect(view, n);
    r = normalize(r + up * max(0.015 - dot(r, up), 0.0));

    // Le ciel calcule, puis ce que l'image montre vraiment dans cette direction : montagnes et nuages.
    // Le ciel calcule ne sert que la ou l'image ne montre pas deja le reflet en entier.
    var found = vec4f(0.0);
    if g.up.w >= 0.0 && dist < 60000.0 {
        found = screen_reflection(i.rel, r, dist);
    }
    var reflection = found.rgb;
    if found.a < 1.0 {
        let air = atmosphere(g.camera_km.xyz + i.rel * 0.001, r, 1.0e9, 6);
        reflection = mix(air.inscatter, found.rgb, found.a);
    }
    var col = mix(body, reflection, fresnel);

    // Chemin de lumiere du soleil : une distribution de pentes vue en rasant s'etire vers l'observateur.
    let light = sun_at_point(up, i.rel, up);
    if dot(light, light) > 0.0 {
        let half_dir = normalize(sun - view);
        let flat_n = normalize(n * 0.55 + up * 0.45);
        let ndh = max(dot(flat_n, half_dir), 1e-3);
        let tan2 = (1.0 - ndh * ndh) / (ndh * ndh);
        // Plus on s'eloigne, plus de vagues tiennent dans un pixel.
        let spread = mix(0.003, 0.012, smoothstep(40.0, 6000.0, dist)) * mix(1.0, 1.5, river);
        let d = exp(-tan2 / spread) / (3.14159 * spread * ndh * ndh * ndh * ndh);
        let fres_sun = 0.02 + 0.98 * pow(1.0 - max(dot(-view, half_dir), 0.0), 5.0);
        // Scintillement : seules certaines facettes renvoient le soleil vers l'oeil.
        let sparkle_a = textureSample(noise3, noise_sampler, i.wrapped / 8.0 + vec3f(t * 0.05, 0.0, t * 0.03)).b;
        let sparkle_b = textureSample(noise3, noise_sampler, i.wrapped / 2.0 - vec3f(t * 0.11, 0.0, -t * 0.08)).b;
        let sparkle = mix(0.5, 2.6, smoothstep(0.42, 0.72, sparkle_a * 0.5 + sparkle_b * 0.5));
        col += light * d * fres_sun * 0.27 * sparkle;
    }

    // Ecume le long du rivage, visible de pres seulement.
    var foam = 0.0;
    if dist < 2500.0 && g.up.w > 0.0 {
        let wobble = textureSample(noise3, noise_sampler, i.wrapped / 16.0 + vec3f(t * 0.012)).a;
        let lace = textureSample(noise3, noise_sampler, i.wrapped / 4.0 + vec3f(t * 0.03)).g;
        foam = smoothstep(1.0, 0.0, thickness + (wobble - 0.5) * 1.6) * smoothstep(0.25, 0.65, lace + 0.2);
    }
    // Les rapides : l'eau blanchit quand le courant s'emballe.
    if dist < 3000.0 && speed > 2.2 {
        let streak = textureSample(noise3, noise_sampler, i.wrapped / 6.4 - flow * t * 0.15).a;
        foam = max(foam, smoothstep(0.75 - (speed - 2.2) * 0.1, 0.8, streak));
    }
    let foam_light = sun_light(up) * max(dot(up, sun), 0.0) + sky_ambient(up);
    col = mix(col, vec3f(0.85) * foam_light, foam * 0.75);

    var alpha = max(1.0 - exp(-thickness * 0.55), max(fresnel, foam * 0.85));
    // Le reflet du soleil est de la lumiere ajoutee : il reste entier meme sur l'eau transparente.
    alpha = max(alpha, clamp(dot(col, vec3f(0.33)) - 1.0, 0.0, 1.0));

    // Banquise polaire.
    if ice > 0.0 {
        let ice_col = vec3f(0.75, 0.82, 0.88) * (sun_light(up) * max(dot(up, sun), 0.0) + sky_ambient(up) * 0.75 + vec3f(0.02));
        col = mix(col, ice_col, ice);
        alpha = mix(alpha, 1.0, ice);
    }
    return vec4f(apply_atmosphere(col, i.rel, 5), alpha);
}
