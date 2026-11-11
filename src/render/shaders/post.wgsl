// Passage en sortie : halo, rayons, exposition, tone mapping, etalonnage et FXAA.

@group(1) @binding(0) var hdr: texture_2d<f32>;
@group(1) @binding(1) var hdr_sampler: sampler;
@group(1) @binding(2) var bloom: texture_2d<f32>;
@group(1) @binding(3) var rays: texture_2d<f32>;
@group(2) @binding(0) var scene_depth: texture_depth_2d;

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.clip = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2f(p.x, 1.0 - p.y);
    return o;
}

fn aces(x: vec3f) -> vec3f {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

// L'oeil s'adapte : on ouvre quand le soleil est bas, et on reste a la valeur de jour depuis l'espace.
fn exposure() -> f32 {
    let mu = dot(g.up.xyz, g.sun_dir.xyz);
    let dusk = mix(1.9, 0.86, smoothstep(-0.2, -0.02, mu));
    let ground = mix(dusk, 0.62, smoothstep(0.03, 0.42, mu));
    return mix(ground, 0.62, smoothstep(30000.0, 200000.0, g.up.w));
}

fn rain_hash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

// Pluie ou neige qui tombe : plusieurs cylindres de gouttes autour de la camera, de 1,5 a 25 m. Chaque
// goutte est une trainee verticale (un flocon, un point qui ondule), cachee par ce qui est plus pres.
fn precipitation(uv: vec2f) -> vec3f {
    let amount = g.weather.y * (1.0 - smoothstep(1500.0, 3000.0, g.up.w));
    if amount < 0.01 || g.up.w < 0.0 {
        return vec3f(0.0);
    }
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let h = g.inv_view_proj * vec4f(ndc, 1.0, 1.0);
    let dir = normalize(h.xyz / h.w);
    let scene = scene_distance(uv);
    let up = g.up.xyz;
    let side = normalize(g.tangent.xyz - up * dot(g.tangent.xyz, up));
    let front = cross(side, up);
    let flat = dir - up * dot(dir, up);
    let across = length(flat);
    if across < 0.02 {
        return vec3f(0.0);
    }
    let snow = g.weather.z;
    let t = g.misc.x;
    let angle = atan2(dot(flat, front), dot(flat, side));
    let light = sky_ambient(up) * 0.9 + sun_light(up) * max(dot(up, g.sun_dir.xyz), 0.0) * 0.12;
    var sum = 0.0;
    for (var k = 0; k < 4; k++) {
        let radius = 1.5 * pow(2.6, f32(k));
        let travel = radius / across;
        if travel > scene {
            continue;
        }
        // Coordonnees sur le cylindre : le tour en metres, et la hauteur.
        let around = angle * radius;
        let height = dot(dir, up) * travel;
        let spacing = mix(0.22, 0.3, snow) * (1.0 + f32(k) * 0.4);
        let column = floor(around / spacing);
        let jitter = rain_hash(vec2f(column, f32(k) * 17.0));
        let speed = mix(9.0, 1.1, snow);
        let length_m = 0.55 + 0.1 * f32(k);
        let period = mix(3.0, 0.7, snow) + jitter * mix(2.0, 0.6, snow);
        let fall = (height + t * speed + jitter * 40.0) / period;
        let along = fract(fall) * period;
        let wobble = snow * sin(t * 1.7 + jitter * 30.0 + height) * 0.08;
        let x = fract(around / spacing) - 0.5 - (jitter - 0.5) * 0.6 + wobble / spacing;
        // Epaisseur a l'ecran : une goutte reste visible meme loin, en s'eclaircissant.
        let pixel = travel * 2.0 / g.misc.w;
        let width = max(mix(0.004, 0.018, snow), pixel * 0.7);
        let dx = x * spacing;
        // Une goutte : une trainee verticale. Un flocon : un disque.
        let drop = (1.0 - smoothstep(width * 0.5, width, abs(dx)))
            * (1.0 - smoothstep(0.0, length_m, along)) * smoothstep(-0.02, 0.0, along);
        let flake = 1.0 - smoothstep(width * 0.4, width, length(vec2f(dx, along - width)));
        let core = mix(drop, flake, snow);
        let streak = 1.0;
        let dense = step(rain_hash(vec2f(column, floor(fall) + f32(k) * 7.0)), amount * mix(0.7, 0.9, snow));
        let fade = 1.0 - f32(k) * 0.18;
        sum += core * streak * dense * fade * min(width / max(pixel, 1e-4), 1.0);
    }
    return light * sum * mix(0.35, 1.1, snow);
}

// Distance de la scene sous un pixel, infinie pour le ciel.
fn scene_distance(uv: vec2f) -> f32 {
    let size = vec2f(textureDimensions(scene_depth));
    let raw = textureLoad(scene_depth, vec2i(clamp(uv * size, vec2f(0.0), size - 1.0)), 0);
    if raw <= 0.0 {
        return 1e9;
    }
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let q = g.inv_view_proj * vec4f(ndc, raw, 1.0);
    return length(q.xyz / q.w);
}

// Pluie et neige bouchent la vue : un rideau gris a quelques kilometres, qui voile aussi le ciel.
fn rain_veil(c: vec3f, uv: vec2f) -> vec3f {
    let rain = g.weather.y * (1.0 - smoothstep(3000.0, 12000.0, g.up.w));
    if rain <= 0.0 {
        return c;
    }
    let d = scene_distance(uv);
    let reach = mix(9000.0, 1600.0, rain) * mix(1.0, 0.45, g.weather.z);
    let sky = select(1.0, 0.8, d > 1e8);
    return mix(c, rain_haze(g.up.xyz), (1.0 - exp(-d / reach)) * min(rain * 1.5, 0.92) * sky);
}

fn tonemapped(uv: vec2f) -> vec3f {
    var c = rain_veil(textureSampleLevel(hdr, hdr_sampler, uv, 0.0).rgb, uv);
    // Six niveaux additionnes : la moyenne reste a l'echelle de l'image.
    let glow = textureSampleLevel(bloom, hdr_sampler, uv, 0.0).rgb / 6.0;
    c = mix(c, glow, 0.075);
    c += textureSampleLevel(rays, hdr_sampler, uv, 0.0).rgb;
    if g.up.w < 0.0 {
        c = c * vec3f(0.22, 0.5, 0.6) + vec3f(0.002, 0.012, 0.02);
    }
    let exposed = c * exposure();
    // Canal par canal, le rouge et le vert saturent ensemble et un coucher de soleil vire au citron.
    // On garde donc une part de tone mapping sur la seule luminance, qui preserve la teinte orangee.
    let input_l = max(luma(exposed), 1e-4);
    let by_luma = min(exposed * aces(vec3f(input_l)).x / input_l, vec3f(1.0));
    let mapped = mix(aces(exposed), by_luma, 0.55);
    // ACES delave les couleurs vives : on en rend une partie.
    let l = luma(mapped);
    var graded = max(mix(vec3f(l), mapped, 1.12), vec3f(0.0));
    // Ombres un peu froides, lumieres un peu chaudes, comme sur une pellicule.
    graded *= mix(vec3f(0.95, 0.99, 1.06), vec3f(1.07, 0.965, 0.9), smoothstep(0.06, 0.7, l));
    let v = uv - 0.5;
    return graded * (1.0 - dot(v, v) * 0.42);
}

fn luma(c: vec3f) -> f32 {
    return dot(c, vec3f(0.299, 0.587, 0.114));
}

fn fxaa(i: VsOut) -> vec3f {
    let px = 1.0 / g.misc.zw;
    let m = tonemapped(i.uv);
    let l_m = luma(m);
    let l_nw = luma(tonemapped(i.uv + vec2f(-px.x, -px.y)));
    let l_ne = luma(tonemapped(i.uv + vec2f(px.x, -px.y)));
    let l_sw = luma(tonemapped(i.uv + vec2f(-px.x, px.y)));
    let l_se = luma(tonemapped(i.uv + vec2f(px.x, px.y)));
    let l_min = min(l_m, min(min(l_nw, l_ne), min(l_sw, l_se)));
    let l_max = max(l_m, max(max(l_nw, l_ne), max(l_sw, l_se)));

    // Pas de contraste local : aucune arete a lisser.
    if l_max - l_min < max(0.03, l_max * 0.1) {
        return m;
    }

    var dir = vec2f(-((l_nw + l_ne) - (l_sw + l_se)), (l_nw + l_sw) - (l_ne + l_se));
    let reduce = max((l_nw + l_ne + l_sw + l_se) * 0.03, 1.0 / 128.0);
    let scale = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    // Portee courte : sur un sol texture (gravier, herbe), une longue recherche voit des aretes partout et
    // etale la texture en trainees floues. Quelques pixels suffisent aux aretes de la geometrie.
    dir = clamp(dir * scale, vec2f(-2.5), vec2f(2.5)) * px;

    let a = 0.5 * (tonemapped(i.uv + dir * (1.0 / 3.0 - 0.5)) + tonemapped(i.uv + dir * (2.0 / 3.0 - 0.5)));
    let b = a * 0.5 + 0.25 * (tonemapped(i.uv - dir * 0.5) + tonemapped(i.uv + dir * 0.5));
    let l_b = luma(b);
    if l_b < l_min || l_b > l_max {
        return a;
    }
    return b;
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4f {
    var c = fxaa(i);
    // Les gouttes, une seule fois par pixel : FXAA les lisserait en bouillie.
    let drops = precipitation(i.uv) * exposure();
    c += (vec3f(1.0) - c) * min(drops, vec3f(1.0));
    // Un soupcon de bruit avant la quantification : plus de bandes dans les degrades du ciel.
    let noise = fract(52.9829189 * fract(dot(i.clip.xy, vec2f(0.06711056, 0.00583715)))) - 0.5;
    let display = pow(c, vec3f(1.0 / 2.2)) + noise / 255.0;
    return vec4f(pow(max(display, vec3f(0.0)), vec3f(2.2)), 1.0);
}
