// Passage en sortie : halo, rayons, exposition, tone mapping, etalonnage et FXAA.

@group(1) @binding(0) var hdr: texture_2d<f32>;
@group(1) @binding(1) var hdr_sampler: sampler;
@group(1) @binding(2) var bloom: texture_2d<f32>;
@group(1) @binding(3) var rays: texture_2d<f32>;

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

fn tonemapped(uv: vec2f) -> vec3f {
    var c = textureSampleLevel(hdr, hdr_sampler, uv, 0.0).rgb;
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
    dir = clamp(dir * scale, vec2f(-8.0), vec2f(8.0)) * px;

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
    let c = fxaa(i);
    // Un soupcon de bruit avant la quantification : plus de bandes dans les degrades du ciel.
    let noise = fract(52.9829189 * fract(dot(i.clip.xy, vec2f(0.06711056, 0.00583715)))) - 0.5;
    let display = pow(c, vec3f(1.0 / 2.2)) + noise / 255.0;
    return vec4f(pow(max(display, vec3f(0.0)), vec3f(2.2)), 1.0);
}
