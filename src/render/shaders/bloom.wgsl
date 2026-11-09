// Halo lumineux : une chaine de reductions puis d'agrandissements de l'image HDR.

@group(1) @binding(0) var src: texture_2d<f32>;
@group(1) @binding(1) var src_sampler: sampler;

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

fn at(uv: vec2f) -> vec3f {
    return textureSampleLevel(src, src_sampler, uv, 0.0).rgb;
}

fn down13(uv: vec2f) -> vec3f {
    let p = 1.0 / vec2f(textureDimensions(src));
    let a = at(uv + p * vec2f(-2.0, -2.0));
    let b = at(uv + p * vec2f(0.0, -2.0));
    let c = at(uv + p * vec2f(2.0, -2.0));
    let d = at(uv + p * vec2f(-2.0, 0.0));
    let e = at(uv);
    let f = at(uv + p * vec2f(2.0, 0.0));
    let h = at(uv + p * vec2f(-2.0, 2.0));
    let i = at(uv + p * vec2f(0.0, 2.0));
    let j = at(uv + p * vec2f(2.0, 2.0));
    let k = at(uv + p * vec2f(-1.0, -1.0));
    let l = at(uv + p * vec2f(1.0, -1.0));
    let m = at(uv + p * vec2f(-1.0, 1.0));
    let n = at(uv + p * vec2f(1.0, 1.0));
    return e * 0.125 + (a + c + h + j) * 0.03125 + (b + d + f + i) * 0.0625 + (k + l + m + n) * 0.125;
}

@fragment
fn fs_down(i: VsOut) -> @location(0) vec4f {
    return vec4f(down13(i.uv), 1.0);
}

// Premiere reduction : un pixel isole tres brillant ne doit pas clignoter dans le halo.
@fragment
fn fs_down_first(i: VsOut) -> @location(0) vec4f {
    let c = down13(i.uv);
    let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
    return vec4f(c * min(1.0, 24.0 / max(l, 1e-3)), 1.0);
}

@fragment
fn fs_up(i: VsOut) -> @location(0) vec4f {
    let p = 1.0 / vec2f(textureDimensions(src));
    var c = at(i.uv) * 0.25;
    c += (at(i.uv + vec2f(p.x, 0.0)) + at(i.uv - vec2f(p.x, 0.0)) + at(i.uv + vec2f(0.0, p.y)) + at(i.uv - vec2f(0.0, p.y))) * 0.125;
    c += (at(i.uv + p) + at(i.uv - p) + at(i.uv + vec2f(p.x, -p.y)) + at(i.uv + vec2f(-p.x, p.y))) * 0.0625;
    return vec4f(c, 1.0);
}

// Rayons du soleil : ce qui brille dans le ciel est etire vers l'astre, les nuages y decoupent des ombres.
@group(2) @binding(0) var scene_depth: texture_depth_2d;

@fragment
fn fs_rays(i: VsOut) -> @location(0) vec4f {
    let sun_clip = g.view_proj * vec4f(g.sun_dir.xyz * 1.0e6, 1.0);
    let camera_up = g.up.xyz;
    let mu = dot(camera_up, g.sun_dir.xyz);
    if sun_clip.w <= 0.0 || g.up.w < 0.0 || g.up.w > 60000.0 {
        return vec4f(0.0);
    }
    let sun_uv = sun_clip.xy / sun_clip.w * vec2f(0.5, -0.5) + 0.5;
    let off = max(max(-sun_uv.x, sun_uv.x - 1.0), max(-sun_uv.y, sun_uv.y - 1.0));
    let strength = (1.0 - smoothstep(0.0, 0.5, off)) * smoothstep(-0.03, 0.03, mu) * 0.16 * (1.0 - smoothstep(0.12, 0.45, mu));
    if strength <= 0.0 {
        return vec4f(0.0);
    }
    let size = vec2f(textureDimensions(scene_depth));
    let jitter = fract(52.9829189 * fract(dot(i.clip.xy, vec2f(0.06711056, 0.00583715))));
    var sum = 0.0;
    var w = 1.0;
    let steps = 36;
    for (var k = 0; k < steps; k++) {
        let f = (f32(k) + jitter) / f32(steps) * 0.9;
        let uv = mix(i.uv, sun_uv, f);
        if all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0)) {
            if textureLoad(scene_depth, vec2i(uv * size), 0) == 0.0 {
                let c = textureSampleLevel(src, src_sampler, uv, 0.0).rgb;
                sum += clamp(dot(c, vec3f(0.3, 0.5, 0.2)) - 1.6, 0.0, 9.0) * w;
            }
        }
        w *= 0.97;
    }
    let tint = sun_light(camera_up);
    return vec4f(tint * sum / f32(steps) * strength, 1.0);
}
