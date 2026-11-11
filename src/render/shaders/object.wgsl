struct VsIn {
    @location(0) pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f, // a = sensibilite au vent
    @location(3) inst_pos: vec3f,
    @location(4) scale: f32,
    @location(5) yaw: f32,
    @location(6) tint: vec4f,  // a = 0 pour les herbes
};

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) rel: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec3f,
    @location(3) leafy: f32,
    @location(4) tip: f32,
    @location(5) herb: f32,
    @location(6) local: vec3f,
    @location(7) bare: f32,
};

@vertex
fn vs(v: VsIn) -> VsOut {
    // Sous la neige, les herbes disparaissent : elles s'aplatissent jusqu'a ne plus depasser.
    var pos = v.pos;
    if v.tint.a == 0.0 {
        let at = v.inst_pos - g.center.xyz;
        pos.y *= 1.0 - 0.95 * snow_cover(normalize(at), length(at) - g.center.w, length(v.inst_pos));
    }
    let rel = object_position(pos, v.color.a, v.inst_pos, v.scale, v.yaw, v.tint.a);
    var o: VsOut;
    o.clip = g.view_proj * vec4f(rel, 1.0);
    o.rel = rel;
    let stretch = select(vec3f(1.0), object_shape(v.yaw).xyz, v.tint.a > 0.0);
    o.normal = object_axes() * yaw_rotate(v.normal / stretch, v.yaw);
    // Les arbres et les rochers ont une teinte sur quatre unites, pour varier autour de leur couleur de base.
    var color = v.color.rgb * v.color.rgb * v.tint.rgb * v.tint.rgb * select(1.0, 4.0, v.tint.a > 0.0);
    // Saisons des feuillus : jeunes feuilles claires au printemps, or, orange ou rouge a l'automne, puis
    // houppier nu, gris-brun de loin.
    o.bare = 0.0;
    let code = deciduous_code(v.tint.a);
    if code > 0.0 && v.color.a > 0.04 {
        let at = v.inst_pos - g.center.xyz;
        let s = leaf_season(normalize(at), length(at) - g.center.w, leaf_jitter(v.yaw));
        let h = fract(v.yaw * 5.31);
        var fall = vec3f(3.0, 1.75 - 0.95 * h, 0.25);
        if code > 1.5 && code < 2.5 {
            fall = mix(vec3f(3.0, 1.6, 0.25), vec3f(2.3, 0.55, 0.12), h);
        } else if code > 2.5 {
            fall = mix(vec3f(3.4, 0.95, 0.18), vec3f(3.1, 0.32, 0.14), h);
        }
        color *= mix(vec3f(1.0), fall, s.y);
        color *= mix(vec3f(1.0), vec3f(1.35, 1.35, 0.75), s.z);
        color = mix(color, vec3f(0.085, 0.07, 0.058), s.x);
        o.bare = s.x;
    }
    o.color = color;
    o.leafy = min(v.color.a * 8.0, 1.0);
    o.tip = v.color.a;
    o.herb = 1.0 - v.tint.a;
    // Chaque arbre a son propre decalage dans le bruit, sinon tous auraient les memes feuilles.
    o.local = v.pos + vec3f(v.yaw * 7.0, v.scale * 13.0, v.yaw * 3.0);
    return o;
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4f {
    var n = normalize(i.normal);
    let up = g.up.xyz;
    let sun = g.sun_dir.xyz;
    let view = normalize(i.rel);
    var color = i.color;

    // Feuillage : paquets de feuilles sombres et clairs, bords ajoures. S'efface au loin.
    let dist = length(i.rel);
    let near_leaf = i.leafy * (1.0 - i.herb) * (1.0 - smoothstep(120.0, 240.0, dist));
    if near_leaf > 0.0 {
        // Trois echelles : bouquets de rameaux, grappes de feuilles, feuilles.
        let leaves = textureSample(noise3, noise_sampler, i.local / 3.2);
        let fine = textureSample(noise3, noise_sampler, i.local / 0.9);
        let micro = textureSample(noise3, noise_sampler, i.local / 0.3);
        let clusters = leaves.g * 0.45 + fine.b * 0.35 + micro.r * 0.2;
        // Sur la tranche du volume, le regard passe entre les feuilles.
        let rim = 1.0 - abs(dot(n, view));
        // Les palmes sont plates, et non des volumes : elles restent entieres (herb un peu au-dessus de 0).
        if near_leaf > 0.5 && i.herb < 0.1 {
            // Un feuillu qui perd ses feuilles ne garde qu'un fin reseau de brindilles.
            let twig = abs(fine.r - 0.5) * 2.0;
            if twig > mix(1.01, 0.09, i.bare) || clusters < 0.27 + rim * rim * 0.6 - 0.2 * i.bare {
                discard;
            }
        }
        let e = 3.2 / 96.0;
        let grad = vec3f(
            textureSample(noise3, noise_sampler, (i.local + vec3f(e, 0.0, 0.0)) / 3.2).g,
            textureSample(noise3, noise_sampler, (i.local + vec3f(0.0, e, 0.0)) / 3.2).g,
            textureSample(noise3, noise_sampler, (i.local + vec3f(0.0, 0.0, e)) / 3.2).g,
        ) - leaves.g;
        n = normalize(n - object_axes() * grad * 5.0 * near_leaf * (1.0 - i.bare));
        // Feuilles claires et creux sombres, et le grain des feuilles elles-memes.
        let speckle = mix(0.4, 1.55, smoothstep(0.3, 0.68, clusters)) * (0.72 + 0.56 * micro.g);
        color *= mix(1.0, speckle, near_leaf * (1.0 - i.bare));
        color *= mix(vec3f(1.0), mix(vec3f(1.2, 1.0, 0.72), vec3f(0.85, 1.05, 1.0), leaves.a), near_leaf);
    } else if i.herb < 0.5 && i.leafy == 0.0 && dist < 240.0 {
        // Ecorce et roche : un grain etire le long du tronc, des bosses qui cassent les facettes.
        let stretched = i.local * vec3f(1.0, 0.16, 1.0);
        let coarse = textureSample(noise3, noise_sampler, stretched / 2.4);
        let grain = textureSample(noise3, noise_sampler, stretched / 0.5);
        let e = 2.4 / 96.0;
        let grad = vec3f(
            textureSample(noise3, noise_sampler, (stretched + vec3f(e, 0.0, 0.0)) / 2.4).a,
            textureSample(noise3, noise_sampler, (stretched + vec3f(0.0, e, 0.0)) / 2.4).a,
            textureSample(noise3, noise_sampler, (stretched + vec3f(0.0, 0.0, e)) / 2.4).a,
        ) - coarse.a;
        n = normalize(n - object_axes() * grad * 3.0);
        color *= 0.55 + 0.45 * coarse.a + 0.55 * grain.g;
    }

    // L'hiver, la neige coiffe les branches et les rochers, et recouvre les herbes.
    let world_up = normalize(i.rel - g.center.xyz);
    let snow = snow_cover(world_up, length(i.rel - g.center.xyz) - g.center.w, dist);
    if snow > 0.0 {
        let on_top = select(smoothstep(0.15, 0.6, dot(n, up)), 1.0, i.herb > 0.5);
        color = mix(color, vec3f(0.74, 0.77, 0.82), snow * on_top * 0.9);
    }
    color *= 1.0 - 0.25 * wetness(dist) * (1.0 - snow) * (1.0 - i.leafy);

    // Face au soleil, toute plante mince devient un filtre de lumiere.
    let facing = max(dot(view, sun), 0.0);
    let against = facing * facing * facing;

    var lit: vec3f;
    if i.herb > 0.5 {
        // Une herbe n'a pas de face : elle recoit le soleil rasant sur toute sa hauteur.
        let light = sun_at_point(up, i.rel, up);
        let wrap = max(dot(up, sun), 0.0) * 0.6 + 0.3;
        lit = color * (light * wrap + ambient_light(up, up));
        lit += color * vec3f(1.35, 1.0, 0.6) * light * (0.1 + against * 0.85) * (0.2 + i.tip);
    } else {
        lit = shade(color, n, i.rel, up);
        // Le soleil traverse les feuilles : un feuillage a contre-jour s'eclaire au lieu de noircir.
        let through = (max(dot(-n, sun), 0.0) * 0.5 + against * 0.7) * i.leafy * (1.0 - i.bare);
        lit += color * vec3f(1.2, 1.25, 0.5) * sun_at_point(up, i.rel, up) * through * 0.5;
    }
    return vec4f(apply_atmosphere(lit, i.rel, 4), 1.0);
}
