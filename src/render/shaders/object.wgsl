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
};

@vertex
fn vs(v: VsIn) -> VsOut {
    let rel = object_position(v.pos, v.color.a, v.inst_pos, v.scale, v.yaw);
    var o: VsOut;
    o.clip = g.view_proj * vec4f(rel, 1.0);
    o.rel = rel;
    o.normal = object_axes() * yaw_rotate(v.normal, v.yaw);
    // Les arbres et les rochers ont une teinte sur quatre unites : l'automne depasse leur couleur de base.
    o.color = v.color.rgb * v.color.rgb * v.tint.rgb * v.tint.rgb * select(1.0, 4.0, v.tint.a > 0.0);
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
        let leaves = textureSample(noise3, noise_sampler, i.local / 5.5);
        let fine = textureSample(noise3, noise_sampler, i.local / 1.6);
        let clusters = leaves.g * 0.5 + fine.b * 0.5;
        // Sur la tranche du volume, le regard passe entre les feuilles.
        let rim = 1.0 - abs(dot(n, view));
        // Les palmes sont plates, et non des volumes : elles restent entieres (herb un peu au-dessus de 0).
        if near_leaf > 0.5 && i.herb < 0.1 && clusters < 0.2 + rim * rim * 0.5 {
            discard;
        }
        let e = 5.5 / 96.0;
        let grad = vec3f(
            textureSample(noise3, noise_sampler, (i.local + vec3f(e, 0.0, 0.0)) / 5.5).g,
            textureSample(noise3, noise_sampler, (i.local + vec3f(0.0, e, 0.0)) / 5.5).g,
            textureSample(noise3, noise_sampler, (i.local + vec3f(0.0, 0.0, e)) / 5.5).g,
        ) - leaves.g;
        n = normalize(n - object_axes() * grad * 6.0 * near_leaf);
        color *= mix(1.0, mix(0.42, 1.55, smoothstep(0.25, 0.7, clusters)), near_leaf);
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
        let through = (max(dot(-n, sun), 0.0) * 0.5 + against * 0.7) * i.leafy;
        lit += color * vec3f(1.2, 1.25, 0.5) * sun_at_point(up, i.rel, up) * through * 0.5;
    }
    return vec4f(apply_atmosphere(lit, i.rel, 4), 1.0);
}
