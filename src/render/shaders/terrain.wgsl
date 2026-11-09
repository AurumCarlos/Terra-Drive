struct VsIn {
    @location(0) pos: vec3f,
    @location(1) morph: vec3f,
    @location(2) normal: vec3f,
    @location(3) color: vec4f, // a = part de foret
    @location(4) offset: vec3f,
    @location(5) size: f32,
    @location(6) wrap: vec3f,
    @location(7) rock: vec4f, // a = force des strates
    @location(8) altitude: f32,
};

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) rel: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec3f,
    @location(3) wrapped: vec3f,
    @location(4) up: vec3f,
    @location(5) forest: f32,
    @location(6) rock: vec4f,
    @location(7) altitude: f32,
};

@vertex
fn vs(v: VsIn) -> VsOut {
    let rel = morphed_position(v.pos, v.morph, v.offset, v.size);
    var o: VsOut;
    o.clip = g.view_proj * vec4f(rel, 1.0);
    o.rel = rel;
    o.normal = v.normal;
    o.color = v.color.rgb * v.color.rgb;
    o.wrapped = v.wrap + v.pos;
    o.up = normalize(rel - g.center.xyz);
    o.forest = v.color.a;
    o.rock = vec4f(v.rock.rgb * v.rock.rgb, v.rock.a);
    o.altitude = v.altitude;
    return o;
}

// Tuiles entieres de WRAP (4096 m) : le motif reste continu d'un chunk a l'autre.
fn tile(p: vec3f, size: f32) -> vec4f {
    return textureSample(noise3, noise_sampler, p / size);
}

fn resolved(footprint: f32, size: f32) -> f32 {
    return smoothstep(0.6, 1.8, footprint / (size / 96.0));
}

// Relief de pixel : la normale suit la derivee d'une hauteur a l'ecran (Mikkelsen).
fn bumped(n: vec3f, rel: vec3f, height: f32) -> vec3f {
    let sx = dpdx(rel);
    let sy = dpdy(rel);
    let r1 = cross(sy, n);
    let r2 = cross(n, sx);
    let det = dot(sx, r1);
    let grad = sign(det) * (dpdx(height) * r1 + dpdy(height) * r2);
    return normalize(abs(det) * n - grad);
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4f {
    let dist = length(i.rel);
    let up = normalize(i.up);
    var n = normalize(i.normal);
    var albedo = i.color;
    let w = i.wrapped;

    // Plusieurs echelles, du gravier au massif. Les mips effacent seuls ce qui passe sous le pixel.
    let t8 = tile(w, 8.0);
    let t32 = tile(w, 32.0);
    let t128 = tile(w, 128.0);
    let t512 = tile(w, 512.0);
    let t2048 = tile(w, 2048.0);
    // Au-dela de WRAP, le motif se cale sur la position en km, juste a grande echelle.
    let km = g.camera_km.xyz + i.rel * 0.001;
    let t16k = textureSample(noise3, noise_sampler, km / 16.3);
    let t96k = textureSample(noise3, noise_sampler, km / 97.0);

    let wide = t2048.a * 0.4 + t16k.a * 0.35 + t96k.a * 0.25;
    let mid = t128.a * 0.55 + t512.a * 0.45;

    // Foret : des cimes qui accrochent la lumiere rasante, jusque dans le lointain.
    let forest = smoothstep(0.08, 0.5, i.forest);
    let crowns = tile(w, 96.0).g;
    let clumps = t512.g;
    // Roche : des strates horizontales, etirees le long de la verticale.
    let layered = textureSample(noise3, noise_sampler, (w + up * dot(w, up) * 15.0) / 256.0).a;
    // Strates franches : elles suivent l'altitude, a peine ondulees par la position.
    let band_wide = textureSample(noise3, noise_sampler, w / 2048.0 + up * i.altitude / 150.0).a;
    let band_thin = textureSample(noise3, noise_sampler, w / 1024.0 + up * i.altitude / 34.0).a;
    let band = band_wide * 0.6 + band_thin * 0.4;
    let strata = i.rock.a;
    let layer_tint = mix(vec3f(0.58, 0.56, 0.6), vec3f(1.38, 1.24, 1.06), smoothstep(0.34, 0.66, band));

    // Une echelle ne compte qu'une fois plus fine que le pixel : agrandie, la texture 8 bits
    // donne des marches d'escalier, et la derivee en ferait des points.
    let footprint = dist * 2.0 / g.misc.w;
    let terrain_h = (t32.a - 0.5) * 2.2 * resolved(footprint, 32.0)
        + (t128.a - 0.5) * 7.0 * resolved(footprint, 128.0)
        + (t512.a - 0.5) * 26.0 * resolved(footprint, 512.0)
        + (t2048.a - 0.5) * 90.0 * resolved(footprint, 2048.0)
        + (t16k.a - 0.5) * 420.0 * resolved(footprint, 16300.0);
    let forest_h = crowns * 13.0 * resolved(footprint, 96.0) + clumps * 34.0 * resolved(footprint, 512.0);
    let rocky = 1.0 - smoothstep(0.6, 0.76, dot(n, up));
    // Une roche massive est bosselee en tous sens ; seule une roche en couches montre ses lits.
    let massive = (t128.g - 0.5) * 16.0 * resolved(footprint, 128.0) + (t512.g - 0.5) * 42.0 * resolved(footprint, 512.0);
    let bedded = (layered - 0.5) * 11.0 * resolved(footprint, 256.0) + (band_wide - 0.5) * 16.0 * resolved(footprint, 512.0);
    let rock_h = mix(massive, bedded, strata);
    n = bumped(n, i.rel, terrain_h * (0.55 + 0.45 * rocky) + forest_h * forest + rock_h * rocky);

    // De pres, les petites bosses du sol en differences finies, sans marches.
    if dist < 120.0 {
        let e = 8.0 / 96.0;
        let dx = tile(w + vec3f(e, 0.0, 0.0), 8.0).a - t8.a;
        let dy = tile(w + vec3f(0.0, e, 0.0), 8.0).a - t8.a;
        let dz = tile(w + vec3f(0.0, 0.0, e), 8.0).a - t8.a;
        var grad = vec3f(dx, dy, dz);
        grad -= n * dot(grad, n);
        n = normalize(n - grad * 2.2 * (1.0 - smoothstep(50.0, 120.0, dist)));
    }

    // Pentes vraiment raides au pixel : roche a bord irregulier.
    let slope = dot(n, up) + (t32.a - 0.5) * 0.12 + (t128.a - 0.5) * 0.1;
    let bare = (1.0 - smoothstep(0.56, 0.68, slope)) * (1.0 - forest * 0.7);
    if bare > 0.0 {
        var rock = i.rock.rgb * mix(0.62, 1.3, smoothstep(0.25, 0.75, mix(t128.a, layered, strata) * 0.7 + t32.a * 0.3));
        rock *= mix(vec3f(1.0), layer_tint, strata);
        albedo = mix(albedo, rock, bare);
    }

    // Les pays en couches portent leurs bandes jusque sur les pentes douces.
    albedo *= mix(vec3f(1.0), layer_tint, strata * strata * 0.45 * (1.0 - forest) * (1.0 - bare));

    // Nuances du sol : herbes seches et herbes fraiches par plaques. Les forets varient autrement.
    let lush = smoothstep(0.32, 0.68, mid * 0.55 + wide * 0.45);
    let open_land = (1.0 - forest) * (1.0 - bare);
    albedo *= mix(vec3f(1.0), mix(vec3f(1.13, 1.02, 0.8), vec3f(0.84, 1.0, 0.86), lush), open_land);
    albedo *= 1.0 + (mid - 0.5) * 0.5 * open_land;

    if forest > 0.0 {
        let gaps = mix(0.34, 1.3, smoothstep(0.3, 0.72, crowns));
        let hue = mix(vec3f(1.14, 0.94, 0.7), vec3f(0.84, 1.08, 1.02), smoothstep(0.3, 0.7, clumps * 0.6 + t2048.a * 0.4));
        albedo *= mix(vec3f(1.0), hue * gaps, forest);
    }

    // Grain de proximite : terre, cailloux et herbes rases.
    let fine = t8.a * 0.6 + t32.a * 0.4;
    albedo *= 0.82 + 0.36 * fine;
    // Sous les herbes, le sol est dans leur ombre : plus sombre que leurs brins.
    albedo *= mix(0.6, 1.0, max(smoothstep(180.0, 340.0, dist), 1.0 - open_land));

    let lit = shade(albedo, n, i.rel, up);
    return vec4f(apply_atmosphere(lit, i.rel, 5), 1.0);
}
