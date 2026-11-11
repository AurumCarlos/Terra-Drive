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

// Comme `tile`, avec un filtrage anisotrope fait a la main : les cartes ne l'appliquent pas aux textures
// 3D, et le sol vu presque a plat deviendrait flou des quelques metres. Deux prises le long du grand axe
// de l'empreinte du pixel, chacune filtree pour sa part de ce grand axe.
fn tile_sharp(p: vec3f, size: f32) -> vec4f {
    let uv = p / size;
    let dx = dpdx(uv);
    let dy = dpdy(uv);
    let lx = length(dx);
    let ly = length(dy);
    let major = select(dy, dx, lx > ly);
    let minor = select(dx, dy, lx > ly);
    let ratio = max(lx, ly) / max(min(lx, ly), 1e-9);
    let along = major * max(0.35, 1.0 / ratio);
    let a = textureSampleGrad(noise3, noise_sampler, uv + major * 0.25, along, minor);
    let b = textureSampleGrad(noise3, noise_sampler, uv - major * 0.25, along, minor);
    return (a + b) * 0.5;
}

// Part de pierrier, avant le grain fin : plaques de cailloux sur les pentes et les bosses.
fn scree_hint(cells: f32, grain: f32, flat: f32) -> f32 {
    return smoothstep(0.5, 0.72, cells * 0.55 + grain * 0.45 + (1.0 - flat) * 0.6);
}

fn resolved(footprint: f32, size: f32) -> f32 {
    return smoothstep(0.6, 1.8, footprint / (size / 96.0));
}

// Relief de pixel : la normale suit la derivee d'une hauteur a l'ecran (Mikkelsen).
// Les derivees se prennent sur un carre de 2 x 2 pixels. Sur une silhouette, ce carre chevauche une
// crete proche et le relief lointain derriere elle ; sur une jupe de chunk (paroi verticale qui bouche
// les fissures entre LOD), la normale stockee, celle du sol, est perpendiculaire au triangle. Dans les
// deux cas la pente calculee est absurde et la normale bascule vers le bas : de fins traits sombres
// le long de l'horizon. La pente se borne donc, et s'efface la ou la surface ne fait pas face a sa normale.
fn bumped(n: vec3f, rel: vec3f, height: f32) -> vec3f {
    let sx = dpdx(rel);
    let sy = dpdy(rel);
    let r1 = cross(sy, n);
    let r2 = cross(n, sx);
    let det = dot(sx, r1);
    let area = max(length(cross(sx, sy)), 1e-20);
    // Pente de la hauteur sur la surface, en m/m.
    var slope = sign(det) * (dpdx(height) * r1 + dpdy(height) * r2) / max(abs(det), 1e-20);
    let steep = length(slope);
    slope *= min(1.0, 1.4 / max(steep, 1e-6));
    // |cos| entre la normale stockee et le triangle : 1 sur le sol, 0 sur une jupe.
    let facing = abs(det) / area;
    slope *= smoothstep(0.15, 0.45, facing);
    // Saut de profondeur dans le carre de pixels : la derivee melange deux surfaces.
    // Sur une surface continue, un pas d'un pixel couvre au plus l'empreinte du pixel etiree par l'incidence.
    let footprint = length(rel) * 2.0 / g.misc.w;
    let incidence = max(abs(dot(normalize(rel), cross(sx, sy) / area)), 0.02);
    let spread = max(length(sx), length(sy)) * incidence / max(footprint, 1e-6);
    slope *= 1.0 - smoothstep(4.0, 8.0, spread);
    return normalize(n - slope);
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4f {
    let dist = length(i.rel);
    let up = normalize(i.up);
    var n = normalize(i.normal);
    var albedo = i.color;
    let w = i.wrapped;

    // Plusieurs echelles, du gravier au massif. Les mips effacent seuls ce qui passe sous le pixel.
    let t8 = tile_sharp(w, 8.0);
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
        // Saisons des feuillus, vus de loin : environ la moitie de la foret, selon les massifs.
        let leaves = leaf_season(up, i.altitude, clumps - 0.5);
        let broadleaf = forest * mix(0.25, 0.75, smoothstep(0.35, 0.65, clumps));
        albedo *= mix(vec3f(1.0), mix(vec3f(3.0, 1.5, 0.3), vec3f(2.5, 0.7, 0.2), t2048.a), leaves.y * broadleaf);
        albedo = mix(albedo, vec3f(0.075, 0.062, 0.05), leaves.x * broadleaf * 0.8);
    }

    // Grain de proximite : terre, cailloux et herbes rases.
    let fine = t8.a * 0.6 + t32.a * 0.4;
    albedo *= 0.82 + 0.36 * fine;

    // Haute montagne : au-dessus des arbres, pas d'herbes hautes pour cacher le sol. Pelouse rase, mottes,
    // pierraille et roche affleurante, sur plusieurs echelles de pres, relief compris.
    let climate = textureSampleLevel(weather_map, noise_sampler, up, 0.0);
    let mean_here = climate.g * 90.0 - 50.0 - 0.006 * max(i.altitude, 0.0);
    let alpine = smoothstep(4.0, -1.0, mean_here) * (1.0 - forest);
    let micro = max(alpine, bare * 0.8) * (1.0 - smoothstep(90.0, 320.0, dist));
    if micro > 0.0 {
        let t2 = tile_sharp(w, 2.0);
        let t05 = tile_sharp(w, 0.5);
        // Cailloux : les cellules de Worley donnent des blocs arrondis ; les pierriers, des plaques.
        let stones = smoothstep(0.7, 0.75, t2.g) * smoothstep(0.35, 0.6, t32.a);
        // Gravillons nets a quelques metres : des bords francs plutot qu'un degrade.
        let near_px = 1.0 - smoothstep(6.0, 25.0, dist);
        let pebbles = smoothstep(0.76, 0.79, t05.g) * near_px * (0.35 + 0.65 * scree_hint(t32.g, t8.a, dot(n, up)));
        let scree = scree_hint(t32.g, t8.a, dot(n, up));
        // Gravier cellulaire (bords francs) sur deux echelles ; la plus fine seulement a quelques metres.
        let t015 = tile_sharp(w, 0.15);
        let grit = mix(t05.b, t015.b, 0.5 * (1.0 - smoothstep(4.0, 12.0, dist)));
        let gravel = 0.62 + 0.38 * t05.a + 0.45 * smoothstep(0.35, 0.75, grit);
        let tuft = smoothstep(0.45, 0.7, t2.a * 0.7 + t8.g * 0.3);
        let stone_color = i.rock.rgb * mix(0.7, 1.25, t05.g) * mix(vec3f(1.0), vec3f(1.08, 1.05, 1.0), t2.b);
        var ground = albedo * mix(vec3f(0.86, 0.84, 0.78), vec3f(0.92, 1.06, 0.88), tuft) * mix(1.0, gravel, 0.6);
        ground = mix(ground, stone_color * gravel, scree * 0.85 * alpine);
        ground = mix(ground, stone_color, stones);
        ground = mix(ground, stone_color * mix(0.65, 1.35, t2.b), pebbles);
        albedo = mix(albedo, ground, micro);

        // Relief en differences finies, a 2 m et 50 cm : bosses de pelouse, cailloux en saillie.
        let e2 = 2.0 / 96.0;
        let e05 = 0.5 / 96.0;
        let h2 = t2.a * 0.6 + t2.g * 0.9 * (0.4 + stones);
        let h05 = t05.a;
        let s2x = tile(w + vec3f(e2, 0.0, 0.0), 2.0);
        let s2y = tile(w + vec3f(0.0, e2, 0.0), 2.0);
        let s2z = tile(w + vec3f(0.0, 0.0, e2), 2.0);
        var grad = (vec3f(s2x.a, s2y.a, s2z.a) * 0.6 + vec3f(s2x.g, s2y.g, s2z.g) * 0.9 * (0.4 + stones) - h2) * 0.55;
        let fade05 = 1.0 - smoothstep(12.0, 40.0, dist);
        if fade05 > 0.0 {
            let dx = tile(w + vec3f(e05, 0.0, 0.0), 0.5).a - h05;
            let dy = tile(w + vec3f(0.0, e05, 0.0), 0.5).a - h05;
            let dz = tile(w + vec3f(0.0, 0.0, e05), 0.5).a - h05;
            grad += vec3f(dx, dy, dz) * 0.35 * fade05;
        }
        grad -= n * dot(grad, n);
        n = normalize(n - grad * 2.4 * micro);
    }
    // Sous les herbes, le sol est dans leur ombre : plus sombre que leurs brins. En haute montagne, il n'y a
    // plus d'herbes hautes pour l'ombrer.
    albedo *= mix(0.6, 1.0, max(max(smoothstep(180.0, 340.0, dist), 1.0 - open_land), alpine * 0.85));

    // Saisons : la neige tient sur les replats l'hiver, en plaques au bord de son domaine ; la foret la cache
    // en partie.
    let snow = snow_cover(up, i.altitude, dist);
    if snow > 0.0 {
        let patchy = smoothstep(0.25, 0.75, snow + (t128.a - 0.5) * 0.5 + (t512.a - 0.5) * 0.4);
        let lies = patchy * smoothstep(0.62, 0.82, dot(n, up)) * (1.0 - forest * 0.55);
        albedo = mix(albedo, vec3f(0.78, 0.8, 0.85) * (0.92 + 0.12 * fine), lies);
    }
    // Sol mouille : plus sombre.
    albedo *= 1.0 - 0.32 * wetness(dist) * (1.0 - snow);

    let lit = shade(albedo, n, i.rel, up);
    return vec4f(apply_atmosphere(lit, i.rel, 5), 1.0);
}
