// Villes : facades et fenetres, toits, chaussees marquees, trottoirs, verre, eau, lumieres de nuit.
// Les materiaux et leurs parametres viennent de world/city_mesh.rs.

struct VsIn {
    @location(0) pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) color: vec4f,
    @location(7) mat: vec4u,
    @location(4) offset: vec3f,
    @location(5) size: f32,
    @location(6) wrap: vec3f,
};

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) rel: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) color: vec4f,
    @location(4) @interpolate(flat) mat: vec4u,
    @location(5) world: vec3f,
};

@vertex
fn vs(v: VsIn) -> VsOut {
    let rel = v.pos + v.offset;
    var o: VsOut;
    o.clip = g.view_proj * vec4f(rel, 1.0);
    o.rel = rel;
    o.normal = v.normal;
    o.uv = v.uv;
    // La carrosserie de la voiture porte sa position locale dans la couleur : pas de decodage.
    o.color = select(vec4f(v.color.rgb * v.color.rgb, v.color.a), v.color, v.mat.x == 18u);
    o.mat = v.mat;
    o.world = v.wrap + v.pos;
    return o;
}

const M_PLAIN: u32 = 0u;
const M_FACADE: u32 = 1u;
const M_ROOF: u32 = 2u;
const M_TILES: u32 = 3u;
const M_METAL: u32 = 4u;
const M_ROAD: u32 = 5u;
const M_WALK: u32 = 6u;
const M_KERB: u32 = 7u;
const M_GLASS: u32 = 8u;
const M_LAMP: u32 = 9u;
const M_RAILING: u32 = 10u;
const M_CROSSING: u32 = 11u;
const M_ZINC: u32 = 12u;
const M_AWNING: u32 = 13u;
const M_WATER: u32 = 14u;
const M_BEACON: u32 = 15u;
const M_HEDGE: u32 = 16u;
const M_PAINT: u32 = 17u;

fn hash2(p: vec2f) -> f32 {
    return hash3(vec3f(p, 17.13));
}

// Largeur de travee, largeur et hauteur de fenetre, allege. Doit suivre city_mesh::bay.
fn style_params(s: u32) -> vec4f {
    switch s {
        case 0u: { return vec4f(3.0, 1.4, 1.5, 0.9); }
        case 1u: { return vec4f(3.2, 1.25, 2.25, 0.35); }
        case 2u: { return vec4f(1.6, 1.5, 1.55, 1.0); }
        case 3u: { return vec4f(1.5, 1.42, 3.3, 0.25); }
        case 4u: { return vec4f(2.6, 0.95, 1.45, 0.9); }
        case 5u: { return vec4f(6.0, 4.6, 2.0, 7.0); }
        case 6u: { return vec4f(3.4, 1.2, 1.35, 0.95); }
        case 7u: { return vec4f(5.0, 1.7, 6.5, 3.2); }
        default: { return vec4f(3.0, 1.2, 1.4, 0.9); }
    }
}

struct Look {
    albedo: vec3f,
    n: vec3f,
    emit: vec3f,
    // Part de reflet du ciel (verre, eau, peinture).
    mirror: f32,
    // Eclat speculaire du soleil.
    gloss: f32,
};

// Fenetre eclairee la nuit : une sur trois environ, de teintes variees.
fn night_light(cell: vec2f, seed: f32, night: f32) -> vec3f {
    let h = hash2(cell + vec2f(seed * 7.1, seed * 3.3));
    if h > 0.36 || night <= 0.0 {
        return vec3f(0.0);
    }
    let warm = mix(vec3f(1.0, 0.62, 0.3), vec3f(0.85, 0.9, 1.0), step(0.27, h));
    return warm * (0.3 + 0.6 * hash2(cell.yx + seed)) * night;
}

fn facade(i: VsOut, base: vec3f, n: vec3f, night: f32, px: f32) -> Look {
    let style = i.mat.y & 0x7Fu;
    let shop = (i.mat.y & 0x80u) != 0u;
    let fh = max(f32(i.mat.z) / 10.0, 1.0);
    let seed = f32(i.mat.w);
    let p = style_params(style);
    let u = i.uv.x;
    let v = i.uv.y;
    // px : taille d'un pixel en metres ; au-dela, le motif se fond dans sa moyenne.
    let fade = smoothstep(0.2, 0.8, px);
    // La nuit, les fenetres allumees restent des points de lumiere plus loin encore.
    let fade_emit = smoothstep(0.35, 1.4, px);

    var look: Look;
    look.n = n;
    look.emit = vec3f(0.0);
    look.mirror = 0.0;
    look.gloss = 0.0;

    // Salissures : le pied des murs et les coulures sous les appuis.
    var wall = base * (0.88 + 0.24 * vnoise(i.world * 0.35, 4096.0 * 0.35));
    wall *= mix(0.78, 1.0, smoothstep(0.0, 2.5, v));
    if style == 5u {
        // Bardage ondule.
        wall *= 0.86 + 0.14 * sin(u * 31.4);
    }
    if style == 1u && v < fh * 2.0 {
        // Refends du rez-de-chaussee haussmannien.
        let groove = smoothstep(0.03, 0.0, abs(fract(v / 0.55) - 0.5) - 0.44);
        wall *= 1.0 - 0.3 * groove * (1.0 - fade);
    }

    let floor_i = floor(v / fh);
    let y = v - floor_i * fh;
    let col_i = floor(u / p.x);
    let x = u - (col_i + 0.5) * p.x;
    var hw = p.y * 0.5;
    var sill = p.z;
    var wh = p.z;
    sill = p.w;
    var is_shop = false;
    if shop && floor_i == 0.0 {
        is_shop = true;
        hw = p.x * 0.43;
        sill = 0.3;
        wh = min(fh - 0.85, 2.9);
    }
    // Vitrines : un bandeau d'enseigne de couleur au-dessus.
    var signboard = 0.0;
    if is_shop {
        signboard = step(sill + wh + 0.15, y) * step(y, sill + wh + 0.75);
    }

    // Distance signee au bord de la fenetre (positive dedans).
    var d = min(hw - abs(x), min(y - sill, sill + wh - y));
    if style == 7u {
        // Vitraux en plein cintre.
        let arc_y = sill + wh - hw;
        if y > arc_y {
            d = min(hw - length(vec2f(x, y - arc_y)), y - sill);
        }
    }
    let valid = v >= 0.0 && v < 400.0;
    let inside = select(0.0, smoothstep(-px * 0.5, px * 0.5, d), valid);
    let frame = select(0.0, smoothstep(-px * 0.5, px * 0.5, d + 0.09) - inside, valid);

    // Le verre : interieur sombre, rideaux, reflet du ciel ; allume la nuit.
    let cell = vec2f(col_i, floor_i);
    let r = hash2(cell + seed * 1.37);
    var interior = vec3f(0.018, 0.02, 0.024) * (0.6 + 0.9 * r);
    if r > 0.7 && !is_shop && style != 3u {
        // Rideaux tires sur le haut de la fenetre.
        interior = mix(interior, vec3f(0.3, 0.27, 0.22), step(sill + wh * (0.35 + 0.4 * fract(r * 7.0)), y));
    }
    var glass = interior;
    var lit = night_light(cell, seed, night);
    if is_shop {
        glass = vec3f(0.03, 0.028, 0.025);
        lit = vec3f(1.0, 0.75, 0.45) * 1.3 * night;
    }
    if style == 7u {
        // Vitrail : petites pieces colorees.
        let piece = floor(vec2f(x, y) / 0.35);
        let hue = hash2(piece + seed);
        glass = mix(vec3f(0.05, 0.02, 0.02), vec3f(0.02, 0.03, 0.08), hue) * (0.6 + hash2(piece.yx));
        lit = glass * 6.0 * night;
    }
    // Fenetres en retrait : l'ombre de l'embrasure sur le haut et un cote.
    let reveal = smoothstep(0.0, 0.16, min(d, sill + wh - y + 0.02));
    glass *= mix(0.45, 1.0, reveal);

    // Volets des vieilles maisons, colores selon la maison.
    var shutter = 0.0;
    var shutter_color = vec3f(0.08, 0.2, 0.1);
    if (style == 4u || style == 6u) && !is_shop && i.color.a > 0.45 {
        let sx = abs(x) - hw - 0.03;
        shutter = select(0.0, step(0.0, sx) * step(sx, hw * 0.95) * step(sill, y) * step(y, sill + wh), valid);
        let k = floor(i.color.a * 7.99);
        shutter_color = select(select(vec3f(0.1, 0.17, 0.3), vec3f(0.35, 0.08, 0.05), k > 5.0), vec3f(0.08, 0.2, 0.1), k < 4.5);
        shutter_color *= 0.8 + 0.25 * sin(y * 60.0);
    }

    let frame_color = select(vec3f(0.55, 0.53, 0.5), wall * 1.25, style == 1u || style == 4u || style == 7u);
    var color = mix(wall, frame_color, frame);
    color = mix(color, shutter_color, shutter);
    color = mix(color, select(vec3f(0.3, 0.05, 0.03), vec3f(0.04, 0.12, 0.25), seed > 128.0), signboard);
    var mirror = inside * 0.7;
    let in_glass = inside * (1.0 - shutter);
    color = mix(color, glass, in_glass);
    var emit = lit * in_glass;
    if style == 3u {
        // Mur-rideau : allege en verre opaque, montants fins, reflets partout.
        mirror = max(mirror, 0.55);
        color = mix(base * 0.7, color, inside);
    }

    // Au loin, la moyenne du motif : murs et verre melanges, une lueur diffuse la nuit.
    let coverage = clamp((2.0 * hw * wh) / (p.x * fh), 0.0, 1.0);
    let far_color = mix(wall, vec3f(0.03, 0.032, 0.036), coverage);
    let far_emit = coverage * night * select(vec3f(1.0, 0.75, 0.5) * 0.025, vec3f(1.0, 0.75, 0.45) * 0.4, is_shop);
    look.albedo = mix(color, far_color, fade);
    look.emit = mix(emit, far_emit, fade_emit);
    look.mirror = mix(mirror * in_glass + select(0.0, 0.55, style == 3u) * (1.0 - in_glass), coverage * 0.6, fade);
    look.gloss = look.mirror;
    return look;
}

// Tuiles en rangees : chaque tuile sa nuance, le bas de chaque rang dans l'ombre du suivant.
fn tiles(i: VsOut, base: vec3f, px: f32) -> vec3f {
    let fade = smoothstep(0.08, 0.3, px);
    let row = floor(i.uv.y / 0.32);
    let tile_x = floor(i.uv.x / 0.26 + row * 0.5);
    let shade = 0.78 + 0.4 * hash2(vec2f(tile_x, row));
    let edge = mix(0.6, 1.0, smoothstep(0.0, 0.1, fract(i.uv.y / 0.32)));
    let detailed = base * shade * edge;
    let moss = vnoise(i.world * 0.4, 4096.0 * 0.4);
    return mix(detailed, base * 0.85, fade) * (0.85 + 0.3 * moss);
}

// Ligne de demi-largeur w, a la distance d de son axe, sur des pixels de px metres. De loin, elle garde
// au moins la largeur d'un pixel et palit d'autant, un peu rehaussee (la peinture routiere renvoie la
// lumiere) : elle reste visible au lieu de s'effacer.
fn paint_line(d: f32, w: f32, px: f32) -> f32 {
    let aa = max(px, 0.02);
    let h = max(w, px * 0.5);
    return smoothstep(h + aa * 0.5, h - aa * 0.5, d) * min(1.0, 1.6 * w / h);
}

// Tirets de `on` (fraction peinte) par periode ; de loin, leur moyenne le long de la route.
fn dashes(s: f32, period: f32, on: f32, px: f32) -> f32 {
    return mix(step(fract(s / period), on), on, smoothstep(period * 0.05, period * 0.25, px));
}

// px : taille d'un pixel en metres en travers (x) et le long (y) de la chaussee. En vue rasante, le
// pixel s'allonge surtout le long de la route : les lignes, fines en travers, restent nettes bien plus loin.
fn road(i: VsOut, base: vec3f, px: vec2f) -> vec3f {
    let x = i.uv.x;
    let s = i.uv.y;
    let cw = f32(i.mat.z) / 4.0;
    let lanes = i.mat.y;
    let pxx = max(px.x, 1e-4);
    let pxs = max(px.y, 1e-4);
    var c = base * (0.75 + 0.5 * vnoise(i.world * 0.6, 4096.0 * 0.6)) * (0.9 + 0.2 * vnoise(i.world * 4.0, 4096.0 * 4.0));
    // Rapiecages plus sombres, ornieres plus claires.
    c *= mix(1.0, 0.7, smoothstep(0.62, 0.66, vnoise(i.world * 0.08, 4096.0 * 0.08)));
    let lane = abs(fract(abs(x) / 3.5) - 0.5);
    c *= 1.0 + 0.12 * smoothstep(0.32, 0.18, lane);
    var paint = 0.0;
    if lanes >= 8u {
        // Autoroute : x part du terre-plein. Ligne de guidage, voies en longs tirets, ligne de rive.
        let n = f32(lanes - 8u);
        let edge = 0.9 + n * 3.5;
        paint = max(paint, paint_line(abs(x - 0.75), 0.08, pxx));
        paint = max(paint, paint_line(abs(x - edge - 0.12), 0.12, pxx));
        let k = (x - 0.9) / 3.5;
        let near = abs(k - round(k)) * 3.5;
        if round(k) >= 1.0 && round(k) <= n - 1.0 {
            paint = max(paint, paint_line(near, 0.08, pxx) * dashes(s, 24.0, 0.42, pxs));
        }
        // La bande d'arret, moins usee, est plus claire ; les voies portent des traces de roues.
        if x > edge + 0.25 {
            c *= 1.18;
        } else {
            let wheel = abs(fract((x - 0.9) / 3.5) - 0.5);
            c *= 1.0 - 0.1 * smoothstep(0.3, 0.12, abs(wheel - 0.22));
        }
    } else if lanes == 7u {
        // Piste de course : lignes blanches de bord, trace plus sombre de la gomme au milieu.
        paint = max(paint, paint_line(abs(abs(x) - (cw - 0.45)), 0.1, pxx));
        c *= 1.0 - 0.18 * smoothstep(cw * 0.5, 0.0, abs(x + 0.8 * sin(s / 37.0)));
    } else if lanes >= 1u {
        // Axe en tirets (3 m peints, 6 m vides) ; double ligne continue sur les avenues.
        if lanes == 1u {
            paint = max(paint, paint_line(abs(x), 0.075, pxx) * dashes(s, 9.0, 0.34, pxs));
        } else {
            paint = max(paint, paint_line(abs(abs(x) - 0.18), 0.08, pxx));
            // Voies de 3.5 m en tirets.
            let k = abs(x) / 3.5;
            let near = abs(k - round(k)) * 3.5;
            if round(k) >= 1.0 && abs(x) < cw - 1.0 {
                paint = max(paint, paint_line(near, 0.07, pxx) * dashes(s, 9.0, 0.34, pxs));
            }
        }
        // Lignes de rive.
        paint = max(paint, paint_line(abs(abs(x) - (cw - 0.35)), 0.07, pxx));
    }
    // L'usure de la peinture ; tres loin seulement, les lignes se fondent dans la chaussee.
    let worn = mix(0.55 + 0.45 * vnoise(i.world * 1.3, 4096.0 * 1.3), 0.8, smoothstep(0.2, 1.0, pxx));
    paint *= worn * (1.0 - smoothstep(2.0, 6.0, pxx));
    return mix(c, vec3f(0.62, 0.62, 0.6), paint);
}

fn slabs(i: VsOut, base: vec3f, px: f32) -> vec3f {
    let q = i.uv / vec2f(0.5, 0.5);
    let cell = floor(q + vec2f(0.0, floor(q.x) * 0.5));
    let f = fract(q + vec2f(0.0, floor(q.x) * 0.5));
    let joint = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
    let fade = smoothstep(0.03, 0.12, px);
    let tone = 0.85 + 0.3 * hash2(cell);
    let detailed = base * tone * mix(0.6, 1.0, smoothstep(0.0, 0.05, joint));
    return mix(detailed, base * 0.95, fade) * (0.85 + 0.3 * vnoise(i.world * 0.5, 4096.0 * 0.5));
}

// ---------------------------------------------------------------------------------------------
// Voiture (vehicle/model.rs) : carrosserie dessinee au pixel depuis les coordonnees locales.

fn car_paint(k: u32) -> vec3f {
    switch k {
        case 0u: { return vec3f(0.33, 0.012, 0.01); }
        case 1u: { return vec3f(0.008, 0.03, 0.15); }
        case 2u: { return vec3f(0.40, 0.41, 0.43); }
        case 3u: { return vec3f(0.70, 0.70, 0.68); }
        case 4u: { return vec3f(0.011, 0.011, 0.013); }
        case 5u: { return vec3f(0.01, 0.065, 0.032); }
        case 6u: { return vec3f(0.58, 0.15, 0.008); }
        default: { return vec3f(0.065, 0.07, 0.075); }
    }
}

// Distance signee a un rectangle aux coins arrondis (negative dedans).
fn rbox(p: vec2f, c: vec2f, h: vec2f, r: f32) -> f32 {
    let q = abs(p - c) - h + vec2f(r);
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// 1 sur un trait de demi-largeur w centre sur d = 0.
fn stroke(d: f32, w: f32, aa: f32) -> f32 {
    return 1.0 - smoothstep(w, w + aa, abs(d));
}

fn filled(d: f32, aa: f32) -> f32 {
    return 1.0 - smoothstep(-aa, aa, d);
}

// Plaque d'immatriculation : fond blanc, bande bleue, sept caracteres en segments.
fn plate(p: vec2f, c: vec2f, seed: f32) -> vec3f {
    let q = p - c; // x de -0.26 a 0.26, y de -0.055 a 0.055
    var col = vec3f(0.72, 0.72, 0.7);
    if q.x < -0.215 {
        return vec3f(0.02, 0.06, 0.3);
    }
    let u = (q.x + 0.2) / 0.43 * 8.0;
    let cell = floor(u);
    let f = vec2f(fract(u), q.y / 0.035 * 0.5 + 0.5);
    if cell >= 0.0 && cell < 8.0 && cell != 2.0 && cell != 6.0 && f.y > 0.0 && f.y < 1.0 {
        let h = hash2(vec2f(cell, seed));
        let segs = array<vec4f, 7>(
            vec4f(0.2, 0.92, 0.8, 0.92), vec4f(0.2, 0.5, 0.8, 0.5), vec4f(0.2, 0.08, 0.8, 0.08),
            vec4f(0.2, 0.5, 0.2, 0.92), vec4f(0.8, 0.5, 0.8, 0.92), vec4f(0.2, 0.08, 0.2, 0.5), vec4f(0.8, 0.08, 0.8, 0.5));
        var ink = 0.0;
        for (var k = 0u; k < 7u; k++) {
            if fract(h * f32(3u + k * 7u)) > 0.42 {
                let s = segs[k];
                let a = s.xy;
                let b = s.zw;
                let t = clamp(dot(f - a, b - a) / dot(b - a, b - a), 0.0, 1.0);
                ink = max(ink, 1.0 - smoothstep(0.07, 0.11, length((f - a - (b - a) * t) * vec2f(1.0, 1.6))));
            }
        }
        col = mix(col, vec3f(0.01), ink);
    }
    return col;
}

// Position locale (x, y) de la carrosserie, codee sur 16 bits par axe dans la couleur.
fn car_local(c: vec4f) -> vec2f {
    return vec2f(c.r * 255.0 * 256.0 + c.g * 255.0, c.b * 255.0 * 256.0 + c.a * 255.0) / 65535.0 * 6.0 - 3.0;
}

// d : fwidth de (x, y, z), calcule hors de tout branchement.
fn car_body(i: VsOut, n: vec3f, view: vec3f, d: vec3f) -> Look {
    let xy = car_local(i.color);
    let x = xy.x;
    let y = xy.y;
    let ax = abs(x);
    let z = i.uv.x;
    let t = i.uv.y;
    let plain = (i.mat.y & 16u) != 0u;
    let lights = i.mat.z;
    let cockpit = (lights & 32u) != 0u;
    let base = car_paint(i.mat.y & 15u);
    let aa = max(length(d), 0.0012);

    var look: Look;
    look.n = n;
    look.emit = vec3f(0.0);
    // Vernis sur une peinture metallisee : paillettes fines.
    let flake = hash3(floor(vec3f(x, y, z) * 420.0));
    look.albedo = base * (0.9 + 0.2 * flake);
    look.mirror = 0.32;
    look.gloss = 1.0;

    let cabin = z > -1.80 && z < 0.98;
    // 0 peinture, 1 vitre, 2 noir brillant, 3 plastique noir mat, 4 chrome, 5 optique.
    var kind = 0u;
    var seam = 0.0;
    var lamp = vec3f(0.0);
    var paint_over = vec3f(-1.0);

    if !plain {
        if t > 0.505 && cabin {
            // Vitrages : surface vitree des portes, pare-brise, lunette, toit panoramique.
            let s = (t - 0.52) / 0.215;
            let front_edge = 0.86 - s * 0.86;
            let rear_edge = -1.62 + s * 0.5;
            let dlo = min(min(z - rear_edge, front_edge - z), min((t - 0.515) * 2.2, (0.733 - t) * 2.2));
            let side_glass = min(dlo, abs(z + 0.17) - 0.065);
            let ws = min(min(z + 0.03, 0.955 - z), (t - 0.795) * 2.2);
            let rw = min(min(z + 1.74, -1.075 - z), (t - 0.795) * 2.2);
            let roof = min(min(z + 1.07, -0.035 - z), (t - 0.81) * 2.2);
            if t < 0.515 {
                kind = 4u;
            } else if side_glass > 0.0 {
                kind = 1u;
                if side_glass < 0.01 {
                    kind = 3u;
                } else if dlo < 0.022 && t > 0.7 {
                    kind = 4u;
                }
            } else if ws > 0.0 || rw > 0.0 {
                kind = 1u;
                let e = max(ws, rw);
                // Serigraphie noire en bord de pare-brise, en points vers l'interieur.
                if e < 0.035 {
                    kind = 2u;
                }
            } else if roof > 0.0 {
                kind = 1u;
            } else {
                kind = 2u;
            }
        } else {
            let side = t > 0.19 && t < 0.505;
            // Joints de portes, poignees, trappe a carburant.
            if side && cabin {
                let zr = -1.0 - max(y - 0.55, 0.0) * 0.55;
                if y > 0.30 {
                    seam = max(seam, stroke(z - 0.92, 0.0016, aa));
                    seam = max(seam, stroke(z + 0.17, 0.0016, aa));
                    seam = max(seam, stroke(z - zr, 0.0016, aa));
                }
                if z > zr && z < 0.92 {
                    seam = max(seam, stroke(y - 0.30, 0.0016, aa));
                }
                for (var k = 0; k < 2; k++) {
                    let c = select(vec2f(-0.86, 0.85), vec2f(0.16, 0.835), k == 0);
                    let d = rbox(vec2f(z, y), c, vec2f(0.085, 0.013), 0.012);
                    if d < 0.0 {
                        kind = 4u;
                    }
                    seam = max(seam, stroke(d, 0.0012, aa) * 0.8);
                }
                if x > 0.0 {
                    seam = max(seam, stroke(length(vec2f(z + 1.62, y - 0.84)) - 0.085, 0.0014, aa));
                }
            }
            // Capot, coffre, boucliers.
            if z > 0.98 && t > 0.44 {
                seam = max(seam, stroke(z - 2.08, 0.0016, aa));
                seam = max(seam, stroke(ax - 0.80, 0.0016, aa) * step(z, 2.08));
                seam = max(seam, stroke(z - 1.0, 0.0016, aa));
            }
            if z < -1.80 && t > 0.44 {
                seam = max(seam, stroke(z + 1.81, 0.0016, aa));
                seam = max(seam, stroke(ax - 0.80, 0.0016, aa));
            }
            if z < -2.15 {
                seam = max(seam, stroke(y - 0.80, 0.0016, aa) * step(ax, 0.66));
            }
            if side && y < 0.64 {
                seam = max(seam, stroke(z - 1.96, 0.0016, aa));
            }
            if side && y < 0.76 {
                seam = max(seam, stroke(z + 1.99, 0.0016, aa));
            }
            // Passages de roues, bas de caisse, dessous.
            if t < 0.08 {
                kind = 3u;
            } else if t < 0.185 {
                kind = 3u;
            } else if y < 0.265 {
                kind = 3u;
            }
            let p = vec2f(x, y);
            if z > 1.98 {
                // Phares : optique effilee, deux projecteurs, signature lumineuse en L.
                let hc = vec2f(0.6, 0.64 - (ax - 0.6) * 0.12);
                let dh = rbox(vec2f(ax, y), hc, vec2f(0.2, 0.04), 0.03);
                if dh < 0.0 {
                    kind = 5u;
                    paint_over = vec3f(0.05, 0.05, 0.055);
                    let drl = smoothstep(-0.014, -0.008, dh) * step(y, hc.y + 0.01);
                    let proj = min(length(vec2f(ax - 0.50, y - hc.y)) - 0.026, length(vec2f(ax - 0.66, y - hc.y + 0.004)) - 0.026);
                    if proj < 0.0 {
                        paint_over = vec3f(0.3, 0.3, 0.32);
                        lamp += vec3f(1.0, 0.96, 0.88) * select(0.4, 40.0, (lights & 1u) != 0u);
                    }
                    lamp += vec3f(1.0, 0.98, 0.95) * drl * 6.0;
                    let left = x < 0.0;
                    let blink = select((lights & 16u) != 0u, (lights & 8u) != 0u, left);
                    if ax > 0.74 && blink {
                        lamp = vec3f(1.0, 0.42, 0.0) * 14.0;
                    }
                }
                // Calandre en nid d'abeilles, prise d'air, plaque, logo, antibrouillards.
                let dg = rbox(p, vec2f(0.0, 0.455), vec2f(0.4, 0.075), 0.04);
                if dg < 0.0 {
                    kind = 3u;
                    let hx = vec2f(x / 0.022, y / 0.019 + floor(x / 0.022) * 0.5);
                    let cell = abs(fract(hx) - 0.5);
                    paint_over = vec3f(0.006) + vec3f(0.035) * smoothstep(0.36, 0.44, max(cell.x, cell.y));
                }
                if abs(dg) < 0.006 {
                    kind = 4u;
                }
                let di = rbox(p, vec2f(0.0, 0.31), vec2f(0.6, 0.05), 0.03);
                if di < 0.0 {
                    kind = 3u;
                    paint_over = vec3f(0.01) * (1.0 + 2.0 * step(0.5, fract(x / 0.03 + y / 0.03)));
                }
                if rbox(p, vec2f(0.0, 0.31), vec2f(0.26, 0.055), 0.008) < 0.0 {
                    kind = 0u;
                    paint_over = plate(p, vec2f(0.0, 0.31), 3.0);
                }
                if length(vec2f(x, y - 0.56)) < 0.034 {
                    kind = 4u;
                }
                if length(vec2f(ax - 0.69, y - 0.31)) < 0.028 {
                    kind = 5u;
                    paint_over = vec3f(0.2);
                    lamp = vec3f(1.0, 0.95, 0.85) * select(0.0, 8.0, (lights & 1u) != 0u);
                }
            }
            if z < -2.04 {
                // Feux arriere a LED reliees par un bandeau, recul, plaque, diffuseur.
                let tc = vec2f(0.64, 0.885);
                let dt = rbox(vec2f(ax, y), tc, vec2f(0.21, 0.034), 0.02);
                let bar = rbox(p, vec2f(0.0, 0.885), vec2f(0.45, 0.005), 0.003);
                if dt < 0.0 || bar < 0.0 {
                    kind = 5u;
                    paint_over = vec3f(0.12, 0.004, 0.003);
                    let led = 0.6 + 0.4 * step(0.5, fract(ax / 0.02));
                    var red = select(0.25, 2.5, (lights & 1u) != 0u) * led;
                    if (lights & 2u) != 0u {
                        red = 16.0 * led;
                    }
                    lamp = vec3f(1.0, 0.03, 0.015) * red;
                    if dt < 0.0 && rbox(vec2f(ax, y), vec2f(0.5, 0.885), vec2f(0.04, 0.02), 0.005) < 0.0 {
                        paint_over = vec3f(0.3);
                        lamp = vec3f(1.0) * select(0.0, 14.0, (lights & 4u) != 0u);
                    }
                    let left = x < 0.0;
                    let blink = select((lights & 16u) != 0u, (lights & 8u) != 0u, left);
                    if dt < 0.0 && ax > 0.78 {
                        paint_over = vec3f(0.2, 0.08, 0.0);
                        lamp = select(vec3f(0.0), vec3f(1.0, 0.42, 0.0) * 14.0, blink);
                    }
                }
                if rbox(p, vec2f(0.0, 0.62), vec2f(0.26, 0.055), 0.008) < 0.0 {
                    kind = 0u;
                    paint_over = plate(p, vec2f(0.0, 0.62), 7.0);
                    lamp = vec3f(0.6, 0.6, 0.55) * select(0.0, 0.6, (lights & 1u) != 0u);
                }
                if length(vec2f(x, y - 0.79)) < 0.03 {
                    kind = 4u;
                }
                if y < 0.35 {
                    kind = 3u;
                    paint_over = vec3f(0.012) * (1.0 + 1.5 * stroke(fract(x / 0.16) - 0.5, 0.06, 0.05));
                }
                if rbox(vec2f(ax, y), vec2f(0.72, 0.40), vec2f(0.06, 0.012), 0.005) < 0.0 {
                    kind = 5u;
                    paint_over = vec3f(0.25, 0.01, 0.005);
                }
            }
        }
    }

    // L'envers de la carrosserie : l'interieur des portes, le ciel de toit.
    if dot(n, view) > 0.0 {
        if kind == 1u && cockpit {
            discard;
        }
        look.n = -n;
        look.albedo = select(vec3f(0.028, 0.026, 0.025), vec3f(0.2, 0.19, 0.18), t > 0.74);
        if t < 0.5 && abs(y - 0.76) < 0.02 {
            look.albedo = vec3f(0.06, 0.05, 0.045);
        }
        look.mirror = 0.0;
        look.gloss = 0.0;
        return look;
    }

    switch kind {
        case 1u: {
            if cockpit {
                discard;
            }
            look.albedo = vec3f(0.01, 0.012, 0.014);
            look.mirror = 0.9;
            look.gloss = 1.0;
        }
        case 2u: {
            look.albedo = vec3f(0.01);
            look.mirror = 0.35;
            look.gloss = 1.0;
        }
        case 3u: {
            look.albedo = vec3f(0.022, 0.022, 0.024);
            look.mirror = 0.03;
            look.gloss = 0.08;
        }
        case 4u: {
            look.albedo = vec3f(0.5, 0.5, 0.52);
            look.mirror = 0.95;
            look.gloss = 1.0;
        }
        case 5u: {
            look.albedo = vec3f(0.04);
            look.mirror = 0.7;
            look.gloss = 1.0;
        }
        default: {}
    }
    if paint_over.x >= 0.0 {
        look.albedo = paint_over;
        if kind == 0u {
            look.mirror = 0.15;
            look.gloss = 0.3;
        }
    }
    look.emit = lamp;
    look.albedo *= 1.0 - 0.9 * seam;
    look.mirror *= 1.0 - seam;
    return look;
}

fn car_tire(i: VsOut, aa: f32) -> vec3f {
    var c = i.color.rgb;
    if i.mat.y == 0u {
        let a = i.uv.y;
        let s = i.uv.x;
        var groove = max(stroke(a + 0.062, 0.005, aa), stroke(a + 0.021, 0.006, aa));
        groove = max(groove, max(stroke(a - 0.021, 0.006, aa), stroke(a - 0.062, 0.005, aa)));
        let sipe = stroke(fract((s + abs(a) * 0.7) / 0.03) - 0.5, 0.04, 0.06) * step(abs(a), 0.105);
        c *= 1.0 - 0.75 * max(groove, sipe * 0.55);
    } else {
        // Flanc : lettrage en relief sur une couronne.
        let r = i.uv.y;
        let ang = i.uv.x / 0.3;
        let band = step(0.285, r) * step(r, 0.302);
        let letter = band * step(0.55, hash2(vec2f(floor(ang * 70.0), floor((r - 0.285) / 0.0045))));
        c *= 1.0 + 1.4 * letter;
    }
    return c;
}

fn car_metal(i: VsOut) -> Look {
    var look: Look;
    look.n = i.normal;
    look.emit = vec3f(0.0);
    look.albedo = i.color.rgb;
    let r = i.uv.y;
    switch i.mat.y {
        case 0u: {
            look.albedo = vec3f(0.55, 0.55, 0.56);
            look.mirror = 0.95;
            look.gloss = 1.0;
        }
        case 1u: {
            look.albedo = i.color.rgb * (0.88 + 0.12 * sin(r * 2600.0));
            look.mirror = 0.55;
            look.gloss = 0.8;
        }
        case 2u: {
            let a = i.uv.x;
            let slot = stroke(fract(a * 3.0 / 6.2832 * 4.0 + r * 10.0) - 0.5, 0.03, 0.03) * step(0.1, r) * step(r, 0.186);
            look.albedo = i.color.rgb * (0.8 + 0.2 * sin(r * 3000.0)) * (1.0 - 0.7 * slot);
            look.mirror = 0.15;
            look.gloss = 0.35;
        }
        default: {
            look.mirror = 0.25;
            look.gloss = 0.7;
        }
    }
    return look;
}

fn car_cabin(i: VsOut) -> Look {
    var look: Look;
    look.n = i.normal;
    look.emit = vec3f(0.0);
    look.albedo = i.color.rgb;
    look.mirror = 0.0;
    look.gloss = 0.0;
    let uv = i.uv;
    switch i.mat.y {
        case 0u: {
            look.mirror = 0.05;
            look.gloss = 0.25;
        }
        case 2u: {
            // Combine : compteur et compte-tours a aiguilles, affichage central.
            let q = vec2f(uv.x * 0.34, uv.y * 0.11);
            var e = vec3f(0.0);
            let speed = f32(i.mat.w);
            for (var k = 0; k < 2; k++) {
                let c = select(vec2f(0.272, 0.055), vec2f(0.068, 0.055), k == 0);
                let d = q - c;
                let r = length(d);
                let ang = atan2(-d.x, -d.y);
                e += vec3f(0.5, 0.75, 1.0) * stroke(r - 0.047, 0.0012, 0.0006) * 1.5;
                let tick = step(abs(fract(ang / 6.2832 * 24.0) - 0.5), 0.12) * step(0.04, r) * step(r, 0.046) * step(abs(ang), 2.3);
                e += vec3f(0.9) * tick;
                var v = speed / 260.0;
                if k == 1 {
                    v = min(0.15 + speed / 300.0, 1.0);
                }
                let na = -2.3 + 4.6 * clamp(v, 0.0, 1.0);
                let dir = vec2f(-sin(na), -cos(na));
                let along = dot(d, dir);
                let off = abs(d.x * dir.y - d.y * dir.x);
                e += vec3f(1.0, 0.35, 0.05) * 4.0 * step(0.0, along) * step(along, 0.042) * (1.0 - smoothstep(0.0008, 0.0016, off));
            }
            let mid = rbox(q, vec2f(0.17, 0.055), vec2f(0.05, 0.035), 0.006);
            e += vec3f(0.12, 0.18, 0.25) * filled(mid, 0.001);
            look.albedo = vec3f(0.005);
            look.emit = e;
            look.mirror = 0.3;
            look.gloss = 1.0;
        }
        case 3u: {
            // Ecran central : une carte routiere.
            let p = uv * vec2f(28.0, 17.0);
            let cell = floor(p);
            let f = fract(p);
            var road = 0.0;
            if hash2(cell) > 0.72 {
                road = max(road, stroke(f.x - 0.5, 0.06, 0.05));
            }
            if hash2(cell + 7.0) > 0.72 {
                road = max(road, stroke(f.y - 0.5, 0.06, 0.05));
            }
            var e = mix(vec3f(0.02, 0.04, 0.07), vec3f(0.35, 0.4, 0.45), road);
            e = mix(e, vec3f(0.2, 0.6, 1.0), stroke(uv.y - 0.55, 0.012, 0.01) * step(0.3, uv.x));
            e += vec3f(1.0, 0.4, 0.1) * filled(length(uv - vec2f(0.5, 0.6)) - 0.03, 0.01);
            look.albedo = vec3f(0.005);
            look.emit = e * 0.9;
            look.mirror = 0.3;
            look.gloss = 1.0;
        }
        case 5u: {
            look.mirror = 0.5;
            look.gloss = 0.7;
        }
        default: {}
    }
    return look;
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4f {
    let up = normalize(i.rel - g.center.xyz);
    var n = normalize(i.normal);
    let view = normalize(i.rel);
    // Les lumieres s'allument au crepuscule, pas en plein jour.
    let night = 1.0 - smoothstep(-0.05, 0.1, dot(up, g.sun_dir.xyz));
    let m = i.mat.x;
    let px = max(length(fwidth(i.uv)), 1e-4);
    // Pour les chaussees : taille du pixel en travers et le long, separement.
    let px_road = fwidth(i.uv);
    var look: Look;
    look.albedo = i.color.rgb;
    look.n = n;
    look.emit = vec3f(0.0);
    look.mirror = 0.0;
    look.gloss = 0.0;
    let grime = 0.88 + 0.24 * vnoise(i.world * 0.5, 4096.0 * 0.5);
    // Derivees pour la voiture, prises ici en flot uniforme.
    let car_d = fwidth(vec3f(car_local(i.color), i.uv.x));

    switch m {
        case 1u: {
            look = facade(i, i.color.rgb, n, night, px);
        }
        case 2u: {
            // Toit-terrasse : gravier, flaques sombres.
            look.albedo = i.color.rgb * (0.7 + 0.5 * vnoise(i.world * 2.0, 4096.0 * 2.0)) * grime;
        }
        case 3u: {
            look.albedo = tiles(i, i.color.rgb, px);
        }
        case 4u: {
            // Tole ondulee.
            let k = sin(i.uv.x * 41.9);
            look.albedo = i.color.rgb * (0.85 + 0.15 * k) * grime;
            look.gloss = 0.15;
            look.mirror = 0.08;
        }
        case 5u: {
            look.albedo = road(i, i.color.rgb, px_road);
            look.gloss = 0.03;
        }
        case 6u: {
            look.albedo = slabs(i, i.color.rgb, px);
        }
        case 7u: {
            look.albedo = i.color.rgb * grime;
        }
        case 8u: {
            look.albedo = i.color.rgb;
            look.mirror = 0.85;
            look.gloss = 1.0;
        }
        case 9u: {
            look.albedo = i.color.rgb * 0.3;
            look.emit = vec3f(1.0, 0.82, 0.55) * 8.0 * night;
        }
        case 10u: {
            // Garde-corps : barreaux de fer, palissade blanche ou verre.
            let style = i.mat.y;
            let u = i.uv.x;
            let v = i.uv.y;
            let h = max(v, 0.0);
            if style == 2u {
                look.albedo = i.color.rgb;
                look.mirror = 0.5;
                look.gloss = 0.6;
                if px > 0.15 && hash2(floor(i.clip.xy)) > 0.5 { discard; }
            } else {
                let spacing = select(0.13, 0.16, style == 1u);
                let width = select(0.022, 0.085, style == 1u);
                let bar = abs(fract(u / spacing) - 0.5) * spacing < width * 0.5;
                let rail = h < 0.08 || h > select(0.92, 0.97, style == 1u) || (style == 1u && abs(h - 0.45) < 0.05);
                if px > 0.05 {
                    // Trop fin pour l'ecran : on garde la part couverte, en tramage.
                    let cover = select(0.3, 0.65, style == 1u);
                    if hash2(floor(i.clip.xy)) > cover { discard; }
                } else if !(bar || rail) {
                    discard;
                }
                look.albedo = i.color.rgb;
            }
            // La normale fait face a la camera : un garde-corps n'a pas d'envers.
            if dot(n, view) > 0.0 { n = -n; look.n = n; }
        }
        case 11u: {
            // Bandes du passage pieton (0) ou ligne d'arret (1).
            if i.mat.y == 0u {
                let stripe = abs(fract((i.uv.x + 0.25) / 1.0) - 0.5) < 0.25;
                if !stripe { discard; }
            }
            look.albedo = i.color.rgb * (0.6 + 0.4 * vnoise(i.world * 1.3, 4096.0 * 1.3));
        }
        case 12u: {
            // Zinc a joints debout.
            let seam = smoothstep(0.05, 0.0, abs(fract(i.uv.x / 0.55) - 0.5) - 0.45);
            look.albedo = i.color.rgb * (1.0 + 0.5 * seam * (1.0 - smoothstep(0.05, 0.2, px))) * grime;
            look.gloss = 0.2;
            look.mirror = 0.12;
        }
        case 13u: {
            let stripe = step(0.5, fract(i.uv.x / 0.6));
            look.albedo = mix(i.color.rgb, vec3f(0.6, 0.58, 0.52), stripe * (1.0 - smoothstep(0.1, 0.3, px)));
        }
        case 14u: {
            let t = g.misc.x;
            let w = i.world;
            let ripple = vec3f(
                vnoise(w * 1.7 + vec3f(t * 0.6, 0.0, 0.0), 4096.0 * 1.7) - 0.5,
                0.0,
                vnoise(w * 1.7 + vec3f(0.0, t * 0.5, 3.0), 4096.0 * 1.7) - 0.5,
            );
            look.n = normalize(n + (ripple - up * dot(ripple, up)) * 0.25);
            look.albedo = i.color.rgb;
            look.mirror = 1.0;
            look.gloss = 1.0;
        }
        case 15u: {
            let blink = step(0.55, fract(g.misc.x * 0.5 + f32(i.mat.w) * 0.1));
            look.albedo = i.color.rgb;
            look.emit = vec3f(1.0, 0.04, 0.02) * (2.0 + 40.0 * night) * blink;
        }
        case 16u: {
            // Haie : feuillage dense, bosselé.
            let leaves = textureSampleLevel(noise3, noise_sampler, i.world / 1.3, 0.0);
            look.albedo = i.color.rgb * (0.5 + 1.0 * leaves.g);
            look.n = normalize(n + (leaves.rgb - 0.5) * 0.8);
        }
        case 17u: {
            // Peinture laquee : voitures, cuves, metal peint.
            look.albedo = i.color.rgb;
            look.mirror = 0.25;
            look.gloss = 0.8;
        }
        case 18u: {
            look = car_body(i, n, view, car_d);
        }
        case 19u: {
            look.albedo = car_tire(i, px);
            look.mirror = 0.02;
            look.gloss = 0.05;
        }
        case 20u: {
            look = car_metal(i);
        }
        case 21u: {
            look = car_cabin(i);
        }
        default: {
            look.albedo = i.color.rgb * grime;
        }
    }

    var lit = shade(look.albedo, look.n, i.rel, up);
    if look.mirror > 0.0 {
        let r = reflect(view, look.n);
        let c = max(dot(-view, look.n), 0.0);
        let fresnel = 0.04 + 0.96 * pow(1.0 - c, 5.0);
        let k = look.mirror * mix(0.12, 1.0, fresnel);
        lit = mix(lit, local_sky(r, up), k);
        if look.gloss > 0.0 {
            let spec = pow(max(dot(r, g.sun_dir.xyz), 0.0), 300.0) * 40.0 + pow(max(dot(r, g.sun_dir.xyz), 0.0), 30.0) * 0.6;
            lit += sun_at_point(look.n, i.rel, up) * spec * look.gloss * k;
        }
    }
    lit += look.emit;
    return vec4f(apply_atmosphere(lit, i.rel, 4), 1.0);
}
