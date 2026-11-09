// Fond de l'image : espace, etoiles, soleil et atmosphere vue de la camera.

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) ndc: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var o: VsOut;
    o.clip = vec4f(p, 0.0, 1.0);
    o.ndc = p;
    return o;
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4f {
    let h = g.inv_view_proj * vec4f(i.ndc, 1.0, 1.0);
    let dir = normalize(h.xyz / h.w);
    let ro = g.camera_km.xyz;

    let ground = ray_sphere(ro, dir, planet_km());
    let hits_planet = ground.x > 0.0 && ground.y > 0.0;
    let tmax = select(1.0e9, ground.x, hits_planet);

    let s = atmosphere(ro, dir, tmax, 14);
    var col = s.inscatter;
    if !hits_planet {
        let cos_sun = dot(dir, g.sun_dir.xyz);
        col += vec3f(40.0, 36.0, 30.0) * smoothstep(0.99985, 0.99995, cos_sun) * s.transmittance;

        // Etoiles fixes, effacees par un ciel lumineux.
        let scaled = dir * 450.0;
        let cell = floor(scaled);
        let star = hash3(cell);
        if star > 0.9975 {
            // Un point rond et decentre dans sa cellule, pas la cellule entiere.
            let jitter = vec3f(hash3(cell + 11.0), hash3(cell + 23.0), hash3(cell + 37.0)) * 0.6 + 0.2;
            let d = length(scaled - cell - jitter);
            let size = 0.12 + 0.2 * hash3(cell + 53.0);
            let tint = mix(vec3f(1.0, 0.82, 0.68), vec3f(0.72, 0.84, 1.0), hash3(cell + 71.0));
            let glare = 1.0 - clamp(dot(s.inscatter, vec3f(0.33)) * 30.0, 0.0, 1.0);
            let power = pow(hash3(cell + 97.0), 3.0) * 2.4 + 0.25;
            col += tint * power * smoothstep(size, size * 0.3, d) * glare * s.transmittance;
        }
    }
    return vec4f(col, 1.0);
}
