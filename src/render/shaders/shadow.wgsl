// Rendu de profondeur des cascades. Le groupe 1 contient la matrice de la cascade en cours.

@group(1) @binding(0) var<uniform> light_view_proj: mat4x4f;

@vertex
fn vs_terrain(
    @location(0) pos: vec3f,
    @location(1) morph: vec3f,
    @location(4) offset: vec3f,
    @location(5) size: f32,
) -> @builtin(position) vec4f {
    return light_view_proj * vec4f(morphed_position(pos, morph, offset, size), 1.0);
}

@vertex
fn vs_city(@location(0) pos: vec3f, @location(4) offset: vec3f) -> @builtin(position) vec4f {
    return light_view_proj * vec4f(pos + offset, 1.0);
}

@vertex
fn vs_object(
    @location(0) pos: vec3f,
    @location(2) color: vec4f,
    @location(3) inst_pos: vec3f,
    @location(4) scale: f32,
    @location(5) yaw: f32,
) -> @builtin(position) vec4f {
    return light_view_proj * vec4f(object_position(pos, color.a, inst_pos, scale, yaw), 1.0);
}
