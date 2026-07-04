#version 440

layout(location = 0) in vec3 position;
layout(location = 1) in float scalar;

layout(std140, binding = 0) uniform buf
{
    mat4 mvp;
    // The scalar variant packs the mapping range into the color slot:
    // (vmin, 1 / (vmax - vmin), unused, unused).
    vec4 color;
} ubuf;

layout(location = 0) out float v_scalar;

void main()
{
    v_scalar = scalar;
    gl_Position = ubuf.mvp * vec4(position, 1.0);
}
