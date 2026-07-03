#version 440

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(std140, binding = 0) uniform buf
{
    mat4 mvp;
    vec4 color;
    vec4 lightDir; // xyz: world-space direction toward the light.
} ubuf;

layout(location = 0) out vec3 v_normal;

void main()
{
    // The model transform is identity, so the normal is already world-space.
    v_normal = normal;
    gl_Position = ubuf.mvp * vec4(position, 1.0);
}
