#version 440

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 vertexColor;

layout(std140, binding = 0) uniform buf
{
    mat4 mvp;
    vec4 color; // only the alpha (opacity) is used by this variant
} ubuf;

layout(location = 0) out vec4 v_color;

void main()
{
    v_color = vec4(vertexColor, ubuf.color.a);
    gl_Position = ubuf.mvp * vec4(position, 1.0);
}
