#version 440

layout(location = 0) in vec3 position;

layout(std140, binding = 0) uniform buf
{
    mat4 mvp;
    vec4 color;
} ubuf;

layout(location = 0) out vec4 v_color;

void main()
{
    v_color = ubuf.color;
    // Give the point-topology mesh representation a visible footprint; ignored
    // when this shader assembles lines.
    gl_PointSize = 4.0;
    gl_Position = ubuf.mvp * vec4(position, 1.0);
}
