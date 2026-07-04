#version 440

layout(location = 0) in float v_scalar;

layout(binding = 1) uniform sampler2D lut;

layout(std140, binding = 0) uniform buf
{
    mat4 mvp;
    // The scalar variant packs the mapping range into the color slot:
    // (vmin, 1 / (vmax - vmin), unused, unused).
    vec4 color;
} ubuf;

layout(location = 0) out vec4 fragColor;

void main()
{
    float t = clamp((v_scalar - ubuf.color.x) * ubuf.color.y, 0.0, 1.0);
    fragColor = texture(lut, vec2(t, 0.5));
}
