#version 440

layout(location = 0) in vec3 v_normal;

layout(std140, binding = 0) uniform buf
{
    mat4 mvp;
    vec4 color;
    vec4 lightDir;
} ubuf;

layout(location = 0) out vec4 fragColor;

void main()
{
    vec3 n = normalize(v_normal);
    vec3 l = normalize(ubuf.lightDir.xyz);
    // Two-sided Lambert so a facet reads lit regardless of its winding, over a
    // fixed ambient floor so faces turned away from the light are not black.
    float ambient = 0.3;
    float diffuse = (1.0 - ambient) * abs(dot(n, l));
    fragColor = vec4(ubuf.color.rgb * (ambient + diffuse), ubuf.color.a);
}
