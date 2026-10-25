#version 450

#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec4 lightPos;
    vec4 camPos;
} ubo;

layout(set = 0, binding = 1) uniform ToonParams {
    vec4  shadowTint;      // rgb = shadow color multiplier, a unused
    vec4  lightColor;      // rgb = light color, a = ambient strength
    float stepThreshold;   // where light turns to shadow (0..1)
    float stepSoftness;    // width of the transition band
    float rimPower;        // rim falloff sharpness
    float rimStrength;     // rim brightness
    float rimEdge0;
    float rimEdge1;
    float specThreshold;   // specular cutoff (0..1)
    float shininess;       // specular tightness
    float specStrength;    // specular brightness
    float _pad;
} toon;

layout(set = 2, binding = 0) uniform sampler uSampler;
layout(set = 2, binding = 1) uniform texture2D uTextures[];

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) flat in uint materialId;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 tex = texture(sampler2D(uTextures[nonuniformEXT(materialId)], uSampler), fragTexCoord);
    if (tex.a < 0.5) discard;
    vec3 albedo = tex.rgb;

    vec3 N = normalize(normal);
    vec3 L = normalize(vec3(ubo.lightPos) - fragPos);

    vec3 V = normalize(vec3(ubo.camPos) - fragPos);

    // Half-Lambert, then a soft step into two tones
    float ndl = dot(N, L) * 0.5 + 0.5;
    float lit = smoothstep(toon.stepThreshold - toon.stepSoftness,
                           toon.stepThreshold + toon.stepSoftness, ndl);

    // Tinted shadow side instead of plain darkening
    vec3 shadowCol = albedo * toon.shadowTint.rgb;
    vec3 base = mix(shadowCol, albedo, lit) * toon.lightColor.rgb;

    // Small ambient floor so shadows never go fully dark
    vec3 ambient = albedo * toon.lightColor.a;

    // Rim light, kept out of the shadowed side
    //float rim = pow(1.0 - max(dot(N, V), 0.0), toon.rimPower);
    //rim = smoothstep(0.5, 0.6, rim) * lit;
    //vec3 rimCol = rim * toon.rimStrength * toon.lightColor.rgb;

    vec3 Ng = normalize(cross(dFdx(fragPos), dFdy(fragPos)));
    Ng = faceforward(Ng, -V, Ng);               // orient toward the camera, avoids Y-flip sign issues

    float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), toon.rimPower); 
    float ramp = smoothstep(toon.rimEdge0, toon.rimEdge1, fres);
    vec3 rimCol = fres * ramp * albedo * toon.rimStrength;

    // Hard-edged stylized specular
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), toon.shininess);
    spec = step(toon.specThreshold, spec) * lit;
    vec3 specCol = spec * toon.specStrength * toon.lightColor.rgb;

    outColor = vec4(base + ambient + rimCol + specCol, 1.0);
    //outColor = vec4(vec3(dot(N, Ng) > 0.5 ? 1.0 : 0.0), 1.0);
}