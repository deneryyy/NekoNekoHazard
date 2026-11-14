#version 450

#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
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

layout(set = 1, binding = 0) uniform sampler uSampler;
layout(set = 1, binding = 1) uniform texture2D uTextures[];

struct Light {
    vec4 positionRange;
    vec4 colorIntensity;
    vec4 directionType;
    vec4 spotParams;
};
layout(std430, set = 3, binding = 0) readonly buffer Lights { Light lights[]; };
layout(set = 3, binding = 1) uniform LightingParams {
    mat4 invProj;
    vec2 screenSize;
    float near;
    float far;
    uvec4 grid;
    vec4 dirDirection;
    vec4 dirColorIntensity;
    uint lightCount;
} lighting;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) flat in uint materialId;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 fragPos;
layout(location = 4) in vec4 baseColorFactor;

layout(location = 0) out vec4 outColor;

float attenuation(float dist, float range) {
    float x = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
    return x * x;
}

float spotFactor(Light l, vec3 L) {          
    float cd = dot(-L, l.directionType.xyz);
    return smoothstep(l.spotParams.y, l.spotParams.x, cd);
}

float toonRamp(vec3 N, vec3 L) {
    float ndl = dot(N, L) * 0.5 + 0.5;
    return smoothstep(toon.stepThreshold - toon.stepSoftness,
                      toon.stepThreshold + toon.stepSoftness, ndl);
}

float toonSpec(vec3 N, vec3 V, vec3 L, float lit) {
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), toon.shininess);
    return step(toon.specThreshold, spec) * lit;
}

vec3 shadeLight(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo) {
    float lit = toonRamp(N, L);

    // No shadow tint here: with many lights it would stack up.
    // The ambient term provides the shadow floor instead.
    vec3 diffuse = albedo * lit * radiance;

    float spec = toonSpec(N, V, L, lit);
    return diffuse + spec * toon.specStrength * radiance;
}

void main() {
    vec4 tex = texture(sampler2D(uTextures[nonuniformEXT(materialId)], uSampler), fragTexCoord) * baseColorFactor;
    if (tex.a < 0.5) discard;
    vec3 albedo = tex.rgb;

    vec3 N = normalize(normal);
    vec3 V = normalize(ubo.camPos.xyz - fragPos);

    // Ambient with tinted shadow color, once
    vec3 color = albedo * toon.shadowTint.rgb * toon.lightColor.a;

    if (lighting.dirDirection.w > 0.5) {
        vec3 L = normalize(lighting.dirDirection.xyz);
        vec3 radiance = lighting.dirColorIntensity.rgb * lighting.dirColorIntensity.w;

        float shadow = 1.0; // TODO shadow mapping
        float lit = toonRamp(N, L) * shadow;

        // The one light that uses tinted shadow instead of plain darkening
        color += mix(albedo * toon.shadowTint.rgb, albedo, lit) * radiance;
        color += toonSpec(N, V, L, lit) * toon.specStrength * radiance;
    }

    for (uint i = 0u; i < lighting.lightCount; i++) {
        Light l = lights[i];
        vec3 toL = l.positionRange.xyz - fragPos;
        float dist = length(toL);
        float range = l.positionRange.w;
        if (dist >= range) continue;

        vec3 L = toL / dist;
        float att = attenuation(dist, range);
        if (l.directionType.w > 0.5) att *= spotFactor(l, L);

        vec3 radiance = l.colorIntensity.rgb * l.colorIntensity.w * att;
        color += shadeLight(N, V, L, radiance, albedo);
    }

    // Rim, once
    float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), toon.rimPower);
    float ramp = smoothstep(toon.rimEdge0, toon.rimEdge1, fres);
    color += fres * ramp * albedo * toon.rimStrength;

    outColor = vec4(color, 1.0);
}