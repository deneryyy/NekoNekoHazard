#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout (location = 0) in vec3 inPos;

layout( push_constant ) uniform constants
{	
    mat4 model;
    uint skinPage;
	int skinDelta;
	uint jointOffset;
} pc;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
	vec4 lightPos;
	vec4 cameraPos;
    mat4 lightSpaceMatrix;
} ubo;

layout(std430, set = 0, binding = 2) readonly buffer Joints { mat4 joints[]; };

struct SkinVertex {
	uvec4 joints;
	vec4 weights;
};

layout(std430, set = 2, binding = 0) readonly buffer Skin { SkinVertex v[]; } skinPages[];

void main() {
    vec4 pos = vec4(inPos, 1.0);
	if (pc.jointOffset != 0xFFFFFFFFu) {
        SkinVertex sv = skinPages[nonuniformEXT(pc.skinPage)].v[gl_VertexIndex + pc.skinDelta];
        mat4 m = joints[pc.jointOffset + sv.joints.x] * sv.weights.x
               + joints[pc.jointOffset + sv.joints.y] * sv.weights.y
               + joints[pc.jointOffset + sv.joints.z] * sv.weights.z
               + joints[pc.jointOffset + sv.joints.w] * sv.weights.w;
        pos = m * pos;
    }

    gl_Position = ubo.lightSpaceMatrix * pc.model * pos;
}