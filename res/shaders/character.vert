#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

struct Vertex {
	vec3 position;
	float uv1;
	vec3 normal;
	float uv2;
}; 

layout(buffer_reference, std430) readonly buffer VertexBuffer { 
	Vertex vertices[];
};

layout( push_constant ) uniform constants
{	
	VertexBuffer vertexBuffer;
	uint materialId;
	mat4 model;
	uint skinPage;
	int skinDelta;
	uint jointOffset;
} pc;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(std430, set = 0, binding = 2) readonly buffer Joints { mat4 joints[]; };

struct SkinVertex {
	uvec4 joints;
	vec4 weights;
};

layout(std430, set = 3, binding = 0) readonly buffer Skin { SkinVertex v[]; } skinPages[];


layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out uint materialId;
layout(location = 2) out vec3 normal;
layout(location = 3) out vec3 fragPos;

void main() {

    Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
	vec4 pos = vec4(v.position, 1.0);
    vec3 nrm = v.normal;
	if (pc.jointOffset != 0xFFFFFFFFu) {
        SkinVertex sv = skinPages[nonuniformEXT(pc.skinPage)].v[gl_VertexIndex + pc.skinDelta];
        mat4 m = joints[pc.jointOffset + sv.joints.x] * sv.weights.x
               + joints[pc.jointOffset + sv.joints.y] * sv.weights.y
               + joints[pc.jointOffset + sv.joints.z] * sv.weights.z
               + joints[pc.jointOffset + sv.joints.w] * sv.weights.w;
        pos = m * pos;
        nrm = mat3(m) * nrm;
    }

	gl_Position =  ubo.proj * ubo.view * pc.model * pos;
	fragTexCoord = vec2(v.uv1, v.uv2);
	materialId = pc.materialId;
	normal = mat3(transpose(inverse(pc.model))) * nrm;
	fragPos = vec3(pc.model * pos);
}