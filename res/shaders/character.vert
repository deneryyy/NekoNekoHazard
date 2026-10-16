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
} PushConstants;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out uint materialId;
layout(location = 2) out vec3 normal;
layout(location = 3) out vec3 fragPos;

void main() {

    Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];

	gl_Position =  ubo.proj * ubo.view * PushConstants.model * vec4(v.position, 1.0F);
	fragTexCoord = vec2(v.uv1, v.uv2);
	materialId = PushConstants.materialId;
	normal = mat3(transpose(inverse(PushConstants.model))) * v.normal;
	fragPos = vec3(ubo.model * vec4(v.position, 1.0F));
}