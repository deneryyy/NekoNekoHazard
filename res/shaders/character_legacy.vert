#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in float inUV1;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in float inUV2;

layout( push_constant ) uniform constants
{	
	uint materialId;
	mat4 model;
} PushConstants;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
	vec4 lightPos;
	vec4 cameraPos;
} ubo;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out uint materialId;
layout(location = 2) out vec3 normal;
layout(location = 3) out vec3 fragPos;

void main() {
	gl_Position =  ubo.proj * ubo.view * PushConstants.model * vec4(inPos, 1.0F);
	fragTexCoord = vec2(inUV1, inUV2);
	materialId = PushConstants.materialId;
	normal = mat3(transpose(inverse(PushConstants.model))) * inNormal;
	fragPos = vec3(PushConstants.model * vec4(inPos, 1.0F));
}