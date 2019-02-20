#version 450

layout(binding = 0) uniform UniformBufferObject {
  mat4 model; // Model in world space relative to origin
  mat4 view;  // Camera view, rotation around origin
  mat4 proj;  // Project to screen, so 3D model can be rendered in 2D
} ubo;

// Per vertex attributes in vertex buffer
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColour;
layout(location = 2) in vec2 inTexCoord;

// output colour, read in by fragment shader
layout(location = 0) out vec3 fragColour;

// pass through texture coordinate to fragment shader
layout(location = 1) out vec2 fragTexCoord;

void main() {
  gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
  fragColour = inColour;
  fragTexCoord = inTexCoord;
}
