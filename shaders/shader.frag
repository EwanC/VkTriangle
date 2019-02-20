#version 450
#extension GL_ARB_separate_shader_objects : enable

// Sampler
layout(binding = 1) uniform sampler2D texSampler;

// output colour for framebuffer
layout(location = 0) out vec4 outColor;

layout(location = 0) in vec3 fragColor; // per-vertex input colour from vertex shader
layout(location = 1) in vec2 fragTexCoord; // Texture coordinate

void main() {
  // Sample texture at coordinate
  outColor = texture(texSampler, fragTexCoord);
}
