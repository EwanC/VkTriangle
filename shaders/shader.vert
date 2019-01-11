#version 450

// Per vertex attributes in vertex buffer
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColour;

// output colour, read in by fragment shader
layout(location = 0) out vec3 fragColour;

void main() {
  gl_Position = vec4(inPosition, 0.0, 1.0);
  fragColour = inColour;
}
