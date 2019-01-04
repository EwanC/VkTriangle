#version 450

vec2 positions[3] = vec2[](
  vec2(0.0, -0.5),  // top
  vec2(0.5, 0.5),   // right
  vec2(-0.5, 0.5)   // left
);


vec3 colours[3] = vec3[](
  vec3(1.0, 0.0, 0.0),  // red
  vec3(0.0, 1.0, 0.0),  // green
  vec3(0.0, 0.0, 1.0)   // blue
);

// output colour, read in by fragment shader
layout(location = 0) out vec3 fragColour;

void main() {
  // built-in gl_VertexIndex variable is index of current vertex
  // Create a vec4 clip-coordinate to output position of vertex
  gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
  fragColour = colours[gl_VertexIndex];
}
