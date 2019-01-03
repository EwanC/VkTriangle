#include <cassert>
#include <iostream>
#include "hello_triangle_application.h"

namespace {
// Callback invoked when window is resized
void framebufferResizeCallback(GLFWwindow* window, int /*width*/,
                               int /*height*/) {
  auto app = reinterpret_cast<HelloTriangleApplication*>(
      glfwGetWindowUserPointer(window));
  app->framebuffer_resized = true;
}
}  // namespace

int main() {
  // Initalize GLFW library, glfwTerminate() teardown called in
  // HelloTriangleApplication destructor
  glfwInit();

  // Don't create an OpenGL context, we're using Vulkan instead
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  // Crreate a window, destroyed in HelloTriangleApplication destructor
  const char* window_title = "Vulkan Triangle";
  const int window_width = 800;
  const int window_height = 600;
  GLFWwindow* window = glfwCreateWindow(window_width, window_height,
                                        window_title, nullptr, nullptr);
  assert(nullptr != window);

  // Construct instance of HelloTriangleApplication which initalizes Vulkan
  std::unique_ptr<HelloTriangleApplication> app;
  try {
    app.reset(new HelloTriangleApplication(window));
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  // Associate pointer to HelloTriangleApplication instance with window, so
  // we can access it if the user resizes the window
  glfwSetWindowUserPointer(window, app.get());
  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

  // Construct instance of HelloTriangleApplication which initalizes Vulkan
  try {
    app->runMainLoop();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
