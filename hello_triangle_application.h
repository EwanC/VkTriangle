#ifndef TRIANGLE_APP_CLASS_H
#define TRIANGLE_APP_CLASS_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>

class HelloTriangleApplication {
 public:
  void runMainLoop();
  HelloTriangleApplication(GLFWwindow* w) : window(w) { initVulkan(); }
  HelloTriangleApplication() = delete;

  ~HelloTriangleApplication();

 private:
  void initVulkan();
  void createInstance();
  void setupDebugCallback();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createSwapChain();

  void createImageViews();
  void createRenderPass();
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void createFramebuffers();
  void createCommandPool();
  void createTextureImage();
  void createTextureImageView();
  void copyBuffer(VkBuffer, VkBuffer, VkDeviceSize);
  void transitionImageLayout(VkImage, VkFormat, VkImageLayout, VkImageLayout);
  void copyBufferToImage(VkBuffer, VkImage, uint32_t, uint32_t);
  void createVertexBuffer();
  void createIndexBuffer();
  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();
  void createCommandBuffers();
  void createSyncObjects();

  void drawFrame();
  void recreateSwapChain();
  void cleanupSwapChain();

  GLFWwindow* window;
  VkInstance instance;
  VkDebugUtilsMessengerEXT validation_callback;
  VkSurfaceKHR surface;
  VkPhysicalDevice physical_device;
  VkDevice logical_device;
  VkQueue graphics_queue;
  VkQueue present_queue;

  VkSwapchainKHR swap_chain;
  std::vector<VkImage> swap_chain_images;           // Presentable images
  VkFormat swap_chain_image_format;                 // Colour depth
  VkExtent2D swap_chain_extent;                     // Resolution of images
  std::vector<VkImageView> swap_chain_image_views;  // Swapchain view

  VkRenderPass render_pass;
  VkDescriptorSetLayout descriptor_set_layout;
  VkPipelineLayout pipeline_layout;
  VkPipeline graphics_pipeline;
  std::vector<VkFramebuffer> swap_chain_framebuffers;
  VkCommandPool command_pool;
  VkDescriptorPool descriptor_pool;
  std::vector<VkDescriptorSet> descriptor_sets;

  VkDeviceMemory vertex_buffer_memory;
  VkDeviceMemory index_buffer_memory;
  VkBuffer vertex_buffer;
  VkBuffer index_buffer;
  VkImage texture_image;
  VkDeviceMemory texture_image_memory;
  VkImageView texture_image_view;
  std::vector<VkCommandBuffer> command_buffers;

  std::vector<VkBuffer> uniform_buffers;
  std::vector<VkDeviceMemory> uniform_buffers_memory;

  // Wrap syncrhonization primitvies together
  struct FrameSync final {
    // Signal that an image has been acquired and is ready for rendering
    VkSemaphore image_available_semaphore;

    // Signal that rendering has finished and presentation can happen
    VkSemaphore render_finished_semaphore;

    // Performs CPU-GPU synchronization
    VkFence in_flight_fence;
  };

  std::vector<FrameSync> frame_sync;  // Synchronization primitives per frame
  size_t current_frame = 0;           // Used to index frame_sync

 public:
  // Public since set to true by GLFW callback when window resized.
  bool framebuffer_resized = false;
};
#endif  // TRIANGLE_APP_CLASS_H
