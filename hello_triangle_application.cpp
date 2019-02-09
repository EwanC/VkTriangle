#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>

#pragma warning(disable : 4201)  // nameless struct/union
#include <glm/glm.hpp>
#pragma warning(default : 4201)
#pragma warning(disable : 4127)  // consider constexpr
#include <glm/gtc/matrix_transform.hpp>
#pragma warning(default : 4127)

#define STB_IMAGE_IMPLEMENTATION
#pragma warning(disable : 4100)  // unreferenced formal parameter
#include <stb_image.h>
#pragma warning(default : 4100)

#include "hello_triangle_application.h"
#include "shaders.h"

namespace {
const int max_frames_in_flight =
    2;  // Number of frames to be process concurrently
const std::vector<const char*> validation_layers = {
    "VK_LAYER_LUNARG_standard_validation"};
const std::vector<const char*> device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* p_create_info,
    const VkAllocationCallbacks* p_allocator,
    VkDebugUtilsMessengerEXT* p_callback) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, p_create_info, p_allocator, p_callback);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT callback,
                                   const VkAllocationCallbacks* p_allocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, callback, p_allocator);
  }
}

// Callback invoked by validation layers
VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
              VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
              const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
              void* /*pUserData*/) {
  std::cerr << "validation layer: " << p_callback_data->pMessage << std::endl;
  // PFN_vkDebugUtilsMessengerCallbackEXT functions should always return
  // VK_FALSE
  return VK_FALSE;
}

// Checks if physical device supports all the extensions we require
bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
  uint32_t extension_count;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                       nullptr);

  std::vector<VkExtensionProperties> available_extensions(extension_count);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                       available_extensions.data());

  std::set<std::string> required_extensions(device_extensions.begin(),
                                            device_extensions.end());

  for (const auto& extension : available_extensions) {
    required_extensions.erase(extension.extensionName);
  }

  return required_extensions.empty();
}

bool checkValidationLayerSupport() {
  uint32_t layer_count;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

  std::vector<VkLayerProperties> available_layers(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

  for (const char* layer_name : validation_layers) {
    bool layer_found = false;

    for (const auto& layer_properties : available_layers) {
      if (strcmp(layer_name, layer_properties.layerName) == 0) {
        layer_found = true;
        break;
      }
    }

    if (!layer_found) {
      return false;
    }
  }

  return true;
}

std::vector<const char*> getRequiredExtensions() {
  uint32_t glfw_extension_count = 0;
  const char** glfw_extensions =
      glfwGetRequiredInstanceExtensions(&glfw_extension_count);

  std::vector<const char*> extensions(glfw_extensions,
                                      glfw_extensions + glfw_extension_count);

#ifdef ENABLE_VALIDATION
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

  return extensions;
}

VkPresentModeKHR chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR> available_present_modes) {
  // VK_PRESENT_MODE_FIFO_KHR is guarantted to be available:
  // The swap chain is a queue where the display takes an image from the front
  // of the queue when the display is refreshed and the program inserts rendered
  // images at the back of the queue.If the queue is full then the program has
  // to wait.This is most similar to vertical sync as found in modern
  //  games.The moment that the display is refreshed is known as
  //"vertical blank"
  VkPresentModeKHR best_mode = VK_PRESENT_MODE_FIFO_KHR;

  for (const auto& mode : available_present_modes) {
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
      // Instead of blocking the application when the queue is full,
      // the images that are already queued are simply replaced with the newer
      // ones.This mode can be used to implement triple buffering,
      // which allows you to avoid tearing with significantly less latency
      // issues than standard vertical sync that uses double buffering.
      return mode;
    } else if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
      // images are transferred to screen right away, may result in tearing
      best_mode = mode;
    }
  }

  return best_mode;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& available_formats) {
  // VkSurfaceFormatKHR is made up of a format(channel + type) and colour space

  // If the surface has no preferred format VK_FORMAT_UNDEFINED is set, in
  // which case use one of the most common RGB formats VK_FORMAT_B8G8R8A8_UNORM
  if (available_formats.size() == 1 &&
      available_formats[0].format == VK_FORMAT_UNDEFINED) {
    return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
  }

  // More than one format available, see if our preffered format is available
  for (const auto& available_format : available_formats) {
    if (available_format.format == VK_FORMAT_B8G8R8A8_UNORM &&
        available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return available_format;
    }
  }

  // Fallback to first format from queried list
  return available_formats[0];
}

VkExtent2D chooseSwapExtent(GLFWwindow* window,
                            const VkSurfaceCapabilitiesKHR& capabilities) {
  // Extent is the resolution of swap chain images and is almost always equal
  // to the resolution of the window.
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    // Window manager allows resolution to differ
    return capabilities.currentExtent;
  } else {
    // GLFW retrieves the size, in pixels, of the framebuffer of the specified
    // window
    assert(window != nullptr);
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D actual_extent = {static_cast<uint32_t>(width),
                                static_cast<uint32_t>(height)};

    // clamp width and height to implementation capabilities
    actual_extent.width =
        std::clamp(actual_extent.width, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.width);

    actual_extent.height =
        std::clamp(actual_extent.height, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);

    return actual_extent;
  }
}

// Queue's in Vulkan are grouped into families encapsulating subsets of
// functionality.
struct QueueFamilyIndices final {
  bool isComplete() const {
    return graphics_family.has_value() && present_family.has_value();
  }

  std::optional<uint32_t> graphics_family;  // graphics queue available
  std::optional<uint32_t> present_family;   // WSI presentation queue available
};

VkShaderModule createShaderModule(VkDevice logical_device,
                                  const std::vector<uint8_t>& code) {
  VkShaderModuleCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size();
  create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule shader_module;
  if (vkCreateShaderModule(logical_device, &create_info, nullptr,
                           &shader_module) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }

  return shader_module;
}

QueueFamilyIndices findQueueFamilies(VkSurfaceKHR surface,
                                     VkPhysicalDevice device) {
  // Find queue families supported by device
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                           nullptr);

  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                           queue_families.data());

  QueueFamilyIndices indices;
  int i = 0;
  for (const auto& queue_family : queue_families) {
    // Check that family contains at least one queue, and that queues family
    // support graphics operations
    if ((queue_family.queueCount > 0) &&
        (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      indices.graphics_family = i;
    }

    // Not all devices support WSI(windows system integration), and not all
    // queue families support presentation to a surface
    VkBool32 present_support = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);

    if ((queue_family.queueCount > 0) && present_support) {
      indices.present_family = i;
    }

    if (indices.isComplete()) {
      break;
    }

    i++;
  }

  return indices;
}

// Holds propeties for our surface relevant to swapchains
struct SwapChainSupportDetails final {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

SwapChainSupportDetails querySwapChainSupport(VkSurfaceKHR surface,
                                              VkPhysicalDevice device) {
  // Query capabilities of surface
  SwapChainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                            &details.capabilities);

  // Query supported surface formats, e.g. RGB, SRGB representations
  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
  if (format_count != 0) {
    details.formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count,
                                         details.formats.data());
  }

  // Query supported surface presentation modes, e.g immediately or blocking
  uint32_t present_mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface,
                                            &present_mode_count, nullptr);
  if (present_mode_count != 0) {
    details.present_modes.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        device, surface, &present_mode_count, details.present_modes.data());
  }

  return details;
}

// Holds data needed by vertex shader
struct Vertex final {
  glm::vec2 pos;
  glm::vec3 colour;

  static VkVertexInputBindingDescription getBindingDescription() {
    // A vertex binding describes how to load data from memory throughout the
    // vertices
    VkVertexInputBindingDescription binding_description = {};
    // Only have 1 binding as all data is packed into a single array
    binding_description.binding = 0;
    // Stride is number of bytes between 2 continugous elements in buffer
    binding_description.stride = sizeof(Vertex);
    // Per vertex rendering rather than instance rendering
    binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return binding_description;
  }

  static std::array<VkVertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    // attribute description desfines how to extract a vertex attribute
    // from a chunk of vertex data originating from a binding description.
    // We have 2 attributes, colour and position
    std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions =
        {};

    // position attribute
    attribute_descriptions[0].binding = 0;  // we only have 1 binding
    attribute_descriptions[0].location =
        0;  // location 0 in shader
            // 2-component vector of 32-bit single precision floats
    attribute_descriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attribute_descriptions[0].offset = offsetof(Vertex, pos);

    // colour attribute
    attribute_descriptions[1].binding = 0;   // we only have 1 binding
    attribute_descriptions[1].location = 1;  // location 1 in shader
    // 3-component vector of 32-bit single precision floats
    attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute_descriptions[1].offset = offsetof(Vertex, colour);

    return attribute_descriptions;
  }
};

uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties,
                        VkPhysicalDevice device) {
  // Physical memory is defined in terms of heaps and types
  // Memory heaps are distinct memory resources like dedicated VRAM and swap
  // space in RAM for when VRAM runs out. The different types of memory exist
  // within these heaps
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(device, &mem_properties);

  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
    // type_filter is bitfield of suitable memory types
    const bool suitable_type = type_filter & (1 << i);
    // check desired properties of memory, like writable & coherence
    if (suitable_type && (mem_properties.memoryTypes[i].propertyFlags &
                          properties) == properties) {
      // Return first matching memory type
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

VkCommandBuffer beginSingleTimeCommands(VkDevice logical_device,
                                        VkCommandPool command_pool) {
  // Allocate command buffer to submit
  VkCommandBufferAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandPool = command_pool;
  alloc_info.commandBufferCount = 1;  // Single

  VkCommandBuffer command_buffer;
  vkAllocateCommandBuffers(logical_device, &alloc_info, &command_buffer);

  // Only submitted once
  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  // start recording
  vkBeginCommandBuffer(command_buffer, &begin_info);
  return command_buffer;
}

void endSingleTimeCommands(VkCommandBuffer command_buffer, VkQueue queue,
                           VkDevice logical_device,
                           VkCommandPool command_pool) {
  // end recording
  vkEndCommandBuffer(command_buffer);

  // Submit single command buffer
  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  // Submit commands
  vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);

  // wait for operations to finish
  vkQueueWaitIdle(queue);

  vkFreeCommandBuffers(logical_device, command_pool, 1, &command_buffer);
}

void createBuffer(VkDevice logical_device, VkPhysicalDevice physical_device,
                  VkDeviceSize size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags properties, VkBuffer& buffer,
                  VkDeviceMemory& buffer_memory) {
  VkBufferCreateInfo buffer_info = {};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  // Puropse of buffer, e.g. vertex or staging
  buffer_info.usage = usage;
  // Only owened by graphics queue
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(logical_device, &buffer_info, nullptr, &buffer) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create vertex buffer!");
  }

  // Query memory properties of device; size, alignment, and type
  VkMemoryRequirements mem_requirements;
  vkGetBufferMemoryRequirements(logical_device, buffer, &mem_requirements);

  // Allocate on device memory
  VkMemoryAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  // required size of allocation to hold buffer
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex = findMemoryType(mem_requirements.memoryTypeBits,
                                              properties, physical_device);

  if (vkAllocateMemory(logical_device, &alloc_info, nullptr, &buffer_memory) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate vertex buffer memory!");
  }

  // Binds device memory to buffer
  vkBindBufferMemory(logical_device, buffer, buffer_memory,
                     0 /* offset into memory region*/);
}

// Needs to match struct in vertex shader source
struct UniformBufferObject final {
  alignas(16) glm::mat4 model;  // Model in world space relative to origin
  alignas(16) glm::mat4 view;   // Camera view, rotation around origin
  alignas(16)
      glm::mat4 proj;  // Project to screen, so 3D model can be rendered in 2D
};

void updateUniformBuffer(VkDevice device, VkDeviceMemory ubo_image,
                         VkExtent2D swap_chain_extent) {
  // Rotate geometry 90 degrees per second regardless of frame rate
  static auto start_time = std::chrono::high_resolution_clock::now();
  const auto current_time = std::chrono::high_resolution_clock::now();
  const float time = std::chrono::duration<float, std::chrono::seconds::period>(
                         current_time - start_time)
                         .count();

  UniformBufferObject ubo = {};

  // Model to world
  const glm::f32 model_angle = time * glm::radians(90.0f);   // 90 degrees
  const glm::vec3 model_axis = glm::vec3(0.0f, 0.0f, 1.0f);  // Z rotation axis
  ubo.model = glm::rotate(glm::mat4(1.0f), model_angle, model_axis);

  // Position of camera's viewpoint
  const glm::vec3 view_eye = glm::vec3(2.0f, 2.0f, 2.0f);
  // Center is where you are looking at, the origin
  const glm::vec3 view_center = glm::vec3(0.0f, 0.0f, 0.0f);
  // Defines worlds upwards direction towards positive
  const glm::vec3 view_up = glm::vec3(0.0f, 0.0f, 1.0f);
  // View from camera space
  ubo.view = glm::lookAt(view_eye, view_center, view_up);

  // 45 degree field of view
  const float proj_fov = glm::radians(45.0f);
  // Aspect ratio should match window size
  const float proj_aspect_ratio = static_cast<float>(swap_chain_extent.width) /
                                  static_cast<float>(swap_chain_extent.height);
  // Near and far clip distance thresholds
  const float proj_near = 0.1f;
  const float proj_far = 10.0f;
  // Create projection matrix from 3D to 2D
  ubo.proj = glm::perspective(proj_fov, proj_aspect_ratio, proj_near, proj_far);

  // In OpenGL the Y coordinate of the clip coordinates is inverted, Flip the
  // sign on the scaling factor of the Y axis to compensate
  ubo.proj[1][1] *= -1;

  // Copy data MVP matricies to uniform buffer so it's available to the device
  void* data;
  vkMapMemory(device, ubo_image, 0, sizeof(ubo), 0, &data);
  std::memcpy(data, &ubo, sizeof(ubo));
  vkUnmapMemory(device, ubo_image);
}

void createImage(VkDevice device, VkPhysicalDevice physical_device,
                 uint32_t width, uint32_t height, VkFormat format,
                 VkImageTiling tiling, VkImageUsageFlags usage,
                 VkMemoryPropertyFlags properties, VkImage& image,
                 VkDeviceMemory& image_memory) {
  VkImageCreateInfo image_info = {};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  // 1D could be used for gradients, or 3D for Voxels, but 2D works for texture
  image_info.imageType = VK_IMAGE_TYPE_2D;
  // Extent described dimenions as texels per axis
  image_info.extent.width = width;
  image_info.extent.height = height;
  image_info.extent.depth = 1;
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.format = format;
  image_info.tiling = tiling;
  // First transition will discard texels, okay since we're first going to
  // transition the image to be a transfer destination and then copy texel
  // data to it from a buffer object
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage = usage;
  // Only 11 sample since we're not mutlisampling
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  // Image only used by our graphics queue family
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS) {
    throw std::runtime_error("failed to create image!");
  }

  // Query image for device specific memory requirements
  VkMemoryRequirements mem_requirements;
  vkGetImageMemoryRequirements(device, image, &mem_requirements);

  // Allocate device memory for image
  VkMemoryAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex = findMemoryType(mem_requirements.memoryTypeBits,
                                              properties, physical_device);
  // Allocate device memory
  if (vkAllocateMemory(device, &alloc_info, nullptr, &image_memory) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate image memory!");
  }

  // Bind device memory to image object
  vkBindImageMemory(device, image, image_memory, 0);
}

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format) {
  VkImageViewCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  create_info.image = image;
  create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  create_info.format = format;

  // Default mapping, don't swizzle channels
  create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

  // Colour view without any mipmapping or layering
  create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  create_info.subresourceRange.baseMipLevel = 0;
  create_info.subresourceRange.levelCount = 1;
  create_info.subresourceRange.baseArrayLayer = 0;
  create_info.subresourceRange.layerCount = 1;

  VkImageView image_view;
  if (vkCreateImageView(device, &create_info, nullptr, &image_view) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create image view!");
  }

  return image_view;
}

// Colour & Position of our rectangles vertices
const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},  // Top-left red
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},   // Top-right green
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},    // Bottom-right blue
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}    // Bottom-right white
};

// Indicies into vertex buffer of 2 triangles used to make
// rectangle, rather than redunancy in vertex buffer.
const std::vector<uint16_t> vertex_indices = {0, 1, 2, 2, 3, 0};
}  // namespace

void HelloTriangleApplication::runMainLoop() {
  // While windows not closed
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();  // check for events
    // Acquire an image from the swap chain, execute the command buffer with
    // that image as attachment in the framebuffer, then finally return the
    // image to the swap chain for presentation
    drawFrame();
  }
}

void HelloTriangleApplication::initVulkan() {
  createInstance();      // Initalize VkInstance member
  setupDebugCallback();  // Set message handler for validation layers

  // Create a surface for our window so we have have something to render to.
  // A VkSurfaceKHR represents an abstract type of surface which is platform
  // independent.
  if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface!");
  }

  // Find a physical device with support for all vulkan features we need
  pickPhysicalDevice();

  // A logical device represents a connection to a physical device describing
  // the features we want to use.
  createLogicalDevice();

  // A swapchain provides the ability to present rendering results to a WSI
  // surface. Used as a framebuffer holding a queue of images that are waitin
  // to be presented to the screen.
  createSwapChain();

  // An image view is a view into an image, describing how to access the image
  // and which part of the image to access. Create a view for our swap chain
  createImageViews();

  // A renderpass contains the structure of the frame, encapsulating the set of
  // framebuffer attachments.
  createRenderPass();

  // We use Uniform Buffer descriptors in our vertex shader, initalize the
  // description set layout so we can pass it to the pipeline layout step of
  // graphics pipeline creation.
  createDescriptorSetLayout();

  // Load SPIR-V vertex & framgement shaders then setup graphics pipeline
  createGraphicsPipeline();

  // A framebuffer object references all of the VkImageView objects that
  // represent the renderpass attachments.
  createFramebuffers();

  // Command pools manage memory used to store command buffers
  createCommandPool();

  // Loads an to use as a textire and turns it into a Vulkan image object.
  createTextureImage();

  // Create view to access our texture image
  createTextureImageView();

  // Create a buffer to store our vertex data
  createVertexBuffer();

  // Create buffer storing indicies into vertex buffer
  createIndexBuffer();

  // Create uniform buffer per swap chain image which our vertex shader reads
  // MVP matricies from.
  createUniformBuffers();

  // Create pool to allocate descriptor sets from, then allocate the sets
  // themselves
  createDescriptorPool();
  createDescriptorSets();

  // Create a command buffer for every image in swapchain
  createCommandBuffers();

  // Create semaphores and fences to synchronise frame processing
  createSyncObjects();
}

void HelloTriangleApplication::createInstance() {
  VkApplicationInfo app_info = {};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Hello Triangle";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "No Engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  const auto extensions = getRequiredExtensions();
  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();

#ifdef ENABLE_VALIDATION
  if (!checkValidationLayerSupport()) {
    throw std::runtime_error("validation layers requested, but not available!");
  }

  create_info.enabledLayerCount =
      static_cast<uint32_t>(validation_layers.size());
  create_info.ppEnabledLayerNames = validation_layers.data();
#else
  create_info.enabledLayerCount = 0;
#endif

  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    throw std::runtime_error("failed to create instance!");
  }

#ifndef NDEBUG
  uint32_t extension_count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);

  std::vector<VkExtensionProperties> vk_extensions(extension_count);
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count,
                                         vk_extensions.data());
  std::cout << "available extensions:" << std::endl;

  for (const auto& extension : vk_extensions) {
    std::cout << "\t" << extension.extensionName << std::endl;
  }
#endif  // !NDEBUG
}

void HelloTriangleApplication::setupDebugCallback() {
#ifdef ENABLE_VALIDATION
  VkDebugUtilsMessengerCreateInfoEXT create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  create_info.messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  create_info.pfnUserCallback =
      debugCallback;  // callback function invoked by validation layer
  create_info.pUserData = nullptr;

  // Assign debug callback triggered when an event occurs
  if (CreateDebugUtilsMessengerEXT(instance, &create_info, nullptr,
                                   &validation_callback) != VK_SUCCESS) {
    throw std::runtime_error("failed to set up debug callback!");
  }
#endif
}

void HelloTriangleApplication::pickPhysicalDevice() {
  physical_device = VK_NULL_HANDLE;
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

  if (device_count == 0) {
    throw std::runtime_error("failed to find any devices with Vulkan support");
  }

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

  // Iterate over all discoverd Vulkan enabled devices
  for (const auto& dev : devices) {
    // Check device supports all the extensions we require
    const bool extensions_supported = checkDeviceExtensionSupport(dev);
    if (!extensions_supported) {
      continue;
    }

    // Check supports queue families with all the functionality we need
    const QueueFamilyIndices q_indices = findQueueFamilies(surface, dev);
    if (!q_indices.isComplete()) {
      continue;
    }

    // Check surface supports properties we need for swapchains, one supported
    // image format and one supported presentation mode will do for now
    const SwapChainSupportDetails swap_chain_support =
        querySwapChainSupport(surface, dev);
    if (!swap_chain_support.formats.empty() &&
        !swap_chain_support.present_modes.empty()) {
      physical_device = dev;
      break;
    }
  }

  if (physical_device == VK_NULL_HANDLE) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
}

void HelloTriangleApplication::createLogicalDevice() {
  const QueueFamilyIndices indices =
      findQueueFamilies(surface, physical_device);
  const std::set<uint32_t> unique_queue_families = {
      indices.graphics_family.value(), indices.present_family.value()};
  // Priority is between 0.0 and 1.0, higher values indicate a higher priority
  const float queue_priority = 1.0f;

  // Create infos for queue families we need
  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  for (uint32_t queue_family : unique_queue_families) {
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_family;
    queue_create_info.queueCount = 1;  // Only want a single queue per family
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_create_info);
  }

  // Don't need any special features, all entries default to VK_FALSE
  VkPhysicalDeviceFeatures device_features = {};

  VkDeviceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.queueCreateInfoCount =
      static_cast<uint32_t>(queue_create_infos.size());
  create_info.pQueueCreateInfos = queue_create_infos.data();
  create_info.pEnabledFeatures = &device_features;
  create_info.enabledExtensionCount =
      static_cast<uint32_t>(device_extensions.size());
  create_info.ppEnabledExtensionNames = device_extensions.data();
#ifdef ENABLE_VALIDATION
  create_info.enabledLayerCount =
      static_cast<uint32_t>(validation_layers.size());
  create_info.ppEnabledLayerNames = validation_layers.data();
#else
  create_info.enabledLayerCount = 0;
#endif

  if (vkCreateDevice(physical_device, &create_info, nullptr, &logical_device) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }

  // Queues are created along with the logical device, but we need to retrieve
  // the handles. Because we've only created a single queue in each family we
  // can hardcode the index to 0.
  vkGetDeviceQueue(logical_device, indices.graphics_family.value(), 0,
                   &graphics_queue);
  vkGetDeviceQueue(logical_device, indices.present_family.value(), 0,
                   &present_queue);
}

void HelloTriangleApplication::createSwapChain() {
  // Find properties surface has relevant to swap chains
  const SwapChainSupportDetails swap_chain_support =
      querySwapChainSupport(surface, physical_device);

  // Select a colour depth from available list
  const VkSurfaceFormatKHR surface_format =
      chooseSwapSurfaceFormat(swap_chain_support.formats);
  // Select the conditions for swapping images to screen from available list
  const VkPresentModeKHR present_mode =
      chooseSwapPresentMode(swap_chain_support.present_modes);
  // Select resolution of images in swapchain from avilable bitfield
  const VkExtent2D extent =
      chooseSwapExtent(window, swap_chain_support.capabilities);

  // Set number of images in swap chain, i.e the queue length, try min+1 to
  // implement triple buffering
  uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
  if (swap_chain_support.capabilities.maxImageCount > 0 &&
      image_count > swap_chain_support.capabilities.maxImageCount) {
    image_count = swap_chain_support.capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  create_info.surface = surface;
  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;  // Layers are only >1 for stereoscopic 3D
  create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  // How to handle images used across multiple queue families
  const QueueFamilyIndices indices =
      findQueueFamilies(surface, physical_device);
  const uint32_t queue_family_indices[] = {indices.graphics_family.value(),
                                           indices.present_family.value()};

  if (indices.graphics_family != indices.present_family) {
    // Images can be used across multiple queues without expilict
    // ownership transfers.
    create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = queue_family_indices;
  } else {
    // Image is owned by 1 queue family at a time, and must be explicitly
    // transferred.
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  create_info.preTransform = swap_chain_support.capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  create_info.presentMode = present_mode;
  create_info.clipped = VK_TRUE;
  create_info.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(logical_device, &create_info, nullptr,
                           &swap_chain) != VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain!");
  }

  // Get array of presentable images associated with a swapchain
  vkGetSwapchainImagesKHR(logical_device, swap_chain, &image_count, nullptr);
  swap_chain_images.resize(image_count);
  vkGetSwapchainImagesKHR(logical_device, swap_chain, &image_count,
                          swap_chain_images.data());

  swap_chain_image_format = surface_format.format;
  swap_chain_extent = extent;
}

void HelloTriangleApplication::createImageViews() {
  // Create a view for every image in our swap chain
  swap_chain_image_views.resize(swap_chain_images.size());
  for (size_t i = 0; i < swap_chain_images.size(); i++) {
    swap_chain_image_views[i] = createImageView(
        logical_device, swap_chain_images[i], swap_chain_image_format);
  }
}

void HelloTriangleApplication::transitionImageLayout(VkImage image,
                                                     VkFormat format,
                                                     VkImageLayout old_layout,
                                                     VkImageLayout new_layout) {
  // Format will be used later for special transitions in the depth buffer
  (void)format;

  // Start recording command buffer for our transition operations
  VkCommandBuffer command_buffer =
      beginSingleTimeCommands(logical_device, command_pool);

  // Pipeline barrier, rather than host<->device sync
  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  // We're not transferring queue ownwership
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  // Specifies operations that should happen before the barrier
  VkPipelineStageFlags source_stage;
  // Specifies operations that should happen after the barrier
  VkPipelineStageFlags destination_stage;

  if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
      new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    // Undefined to copy transition has no dependancies to wait on
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    // Earliest possible stage
    source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    // Shader read layouy has a dependency on shader writes
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else {
    throw std::invalid_argument("unsupported layout transition!");
  }

  vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0,
                       nullptr, 0, nullptr, 1, &barrier);

  endSingleTimeCommands(command_buffer, graphics_queue, logical_device,
                        command_pool);
}

void HelloTriangleApplication::copyBufferToImage(VkBuffer buffer, VkImage image,
                                                 uint32_t width,
                                                 uint32_t height) {
  VkCommandBuffer command_buffer =
      beginSingleTimeCommands(logical_device, command_pool);

  VkBufferImageCopy region = {};
  // Byte offset at which pixel to start
  region.bufferOffset = 0;
  // Specify how the pixels are laid out in memory, 0 signififes tightly packed
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  // Which part of image we want to copy the pixe tols
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {width, height, 1};

  // layour for optimial copying
  const VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  vkCmdCopyBufferToImage(command_buffer, buffer, image, layout, 1, &region);

  endSingleTimeCommands(command_buffer, graphics_queue, logical_device,
                        command_pool);
}

void HelloTriangleApplication::copyBuffer(VkBuffer src_buffer,
                                          VkBuffer dst_buffer,
                                          VkDeviceSize size) {
  VkCommandBuffer command_buffer =
      beginSingleTimeCommands(logical_device, command_pool);

  // Copy operation
  VkBufferCopy copy_region = {};
  copy_region.size = size;
  copy_region.srcOffset = 0;
  copy_region.dstOffset = 0;
  vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);

  endSingleTimeCommands(command_buffer, graphics_queue, logical_device,
                        command_pool);
}

void HelloTriangleApplication::createIndexBuffer() {
  const VkDeviceSize buffer_size =
      sizeof(vertex_indices[0]) * vertex_indices.size();

  // Staging buffer from cpu accessible memory to local
  VkBuffer staging_buffer;
  VkDeviceMemory staging_buffer_memory;

  // Our staging buffer needs host visible memory that will be immediately
  // visible to device when written to
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  createBuffer(logical_device, physical_device, buffer_size, usage, properties,
               staging_buffer, staging_buffer_memory);

  // Map staging buffer and copy the indicies data into it we want in our
  // index buffer
  void* data;
  vkMapMemory(logical_device, staging_buffer_memory, 0, buffer_size, 0, &data);
  std::memcpy(data, vertex_indices.data(), (size_t)buffer_size);
  vkUnmapMemory(logical_device, staging_buffer_memory);

  // Index buffer should be on device memory used for indexing.
  usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  createBuffer(logical_device, physical_device, buffer_size, usage, properties,
               index_buffer, index_buffer_memory);

  // Copy staging buffer data to index buffer
  copyBuffer(staging_buffer, index_buffer, buffer_size);

  // Free our staging buffer now data has been copied to device
  vkDestroyBuffer(logical_device, staging_buffer, nullptr);
  vkFreeMemory(logical_device, staging_buffer_memory, nullptr);
}

void HelloTriangleApplication::createUniformBuffers() {
  const VkDeviceSize buffer_size = sizeof(UniformBufferObject);

  // We want a uniform buffer per swap chain image because multiple frames may
  // be in flight at the same time
  const size_t swap_chain_len = swap_chain_images.size();
  uniform_buffers.resize(swap_chain_len);
  uniform_buffers_memory.resize(swap_chain_len);

  // We're not using a staging buffer so use host visible memory that will be
  // immediately visible to device when written to
  const VkBufferUsageFlags usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  const VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  for (size_t i = 0; i < swap_chain_len; i++) {
    createBuffer(logical_device, physical_device, buffer_size, usage,
                 properties, uniform_buffers[i], uniform_buffers_memory[i]);
  }
}

void HelloTriangleApplication::createDescriptorPool() {
  // We'll allocate a descriptor for every frame
  VkDescriptorPoolSize pool_size = {};
  pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  pool_size.descriptorCount = static_cast<uint32_t>(swap_chain_images.size());

  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  pool_info.maxSets = static_cast<uint32_t>(swap_chain_images.size());

  if (vkCreateDescriptorPool(logical_device, &pool_info, nullptr,
                             &descriptor_pool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void HelloTriangleApplication::createDescriptorSets() {
  // create one descriptor set for each swap chain image, all with the same
  // layout.
  const size_t swap_chain_len = swap_chain_images.size();
  std::vector<VkDescriptorSetLayout> layouts(swap_chain_len,
                                             descriptor_set_layout);
  VkDescriptorSetAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = descriptor_pool;
  alloc_info.descriptorSetCount = static_cast<uint32_t>(swap_chain_len);
  alloc_info.pSetLayouts = layouts.data();

  descriptor_sets.resize(swap_chain_len);
  if (vkAllocateDescriptorSets(logical_device, &alloc_info,
                               descriptor_sets.data()) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate descriptor sets!");
  }

  // Configure descriptors now they've been allocated
  for (size_t i = 0; i < swap_chain_len; i++) {
    VkDescriptorBufferInfo buffer_info = {};
    buffer_info.buffer = uniform_buffers[i];  // describes our uniform buffers
    buffer_info.offset = 0;
    buffer_info.range = sizeof(UniformBufferObject);

    VkWriteDescriptorSet descriptor_write = {};
    descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_write.dstSet = descriptor_sets[i];
    descriptor_write.dstBinding = 0;  // binding index from shader
    descriptor_write.dstArrayElement = 0;
    descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_write.descriptorCount = 1;
    descriptor_write.pBufferInfo = &buffer_info;

    vkUpdateDescriptorSets(logical_device, 1, &descriptor_write, 0, nullptr);
  }
}

void HelloTriangleApplication::createVertexBuffer() {
  const VkDeviceSize buffer_size = sizeof(Vertex) * vertices.size();

  // Use a staging buffer in CPU accessible memory to upload the data from the
  // vertex array to. Before we then copy buffer to a more optimial memory
  // location.
  VkBuffer staging_buffer;
  VkDeviceMemory staging_buffer_memory;
  // We want host visible memory so we can map it, and coherent so when we
  // write data to mapped memory it's immediately visiable
  VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  // Buffer can be used as source in a memory transfer operation
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  createBuffer(logical_device, physical_device, buffer_size, usage, properties,
               staging_buffer, staging_buffer_memory);

  // Map device memory to host pointer so we can copy our vertex data into it
  void* data;
  vkMapMemory(logical_device, staging_buffer_memory, 0 /* offset */,
              buffer_size, 0 /* flags */, &data);
  std::memcpy(data, vertices.data(), (size_t)buffer_size);
  vkUnmapMemory(logical_device, staging_buffer_memory);

  // local memory is more performant, but not accessible from CPU
  properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  // Buffer can be used as destination in a memory transfer operation and is
  // a vertex buffer
  usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  createBuffer(logical_device, physical_device, buffer_size, usage, properties,
               vertex_buffer, vertex_buffer_memory);

  // Enqueue operation to copy our staging buffer into vertex buffer
  copyBuffer(staging_buffer, vertex_buffer, buffer_size);

  vkDestroyBuffer(logical_device, staging_buffer, nullptr);
  vkFreeMemory(logical_device, staging_buffer_memory, nullptr);
}

void HelloTriangleApplication::createCommandBuffers() {
  // Command buffer for every image in swapchain
  command_buffers.resize(swap_chain_framebuffers.size());

  VkCommandBufferAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = command_pool;

  // Primary buffers can be submitted to a queue for execution, but cannot be
  // called from other command buffers.
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = static_cast<uint32_t>(command_buffers.size());

  if (vkAllocateCommandBuffers(logical_device, &alloc_info,
                               command_buffers.data()) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }

  // Recording command buffer for each image
  for (size_t i = 0; i < command_buffers.size(); i++) {
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    // Simultaneous use means that the buffer can be resubmitted while already
    // awaiting execution, chosen since we may already be scheduling the
    // drawing commands for the next frame while the last frame isn't finished
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    begin_info.pInheritanceInfo = nullptr;

    if (vkBeginCommandBuffer(command_buffers[i], &begin_info) != VK_SUCCESS) {
      throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = render_pass;
    render_pass_info.framebuffer = swap_chain_framebuffers[i];

    // render area defines where shader loads and stores will take place.
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = swap_chain_extent;

    // Clear colour is 100% opacity black, used for VK_ATTACHMENT_LOAD_OP_CLEAR
    // which specifies that the contents within the render area will be cleared
    // to a uniform value.
    VkClearValue clear_colour = {0.0f, 0.0f, 0.0f, 1.0f};
    render_pass_info.clearValueCount = 1;
    render_pass_info.pClearValues = &clear_colour;

    // Start drawing
    vkCmdBeginRenderPass(command_buffers[i], &render_pass_info,
                         VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics_pipeline);

    // Bind vertex buffer to a command buffer
    const VkBuffer vertex_buffers[] = {vertex_buffer};
    const VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(command_buffers[i], 0 /* offset */, 1 /* count */,
                           vertex_buffers, offsets);

    // Bind 16-bit index buffer, indices can be 16 or 32-bit
    vkCmdBindIndexBuffer(command_buffers[i], index_buffer, 0,
                         VK_INDEX_TYPE_UINT16);

    // Descriptor sets can be used for compute or graphics pipelines
    const VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
    // Bind the right descriptor set for each swap chain image
    vkCmdBindDescriptorSets(command_buffers[i], bind_point, pipeline_layout, 0,
                            1, &descriptor_sets[i], 0, nullptr);

    const uint32_t index_count = static_cast<uint32_t>(vertex_indices.size());
    const uint32_t instance_count =
        1;  // 1 since we're not doing instance rendering
    const uint32_t first_indice = 0;    // offset into vertex buffer
    const uint32_t first_instance = 0;  // offset into instance rendering
    const int32_t vertex_offset = 0;  // Value added before indexing into buffer

    vkCmdDrawIndexed(command_buffers[i], index_count, instance_count,
                     first_indice, vertex_offset, first_indice);
    vkCmdEndRenderPass(command_buffers[i]);

    if (vkEndCommandBuffer(command_buffers[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to record command buffer!");
    }
  }
}

void HelloTriangleApplication::createRenderPass() {
  // Colour attachment acts as a output buffer for framebuffer, for
  // offscreen rendering
  VkAttachmentDescription colour_attachment = {};
  colour_attachment.format = swap_chain_image_format;
  colour_attachment.samples = VK_SAMPLE_COUNT_1_BIT;

  // The loadOp and storeOp determine what to do with the data in the attachment
  // before rendering and after rendering.

  // Clear the values to a constant at the start
  colour_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  // Rendered contents will be stored in memory and can be read later
  colour_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

  // undefined behaviour, we  don't care about sencil
  colour_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colour_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

  // Don't care about previous layout of image
  colour_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  // We want the image to be ready for presentation using the swap chain after
  // rendering
  colour_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference colour_attachment_ref = {};
  colour_attachment_ref.attachment = 0;
  colour_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  // A single render pass can consist of multiple subpasses. Subpasses are
  // subsequent rendering operations that depend on the contents of
  // framebuffers in previous passes.
  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  // index directly referenced from the fragment shader
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colour_attachment_ref;

  // Wait for the swap chain to finish reading from the image before
  // accessing it but setting an attachment dependency
  VkSubpassDependency dependency = {};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo render_pass_info = {};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount = 1;
  render_pass_info.pAttachments = &colour_attachment;
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  render_pass_info.dependencyCount = 1;
  render_pass_info.pDependencies = &dependency;

  if (vkCreateRenderPass(logical_device, &render_pass_info, nullptr,
                         &render_pass) != VK_SUCCESS) {
    throw std::runtime_error("failed to create render pass!");
  }
}

void HelloTriangleApplication::recreateSwapChain() {
  // Window has been resized, we need to recreate the swapchain
  // and it's dependencies so it's compatible with the new window surface

  int width = 0, height = 0;
  while (width == 0 || height == 0) {
    // Window has been minimized, stall until window is maximized again
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
  }

  // Don't touch resources still in use
  vkDeviceWaitIdle(logical_device);

  // Free swapchain dependent resources before allocating new ones
  cleanupSwapChain();

  // Create objects again with new window size
  createSwapChain();
  createImageViews();
  createRenderPass();
  createGraphicsPipeline();
  createFramebuffers();
  createCommandBuffers();
}

void HelloTriangleApplication::createDescriptorSetLayout() {
  // We want a single unform buffer for our vertex shader
  VkDescriptorSetLayoutBinding ubo_layout_binding = {};
  ubo_layout_binding.binding = 0;
  ubo_layout_binding.descriptorCount = 1;
  ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  ubo_layout_binding.pImmutableSamplers = nullptr;
  ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkDescriptorSetLayoutCreateInfo layout_info = {};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = 1;
  layout_info.pBindings = &ubo_layout_binding;

  // Initalize descriptor_set_layout used in graphics pipeline creation
  if (vkCreateDescriptorSetLayout(logical_device, &layout_info, nullptr,
                                  &descriptor_set_layout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }
}

void HelloTriangleApplication::createGraphicsPipeline() {
  // SPIR-V vertex shader buffer
  const std::vector<uint8_t> vert_shader_code(
      vertex_shader, vertex_shader + vertex_shader_length);
  // SPIR-V fragment shader buffer
  const std::vector<uint8_t> frag_shader_code(frag_shader,
                                              frag_shader + frag_shader_length);

  // Vertex Shader is run on every vertex and applies transformations to turn
  // positions from model space to screen space
  const VkShaderModule vert_shader_module =
      createShaderModule(logical_device, vert_shader_code);

  // Framents are pixels elements that fill the framebuffer, the fragment shader
  // is invoked on every fragment which survives rasterization to determine
  // which framebuffer the fragments are written to and with which color and
  // depth values.
  const VkShaderModule frag_shader_module =
      createShaderModule(logical_device, frag_shader_code);

  VkPipelineShaderStageCreateInfo vert_shader_stage_info = {};
  vert_shader_stage_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vert_shader_stage_info.module = vert_shader_module;
  vert_shader_stage_info.pName = "main";

  VkPipelineShaderStageCreateInfo frag_shader_stage_info = {};
  frag_shader_stage_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  frag_shader_stage_info.module = frag_shader_module;
  frag_shader_stage_info.pName = "main";

  const VkPipelineShaderStageCreateInfo shaderStages[] = {
      vert_shader_stage_info, frag_shader_stage_info};

  // Struct is empty for now because we're hardcoding the data
  VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
  vertex_input_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  // Binding defines the spacings between data
  const auto binding_description = Vertex::getBindingDescription();
  vertex_input_info.vertexBindingDescriptionCount = 1;
  vertex_input_info.pVertexBindingDescriptions = &binding_description;

  // Attribute descriptions define type of the bindingDescription passed to the
  // vertex shader, which binding to load them from and at which offset
  const auto attribute_descriptions = Vertex::getAttributeDescriptions();
  vertex_input_info.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attribute_descriptions.size());
  vertex_input_info.pVertexAttributeDescriptions =
      attribute_descriptions.data();

  // Input assembler stage of graphics pipeline collects the raw vertex data
  // from the buffers. Defining what kind of geometry will be drawn from the
  // vertices and if primitive restart should be enabled
  VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
  input_assembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  // triangle from every 3 vertices without reuse
  input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly.primitiveRestartEnable = VK_FALSE;

  // A viewport describes the region of the framebuffer that the output will
  // be rendered to, i.e transformation from image to framebuffer
  VkViewport viewport = {};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = static_cast<float>(swap_chain_extent.width);
  viewport.height = static_cast<float>(swap_chain_extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  // Scissor rectangles define in which regions pixels will actually be stored
  // any pixels outside the rectangles will be discarded by the rasterizer.
  // We want the entire framebuffer, so don't filter anything out.
  VkRect2D scissor = {};
  scissor.offset = {0, 0};
  scissor.extent = swap_chain_extent;

  VkPipelineViewportStateCreateInfo viewport_state = {};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = &viewport;
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = &scissor;

  // The rasterizer takes the geometry that is shaped by the vertices from the
  // vertex shader and turns it into fragments to be colored by the fragment
  // shader
  VkPipelineRasterizationStateCreateInfo rasterizer = {};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  // Counter-clockwise because we Y-Axis flipped in the projection matrix
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.0f;
  rasterizer.depthBiasClamp = 0.0f;
  rasterizer.depthBiasSlopeFactor = 0.0f;

  // Multisampling combines the fragment shader results of multiple polygons
  // that rasterize to the same pixel, a way to perform anti-aliasing.
  // We disable it for now.
  VkPipelineMultisampleStateCreateInfo multisampling = {};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading = 1.0f;
  multisampling.pSampleMask = nullptr;
  multisampling.alphaToCoverageEnable = VK_FALSE;
  multisampling.alphaToOneEnable = VK_FALSE;

  // The color blending stage applies operations to mix different fragments
  // that map to the same pixel in the framebuffer. Fragments can simply
  // overwrite each other, add up or be mixed based upon transparency.

  // VkPipelineColorBlendAttachmentState is configuration per framebuffer
  VkPipelineColorBlendAttachmentState colour_blend_attachment = {};
  // Bitwise operation to combine old and new colours
  colour_blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  // Fragment shader colour is passed through unmodified with VK_FALSE
  colour_blend_attachment.blendEnable = VK_FALSE;
  colour_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  colour_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  colour_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
  colour_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  colour_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  colour_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

  // VkPipelineColorBlendStateCreateInfo is global colour blending settings
  VkPipelineColorBlendStateCreateInfo colour_blending = {};
  colour_blending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colour_blending.logicOpEnable = VK_FALSE;
  colour_blending.logicOp = VK_LOGIC_OP_COPY;
  colour_blending.attachmentCount = 1;
  colour_blending.pAttachments = &colour_blend_attachment;
  colour_blending.blendConstants[0] = 0.0f;
  colour_blending.blendConstants[1] = 0.0f;
  colour_blending.blendConstants[2] = 0.0f;
  colour_blending.blendConstants[3] = 0.0f;

  // Dynamic State can be modified without recreating the whole pipeline,
  // like viewport size and line width
  const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                           VK_DYNAMIC_STATE_LINE_WIDTH};

  VkPipelineDynamicStateCreateInfo dynamic_state = {};
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dynamic_states;

  // Pipeline layouts allow setting of uniform values that can be changed at
  // drawing time to alter the behavior of your shaders without having to
  // recreate them
  VkPipelineLayoutCreateInfo pipeline_layout_info = {};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

  // Unfiorm buffer vertex shader layout
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
  pipeline_layout_info.pushConstantRangeCount = 0;
  pipeline_layout_info.pPushConstantRanges = nullptr;

  if (vkCreatePipelineLayout(logical_device, &pipeline_layout_info, nullptr,
                             &pipeline_layout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }

  VkGraphicsPipelineCreateInfo pipeline_info = {};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = 2;  // vertex & fragment
  pipeline_info.pStages = shaderStages;
  pipeline_info.pVertexInputState = &vertex_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pDepthStencilState = nullptr;
  pipeline_info.pColorBlendState = &colour_blending;
  pipeline_info.pDynamicState = nullptr;
  pipeline_info.layout = pipeline_layout;
  pipeline_info.renderPass = render_pass;
  // Index of the sub pass where this graphics pipeline will be used
  pipeline_info.subpass = 0;
  // Allows you to create a new graphics pipeline deriving from existing
  // pipeline
  pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
  pipeline_info.basePipelineIndex = -1;

  if (vkCreateGraphicsPipelines(logical_device, VK_NULL_HANDLE, 1,
                                &pipeline_info, nullptr,
                                &graphics_pipeline) != VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }

  vkDestroyShaderModule(logical_device, frag_shader_module, nullptr);
  vkDestroyShaderModule(logical_device, vert_shader_module, nullptr);
}

void HelloTriangleApplication::createCommandPool() {
  // A command pool can only allocated command buffers to be submitted on a
  // single type of queue. We're only going to submit drawing commands, so use
  // graphics queue family
  QueueFamilyIndices queue_family_indices =
      findQueueFamilies(surface, physical_device);

  VkCommandPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = queue_family_indices.graphics_family.value();
  pool_info.flags = 0;

  if (vkCreateCommandPool(logical_device, &pool_info, nullptr, &command_pool) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create command pool!");
  }
}

void HelloTriangleApplication::createTextureImage() {
  int tex_width, tex_height, tex_channels;

  // STBI_rgb_alpha value forces the image to be loaded with an alpha channel,
  // even if it doesn't have one. Return value is first element in pixel array
  stbi_uc* pixels = stbi_load(STATUE_TEXTURE, &tex_width, &tex_height,
                              &tex_channels, STBI_rgb_alpha);

  if (!pixels) {
    throw std::runtime_error("failed to load texture image!");
  }

  // 4 bytes per pixel for 'STBI_rgba_alpha'
  const VkDeviceSize image_size = tex_width * tex_height * 4;

  // Create host visible buffer to copy image pixel data to
  VkBuffer staging_buffer;
  VkDeviceMemory staging_buffer_memory;
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  createBuffer(logical_device, physical_device, image_size, usage, properties,
               staging_buffer, staging_buffer_memory);

  void* data;
  vkMapMemory(logical_device, staging_buffer_memory, 0, image_size, 0, &data);
  std::memcpy(data, pixels, static_cast<size_t>(image_size));
  vkUnmapMemory(logical_device, staging_buffer_memory);

  // Free pixel data now it's in a buffer
  stbi_image_free(pixels);

  // Same format for the texels as the pixels in the buffer,
  const VkFormat image_format = VK_FORMAT_R8G8B8A8_UNORM;
  // Texels are laid out in an implementation defined order for optimal access
  const VkImageTiling image_tiling = VK_IMAGE_TILING_OPTIMAL;

  // Sampled bit so we can access image from shader
  usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  // Allocate image on fast device side memory
  properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  createImage(logical_device, physical_device, tex_width, tex_height,
              image_format, image_tiling, usage, properties, texture_image,
              texture_image_memory);

  // Transition image layout to one optimal for copying
  const VkImageLayout copy_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  transitionImageLayout(texture_image, image_format, VK_IMAGE_LAYOUT_UNDEFINED,
                        copy_layout);
  // Perform copy
  copyBufferToImage(staging_buffer, texture_image,
                    static_cast<uint32_t>(tex_width),
                    static_cast<uint32_t>(tex_height));

  // Tranition layout to prepare for sampling from the shader
  transitionImageLayout(texture_image, image_format, copy_layout,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  // Cleanup staging buffers since data lives in image object
  vkDestroyBuffer(logical_device, staging_buffer, nullptr);
  vkFreeMemory(logical_device, staging_buffer_memory, nullptr);
}

void HelloTriangleApplication::createTextureImageView() {
  texture_image_view =
      createImageView(logical_device, texture_image, VK_FORMAT_R8G8B8A8_UNORM);
}

void HelloTriangleApplication::drawFrame() {
  FrameSync& sync = frame_sync[current_frame];

  // Blocking wait for frame to finish presentation
  vkWaitForFences(logical_device, 1, &sync.in_flight_fence, VK_TRUE,
                  std::numeric_limits<uint64_t>::max());

  // Aquire image from the swap chain, signal sempahore when ready
  uint32_t image_index;
  VkResult result = vkAcquireNextImageKHR(
      logical_device, swap_chain,
      std::numeric_limits<uint64_t>::max() /* timeout */,
      sync.image_available_semaphore, VK_NULL_HANDLE, &image_index);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    // Window has been resized, swap chain no longer compatible with the surface
    recreateSwapChain();
    return;
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("failed to acquire swap chain image!");
  }

  // Generate transformation matrix for image to rotate geometry
  updateUniformBuffer(logical_device, uniform_buffers_memory[image_index],
                      swap_chain_extent);

  // Submit command buffer for execution
  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  // wait on image being available
  const VkSemaphore wait_semaphores[] = {sync.image_available_semaphore};
  VkPipelineStageFlags wait_stages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = wait_semaphores;
  submit_info.pWaitDstStageMask = wait_stages;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffers[image_index];

  // Signal once command buffer has completed
  const VkSemaphore signal_semaphores[] = {sync.render_finished_semaphore};
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = signal_semaphores;

  // Reset fence to unsignalled state, will be signalled once command buffer
  // has completed execution
  vkResetFences(logical_device, 1, &sync.in_flight_fence);

  if (vkQueueSubmit(graphics_queue, 1, &submit_info, sync.in_flight_fence) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  // Wait on render completing before drawing image to screen
  VkPresentInfoKHR present_info = {};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = signal_semaphores;

  VkSwapchainKHR swap_chains[] = {swap_chain};
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swap_chains;
  present_info.pImageIndices = &image_index;

  // submit the request to present an image to the swap chain
  result = vkQueuePresentKHR(present_queue, &present_info);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      framebuffer_resized) {
    // Window has been resized
    framebuffer_resized = false;
    recreateSwapChain();
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to present swap chain image!");
  }

  // advance frame after drawing
  current_frame = (current_frame + 1) % max_frames_in_flight;
}

void HelloTriangleApplication::createSyncObjects() {
  frame_sync.resize(max_frames_in_flight);

  VkSemaphoreCreateInfo semaphore_info = {};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  // Create fence in signalled state, is unsignalled by default
  VkFenceCreateInfo fence_info = {};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < max_frames_in_flight; i++) {
    FrameSync& sync = frame_sync[i];
    if (vkCreateSemaphore(logical_device, &semaphore_info, nullptr,
                          &sync.image_available_semaphore) != VK_SUCCESS ||
        vkCreateSemaphore(logical_device, &semaphore_info, nullptr,
                          &sync.render_finished_semaphore) != VK_SUCCESS ||
        vkCreateFence(logical_device, &fence_info, nullptr,
                      &sync.in_flight_fence) != VK_SUCCESS) {
      throw std::runtime_error(
          "failed to create synchronization objects for a frame!");
    }
  }
}

void HelloTriangleApplication::createFramebuffers() {
  swap_chain_framebuffers.resize(swap_chain_image_views.size());

  for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
    VkImageView attachments[] = {swap_chain_image_views[i]};

    // Create a framebuffer for all of the images in the swap chain and use
    // the one that corresponds to the retrieved image at drawing time.
    VkFramebufferCreateInfo framebuffer_info = {};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = render_pass;
    framebuffer_info.attachmentCount = 1;
    framebuffer_info.pAttachments = attachments;
    framebuffer_info.width = swap_chain_extent.width;
    framebuffer_info.height = swap_chain_extent.height;
    framebuffer_info.layers = 1;

    if (vkCreateFramebuffer(logical_device, &framebuffer_info, nullptr,
                            &swap_chain_framebuffers[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create framebuffer!");
    }
  }
}

HelloTriangleApplication::~HelloTriangleApplication() {
  // Wait on outstanding queue operations to complete before freeing resources
  // that still may be in use
  vkDeviceWaitIdle(logical_device);

  cleanupSwapChain();

  vkDestroyImageView(logical_device, texture_image_view, nullptr);
  vkDestroyImage(logical_device, texture_image, nullptr);
  vkFreeMemory(logical_device, texture_image_memory, nullptr);

  vkDestroyDescriptorPool(logical_device, descriptor_pool,
                          nullptr /* allocator */);

  vkDestroyDescriptorSetLayout(logical_device, descriptor_set_layout, nullptr);

  for (size_t i = 0; i < swap_chain_images.size(); i++) {
    vkDestroyBuffer(logical_device, uniform_buffers[i], nullptr);
    vkFreeMemory(logical_device, uniform_buffers_memory[i], nullptr);
  }

  vkDestroyBuffer(logical_device, index_buffer, nullptr);
  vkFreeMemory(logical_device, index_buffer_memory, nullptr);

  vkDestroyBuffer(logical_device, vertex_buffer, nullptr);
  vkFreeMemory(logical_device, vertex_buffer_memory, nullptr);

  for (size_t i = 0; i < max_frames_in_flight; i++) {
    FrameSync& sync = frame_sync[i];
    vkDestroySemaphore(logical_device, sync.render_finished_semaphore, nullptr);
    vkDestroySemaphore(logical_device, sync.image_available_semaphore, nullptr);
    vkDestroyFence(logical_device, sync.in_flight_fence, nullptr);
  }

  vkDestroyCommandPool(logical_device, command_pool, nullptr);

  vkDestroyDevice(logical_device, nullptr);

#ifdef ENABLE_VALIDATION
  DestroyDebugUtilsMessengerEXT(instance, validation_callback, nullptr);
#endif

  vkDestroySurfaceKHR(instance, surface, nullptr);
  vkDestroyInstance(instance, nullptr);

  // Destroy window then free resources allocated by GLFW library
  glfwDestroyWindow(window);
  glfwTerminate();
}

void HelloTriangleApplication::cleanupSwapChain() {
  // Destroy swapchain dependent resources so we can recreate them when
  // incompatible with window surface.
  for (auto& framebuffer : swap_chain_framebuffers) {
    vkDestroyFramebuffer(logical_device, framebuffer, nullptr);
  }

  vkFreeCommandBuffers(logical_device, command_pool,
                       static_cast<uint32_t>(command_buffers.size()),
                       command_buffers.data());

  vkDestroyPipeline(logical_device, graphics_pipeline, nullptr);
  vkDestroyPipelineLayout(logical_device, pipeline_layout, nullptr);
  vkDestroyRenderPass(logical_device, render_pass, nullptr);

  for (auto& view : swap_chain_image_views) {
    vkDestroyImageView(logical_device, view, nullptr);
  }

  vkDestroySwapchainKHR(logical_device, swap_chain, nullptr);
}
