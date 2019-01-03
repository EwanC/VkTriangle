cmake_minimum_required(VERSION 3.13.2)
project (triangle)

if (NOT WIN32)
  message(FATAL_ERROR "Project only supports Windows")
endif()

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
  message(FATAL_ERROR "Project only supports 64-bit systems")
endif()

if (NOT MSVC AND MSVC_VERSION LESS 1910)
  message(FATAL_ERROR "Project only supports Visual Studio 2019")
endif()

if (NOT DEFINED ENV{VULKAN_SDK})
  message(FATAL_ERROR "Could not find VULKAN_SDK environment variable")
else()
  message(STATUS "Using Vulkan SDK $ENV{VULKAN_SDK}")
endif()

find_path(VULKAN_INCLUDE_DIR
  NAMES vulkan/vulkan.h
  PATHS
  "$ENV{VULKAN_SDK}/Include"
)
include_directories(${VULKAN_INCLUDE_DIR})

set(VK_VERSION_MAJOR "1")
set(VK_LOADER_NAME "vulkan-${VK_VERSION_MAJOR}")

find_library(VULKAN_LOADER
  NAMES "${VK_LOADER_NAME}"
  PATHS "$ENV{VULKAN_SDK}/Lib" "$ENV{VULKAN_SDK}/Bin"
)

SET(GLM_INSTALL "" CACHE PATH "Path to glm install directoy")
if (NOT EXISTS ${GLM_INSTALL})
  message(FATAL_ERROR "Could find glm install at ${GLM_INSTALL}")
endif()
include_directories(${GLM_INSTALL})

SET(GLFW_INSTALL "" CACHE PATH "Path to glfw install directoy")
if (NOT EXISTS ${GLFW_INSTALL})
  message(FATAL_ERROR "Could find glfw install at ${GLFW_INSTALL}")
endif()
include_directories("${GLFW_INSTALL}\\include")
find_library(GLFW_LIB NAMES "glfw3" PATHS "${GLFW_INSTALL}\\lib-vc2015")

set(ENABLE_VALIDATION_LAYERS ON CACHE BOOL "Build with Vulkan validation layers")
if (ENABLE_VALIDATION_LAYERS)
  add_compile_definitions("ENABLE_VALIDATION")
endif()

#Compile shaders to SPIRV-v at configure time
execute_process(COMMAND compile_shaders.bat WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Load SPIR-v from file and write into header as byte array.
# More efficient than reading file at runtime
macro(EMBED_SPIRV SHADER_NAME)
  file(READ "${SHADER_NAME}.spv" SPV_BINARY HEX)
  string(LENGTH "${SPV_BINARY}" BINARY_LENGTH)

  math(EXPR BINARY_LENGTH "${BINARY_LENGTH} - 1")
  set(SPV_STRING "")
  foreach(iter RANGE 0 ${BINARY_LENGTH} 2)
    string(SUBSTRING ${SPV_BINARY} ${iter} 2 line)
    set(SPV_STRING "${SPV_STRING} 0x${line},\n")
  endforeach()
  set("${SHADER_NAME}_SPV_STRING" ${SPV_STRING})
endmacro()

EMBED_SPIRV(VERT)  # Vertex shader
EMBED_SPIRV(FRAG)  # Fragment shader

configure_file(shaders.h.in shaders.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set (CMAKE_CXX_STANDARD 17)  # std::optional

add_executable(triangle main.cpp hello_triangle_application.cpp hello_triangle_application.h shaders.h)
target_compile_options(triangle PRIVATE "/W4" "/WX")
target_link_libraries(triangle ${VULKAN_LOADER} ${GLFW_LIB})
