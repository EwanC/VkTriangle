cmake_minimum_required(VERSION 2.8.9)
project (triangle)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Vulkan)

message(STATUS "Vulkan FOUND = ${Vulkan_FOUND}")
message(STATUS "Vulkan Include = ${Vulkan_INCLUDE_DIR}")
message(STATUS "Vulkan Lib = ${Vulkan_LIBRARY}")

include_directories(${Vulkan_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}\\..\\glm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}\\..\\glfw-3.2.1.bin.WIN64\\include)

set(VULKAN_LOADER_NAME "vulkan-1")
find_library(VULKAN_LOADER NAMES ${VULKAN_LOADER_NAME} HINTS "${Vulkan_LIBRARY}\\..")

set(GLFW_PATH "${CMAKE_CURRENT_SOURCE_DIR}\\..\\glfw-3.2.1.bin.WIN64\\lib-vc2013")
find_library(GLFW_LIB NAMES "glfw3" HINTS ${GLFW_PATH})


add_executable(triangle main.cpp)
target_link_libraries(triangle ${VULKAN_LOADER} ${GLFW_LIB})