cmake_minimum_required(VERSION 3.20)
project(lux-vulkan VERSION 0.1.0 LANGUAGES CXX)

# lux-vulkan is a PLUGIN (shared library) loaded by lux-accel at runtime.
# It links nothing from lux-accel — it only consumes backend_api.h headers.

# Locate lux-accel for the backend_api.h header (header-only consumption).
find_package(lux-accel CONFIG QUIET)
if(NOT lux-accel_FOUND)
  if(NOT LUXACCEL_INCLUDE_DIR)
    set(LUXACCEL_INCLUDE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/../lux-accel/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/../install/include")
  endif()
endif()

add_library(lux_vulkan MODULE
  src/vulkan_plugin.cpp
)

target_compile_features(lux_vulkan PRIVATE cxx_std_17)

target_include_directories(lux_vulkan
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${LUXACCEL_INCLUDE_DIR}
)

if(lux-accel_FOUND AND TARGET lux::accel)
  target_include_directories(lux_vulkan PRIVATE
    $<TARGET_PROPERTY:lux::accel,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

# ── Vulkan runtime ──────────────────────────────────────────────────────────
# Prefer CMake's FindVulkan (honours $VULKAN_SDK). Fall back to a manual search
# that covers Homebrew (vulkan-headers + vulkan-loader) and system installs.
set(_lux_vk_found FALSE)
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
  target_link_libraries(lux_vulkan PRIVATE Vulkan::Vulkan)
  set(_lux_vk_found TRUE)
  message(STATUS "lux-vulkan: using Vulkan::Vulkan (${Vulkan_LIBRARY})")
else()
  find_path(VK_INCLUDE_DIR vulkan/vulkan.h
    HINTS /opt/homebrew/opt/vulkan-headers/include
          /usr/local/include /opt/homebrew/include /usr/include)
  find_library(VK_LIBRARY NAMES vulkan vulkan-1
    HINTS /opt/homebrew/opt/vulkan-loader/lib
          /usr/local/lib /opt/homebrew/lib /usr/lib)
  if(VK_INCLUDE_DIR AND VK_LIBRARY)
    target_include_directories(lux_vulkan PRIVATE ${VK_INCLUDE_DIR})
    target_link_libraries(lux_vulkan PRIVATE ${VK_LIBRARY})
    set(_lux_vk_found TRUE)
    message(STATUS "lux-vulkan: using ${VK_LIBRARY} (headers: ${VK_INCLUDE_DIR})")
  endif()
endif()

if(_lux_vk_found)
  target_compile_definitions(lux_vulkan PRIVATE LUX_HAS_VULKAN=1)

  # ── shaderc (optional): enables create_kernel_from_source (GLSL -> SPIR-V) ──
  find_path(SHADERC_INCLUDE_DIR shaderc/shaderc.hpp
    HINTS /opt/homebrew/opt/shaderc/include
          /usr/local/include /opt/homebrew/include /usr/include)
  find_library(SHADERC_LIBRARY NAMES shaderc_shared shaderc_combined shaderc
    HINTS /opt/homebrew/opt/shaderc/lib
          /usr/local/lib /opt/homebrew/lib /usr/lib)
  if(SHADERC_INCLUDE_DIR AND SHADERC_LIBRARY)
    target_include_directories(lux_vulkan PRIVATE ${SHADERC_INCLUDE_DIR})
    target_link_libraries(lux_vulkan PRIVATE ${SHADERC_LIBRARY})
    target_compile_definitions(lux_vulkan PRIVATE LUX_HAS_SHADERC=1)
    message(STATUS "lux-vulkan: GLSL source compilation enabled via ${SHADERC_LIBRARY}")
  else()
    message(STATUS "lux-vulkan: libshaderc not found; create_kernel_from_source "
                   "disabled (create_kernel_from_bundle with SPIR-V still works)")
  endif()
else()
  message(STATUS "lux-vulkan: no Vulkan runtime found; building stub plugin "
                 "(is_available() returns 0)")
endif()

# Plugin settings — hidden visibility, .plugin extension matching loader pattern.
set_target_properties(lux_vulkan PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  PREFIX ""
  SUFFIX ".plugin"
)

include(GNUInstallDirs)

install(TARGETS lux_vulkan
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/lux/plugins
)

# Smoke test — dlopens the plugin; on a host with a Vulkan ICD it runs a real
# GLSL vector-add dispatch and verifies the result.
option(LUX_VULKAN_BUILD_TESTS "Build the lux-vulkan plugin smoke test" ON)
if(LUX_VULKAN_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_load.cpp")
  add_executable(lux_vulkan_test_load test/test_load.cpp)
  target_compile_features(lux_vulkan_test_load PRIVATE cxx_std_17)
  target_include_directories(lux_vulkan_test_load PRIVATE ${LUXACCEL_INCLUDE_DIR})
  if(lux-accel_FOUND AND TARGET lux::accel)
    target_include_directories(lux_vulkan_test_load PRIVATE
      $<TARGET_PROPERTY:lux::accel,INTERFACE_INCLUDE_DIRECTORIES>)
  endif()
  if(NOT WIN32)
    target_link_libraries(lux_vulkan_test_load PRIVATE ${CMAKE_DL_LIBS})
  endif()
  add_dependencies(lux_vulkan_test_load lux_vulkan)

  enable_testing()
  add_test(NAME lux_vulkan_load
           COMMAND lux_vulkan_test_load $<TARGET_FILE:lux_vulkan>)
endif()
