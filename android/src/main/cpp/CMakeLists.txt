cmake_minimum_required(VERSION 3.18.1)
project("native_llama")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization and defines
add_compile_options(-O3 -fexceptions -frtti)

# --- Enable Vulkan for Android GPU Acceleration ---
set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan" FORCE)
set(GGML_VULKAN_PERF_FA OFF CACHE BOOL "Disable Flash Attention" FORCE)

# --- Force compiler to ignore hardware audio frameworks ---
add_definitions(-DMA_NO_OPENSL=1)
add_definitions(-DMA_NO_AAUDIO=1)
add_definitions(-DMA_NO_AVFOUNDATION=1)
add_definitions(-DMA_NO_COREAUDIO=1)

# --- Fetch Vulkan & SPIR-V C++ Headers ---
include(FetchContent)

FetchContent_Declare(
    VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(VulkanHeaders)

FetchContent_Declare(
    SPIRVHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(SPIRVHeaders)
# ------------------------------------------------------

set(SHARED_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared_cpp")

include_directories(
    "${vulkanheaders_SOURCE_DIR}/include"
    "${spirvheaders_SOURCE_DIR}/include"
)

# --- Let the official llama.cpp CMake system handle its own build ---
add_subdirectory("${SHARED_CPP_DIR}" llama_cpp_build)

file(GLOB MTMD_MODEL_SRCS "${SHARED_CPP_DIR}/tools/mtmd/models/*.cpp")

# Add our custom bridge files
add_library(native_llama SHARED
    native-lib.cpp
    build-info.cpp
    "${SHARED_CPP_DIR}/tools/mtmd/clip.cpp"
    "${SHARED_CPP_DIR}/tools/mtmd/mtmd-helper.cpp"
    "${SHARED_CPP_DIR}/tools/mtmd/mtmd.cpp"
    "${SHARED_CPP_DIR}/tools/mtmd/mtmd-image.cpp"
    "${SHARED_CPP_DIR}/tools/mtmd/mtmd-audio.cpp"
    "${SHARED_CPP_DIR}/tools/mtmd/deprecation-warning.cpp"
    ${MTMD_MODEL_SRCS}
)

target_include_directories(native_llama PRIVATE
    .
    "${SHARED_CPP_DIR}/include"
    "${SHARED_CPP_DIR}/common"
    "${SHARED_CPP_DIR}/src"
    "${SHARED_CPP_DIR}/ggml/include"
    "${SHARED_CPP_DIR}/tools/mtmd"
    "${SHARED_CPP_DIR}/vendor"
)

find_library(log-lib log)
# CRITICAL FIX: Add vulkan back to the linker so dynamic loading can find libvulkan.so
target_link_libraries(native_llama llama dl vulkan ${log-lib})