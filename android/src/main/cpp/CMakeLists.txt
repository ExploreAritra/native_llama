cmake_minimum_required(VERSION 3.18.1)
project("native_llama")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization and defines
add_compile_options(-O3 -fexceptions -frtti)

# --- Configuration for local llama.cpp ---
set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan" FORCE)
set(GGML_VULKAN_PERF_FA OFF CACHE BOOL "Disable Flash Attention" FORCE)

# Disable unnecessary heavy builds from the local repo
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable server" FORCE)

# --- Force compiler to ignore hardware audio frameworks ---
add_definitions(-DMA_NO_OPENSL=1)
add_definitions(-DMA_NO_AAUDIO=1)
add_definitions(-DMA_NO_AVFOUNDATION=1)
add_definitions(-DMA_NO_COREAUDIO=1)

include(FetchContent)

# 1. Fetch Vulkan Headers (Required for Android GPU)
FetchContent_Declare(VulkanHeaders GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git GIT_TAG main)
FetchContent_MakeAvailable(VulkanHeaders)

# 2. Fetch SPIR-V Headers (Required for Android GPU)
FetchContent_Declare(SPIRVHeaders GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git GIT_TAG main)
FetchContent_MakeAvailable(SPIRVHeaders)

# 3. CRITICAL FIX: Point directly to the single source of truth in the iOS folder!
# Path traversal: cpp -> main -> src -> android -> plugin_root -> ios/shared_cpp
set(SHARED_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../ios/shared_cpp")

# Add the local directory to the build
# (Requires a secondary binary_dir argument since the source is outside the current CMake tree)
add_subdirectory("${SHARED_CPP_DIR}" llama_cpp_build)

# Apply Vulkan headers to the ggml-vulkan target
target_include_directories(ggml-vulkan PRIVATE
    "${vulkanheaders_SOURCE_DIR}/include"
    "${spirvheaders_SOURCE_DIR}/include"
)

# Grab all your custom MTMD model files
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

# Explicitly specify include directories mapping to the unified local folder
target_include_directories(native_llama PRIVATE
    .
    "${SHARED_CPP_DIR}/include"
    "${SHARED_CPP_DIR}/common"
    "${SHARED_CPP_DIR}/src"
    "${SHARED_CPP_DIR}/ggml/include"
    "${SHARED_CPP_DIR}/ggml/src"
    "${SHARED_CPP_DIR}/tools/mtmd"
    "${SHARED_CPP_DIR}/vendor"
)

find_library(log-lib log)
# Link against the locally built llama library
target_link_libraries(native_llama llama dl vulkan ${log-lib})