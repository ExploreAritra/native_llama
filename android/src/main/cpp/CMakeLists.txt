cmake_minimum_required(VERSION 3.18.1)
project("native_llama")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization and defines
add_compile_options(-O3 -fexceptions -frtti)

# --- Configuration for llama.cpp ---
set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan" FORCE)
set(GGML_VULKAN_PERF_FA OFF CACHE BOOL "Disable Flash Attention" FORCE)

# Disable unnecessary heavy builds from the fetched repo
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable server" FORCE)

# --- Force compiler to ignore hardware audio frameworks ---
add_definitions(-DMA_NO_OPENSL=1)
add_definitions(-DMA_NO_AAUDIO=1)
add_definitions(-DMA_NO_AVFOUNDATION=1)
add_definitions(-DMA_NO_COREAUDIO=1)

include(FetchContent)

# 1. Fetch Vulkan Headers
FetchContent_Declare(VulkanHeaders GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git GIT_TAG main)
FetchContent_MakeAvailable(VulkanHeaders)

# 2. Fetch SPIR-V Headers
FetchContent_Declare(SPIRVHeaders GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git GIT_TAG main)
FetchContent_MakeAvailable(SPIRVHeaders)

# 3. CRITICAL FIX: Fetch llama.cpp directly from GitHub!
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        master # Note: You can lock this to a specific commit hash for absolute stability
)
FetchContent_MakeAvailable(llama_cpp)

# Apply Vulkan headers to the downloaded target
target_include_directories(ggml-vulkan PRIVATE
    "${vulkanheaders_SOURCE_DIR}/include"
    "${spirvheaders_SOURCE_DIR}/include"
)

set(SHARED_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared_cpp")

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

# Explicitly specify include directories directly from the fetched source and local mtmd
target_include_directories(native_llama PRIVATE
    .
    "${llama_cpp_SOURCE_DIR}/include"
    "${llama_cpp_SOURCE_DIR}/common"
    "${llama_cpp_SOURCE_DIR}/src"
    "${llama_cpp_SOURCE_DIR}/ggml/include"
    "${llama_cpp_SOURCE_DIR}/ggml/src"
    "${SHARED_CPP_DIR}/tools/mtmd"
    "${SHARED_CPP_DIR}/vendor"
)

find_library(log-lib log)
# Link against the fetched llama library
target_link_libraries(native_llama llama dl vulkan ${log-lib})