Pod::Spec.new do |s|
  s.name             = 'native_llama'
  s.version          = '1.0.1'
  s.summary          = 'On-device LLM plugin'
  s.description      = 'A highly optimized, hardware-accelerated Flutter plugin.'
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Timebox' => 'explorearitra@gmail.com' }
  s.source           = { :path => '.' }

  # 1. Using a physical copy for absolute reliability with CocoaPods
  s.source_files = [
    'Classes/**/*.{h,m,mm,swift,cpp,c}',
    'shared_cpp/include/*.h',
    'shared_cpp/ggml/include/*.h',
    'shared_cpp/src/*.{h,cpp,c}',
    'shared_cpp/src/models/*.{h,cpp}',
    'shared_cpp/common/*.{h,cpp,c}',
    'shared_cpp/common/jinja/*.{h,cpp}',
    'shared_cpp/ggml/src/*.{h,c,cpp}',
    'shared_cpp/ggml/src/ggml-cpu/*.{h,c,cpp}',
    'shared_cpp/ggml/src/ggml-cpu/arch/arm/*.{h,c,cpp}',
    'shared_cpp/ggml/src/ggml-cpu/llamafile/*.{h,cpp}',
    'shared_cpp/vendor/**/*.{h,c,cpp}',
    'shared_cpp/ggml/src/ggml-metal/*.{h,m,mm,cpp}',
    'shared_cpp/tools/mtmd/**/*.{h,cpp,c}'
  ]

  # Exclude ALL standalone CLI and Debug tools to prevent duplicate main() symbols
  s.exclude_files = [
    'shared_cpp/tools/mtmd/mtmd-cli.cpp',
    'shared_cpp/tools/mtmd/debug/mtmd-debug.cpp'
  ]

  s.public_header_files = 'Classes/**/*.h'

  s.project_header_files = [
    'shared_cpp/**/*.h',
    'shared_cpp/**/*.hpp'
  ]

  s.dependency 'Flutter'
  s.platform = :ios, '17.0'

  # 2. Compile Metal shaders
  s.resources = ['shared_cpp/ggml/src/ggml-metal/*.metal']

  s.compiler_flags = '-fno-objc-arc -DMA_NO_AVFOUNDATION=1 -DMA_NO_COREAUDIO=1'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',

    'MTL_PREPROCESSOR_DEFINITIONS' => 'GGML_METAL_HAS_BF16=1',
    'MTL_LANGUAGE_REVISION' => 'Metal31',

    'OTHER_LDFLAGS' => '$(inherited) -framework Metal -framework Foundation',

    # --- CRITICAL FIX: Undefine the broken Apple cache line macro and Force Obj-C++ ---
    'OTHER_CPLUSPLUSFLAGS' => '$(inherited) -fno-modules -x objective-c++ -U__cpp_lib_hardware_interference_size',

    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY' => 'libc++',

    'HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/common"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/src/models"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-cpu"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-cpu/arch/arm"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-cpu/llamafile"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-metal"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/vendor"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/tools/mtmd"',
      '"$(PODS_ROOT)/Headers/Public/native_llama"'
    ].join(' '),

    'USER_HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/common"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/src/models"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-cpu"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-cpu/arch/arm"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-cpu/llamafile"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-metal"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/vendor"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/tools/mtmd"'
    ].join(' '),

    'MTL_HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-metal"'
    ].join(' '),

    'GCC_PREPROCESSOR_DEFINITIONS' => [
      '$(inherited)',
      'GGML_USE_METAL=1',
      'GGML_USE_ACCELERATE=1',
      'GGML_USE_CPU=1',
      'GGML_METAL_NDEBUG=1',
      'GGML_METAL_HAS_BF16=1',
      'GGML_VERSION="\\"4412\\""',
      'GGML_COMMIT="\\"82f7e77\\""'
    ].join(' ')
  }

  s.frameworks = 'Accelerate', 'Metal', 'MetalKit', 'MetalPerformanceShaders'
  s.swift_version = '5.0'
end