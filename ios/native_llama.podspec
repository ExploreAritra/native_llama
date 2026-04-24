Pod::Spec.new do |s|
  s.name             = 'native_llama'
  s.version          = '1.0.0'
  s.summary          = 'On-device LLM plugin'
  s.description      = 'A highly optimized, hardware-accelerated Flutter plugin.'
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Timebox' => 'explorearitra@gmail.com' }
  s.source           = { :path => '.' }

  # 1. Using a physical copy for absolute reliability with CocoaPods
  s.source_files = [
    'Classes/**/*.{h,m,mm,swift,cpp,c}',
    'shared_cpp_copy/include/*.h',
    'shared_cpp_copy/ggml/include/*.h',
    'shared_cpp_copy/src/*.{h,cpp,c}',
    'shared_cpp_copy/src/models/*.{h,cpp}',
    'shared_cpp_copy/common/*.{h,cpp,c}',
    'shared_cpp_copy/common/jinja/*.{h,cpp}',
    'shared_cpp_copy/ggml/src/*.{h,c,cpp}',
    'shared_cpp_copy/ggml/src/ggml-cpu/*.{h,c,cpp}',
    'shared_cpp_copy/ggml/src/ggml-cpu/arch/arm/*.{h,c,cpp}',
    'shared_cpp_copy/ggml/src/ggml-cpu/llamafile/*.{h,cpp}',
    'shared_cpp_copy/vendor/**/*.{h,c,cpp}',
    'shared_cpp_copy/ggml/src/ggml-metal/*.{h,m,mm,cpp}'
  ]

  s.public_header_files = 'Classes/**/*.h'

  # Prevent header collisions by not making them public or private (keeping them as "project" headers)
  # CocoaPods will still allow them to be used for compilation but won't copy them to the framework's Headers folder.
  s.project_header_files = [
    'shared_cpp_copy/**/*.h',
    'shared_cpp_copy/**/*.hpp'
  ]

  s.dependency 'Flutter'
  s.platform = :ios, '17.0'

  # 2. Compile Metal shaders
  s.resources = ['shared_cpp_copy/ggml/src/ggml-metal/*.metal']

  s.compiler_flags = '-fno-objc-arc'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',

    # --- NEW: Metal 3.1 & BFloat16 Compiler Flags ---
    'MTL_PREPROCESSOR_DEFINITIONS' => 'GGML_METAL_HAS_BF16=1',
    'MTL_LANGUAGE_REVISION' => 'Metal31',
    # ------------------------------------------------

    'HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/common"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/src/models"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/src/ggml-cpu"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/src/ggml-cpu/arch/arm"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/src/ggml-cpu/llamafile"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/src/ggml-metal"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/vendor"',
      '"$(PODS_ROOT)/Headers/Public/native_llama"'
    ].join(' '),

    'USER_HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/common"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/src/models"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/src/ggml-cpu"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/src/ggml-cpu/arch/arm"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/src/ggml-cpu/llamafile"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/src/ggml-metal"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/vendor"'
    ].join(' '),

    'MTL_HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp_copy/ggml/src/ggml-metal"'
    ].join(' '),

    # Changed from c++17 to gnu++17
    'CLANG_CXX_LANGUAGE_STANDARD' => 'gnu++17',
    'CLANG_CXX_LIBRARY' => 'libc++',

    # 3. Enable Hardware Acceleration Flags
    'GCC_PREPROCESSOR_DEFINITIONS' => [
      '$(inherited)',
      'GGML_USE_METAL=1',
      'GGML_USE_ACCELERATE=1',
      'GGML_USE_CPU=1',
      'GGML_METAL_NDEBUG=1',
      'GGML_METAL_HAS_BF16=1', # --- NEW: Added BF16 Macro here ---
      'GGML_VERSION="\\"4412\\""',
      'GGML_COMMIT="\\"82f7e77\\""'
    ].join(' ')
  }

  s.frameworks = 'Accelerate', 'Metal', 'MetalKit', 'MetalPerformanceShaders'
  s.swift_version = '5.0'
end