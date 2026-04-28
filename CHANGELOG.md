## 1.0.1

* **Architectural Overhaul:** Transitioned from dynamic `FetchContent`/Git cloning to a unified, bundled `shared_cpp` architecture for flawless local symlink development and CocoaPods reliability.
* **Apple Silicon & iOS Fixes:**
    * Fixed Apple Metal Shader compilation crashes by disabling Clang modules (`-fno-modules`) and forcing explicit Objective-C++ linkage (`-x objective-c++`).
    * Resolved Apple Clang C++17 `std::hardware_destructive_interference_size` bugs to enable CPU optimization on iOS.
    * Explicitly linked the `Metal` and `Foundation` frameworks in the `.podspec`.
* **Android Fixes:** Re-routed CMake to use the unified C++ source, improving build stability and caching.
* **Package Optimization:** Implemented a highly aggressive `.pubignore` to bypass pub.dev's 100MB limit by stripping unused server/desktop GPU backends (CUDA, SYCL, etc.) and massive Git histories, while preserving core on-device LLM capabilities.


## 1.0.0

* **Breaking Change**: Renamed `imagePaths` to `mediaPaths` in `startGeneration` to support broader multi-modal capabilities.
* **Performance**: Optimized default thread allocation based on high-performance CPU cores (Core detection for Android & iOS).
* **Stability**: Improved memory management during embedding extraction and vision processing.
* **Inference**: Added support for speculative decoding and automatic KV cache management for long-context generation.
* **Platform**: Full Vulkan (Android) and Metal (iOS) GPU offloading.
* **Packaging**: Reduced package size by excluding binary artifacts and redundant source copies.
