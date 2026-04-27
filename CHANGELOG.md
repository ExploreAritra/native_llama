## 1.0.0

* **Breaking Change**: Renamed `imagePaths` to `mediaPaths` in `startGeneration` to support broader multi-modal capabilities.
* **Performance**: Optimized default thread allocation based on high-performance CPU cores (Core detection for Android & iOS).
* **Stability**: Improved memory management during embedding extraction and vision processing.
* **Inference**: Added support for speculative decoding and automatic KV cache management for long-context generation.
* **Platform**: Full Vulkan (Android) and Metal (iOS) GPU offloading.
* **Packaging**: Reduced package size by excluding binary artifacts and redundant source copies.
