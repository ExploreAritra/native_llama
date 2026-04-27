# native_llama

A high-performance Flutter plugin for running **Llama** and other LLMs locally on mobile devices. Powered by `llama.cpp` and `MTMD`, it supports Android (Vulkan) and iOS (Metal) acceleration.

## Features

- 🚀 **Local Inference**: Run LLMs entirely on-device (no internet required).
- ⚡ **Hardware Acceleration**: 
  - **Android**: Vulkan support for high-performance GPU offloading.
  - **iOS**: Metal support optimized for Apple Silicon.
- 👁️ **Multimodal (Vision)**: Support for vision models (e.g., Qwen2-VL) via multimodal projectors (`mmproj`).
- ⏩ **Speculative Decoding**: Use a smaller "draft" model to accelerate generation from a larger "target" model.
- 🧠 **Embeddings**: Extract high-quality text embeddings for RAG and semantic search.
- 🛠️ **Fine-grained Control**:
  - Control CPU thread count (auto-detects performance cores).
  - Adjust GPU layer offloading.
  - Dynamic context window management.

## Installation

Add `native_llama` to your `pubspec.yaml`:

```yaml
dependencies:
  native_llama: ^1.0.0
```

### Android Setup
Ensure your `minSdkVersion` is at least **28** in `android/app/build.gradle`.
For optimal performance, add `android:largeHeap="true"` and `android:extractNativeLibs="true"` to your `AndroidManifest.xml`.

### iOS Setup
Ensure your deployment target is at least **iOS 13.0**.
Metal acceleration is enabled by default on compatible devices.

## Usage

### Initialize Model
```dart
final llama = NativeLlama();

await llama.initModel(
  "/path/to/model.gguf",
  nCtx: 2048,
  nThreads: 4,
  nGpuLayers: -1, // -1 for full GPU offload
);
```

### Text Generation (Streaming)
```dart
final messages = [
  {'role': 'system', 'text': 'You are a helpful assistant.'},
  {'role': 'user', 'text': 'Hello!'},
];

llama.generateResponse(messages).listen((token) {
  print("Received token: $token");
});
```

### Multimodal Vision
```dart
await llama.initVision("/path/to/mmproj.gguf");

llama.generateResponse(
  messages,
  mediaPaths: ["/path/to/image.jpg"],
).listen((token) {
  // ...
});
```

### Speculative Decoding
```dart
await llama.initDraftModel("/path/to/small_draft_model.gguf");

// Generation will automatically use the draft model for speedup
llama.generateResponse(messages).listen((token) => ...);
```

## Example App
Check the `example` directory for a full-featured chat application that includes:
- Model downloading and management.
- Hardware configuration settings.
- Image attachment support.
- Real-time performance metrics.

## License
MIT
