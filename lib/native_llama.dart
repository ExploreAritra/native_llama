import 'dart:async';
import 'package:flutter/services.dart';

class NativeLlama {
  static const MethodChannel _methodChannel = MethodChannel('native_llama/methods');
  static const EventChannel _eventChannel = EventChannel('native_llama/events');

  bool _isInitialized = false;
  bool _isDraftInitialized = false;
  bool _isVisionInitialized = false; // Note: Used for Multimodal Projector (mmproj)

  bool get isInitialized => _isInitialized;
  bool get isDraftInitialized => _isDraftInitialized;
  bool get isVisionInitialized => _isVisionInitialized;

  /// Initializes the main base model
  /// [nCtx] Optional override for context window. If null, calculates based on device RAM.
  /// [nThreads] Optional override for CPU threads. Defaults to 4.
  /// [nGpuLayers] Number of layers to offload to GPU. -1 for auto (all), 0 for CPU only.
  Future<void> initModel(String absolutePath, {int? nCtx, int? nThreads, int? nGpuLayers}) async {
    try {
      final bool result = await _methodChannel.invokeMethod('initModel', {
        'modelPath': absolutePath,
        'nCtx': nCtx,
        'nThreads': nThreads,
        'nGpuLayers': nGpuLayers ?? 0,
      });
      _isInitialized = result;
      if (!result) throw Exception("Native initialization failed. The model may be too large for this device's memory.");
    } on PlatformException catch (e) {
      _isInitialized = false;
      throw Exception("Platform Exception during init: ${e.message}");
    }
  }

  /// Initializes the draft model for speculative decoding
  Future<void> initDraftModel(String absolutePath, {int? nCtx, int? nThreads, int? nGpuLayers}) async {
    try {
      final bool result = await _methodChannel.invokeMethod('initDraftModel', {
        'modelPath': absolutePath,
        'nCtx': nCtx,
        'nThreads': nThreads,
        'nGpuLayers': nGpuLayers ?? 0,
      });
      _isDraftInitialized = result;
      if (!result) throw Exception("Native initialization failed for draft model.");
    } on PlatformException catch (e) {
      _isDraftInitialized = false;
      throw Exception("Platform Exception during draft init: ${e.message}");
    }
  }

  /// Initialize the Multimodal Projector (MTMD)
  Future<void> initVision(String absolutePath) async {
    try {
      final bool result = await _methodChannel.invokeMethod('initVision', {
        'mmprojPath': absolutePath,
      });
      _isVisionInitialized = result;
      if (!result) throw Exception("Native initialization failed for multimodal projector.");
    } on PlatformException catch (e) {
      _isVisionInitialized = false;
      throw Exception("Platform Exception during vision init: ${e.message}");
    }
  }

  /// Extracts embeddings for Vector DB/RAG integration
  Future<List<double>> getEmbedding(String text) async {
    if (!_isInitialized) throw Exception("Model not initialized.");
    try {
      final List<dynamic>? result = await _methodChannel.invokeMethod('getEmbedding', {
        'text': text,
      });
      return result?.cast<double>() ?? [];
    } catch (e) {
      print("Embedding Error: $e");
      return [];
    }
  }

  /// Generates response and streams tokens back to the UI
  Stream<String> generateResponse(
      List<Map<String, String>> messages, {
        List<String>? mediaPaths, // --- MODIFIED: Accepts images & audio files ---
        double temperature = 0.7,
        int topK = 40,
        double topP = 0.9,
      }) {
    if (!_isInitialized) {
      return Stream.error("Model not initialized");
    }

    final StreamController<String> controller = StreamController<String>();
    StreamSubscription? subscription;

    controller.onListen = () {
      subscription = _eventChannel.receiveBroadcastStream().listen(
            (event) {
          final token = event.toString();

          if (token == "__END_OF_STREAM__") {
            if (!controller.isClosed) controller.close();
          } else {
            controller.add(token);
          }
        },
        onDone: () {
          if (!controller.isClosed) controller.close();
        },
        onError: (e) {
          if (!controller.isClosed) controller.addError(e);
        },
      );

      controller.onCancel = () {
        abortGeneration();
        subscription?.cancel();
      };

      final roles = messages.map((m) => m['role']!).toList();
      final contents = messages.map((m) => m['text']!).toList();

      _methodChannel.invokeMethod('startGeneration', {
        'roles': roles,
        'contents': contents,
        'mediaPaths': mediaPaths ?? [], // --- MODIFIED: Pass media array to Native ---
        'temperature': temperature,
        'topK': topK,
        'topP': topP,
      }).catchError((e) {
        if (!controller.isClosed) controller.addError(e);
      });
    };

    return controller.stream;
  }

  /// Force stops the current generation loop
  Future<void> abortGeneration() async {
    await _methodChannel.invokeMethod('abortGeneration');
  }

  /// Safely disposes of models to prevent OOM crashes
  Future<void> dispose() async {
    try {
      await _methodChannel.invokeMethod('dispose');
      _isInitialized = false;
      _isDraftInitialized = false;
      _isVisionInitialized = false;
    } catch (e) {
      print("Error disposing model: $e");
    }
  }

  /// Gets the number of CPU cores.
  /// [performanceOnly] If true, attempts to return only high-performance cores (Android only).
  Future<int> getCpuCores({bool performanceOnly = false}) async {
    try {
      return await _methodChannel.invokeMethod('getCpuCores', {
        'performanceOnly': performanceOnly,
      }) ?? 4;
    } catch (e) {
      return 4;
    }
  }
}