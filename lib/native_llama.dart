import 'dart:async';
import 'package:flutter/services.dart';

class NativeLlama {
  static const MethodChannel _methodChannel = MethodChannel('native_llama/methods');
  static const EventChannel _eventChannel = EventChannel('native_llama/events');

  bool _isInitialized = false;
  bool _isDraftInitialized = false;

  bool get isInitialized => _isInitialized;
  bool get isDraftInitialized => _isDraftInitialized;

  /// Initializes the main base model
  /// [nCtx] Optional override for context window. If null, calculates based on device RAM.
  /// [nThreads] Optional override for CPU threads. Defaults to 4.
  Future<void> initModel(String absolutePath, {int? nCtx, int? nThreads}) async {
    try {
      final bool result = await _methodChannel.invokeMethod('initModel', {
        'modelPath': absolutePath,
        'nCtx': nCtx,
        'nThreads': nThreads,
      });
      _isInitialized = result;
      if (!result) throw Exception("Native initialization failed. The model may be too large for this device's memory.");
    } on PlatformException catch (e) {
      _isInitialized = false;
      throw Exception("Platform Exception during init: ${e.message}");
    }
  }

  /// Initializes the draft model for speculative decoding
  Future<void> initDraftModel(String absolutePath, {int? nCtx, int? nThreads}) async {
    try {
      final bool result = await _methodChannel.invokeMethod('initDraftModel', {
        'modelPath': absolutePath,
        'nCtx': nCtx,
        'nThreads': nThreads,
      });
      _isDraftInitialized = result;
      if (!result) throw Exception("Native initialization failed for draft model.");
    } on PlatformException catch (e) {
      _isDraftInitialized = false;
      throw Exception("Platform Exception during draft init: ${e.message}");
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
          controller.add(event.toString());
        },
        onDone: () => controller.close(),
        onError: (e) => controller.addError(e),
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
        'temperature': temperature,
        'topK': topK,
        'topP': topP,
      }).catchError((e) {
        controller.addError(e);
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
    } catch (e) {
      print("Error disposing model: $e");
    }
  }
}