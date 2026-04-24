import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:native_llama/native_llama.dart';

class ModelInfo {
  final String name;
  final String description;
  final String url;
  final String filename;
  final double sizeGB;
  final int minRamGB;

  ModelInfo({
    required this.name,
    required this.description,
    required this.url,
    required this.filename,
    required this.sizeGB,
    required this.minRamGB,
  });
}

class LlamaService {
  static final LlamaService instance = LlamaService._internal();
  LlamaService._internal();

  final NativeLlama plugin = NativeLlama();

  bool get isInitialized => plugin.isInitialized;
  bool get isDraftInitialized => plugin.isDraftInitialized;

  final List<ModelInfo> availableModels = [
    ModelInfo(
      name: "Qwen 2.5 1.5B (Q4_K_M)",
      description: "Fast, capable model for mobile.",
      url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
      filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
      sizeGB: 1.1,
      minRamGB: 3,
    ),
    ModelInfo(
      name: "SmolLM2 135M (Q8_0)",
      description: "Ultra-compact model for fast draft generation.",
      url: "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf",
      filename: "SmolLM2-135M-Instruct-Q8_0.gguf",
      sizeGB: 0.15,
      minRamGB: 1,
    ),
  ];

  // ADDED: Optional nCtx and nThreads overrides
  Future<void> initModel(String path, {int? nCtx, int? nThreads}) async {
    await plugin.initModel(await _getTruePath(path), nCtx: nCtx, nThreads: nThreads);
  }

  // ADDED: Optional nCtx and nThreads overrides
  Future<void> initDraftModel(String path, {int? nCtx, int? nThreads}) async {
    await plugin.initDraftModel(await _getTruePath(path), nCtx: nCtx, nThreads: nThreads);
  }

  Future<List<double>> getEmbedding(String text) async {
    return await plugin.getEmbedding(text);
  }

  // ADDED: Sampler controls exposed
  Stream<String> generateResponse(
      List<Map<String, String>> messages, {
        double temperature = 0.7,
        int topK = 40,
        double topP = 0.9,
      }) {
    return plugin.generateResponse(
      messages,
      temperature: temperature,
      topK: topK,
      topP: topP,
    );
  }

  Future<void> abortGeneration() async {
    await plugin.abortGeneration();
  }

  void dispose() {
    plugin.dispose();
  }

  Future<String> _getTruePath(String path) async {
    if (path.isEmpty) return "";
    if (path.contains('/Documents/')) {
      final fileName = path.split('/').last;
      final docsDir = await getApplicationDocumentsDirectory();
      return "${docsDir.path}/models/$fileName";
    }
    return path;
  }

  Future<String> getLocalModelsPath() async {
    final directory = await getApplicationDocumentsDirectory();
    final path = "${directory.path}/models";
    final dir = Directory(path);
    if (!await dir.exists()) await dir.create(recursive: true);
    return path;
  }

  Future<List<File>> getDownloadedModels() async {
    final path = await getLocalModelsPath();
    final dir = Directory(path);
    if (!await dir.exists()) return [];
    return dir.listSync().whereType<File>().where((file) => file.path.endsWith(".gguf")).toList();
  }

  Future<String> importModel(String sourcePath) async {
    final destinationDir = await getLocalModelsPath();
    final fileName = sourcePath.split('/').last;
    final destinationPath = "$destinationDir/$fileName";
    await File(sourcePath).copy(destinationPath);
    return destinationPath;
  }
}