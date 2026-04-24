import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../../../core/services/llama_service.dart';
import '../../../core/services/database_service.dart';
import '../../../core/services/semantic_cache_service.dart';
import '../../../core/models/chat_model.dart';

class AssistantController extends GetxController {
  final LlamaService _llamaService = LlamaService.instance;
  final DatabaseService _dbService = DatabaseService.instance;
  final SemanticCacheService _cacheService = SemanticCacheService();

  var messages = <ChatMessage>[].obs;
  var currentSessionId = Rxn<int>();
  var isTyping = false.obs;
  var isThinkingEnabled = true.obs;
  final ScrollController scrollController = ScrollController();

  @override
  void onInit() {
    super.onInit();
    createNewChat();
  }

  Future<void> createNewChat() async {
    currentSessionId.value = await _dbService.createChatSession("Plugin Test Chat");
    messages.clear();
  }

  void scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (scrollController.hasClients) {
        scrollController.animateTo(scrollController.position.maxScrollExtent, duration: const Duration(milliseconds: 300), curve: Curves.easeOut);
      }
    });
  }


  void sendMessage(String text) async {
    if (text.isEmpty) return;
    int sessionId = currentSessionId.value!;

    messages.add(ChatMessage(sessionId: sessionId, role: "user", content: text, createdAt: DateTime.now()));
    await _dbService.saveChatMessage(sessionId, "user", text);
    scrollToBottom();
    isTyping.value = true;

    // Check Initialization FIRST
    if (!_llamaService.isInitialized) {
      messages.add(ChatMessage(
          sessionId: sessionId,
          role: "assistant",
          content: "Model not initialized. The model may have been too large for your device's RAM. Please try a smaller model in Settings.",
          createdAt: DateTime.now()
      ));
      isTyping.value = false;
      scrollToBottom();
      return;
    }

    // Test Semantic Cache (Embeddings Feature)
    final cachedResponse = await _cacheService.getCachedResponse(text);
    if (cachedResponse != null) {
      messages.add(ChatMessage(sessionId: sessionId, role: "assistant", content: "[CACHED] $cachedResponse", createdAt: DateTime.now()));
      isTyping.value = false;
      scrollToBottom();
      return;
    }

    if (!_llamaService.isInitialized) {
      messages.add(ChatMessage(sessionId: sessionId, role: "assistant", content: "Model not initialized.", createdAt: DateTime.now()));
      isTyping.value = false;
      return;
    }

    final assistantIndex = messages.length;
    messages.add(ChatMessage(
      sessionId: sessionId,
      role: "assistant",
      content: "",
      createdAt: DateTime.now(),
      expectTags: isThinkingEnabled.value,
    ));

    String systemPrompt;
    if (isThinkingEnabled.value) {
      systemPrompt =
          "You are a helpful assistant. Provide a brief, high-level chain of thought inside <thought> tags before the final answer. Keep the thinking process concise and avoid unnecessary length. If the reasoning is complex, break it down into clear, successive steps. Then, provide the final answer inside <final_answer> tags. Example: <thought>1. Analysis... 2. Steps...</thought><final_answer>The answer.</final_answer>";
    } else {
      systemPrompt = "You are a helpful assistant. Provide a direct and concise answer without any internal reasoning tags.";
    }

    List<Map<String, String>> chatHistory = [
      {"role": "system", "text": systemPrompt}
    ];
    for (var m in messages.sublist(0, assistantIndex)) {
      chatHistory.add({"role": m.role, "text": m.content});
    }

    String fullResponse = "";
    try {
      await for (final token in _llamaService.generateResponse(chatHistory)) {
        fullResponse += token;
        messages[assistantIndex].updateContent(fullResponse);
        messages.refresh();
        scrollToBottom();
      }
      messages[assistantIndex].updateContent(fullResponse, isFinalized: true);
      messages.refresh();
      await _dbService.saveChatMessage(sessionId, "assistant", fullResponse);
      await _cacheService.cacheResponse(text, fullResponse);
    } catch (e) {
      messages[assistantIndex].updateContent("Error: $e");
    } finally {
      isTyping.value = false;
    }
  }

  void stopGeneration() {
    _llamaService.abortGeneration();
    isTyping.value = false;
  }
}