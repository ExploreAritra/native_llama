import 'dart:io';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../controller/settings_controller.dart';
import '../../../core/services/llama_service.dart';

class SettingsView extends GetView<SettingsController> {
  const SettingsView({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Plugin Settings'),
        actions: [
          IconButton(icon: const Icon(Icons.file_download), onPressed: controller.importModelFile),
        ],
      ),
      body: Obx(() => ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildSectionHeader('Active Configuration'),
          Card(
            color: Colors.blueGrey[900],
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  _buildModelDropdown(
                    label: "Main LLM Engine", icon: Icons.psychology,
                    value: controller.selectedModelPath.value,
                    options: controller.mainModels,
                    onChanged: (p) => controller.loadModel(p!),
                  ),
                  const Divider(height: 32, color: Colors.white10),
                  _buildModelDropdown(
                    label: "Draft Model (Speculative Decoding)", icon: Icons.bolt,
                    value: controller.selectedDraftModelPath.value,
                    options: controller.draftModels,
                    onChanged: (p) => controller.loadDraftModel(p!),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),
          _buildSectionHeader('Available to Download'),
          ...LlamaService.instance.availableModels.map((m) => _buildModelDownloadCard(m)),
          const SizedBox(height: 24),
          _buildSectionHeader('Local Storage'),
          ...controller.downloadedModels.map((file) => _buildLocalModelCard(file)),
          if (controller.isDownloading.value) ...[
            const SizedBox(height: 24),
            _buildSectionHeader('Downloading: ${controller.currentDownloadingModel.value}'),
            LinearProgressIndicator(value: controller.downloadProgress.value),
          ],
        ],
      )),
    );
  }

  Widget _buildModelDropdown({required String label, required IconData icon, required String value, required List<File> options, required void Function(String?) onChanged}) {
    final currentValue = options.any((f) => f.path == value) ? value : null;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(children: [Icon(icon, size: 16, color: Colors.blueAccent), const SizedBox(width: 8), Text(label)]),
        const SizedBox(height: 8),
        DropdownButtonFormField<String>(
          value: currentValue,
          dropdownColor: Colors.blueGrey[900],
          isExpanded: true,
          decoration: InputDecoration(filled: true, fillColor: Colors.black26, border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none)),
          items: options.map((f) => DropdownMenuItem(value: f.path, child: Text(f.path.split('/').last, overflow: TextOverflow.ellipsis))).toList(),
          onChanged: options.isEmpty ? null : onChanged,
        ),
      ],
    );
  }

  Widget _buildModelDownloadCard(ModelInfo model) {
    final isDownloaded = controller.downloadedModels.any((f) => f.path.endsWith(model.filename));
    return Card(
      color: Colors.grey[900],
      child: ListTile(
        title: Text(model.name, style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text("${model.sizeGB}GB"),
        trailing: isDownloaded ? const Icon(Icons.check_circle, color: Colors.green) : IconButton(icon: const Icon(Icons.download, color: Colors.blue), onPressed: () => controller.downloadModel(model)),
      ),
    );
  }

  Widget _buildLocalModelCard(File file) {
    return Card(
      color: Colors.grey[900],
      child: ListTile(
        title: Text(file.path.split('/').last),
        trailing: IconButton(icon: const Icon(Icons.delete, color: Colors.red), onPressed: () => controller.deleteModel(file)),
      ),
    );
  }

  Widget _buildSectionHeader(String title) => Padding(padding: const EdgeInsets.only(bottom: 8), child: Text(title.toUpperCase(), style: const TextStyle(color: Colors.blueAccent, fontWeight: FontWeight.bold)));
}