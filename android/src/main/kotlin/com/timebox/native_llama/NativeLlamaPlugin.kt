package com.timebox.native_llama

import android.os.Handler
import android.os.Looper
import androidx.annotation.Keep
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import java.util.concurrent.Executors

class NativeLlamaPlugin: FlutterPlugin, MethodCallHandler, EventChannel.StreamHandler {
    private lateinit var methodChannel: MethodChannel
    private lateinit var eventChannel: EventChannel

    private var eventSink: EventChannel.EventSink? = null
    private val handler = Handler(Looper.getMainLooper())
    // Executes C++ inference on a background thread to keep Flutter UI smooth
    private val executor = Executors.newSingleThreadExecutor()

    companion object {
        init {
            System.loadLibrary("native_llama")
        }
    }

    // Native JNI bindings
    private external fun initLlama(modelPath: String, nCtx: Int, nThreads: Int): Boolean
    private external fun initDraftModel(modelPath: String, nCtx: Int, nThreads: Int): Boolean
    private external fun getEmbedding(text: String): DoubleArray?
    private external fun startNativeGeneration(roles: Array<String>, contents: Array<String>, temperature: Float, topK: Int, topP: Float)
    private external fun abortGeneration()
    private external fun disposeLlama()

    override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        methodChannel = MethodChannel(flutterPluginBinding.binaryMessenger, "native_llama/methods")
        methodChannel.setMethodCallHandler(this)

        eventChannel = EventChannel(flutterPluginBinding.binaryMessenger, "native_llama/events")
        eventChannel.setStreamHandler(this)
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "initModel" -> {
                val modelPath = call.argument<String>("modelPath")
                val nCtx = call.argument<Int>("nCtx") ?: -1
                val nThreads = call.argument<Int>("nThreads") ?: -1

                if (modelPath != null) {
                    executor.execute {
                        val success = initLlama(modelPath, nCtx, nThreads)
                        handler.post { result.success(success) }
                    }
                } else {
                    result.error("INVALID_ARGUMENT", "Model path is null", null)
                }
            }
            "initDraftModel" -> {
                val modelPath = call.argument<String>("modelPath")
                val nCtx = call.argument<Int>("nCtx") ?: -1
                val nThreads = call.argument<Int>("nThreads") ?: -1

                if (modelPath != null) {
                    executor.execute {
                        val success = initDraftModel(modelPath, nCtx, nThreads)
                        handler.post { result.success(success) }
                    }
                } else {
                    result.error("INVALID_ARGUMENT", "Draft model path is null", null)
                }
            }
            "getEmbedding" -> {
                val text = call.argument<String>("text")
                if (text != null) {
                    executor.execute {
                        val embedding = getEmbedding(text)
                        handler.post {
                            if (embedding != null) result.success(embedding.toList())
                            else result.error("EMBEDDING_ERROR", "Failed to get embedding", null)
                        }
                    }
                } else {
                    result.error("INVALID_ARGUMENT", "Text is null", null)
                }
            }
            "startGeneration" -> {
                val roles = call.argument<List<String>>("roles")?.toTypedArray()
                val contents = call.argument<List<String>>("contents")?.toTypedArray()

                // Cast Double from Dart to Float for Kotlin/C++ boundary
                val temperature = call.argument<Double>("temperature")?.toFloat() ?: 0.7f
                val topK = call.argument<Int>("topK") ?: 40
                val topP = call.argument<Double>("topP")?.toFloat() ?: 0.9f

                if (roles != null && contents != null) {
                    executor.execute {
                        startNativeGeneration(roles, contents, temperature, topK, topP)
                    }
                    result.success(null)
                } else {
                    result.error("INVALID_ARGUMENT", "Roles or contents are null", null)
                }
            }
            "abortGeneration" -> {
                abortGeneration()
                result.success(true)
            }
            "dispose" -> {
                executor.execute {
                    disposeLlama()
                    handler.post { result.success(true) }
                }
            }
            else -> result.notImplemented()
        }
    }

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        eventSink = events
    }

    override fun onCancel(arguments: Any?) {
        eventSink = null
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        methodChannel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        disposeLlama()
    }

    // Called from C++ JNI
    @Keep
    fun onTokenReceived(token: String) {
        handler.post {
            if (token == "__END_OF_STREAM__") {
                eventSink?.endOfStream()
            } else {
                eventSink?.success(token)
            }
        }
    }
}