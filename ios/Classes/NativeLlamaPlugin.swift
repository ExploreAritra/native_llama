import Flutter
import UIKit

public class NativeLlamaPlugin: NSObject, FlutterPlugin, FlutterStreamHandler {
    private var eventSink: FlutterEventSink?

    public static func register(with registrar: FlutterPluginRegistrar) {
        let methodChannel = FlutterMethodChannel(name: "native_llama/methods", binaryMessenger: registrar.messenger())
        let eventChannel = FlutterEventChannel(name: "native_llama/events", binaryMessenger: registrar.messenger())

        let instance = NativeLlamaPlugin()
        registrar.addMethodCallDelegate(instance, channel: methodChannel)
        eventChannel.setStreamHandler(instance)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "initModel":
            guard let args = call.arguments as? [String: Any],
                  let modelPath = args["modelPath"] as? String else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "Model path is null", details: nil))
                return
            }
            // Move heavy initialization to a background thread
            DispatchQueue.global(qos: .userInitiated).async {
                let success = LlamaBridge.shared().initModel(modelPath)
                DispatchQueue.main.async { result(success) }
            }

        case "initDraftModel":
            guard let args = call.arguments as? [String: Any],
                  let modelPath = args["modelPath"] as? String else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "Model path is null", details: nil))
                return
            }
            DispatchQueue.global(qos: .userInitiated).async {
                let success = LlamaBridge.shared().initDraftModel(modelPath)
                DispatchQueue.main.async { result(success) }
            }

        case "getEmbedding":
            guard let args = call.arguments as? [String: Any],
                  let text = args["text"] as? String else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "Text is null", details: nil))
                return
            }
            DispatchQueue.global(qos: .userInitiated).async {
                let embedding = LlamaBridge.shared().getEmbedding(text)
                DispatchQueue.main.async { result(embedding) }
            }

        case "startGeneration":
            guard let args = call.arguments as? [String: Any],
                  let roles = args["roles"] as? [String],
                  let contents = args["contents"] as? [String] else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "Roles or contents are null", details: nil))
                return
            }

            // Prevent screen from sleeping during long generations
            UIApplication.shared.isIdleTimerDisabled = true

            DispatchQueue.global(qos: .userInitiated).async {
                LlamaBridge.shared().startGeneration(withRoles: roles, contents: contents) { [weak self] token in
                    guard let token = token else { return }
                    DispatchQueue.main.async {
                        if token == "__END_OF_STREAM__" {
                            UIApplication.shared.isIdleTimerDisabled = false
                            self?.eventSink?(FlutterEndOfEventStream)
                        } else {
                            self?.eventSink?(token)
                        }
                    }
                }
            }
            result(nil)

        case "abortGeneration":
            LlamaBridge.shared().abortGeneration()
            result(true)

        case "dispose":
            DispatchQueue.global(qos: .userInitiated).async {
                LlamaBridge.shared().dispose()
                DispatchQueue.main.async { result(true) }
            }

        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // MARK: - FlutterStreamHandler
    public func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        self.eventSink = events
        return nil
    }

    public func onCancel(withArguments arguments: Any?) -> FlutterError? {
        self.eventSink = nil
        return nil
    }
}