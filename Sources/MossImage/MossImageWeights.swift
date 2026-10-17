// Weight loading for the three MOSS-Image components from a diffusers snapshot
// (transformer/, text_encoder/, vae/). Two-way strict: every module key filled and every checkpoint
// key consumed (after the explicit drops), or throw — a partial load emits garbage with no other
// symptom. Loads ride the CPU stream (lazy Load ops bind to the stream current at creation; a
// multi-GB read on the GPU stream trips the Metal watchdog).

import Foundation
import Flux2VAE
import MLX
import MLXNN

public enum MossImageError: Error, CustomStringConvertible {
    case loading(String)
    case invalidInput(String)
    public var description: String {
        switch self {
        case .loading(let m): return "MossImage load: \(m)"
        case .invalidInput(let m): return "MossImage input: \(m)"
        }
    }
}

public enum MossImageWeights {

    static func loadAllArrays(directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw MossImageError.loading("no .safetensors under \(directory.path)") }
        var merged: [String: MLXArray] = [:]
        for f in files { merged.merge(try MLX.loadArrays(url: f)) { a, _ in a } }
        return merged
    }

    static func verifyAndLoad(_ model: Module, _ weights: [String: MLXArray], label: String) throws {
        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(weights.keys)
        let missing = moduleKeys.subtracting(fileKeys).sorted()
        guard missing.isEmpty else {
            throw MossImageError.loading("\(label): checkpoint missing \(missing.count) keys, e.g. "
                + missing.prefix(4).joined(separator: ", "))
        }
        let unused = fileKeys.subtracting(moduleKeys).sorted()
        guard unused.isEmpty else {
            throw MossImageError.loading("\(label): \(unused.count) unconsumed keys, e.g. "
                + unused.prefix(4).joined(separator: ", "))
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)
    }

    /// `transformer/` → MossImageTransformer in `dtype`. `rope.bands` is a derived constant the
    /// reference keeps only for checkpoint compatibility; it is rebuilt, never loaded.
    public static func loadTransformer(directory: URL, dtype: DType = .bfloat16) throws -> MossImageTransformer {
        let config = try MossImageTransformerConfiguration.load(transformerDirectory: directory)
        let model = MossImageTransformer(config)
        try Device.withDefaultDevice(.cpu) {
            var state: [String: MLXArray] = [:]
            for (k, v) in try loadAllArrays(directory: directory) where k != "rope.bands" {
                state[transformerKey(k)] = v.asType(dtype)
            }
            try verifyAndLoad(model, state, label: "transformer")
        }
        return model
    }

    /// Checkpoint key → module path (only the Sequential time MLP differs).
    public static func transformerKey(_ key: String) -> String {
        key.replacingOccurrences(of: "time_embedding.0.", with: "time_embedding.linear_1.")
            .replacingOccurrences(of: "time_embedding.2.", with: "time_embedding.linear_2.")
    }

    /// `text_encoder/` → Qwen35Encoder (language model only) in `dtype`.
    public static func loadTextEncoder(directory: URL, dtype: DType = .bfloat16) throws -> Qwen35Encoder {
        let config = try Qwen35EncoderConfiguration.load(textEncoderDirectory: directory)
        let model = Qwen35Encoder(config)
        try Device.withDefaultDevice(.cpu) {
            let state = Qwen35Encoder.sanitize(try loadAllArrays(directory: directory), dtype: dtype)
            try verifyAndLoad(model, state, label: "text_encoder")
        }
        return model
    }

    /// `vae/` → the FLUX.2 VAE decoder; the snapshot's config selects the small decoder widths.
    public static func loadVAE(directory: URL, dtype: DType = .float32) throws -> Flux2VAE {
        var vae: Flux2VAE!
        try Device.withDefaultDevice(.cpu) {
            vae = try Flux2VAEWeights.loadVAE(directory: directory, dtype: dtype)
        }
        return vae
    }
}

// MARK: - Quantization (int8 / int4 tiers, on load from the bf16 snapshot)

extension MossImageWeights {
    /// Layers kept at the load dtype: the latent in/out projections, the timestep MLP, and the two
    /// modulation producers (small, precision-sensitive scale/gate producers shared by every block).
    static func keepHiPrecision(path: String) -> Bool {
        ["video_embedder", "final_linear_video", "time_embedding", "shared_block_modulation",
         "video_head_modulation", "text_embedder.proj"].contains { path.hasPrefix($0) }
    }

    /// Quantize the DiT's block and text-refiner Linears in place (affine, group 64).
    public static func quantizeTransformer(_ model: MossImageTransformer, bits: Int, groupSize: Int = 64) {
        quantize(model: model, filter: { path, module in
            guard module is Linear, !keepHiPrecision(path: path) else { return nil }
            return (groupSize, bits, .affine)
        })
        eval(model)
    }
}
