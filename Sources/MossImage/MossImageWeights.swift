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

    /// `evaluate: false` leaves the module holding the lazy loads, so a caller can quantize them
    /// before anything is materialized (the bf16 copy is then never resident in full).
    static func verifyAndLoad(_ model: Module, _ weights: [String: MLXArray], label: String,
                              evaluate: Bool = true) throws {
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
        if evaluate { eval(model) }
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

    /// `text_encoder/` → Qwen35Encoder (language model only) in `dtype`; `quantBits` quantizes the
    /// Linears on load (`quantizeTextEncoder`) straight from the lazy loads, layer by layer.
    public static func loadTextEncoder(directory: URL, dtype: DType = .bfloat16,
                                       quantBits: Int? = nil, groupSize: Int = 64) throws -> Qwen35Encoder {
        let config = try Qwen35EncoderConfiguration.load(textEncoderDirectory: directory)
        let model = Qwen35Encoder(config)
        try Device.withDefaultDevice(.cpu) {
            try verifyAndLoad(model, Qwen35Encoder.sanitize(try loadAllArrays(directory: directory), dtype: dtype),
                              label: "text_encoder", evaluate: quantBits == nil)
            if let quantBits { quantizeTextEncoder(model, bits: quantBits, groupSize: groupSize) }
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

    /// Quantize the DiT's block and text-refiner Linears in place (affine). `eightBit`: path
    /// substrings that stay at 8 bits inside a 4-bit tier (mixed recipe). `MOSS_QUANT_EIGHT_BIT` /
    /// `MOSS_QUANT_GROUP` override for recipe exploration.
    public static func quantizeTransformer(_ model: MossImageTransformer, bits: Int, groupSize: Int = 64,
                                           eightBit: [String] = []) {
        let env = ProcessInfo.processInfo.environment
        let keep8 = env["MOSS_QUANT_EIGHT_BIT"].map { $0.split(separator: ",").map(String.init) } ?? eightBit
        let group = env["MOSS_QUANT_GROUP"].flatMap(Int.init) ?? groupSize
        quantize(model: model, filter: { path, module in
            guard module is Linear, !keepHiPrecision(path: path) else { return nil }
            let b = keep8.contains { path.contains($0) } ? 8 : bits
            return (group, b, .affine)
        })
        eval(model)
    }

    /// Encoder layers kept at the load dtype inside a quantized encoder: the GDN decay-gate inputs
    /// `in_proj_a` / `in_proj_b` (32 outputs each, 0.5 MB per layer — they feed exp(−A·softplus(a + dt))
    /// and sigmoid(b) for the fp32 delta rule). Everything that is not a Linear — embed_tokens, every
    /// RMSNorm, conv1d, A_log, dt_bias — is never touched by the filter.
    static func keepEncoderHiPrecision(path: String) -> Bool {
        path.hasSuffix(".in_proj_a") || path.hasSuffix(".in_proj_b")
    }

    /// Quantize the Qwen3.5-9B encoder's Linears in place (affine, `groupSize` along the input dim):
    /// attention q/k/v/o, GDN in_proj_qkv / in_proj_z / out_proj, and the MLPs. Evaluated one decoder
    /// layer at a time, so when the model still holds lazy loads (`loadTextEncoder(quantBits:)`) each
    /// layer's bf16 weights are read, quantized and dropped before the next — the load high-water is
    /// the quantized model plus one bf16 layer. `MOSS_TE_QUANT_ALL=1` also quantizes in_proj_a/b.
    public static func quantizeTextEncoder(_ model: Qwen35Encoder, bits: Int, groupSize: Int = 64) {
        let all = ProcessInfo.processInfo.environment["MOSS_TE_QUANT_ALL"] == "1"
        quantize(model: model, filter: { path, module in
            guard module is Linear, all || !keepEncoderHiPrecision(path: path) else { return nil }
            return (groupSize, bits, .affine)
        })
        eval(model.languageModel.embedTokens)
        eval(model.languageModel.norm)
        for layer in model.languageModel.layers { eval(layer) }
    }
}
