// Qwen3.5-9B text encoder (prefill only) for MOSS-Image — HF transformers `Qwen3_5Model` text path,
// adapted from mlx-swift-lm's MLXLLM/Models/Qwen35.swift (decode machinery, caches, MoE and the
// fused projections dropped). MOSS-Image conditions on `last_hidden_state` (AFTER the final norm)
// of one unpadded chat-templated prompt, so there is no cache, no padding mask and no LM head.
//
// HF facts this mirrors (modeling_qwen3_5.py):
//   - Qwen3_5RMSNorm is zero-centred: x̂ · (1 + w), computed in fp32, cast back — input/post-attention
//     layernorms, the final norm, and the attention q/k norms. The weights are kept in fp32 here
//     with the 1 already added (`sanitize`), so bf16 runs don't round (1 + w).
//   - The GDN's gated norm (Qwen3_5RMSNormGated) uses w directly: (x̂ · w) · silu(z).
//   - full attention: q_proj emits [query | output gate] per head; the gate is sigmoid-multiplied
//     before o_proj; partial RoPE over the leading 64 of 256 dims, θ = 1e7. Interleaved M-RoPE
//     with identical t/h/w ids (text-only) is plain rotate-half RoPE.
//   - GDN: causal depthwise conv (k = 4, zero left pad) → silu → split q/k/v; q, k l2-normalised
//     with FLA's eps-on-the-sum l2norm, q scaled by Dk^-0.5, the rule in fp32; key heads
//     repeat_interleave'd onto the value heads.
//   - layer_types: every 4th layer (3, 7, …, 31) is full attention, the rest GDN.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

public struct Qwen35EncoderConfiguration: Codable, Sendable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var fullAttentionInterval: Int
    public var linearConvKernelDim: Int
    public var linearKeyHeadDim: Int
    public var linearNumKeyHeads: Int
    public var linearNumValueHeads: Int
    public var linearValueHeadDim: Int
    public var partialRotaryFactor: Float
    public var ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case fullAttentionInterval = "full_attention_interval"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearValueHeadDim = "linear_value_head_dim"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeParameters = "rope_parameters"
    }

    enum RopeKeys: String, CodingKey { case ropeTheta = "rope_theta" }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        fullAttentionInterval = try c.decode(Int.self, forKey: .fullAttentionInterval)
        linearConvKernelDim = try c.decode(Int.self, forKey: .linearConvKernelDim)
        linearKeyHeadDim = try c.decode(Int.self, forKey: .linearKeyHeadDim)
        linearNumKeyHeads = try c.decode(Int.self, forKey: .linearNumKeyHeads)
        linearNumValueHeads = try c.decode(Int.self, forKey: .linearNumValueHeads)
        linearValueHeadDim = try c.decode(Int.self, forKey: .linearValueHeadDim)
        partialRotaryFactor = try c.decode(Float.self, forKey: .partialRotaryFactor)
        let rope = try c.nestedContainer(keyedBy: RopeKeys.self, forKey: .ropeParameters)
        ropeTheta = try rope.decode(Float.self, forKey: .ropeTheta)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
    }

    /// `text_config` of the snapshot's `text_encoder/config.json`.
    public static func load(textEncoderDirectory: URL) throws -> Qwen35EncoderConfiguration {
        let data = try Data(contentsOf: textEncoderDirectory.appendingPathComponent("config.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = root["text_config"] else {
            throw MossImageError.loading("text_encoder/config.json lacks text_config")
        }
        return try JSONDecoder().decode(
            Qwen35EncoderConfiguration.self, from: JSONSerialization.data(withJSONObject: text))
    }

    public func isFullAttention(_ layer: Int) -> Bool { (layer + 1) % fullAttentionInterval == 0 }
}

/// Qwen3_5RMSNorm with the zero-centred weight pre-shifted: weight holds (1 + w), fp32.
final class Qwen35RMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps).asType(x.dtype)
    }
}

/// Qwen3_5RMSNormGated: (x̂ · w) in the input dtype, then · silu(z) in fp32.
final class Qwen35RMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let normed = MLXFast.rmsNorm(x.asType(.float32), weight: MLXArray.mlxNone, eps: eps).asType(dtype)
        let weighted = weight.asType(dtype) * normed
        return (weighted.asType(.float32) * silu(gate.asType(.float32))).asType(dtype)
    }
}

final class Qwen35EncoderGDN: Module {
    let numVHeads, numKHeads, headKDim, headVDim, keyDim, valueDim, convKernelSize, convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Qwen35RMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen35EncoderConfiguration) {
        numVHeads = args.linearNumValueHeads
        numKHeads = args.linearNumKeyHeads
        headKDim = args.linearKeyHeadDim
        headVDim = args.linearValueHeadDim
        keyDim = headKDim * numKHeads
        valueDim = headVDim * numVHeads
        convKernelSize = args.linearConvKernelDim
        convDim = keyDim * 2 + valueDim
        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim, kernelSize: convKernelSize,
            padding: 0, groups: convDim, bias: false)
        _inProjQKV.wrappedValue = Linear(args.hiddenSize, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(args.hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(args.hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(args.hiddenSize, numVHeads, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        _aLog.wrappedValue = MLXArray.zeros([numVHeads])
        _norm.wrappedValue = Qwen35RMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, args.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, S) = (x.dim(0), x.dim(1))
        let qkv = inProjQKV(x)
        let z = inProjZ(x).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(x)
        let a = inProjA(x)

        // Causal depthwise conv: zero left pad of k - 1.
        let padded = concatenated(
            [MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: qkv.dtype), qkv], axis: 1)
        let convOut = silu(conv1d(padded))
        let parts = split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = parts[0].reshaped(B, S, numKHeads, headKDim)
        let k = parts[1].reshaped(B, S, numKHeads, headKDim)
        let v = parts[2].reshaped(B, S, numVHeads, headVDim)

        // torch_chunk_gated_delta_rule: q, k, v cast to fp32 FIRST, then FLA's l2norm
        // x · rsqrt(Σx² + 1e-6) — eps on the SUM. (mlx-lm's rmsNorm · Dk^-½ form puts the eps on the
        // mean, i.e. 128e-6 on the sum: a uniform ~1e-3 relL2 drift from the first GDN layer whose
        // q/k norms are small — measured here before this fix.) q is then scaled by Dk^-½; the
        // rule runs in fp32 and its output is cast back to the activation dtype.
        let dtype = q.dtype
        func l2norm(_ x: MLXArray) -> MLXArray {
            x * rsqrt(sum(x * x, axis: -1, keepDims: true) + 1e-6)
        }
        let qNormed = l2norm(q.asType(.float32)) * pow(Float(headKDim), -0.5)
        let kNormed = l2norm(k.asType(.float32))

        let (out, _) = gatedDeltaUpdate(
            q: qNormed, k: kNormed, v: v.asType(.float32), a: a, b: b, aLog: aLog, dtBias: dtBias)
        return outProj(norm(out.asType(dtype), gate: z).reshaped(B, S, -1))
    }
}

final class Qwen35EncoderAttention: Module {
    let heads, kvHeads, headDim, ropeDims: Int
    let ropeTheta: Float
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen35RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen35RMSNorm

    init(_ args: Qwen35EncoderConfiguration) {
        heads = args.attentionHeads
        kvHeads = args.kvHeads
        headDim = args.headDim
        ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        ropeTheta = args.ropeTheta
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(args.hiddenSize, heads * headDim * 2, bias: false)
        _kProj.wrappedValue = Linear(args.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(args.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, args.hiddenSize, bias: false)
        _qNorm.wrappedValue = Qwen35RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = Qwen35RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let qg = split(qProj(x).reshaped(B, L, heads, headDim * 2), parts: 2, axis: -1)
        let gate = qg[1].reshaped(B, L, -1)
        var q = qNorm(qg[0]).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(q, dimensions: ropeDims, traditional: false, base: ropeTheta, scale: 1, offset: 0)
        k = MLXFast.RoPE(k, dimensions: ropeDims, traditional: false, base: ropeTheta, scale: 1, offset: 0)
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal)
        let merged = o.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(merged * sigmoid(gate))
    }
}

final class Qwen35EncoderMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hidden: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hidden, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hidden, bias: false)
        _downProj.wrappedValue = Linear(hidden, dimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

final class Qwen35EncoderLayer: Module {
    let isFullAttention: Bool
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35EncoderAttention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35EncoderGDN?
    @ModuleInfo(key: "mlp") var mlp: Qwen35EncoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Qwen35RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Qwen35RMSNorm

    init(_ args: Qwen35EncoderConfiguration, layer: Int) {
        isFullAttention = args.isFullAttention(layer)
        if isFullAttention {
            _selfAttn.wrappedValue = Qwen35EncoderAttention(args)
        } else {
            _linearAttn.wrappedValue = Qwen35EncoderGDN(args)
        }
        _mlp.wrappedValue = Qwen35EncoderMLP(dimensions: args.hiddenSize, hidden: args.intermediateSize)
        _inputLayerNorm.wrappedValue = Qwen35RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = Qwen35RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = inputLayerNorm(x)
        let r = isFullAttention ? selfAttn!(normed) : linearAttn!(normed)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// The `language_model` of Qwen3_5Model (vision tower omitted — MOSS-Image never feeds images).
final class Qwen35EncoderLanguageModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen35EncoderLayer]
    @ModuleInfo(key: "norm") var norm: Qwen35RMSNorm

    init(_ args: Qwen35EncoderConfiguration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        _layers.wrappedValue = (0 ..< args.hiddenLayers).map { Qwen35EncoderLayer(args, layer: $0) }
        _norm.wrappedValue = Qwen35RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }
}

public final class Qwen35Encoder: Module {
    public let configuration: Qwen35EncoderConfiguration
    @ModuleInfo(key: "language_model") var languageModel: Qwen35EncoderLanguageModel

    public init(_ args: Qwen35EncoderConfiguration) {
        configuration = args
        _languageModel.wrappedValue = Qwen35EncoderLanguageModel(args)
        super.init()
    }

    /// `last_hidden_state` for [1, S] token ids → [1, S, hidden]. `captureLayers` (1-based, as HF's
    /// `hidden_states[i]` = output of layer i - 1) are returned alongside, for bisecting.
    public func lastHiddenState(_ ids: MLXArray, captureLayers: Set<Int> = []) -> (MLXArray, [Int: MLXArray]) {
        var h = languageModel.embedTokens(ids)
        var captured: [Int: MLXArray] = [:]
        for (i, layer) in languageModel.layers.enumerated() {
            h = layer(h)
            if captureLayers.contains(i + 1) { captured[i + 1] = h }
            eval(h)  // per-layer checkpoint: bounds the prefill high-water on long (≤ 4096-token) prompts
        }
        return (languageModel.norm(h), captured)
    }

    /// HF checkpoint → module keys: drop the vision tower (and any MTP head), move the depthwise
    /// conv to MLX layout, shift the zero-centred RMSNorm weights by one and keep them in fp32.
    public static func sanitize(_ weights: [String: MLXArray], dtype: DType) -> [String: MLXArray] {
        let zeroCentred = [".input_layernorm.weight", ".post_attention_layernorm.weight",
                           "language_model.norm.weight", ".q_norm.weight", ".k_norm.weight"]
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            guard key.hasPrefix("language_model."), !key.contains("mtp.") else { continue }
            var v = value
            if key.hasSuffix("conv1d.weight") && v.dim(-1) != 1 {
                v = v.movedAxis(source: 2, destination: 1)  // [C, 1, K] → [C, K, 1]
            }
            if zeroCentred.contains(where: { key.hasSuffix($0) }) {
                out[key] = v.asType(.float32) + 1
            } else if key.hasSuffix(".A_log") || key.hasSuffix(".dt_bias") {
                out[key] = v.asType(.float32)
            } else {
                out[key] = v.asType(dtype)
            }
        }
        return out
    }
}
