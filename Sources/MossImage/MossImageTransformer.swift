// MOSSImageTransformer — 1:1 with the upstream pipeline_moss_image.py (same class decomposition,
// constants and dtype points). Single-stream DiT: the image tokens (row-major over the /16 latent
// grid) followed by the refined text tokens run through 42 identical blocks that all read ONE
// shared adaLN modulation; the head keeps only the image tokens.
//
// Dtype points of the reference (bf16 weights): the timestep sinusoid is float64 → fp32 → the MLP in
// the weight dtype; modulation and head modulation are produced in the weight dtype; RoPE phases are
// fp32 and the sin/cos table is cast to the activation dtype; every RMSNorm normalises in fp32 and
// casts back BEFORE the weight multiply.

import Foundation
import MLX
import MLXFast
import MLXNN

public struct MossImageTransformerConfiguration: Codable, Sendable {
    public var dim = 3072
    public var ffnDim = 8192
    public var freqDim = 256
    public var latentChannels = 128
    public var numHeads = 24
    public var numLayers = 42
    public var ropeBase: Float = 10000
    public var textDim = 4096

    enum CodingKeys: String, CodingKey {
        case dim, ffnDim = "ffn_dim", freqDim = "freq_dim", latentChannels = "latent_channels"
        case numHeads = "num_heads", numLayers = "num_layers", ropeBase = "rope_base", textDim = "text_dim"
    }

    public init() {}

    public var headDim: Int { dim / numHeads }

    public static func load(transformerDirectory: URL) throws -> MossImageTransformerConfiguration {
        try JSONDecoder().decode(
            Self.self, from: Data(contentsOf: transformerDirectory.appendingPathComponent("config.json")))
    }
}

/// `_RMSNorm`: y = x · rsqrt(mean(x²) + 1e-6) in fp32, cast to x.dtype, THEN · weight.
final class MossRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(_ dim: Int) {
        _weight.wrappedValue = MLXArray.ones([dim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: MLXArray.mlxNone, eps: 1e-6).asType(x.dtype) * weight
    }
}

/// `_situ`: (4·tanh(gate/4)·sigmoid(gate)) · (25·tanh(hidden/25)).
@inline(__always)
func situ(_ hidden: MLXArray, _ gate: MLXArray) -> MLXArray {
    (4.0 * tanh(gate / 4.0) * sigmoid(gate)) * (25.0 * tanh(hidden / 25.0))
}

/// `_rotate`: rope = [sin | cos] (each head_dim/2); rotate-half with the halves of x.
func mossRotate(_ x: MLXArray, _ rope: MLXArray) -> MLXArray {
    // x: [1, N, H, D]; rope: [1, N, D] → broadcast over heads.
    let half = rope.dim(-1) / 2
    let sinHalf = rope[.ellipsis, ..<half]
    let cosHalf = rope[.ellipsis, half...]
    let sin = concatenated([sinHalf, sinHalf], axis: -1).expandedDimensions(axis: -2)
    let cos = concatenated([cosHalf, cosHalf], axis: -1).expandedDimensions(axis: -2)
    let d = x.dim(-1) / 2
    let a = x[.ellipsis, ..<d]
    let b = x[.ellipsis, d...]
    return x * cos + concatenated([-b, a], axis: -1) * sin
}

/// `_attention`: q/k rotated per head, SDPA (no mask — one sample at its true length).
func mossAttention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, rope: MLXArray, heads: Int) -> MLXArray {
    let (B, N, D) = (q.dim(0), q.dim(1), q.dim(2))
    let hd = D / heads
    let qr = mossRotate(q.reshaped(B, N, heads, hd), rope).transposed(0, 2, 1, 3)
    let kr = mossRotate(k.reshaped(B, N, heads, hd), rope).transposed(0, 2, 1, 3)
    let vr = v.reshaped(B, N, heads, hd).transposed(0, 2, 1, 3)
    let o = MLXFast.scaledDotProductAttention(
        queries: qr.asType(v.dtype), keys: kr.asType(v.dtype), values: vr,
        scale: 1 / Float(hd).squareRoot(), mask: .none)
    return o.transposed(0, 2, 1, 3).reshaped(B, N, D)
}

final class MossTextBlock: Module {
    let heads: Int
    @ModuleInfo(key: "norm1") var norm1: MossRMSNorm
    @ModuleInfo(key: "norm2") var norm2: MossRMSNorm
    @ModuleInfo(key: "q_norm") var qNorm: MossRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: MossRMSNorm
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "ffn_gate") var ffnGate: Linear
    @ModuleInfo(key: "ffn_up") var ffnUp: Linear
    @ModuleInfo(key: "ffn_down") var ffnDown: Linear

    init(dim: Int, ffnDim: Int, heads: Int) {
        self.heads = heads
        _norm1.wrappedValue = MossRMSNorm(dim)
        _norm2.wrappedValue = MossRMSNorm(dim)
        _qNorm.wrappedValue = MossRMSNorm(dim / heads)
        _kNorm.wrappedValue = MossRMSNorm(dim / heads)
        _qkv.wrappedValue = Linear(dim, 3 * dim)
        _proj.wrappedValue = Linear(dim, dim)
        _ffnGate.wrappedValue = Linear(dim, ffnDim)
        _ffnUp.wrappedValue = Linear(dim, ffnDim)
        _ffnDown.wrappedValue = Linear(ffnDim, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, rope: MLXArray) -> MLXArray {
        let (B, N, D) = (x.dim(0), x.dim(1), x.dim(2))
        let parts = split(qkv(norm1(x)), parts: 3, axis: -1)
        let q = qNorm(parts[0].reshaped(B, N, heads, -1)).reshaped(B, N, D)
        let k = kNorm(parts[1].reshaped(B, N, heads, -1)).reshaped(B, N, D)
        var h = x + proj(mossAttention(q, k, parts[2], rope: rope, heads: heads))
        let y = norm2(h)
        h = h + ffnDown(situ(ffnUp(y), ffnGate(y)))
        return h
    }
}

final class MossTextProjection: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "blocks") var blocks: [MossTextBlock]

    init(textDim: Int, dim: Int, ffnDim: Int, heads: Int) {
        _proj.wrappedValue = Linear(textDim, dim)
        _blocks.wrappedValue = (0 ..< 2).map { _ in MossTextBlock(dim: dim, ffnDim: ffnDim, heads: heads) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, rope: MLXArray) -> MLXArray {
        var h = proj(x.asType(proj.weight.dtype))
        for block in blocks { h = block(h, rope: rope) }
        return h
    }
}

/// `_Block.proj` (an nn.ModuleDict): the bias-free projections.
final class MossBlockProjections: Module {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "ffn_up_gate") var ffnUpGate: Linear
    @ModuleInfo(key: "ffn_down") var ffnDown: Linear

    init(dim: Int, ffnDim: Int) {
        _q.wrappedValue = Linear(dim, dim, bias: false)
        _k.wrappedValue = Linear(dim, dim, bias: false)
        _v.wrappedValue = Linear(dim, dim, bias: false)
        _o.wrappedValue = Linear(dim, dim, bias: false)
        _ffnUpGate.wrappedValue = Linear(dim, 2 * ffnDim, bias: false)
        _ffnDown.wrappedValue = Linear(ffnDim, dim, bias: false)
        super.init()
    }
}

final class MossBlock: Module {
    let heads: Int
    @ModuleInfo(key: "norm1") var norm1: MossRMSNorm
    @ModuleInfo(key: "norm2") var norm2: MossRMSNorm
    @ModuleInfo(key: "q_norm") var qNorm: MossRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: MossRMSNorm
    @ModuleInfo(key: "proj") var proj: MossBlockProjections

    init(dim: Int, ffnDim: Int, heads: Int) {
        self.heads = heads
        _norm1.wrappedValue = MossRMSNorm(dim)
        _norm2.wrappedValue = MossRMSNorm(dim)
        _qNorm.wrappedValue = MossRMSNorm(dim / heads)
        _kNorm.wrappedValue = MossRMSNorm(dim / heads)
        _proj.wrappedValue = MossBlockProjections(dim: dim, ffnDim: ffnDim)
        super.init()
    }

    /// `modulation`: [1, 6, dim] = shift_a, scale_a, gate_a, shift_f, scale_f, gate_f.
    func callAsFunction(_ x: MLXArray, rope: MLXArray, modulation m: MLXArray) -> MLXArray {
        let (B, N, D) = (x.dim(0), x.dim(1), x.dim(2))
        let shiftA = m[0..., 0 ..< 1, 0...], scaleA = m[0..., 1 ..< 2, 0...], gateA = m[0..., 2 ..< 3, 0...]
        let shiftF = m[0..., 3 ..< 4, 0...], scaleF = m[0..., 4 ..< 5, 0...], gateF = m[0..., 5 ..< 6, 0...]
        // The reference reads proj['q'].weight.dtype; a quantized tier packs that weight as uint32,
        // so take the (never quantized) norm weight's dtype — the same model dtype.
        let dtype = norm1.weight.dtype
        let y = (norm1(x) * (1 + scaleA) + shiftA).asType(dtype)
        let q = qNorm(proj.q(y).reshaped(B, N, heads, -1)).reshaped(B, N, D)
        let k = kNorm(proj.k(y).reshaped(B, N, heads, -1)).reshaped(B, N, D)
        var h = x + proj.o(mossAttention(q, k, proj.v(y), rope: rope, heads: heads)) * gateA
        let y2 = norm2(h) * (1 + scaleF) + shiftF
        let hg = split(proj.ffnUpGate(y2.asType(dtype)), parts: 2, axis: -1)
        h = h + proj.ffnDown(situ(hg[0], hg[1])) * gateF
        return h
    }
}

/// `time_embedding` (nn.Sequential(Linear, SiLU, Linear)). The checkpoint's "0" / "2" keys read as
/// array indices to `unflattened`, so the loader renames them to "linear_1" / "linear_2".
final class MossTimeEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(freqDim: Int, dim: Int) {
        _linear1.wrappedValue = Linear(freqDim, dim)
        _linear2.wrappedValue = Linear(dim, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

public final class MossImageTransformer: Module {
    public let config: MossImageTransformerConfiguration
    /// `_RoPE.bands`: fp32 constants, rebuilt (never loaded) — held outside Module reflection.
    let bands: [Float]

    @ModuleInfo(key: "video_embedder") var videoEmbedder: Linear
    @ModuleInfo(key: "text_embedder") var textEmbedder: MossTextProjection
    @ModuleInfo(key: "time_embedding") var timeEmbedding: MossTimeEmbedding
    @ModuleInfo(key: "shared_block_modulation") var sharedBlockModulation: Linear
    @ModuleInfo(key: "layers") var layers: [MossBlock]
    @ModuleInfo(key: "final_norm_video") var finalNormVideo: MossRMSNorm
    @ModuleInfo(key: "final_linear_video") var finalLinearVideo: Linear
    @ModuleInfo(key: "video_head_modulation") var videoHeadModulation: Linear

    public init(_ c: MossImageTransformerConfiguration) {
        precondition(c.dim % c.numHeads == 0 && c.headDim % 8 == 0 && c.freqDim % 2 == 0)
        config = c
        // `_default_bands`: 1 / base ** (arange(n) / n), fp32 on CPU.
        let n = c.headDim / 8
        bands = (0 ..< n).map { i in Float(1) / powf(c.ropeBase, Float(i) / Float(n)) }
        _videoEmbedder.wrappedValue = Linear(c.latentChannels, c.dim)
        _textEmbedder.wrappedValue = MossTextProjection(
            textDim: c.textDim, dim: c.dim, ffnDim: c.ffnDim, heads: c.numHeads)
        _timeEmbedding.wrappedValue = MossTimeEmbedding(freqDim: c.freqDim, dim: c.dim)
        _sharedBlockModulation.wrappedValue = Linear(c.dim, 6 * c.dim)
        _layers.wrappedValue = (0 ..< c.numLayers).map { _ in
            MossBlock(dim: c.dim, ffnDim: c.ffnDim, heads: c.numHeads)
        }
        _finalNormVideo.wrappedValue = MossRMSNorm(c.dim)
        _finalLinearVideo.wrappedValue = Linear(c.dim, c.latentChannels, bias: false)
        _videoHeadModulation.wrappedValue = Linear(c.dim, 2 * c.dim)
        super.init()
    }

    public var dtype: DType { videoEmbedder.weight.dtype }

    /// `_RoPE.forward`: coordinates (r, f, h, w) per token, image first (r = 0, f = text_length,
    /// h/w centred), then text (f = index). Phase = coord · band in fp32; table = [sin | cos]
    /// laid out (axis, band) → [N, head_dim].
    public func rope(height: Int, width: Int, textLength: Int) -> MLXArray {
        let nb = bands.count
        let n = height * width + textLength
        var table = [Float](repeating: 0, count: n * 8 * nb)
        let ch = Float(height - 1) / 2, cw = Float(width - 1) / 2
        func fill(_ token: Int, _ coords: (Float, Float, Float, Float)) {
            let axes = [coords.0, coords.1, coords.2, coords.3]
            let base = token * 8 * nb
            for (a, coord) in axes.enumerated() {
                for j in 0 ..< nb {
                    let phase = coord * bands[j]  // fp32 product, as torch
                    table[base + a * nb + j] = Float(Foundation.sin(Double(phase)))
                    table[base + (4 + a) * nb + j] = Float(Foundation.cos(Double(phase)))
                }
            }
        }
        for i in 0 ..< height {
            for j in 0 ..< width {
                fill(i * width + j, (0, Float(textLength), Float(i) - ch, Float(j) - cw))
            }
        }
        for t in 0 ..< textLength {
            fill(height * width + t, (0, Float(t), 0, 0))
        }
        return MLXArray(table, [1, n, 8 * nb])
    }

    /// Timestep sinusoid (float64, as training) → fp32 → time MLP in the weight dtype.
    func timeVector(_ timestep: Double) -> MLXArray {
        let half = config.freqDim / 2
        var emb = [Float](repeating: 0, count: 2 * half)
        for i in 0 ..< half {
            let freq = Foundation.pow(10000.0, -Double(i) / Double(half))
            let phase = timestep * freq
            emb[i] = Float(Foundation.cos(phase))
            emb[half + i] = Float(Foundation.sin(phase))
        }
        let e = MLXArray(emb, [1, 2 * half]).asType(dtype)
        return timeEmbedding(e).asType(.float32)
    }

    /// Shared block modulation [1, 6, dim] and head modulation [1, 2, dim] for one timestep.
    public func modulation(timestep: Double) -> (blocks: MLXArray, head: MLXArray) {
        let t = timeVector(timestep).asType(dtype)
        return (sharedBlockModulation(t).reshaped(1, 6, -1), videoHeadModulation(t).reshaped(1, 2, -1))
    }

    /// One forward. `latents` [1, latent_channels, H, W]; `context` [1, S, text_dim] (unpadded);
    /// `skipLayers`: zero-based blocks to skip (the negative branch's [22]).
    public func callAsFunction(
        _ latents: MLXArray, timestep: Double, context: MLXArray, skipLayers: Set<Int> = [],
        capture: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        precondition(latents.ndim == 4 && latents.dim(1) == config.latentChannels)
        precondition(latents.dim(0) == 1 && context.dim(0) == 1, "batch 1 (per-sample, as the reference loop)")
        let (height, width) = (latents.dim(2), latents.dim(3))
        let hw = height * width
        let (mod, head) = modulation(timestep: timestep)
        var video = latents.reshaped(1, config.latentChannels, hw).transposed(0, 2, 1)
        video = videoEmbedder(video.asType(dtype))
        let rope = self.rope(height: height, width: width, textLength: context.dim(1)).asType(video.dtype)
        let text = textEmbedder(context, rope: rope[0..., hw..., 0...])
        capture?("text_refined", text)
        var x = concatenated([video, text], axis: 1)
        for (i, layer) in layers.enumerated() where !skipLayers.contains(i) {
            x = layer(x, rope: rope, modulation: mod)
            capture?("block\(i)", x)
            eval(x)
        }
        var y = finalNormVideo(x[0..., ..<hw, 0...].asType(finalNormVideo.weight.dtype))
        y = y * (1 + head[0..., 1 ..< 2, 0...]) + head[0..., 0 ..< 1, 0...]
        y = finalLinearVideo(y.asType(finalLinearVideo.weight.dtype))
        return y.transposed(0, 2, 1).reshaped(1, config.latentChannels, height, width)
    }
}
