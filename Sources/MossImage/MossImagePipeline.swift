// MOSSImagePipeline.__call__ for batch 1: CFG with the negative branch run with blocks skipped
// (default [22] — "uncond_skip_layers"; two forwards per step), fp32 CFG combine and Euler update,
// fp32 initial noise, FLUX.2 small-decoder decode in fp32 (bn de-norm in packed space, unpatchify,
// decode, clamp). Optional positive-branch skip-layer guidance (slg) as the reference.
//
// Memory: the text encoder (16.8 GB bf16) is only needed before the loop. `encodeAndRelease`
// conditions both branches and drops it, so the denoise peak is the DiT + activations.

import Foundation
import Flux2VAE
import MLX
import MLXRandom

/// The FLUX.2 VAE (small decoder), re-exported so wrappers need not import Flux2VAE.
public typealias MossImageVAE = Flux2VAE

public struct MossImageGenerationParameters: Sendable {
    public var width = 1024
    public var height = 1024
    public var steps = 50
    public var guidanceScale: Float = 3.0
    public var negativePrompt = ""
    public var uncondSkipLayers: Set<Int> = [22]
    public var slgLayers: Set<Int> = []
    public var slgScale: Float = 0
    public var slgStart: Float = 0
    public var slgStop: Float = 1
    public var shift: Float = 5.0
    public var seed: UInt64 = 0

    public init() {}

    func validate(depth: Int) throws {
        guard width > 0, height > 0, width % 16 == 0, height % 16 == 0 else {
            throw MossImageError.invalidInput("width and height must be positive multiples of 16")
        }
        guard steps >= 1 else { throw MossImageError.invalidInput("steps must be ≥ 1") }
        guard guidanceScale.isFinite, guidanceScale >= 0 else {
            throw MossImageError.invalidInput("guidance scale must be finite and ≥ 0")
        }
        guard slgScale.isFinite, slgScale >= 0, 0 <= slgStart, slgStart <= slgStop, slgStop <= 1 else {
            throw MossImageError.invalidInput("require slg_scale ≥ 0 and 0 ≤ slg_start ≤ slg_stop ≤ 1")
        }
        for i in uncondSkipLayers.union(slgLayers) where !(0 ..< depth).contains(i) {
            throw MossImageError.invalidInput("skip layer \(i) outside [0, \(depth))")
        }
    }
}

public struct MossImageConditioning {
    public let positive: MLXArray
    public let negative: MLXArray?

    public init(positive: MLXArray, negative: MLXArray?) {
        self.positive = positive
        self.negative = negative
    }
}

public final class MossImagePipeline {
    public let transformer: MossImageTransformer
    public private(set) var promptEncoder: MossImagePromptEncoder?
    public let vae: Flux2VAE

    public init(transformer: MossImageTransformer, promptEncoder: MossImagePromptEncoder?, vae: Flux2VAE) {
        self.transformer = transformer
        self.promptEncoder = promptEncoder
        self.vae = vae
    }

    /// Load from snapshot roots. Base and SFT share text_encoder/tokenizer/vae byte-for-byte; only
    /// `transformerSnapshot` differs, so a Base install may point `sharedSnapshot` at the SFT root.
    public static func load(transformerSnapshot: URL, sharedSnapshot: URL? = nil,
                            dtype: DType = .bfloat16,
                            textEncoderQuant: MossImageTextEncoderQuant = .bf16) async throws -> MossImagePipeline {
        let shared = sharedSnapshot ?? transformerSnapshot
        let transformer = try MossImageWeights.loadTransformer(
            directory: transformerSnapshot.appendingPathComponent("transformer"), dtype: dtype)
        let encoder = try await MossImagePromptEncoder.load(
            snapshot: shared, dtype: dtype, textEncoderQuant: textEncoderQuant)
        let vae = try MossImageWeights.loadVAE(directory: shared.appendingPathComponent("vae"), dtype: .float32)
        return MossImagePipeline(transformer: transformer, promptEncoder: encoder, vae: vae)
    }

    public func encode(prompt: String, parameters p: MossImageGenerationParameters) throws -> MossImageConditioning {
        guard let encoder = promptEncoder else {
            throw MossImageError.invalidInput("text encoder released; pass conditioning")
        }
        let positive = encoder.encode(prompt)
        let negative = p.guidanceScale != 1 ? encoder.encode(p.negativePrompt) : nil
        return MossImageConditioning(positive: positive, negative: negative)
    }

    /// Encode, then drop the text encoder (and its cache) before denoising.
    public func encodeAndRelease(prompt: String, parameters p: MossImageGenerationParameters) throws -> MossImageConditioning {
        let c = try encode(prompt: prompt, parameters: p)
        promptEncoder = nil
        MLX.GPU.clearCache()
        return c
    }

    /// Initial noise [1, 128, H/16, W/16], fp32.
    public func noise(parameters p: MossImageGenerationParameters) -> MLXArray {
        MLXRandom.normal([1, transformer.config.latentChannels, p.height / 16, p.width / 16],
                         dtype: .float32, key: MLXRandom.key(p.seed))
    }

    /// The denoise loop → final packed latents [1, 128, H/16, W/16] (fp32).
    public func denoise(conditioning c: MossImageConditioning, parameters p: MossImageGenerationParameters,
                        latents initial: MLXArray? = nil,
                        onStep: ((Int, Int, MLXArray) -> Void)? = nil) throws -> MLXArray {
        try p.validate(depth: transformer.layers.count)
        let shape = [1, transformer.config.latentChannels, p.height / 16, p.width / 16]
        var latents = (initial ?? noise(parameters: p)).asType(.float32)
        guard latents.shape == shape else { throw MossImageError.invalidInput("latents must be \(shape)") }
        let doCFG = p.guidanceScale != 1
        guard !doCFG || c.negative != nil else { throw MossImageError.invalidInput("CFG needs a negative branch") }
        let scheduler = MossImageScheduler(shift: p.shift)
        let sigmas = scheduler.sigmas(steps: p.steps)
        let timesteps = scheduler.timesteps(sigmas: sigmas)
        let n = timesteps.count
        for (i, t) in timesteps.enumerated() {
            try Task.checkCancellation()  // CAN: one checkpoint per denoise step
            let skipActive = p.slgStart <= Float(i) / Float(n) && Float(i) / Float(n) < p.slgStop
            let ts = Double(t)
            let conditional = transformer(latents, timestep: ts, context: c.positive).asType(.float32)
            var prediction = conditional
            if doCFG, let negative = c.negative {
                let skip = skipActive ? p.uncondSkipLayers : []
                let uncond = transformer(latents, timestep: ts, context: negative, skipLayers: skip)
                    .asType(.float32)
                prediction = uncond + p.guidanceScale * (conditional - uncond)
            }
            if !p.slgLayers.isEmpty, p.slgScale > 0, skipActive {
                let skipped = transformer(latents, timestep: ts, context: c.positive, skipLayers: p.slgLayers)
                    .asType(.float32)
                if !doCFG { prediction = skipped }
                prediction = prediction + p.slgScale * (conditional - skipped)
            }
            // FlowMatchEulerDiscreteScheduler.step: x + (σ_next − σ) · v, fp32.
            latents = latents + (sigmas[i + 1] - sigmas[i]) * prediction
            eval(latents)
            onStep?(i, n, latents)
            MLX.GPU.clearCache()
        }
        return latents
    }

    /// Packed latents → image [1, 3, H, W] in [-1, 1] (fp32).
    public func decode(_ latents: MLXArray) -> MLXArray {
        let image = clip(vae.decodePackedLatents(latents.asType(.float32)), min: -1, max: 1)
        eval(image)
        return image
    }

    /// VaeImageProcessor.postprocess(pil): (x / 2 + 0.5).clamp(0, 1) · 255, rounded → HWC uint8.
    public static func toRGB8(_ image: MLXArray) -> (pixels: [UInt8], width: Int, height: Int) {
        let x = clip(image[0] / 2 + 0.5, min: 0, max: 1)
        let hwc = (x.transposed(1, 2, 0) * 255).round().asType(.uint8)
        eval(hwc)
        return (hwc.asArray(UInt8.self), hwc.dim(1), hwc.dim(0))
    }

    public func generate(prompt: String, parameters p: MossImageGenerationParameters,
                         releaseTextEncoder: Bool = false,
                         onStep: ((Int, Int, MLXArray) -> Void)? = nil) throws -> MLXArray {
        let c = releaseTextEncoder
            ? try encodeAndRelease(prompt: prompt, parameters: p) : try encode(prompt: prompt, parameters: p)
        return decode(try denoise(conditioning: c, parameters: p, onStep: onStep))
    }
}
