// MLXEngine `textToImage` packages over the MossImage core — MOSS-Image-SFT (the default tier) and
// MOSS-Image-Base. OpenMOSS, Apache-2.0 weights; shipping approved by the operator on the tech
// report's data account (AB-D-0150: enumerated, ~98% web-scrape corpus — not posture A).
//
// Core (PORTING-SPEC.md): 5B single-stream DiT (42 blocks, one shared adaLN modulation, SiTU-GLU,
// 4-axis RoPE) over a frozen Qwen3.5-9B encoder (last hidden state of the chat-templated prompt,
// thinking enabled) and the FLUX.2 VAE with BFL's small decoder. Sampling: 50 Euler steps, shift 5
// snapped to the training grid, CFG 3 whose negative branch skips block 22 (two forwards per step).
// Prompts may be prose, tag lists, or a JSON layout (summary.caption + items with yxyx boxes in
// centred absolute pixels) — the JSON is just the prompt text.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXToolKit
import MossImage
import UniformTypeIdentifiers

public enum MossImagePackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case pngEncode
    case sizeOutOfEnvelope(width: Int, height: Int)
    case unsupportedQuant(Quant)

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "MOSS-Image snapshot not readable at \(p)."
        case .pngEncode: return "PNG encoding failed."
        case .unsupportedQuant(let q):
            return "MOSS-Image has no \(q) tier (bf16 and int8 are declared; int4 fails the quality gate)."
        case .sizeOutOfEnvelope(let w, let h):
            return "MOSS-Image output \(w)×\(h) is outside the supported envelope "
                + "(sides multiples of 16, each ≥ 256 px, area ≤ 1456×1456)."
        }
    }
}

/// The input envelope the declared footprints were measured at. `run()` refuses anything outside
/// it so the declaration stays an upper bound (AB-L-0202).
public enum MossImageEnvelope {
    /// Output area cap: 1456² — upstream's own large example, inside the ~2×1024² area the final
    /// training stage reached (tech report Table 1).
    public static let maxOutputPixels = 1456 * 1456
    public static let minSide = 256

    /// Request size → generated size (default 1024², floored to multiples of 16), or a refusal.
    public static func outputSize(_ width: Int?, _ height: Int?) throws -> (Int, Int) {
        let w = ((width ?? 1024) / 16) * 16
        let h = ((height ?? 1024) / 16) * 16
        guard w >= minSide, h >= minSide, w * h <= maxOutputPixels else {
            throw MossImagePackageError.sizeOutOfEnvelope(width: w, height: h)
        }
        return (w, h)
    }
}

/// Resident DiT + VAE; the 16.8 GB encoder loads per request (released before the denoise loop)
/// unless `keepEncoderResident`.
@InferenceActor
final class MossImageGenerator {
    let transformer: MossImageTransformer
    let vae: MossImageVAE
    let sharedSnapshot: URL
    let keepEncoderResident: Bool
    private var residentEncoder: MossImagePromptEncoder?

    init(transformer: MossImageTransformer, vae: MossImageVAE, sharedSnapshot: URL, keepEncoderResident: Bool) {
        self.transformer = transformer
        self.vae = vae
        self.sharedSnapshot = sharedSnapshot
        self.keepEncoderResident = keepEncoderResident
    }

    static func load(configuration c: MossImageConfiguration, snapshot: URL, shared: URL) throws -> MossImageGenerator {
        let transformer = try MossImageWeights.loadTransformer(
            directory: snapshot.appendingPathComponent("transformer"), dtype: .bfloat16)
        switch c.quant {
        case .bf16: break
        case .int8: MossImageWeights.quantizeTransformer(transformer, bits: 8)
        default: throw MossImagePackageError.unsupportedQuant(c.quant)
        }
        let vae = try MossImageWeights.loadVAE(directory: shared.appendingPathComponent("vae"), dtype: .float32)
        MLX.Memory.clearCache()
        return MossImageGenerator(transformer: transformer, vae: vae, sharedSnapshot: shared,
                                  keepEncoderResident: c.keepEncoderResident)
    }

    /// Both branches' conditioning. A per-request encoder is created and dropped inside this scope,
    /// so nothing outlives it into the denoise loop (a caller-scope binding kept all 16.8 GB
    /// resident through denoising — measured +6 GB on the 1024² peak before this was scoped).
    private func condition(prompt: String, parameters p: MossImageGenerationParameters) async throws -> MossImageConditioning {
        if residentEncoder == nil && keepEncoderResident {
            residentEncoder = try await MossImagePromptEncoder.load(snapshot: sharedSnapshot, dtype: .bfloat16)
        }
        var encoder = residentEncoder
        if encoder == nil { encoder = try await MossImagePromptEncoder.load(snapshot: sharedSnapshot, dtype: .bfloat16) }
        return try MossImagePipeline(transformer: transformer, promptEncoder: encoder!, vae: vae)
            .encode(prompt: prompt, parameters: p)
    }

    func generate(prompt: String, parameters p: MossImageGenerationParameters) async throws -> (pixels: [UInt8], width: Int, height: Int) {
        let conditioning = try await condition(prompt: prompt, parameters: p)
        MLX.Memory.clearCache()
        try Task.checkCancellation()  // post-encode seam
        let pipe = MossImagePipeline(transformer: transformer, promptEncoder: nil, vae: vae)
        let latents = try pipe.denoise(conditioning: conditioning, parameters: p) { step, total, _ in
            RunProgress.report(.denoise, step: step + 1, totalSteps: total)
        }
        try Task.checkCancellation()  // pre-decode seam
        let image = pipe.decode(latents)
        let rgb = MossImagePipeline.toRGB8(image)
        MLX.Memory.clearCache()
        return rgb
    }
}

/// The half both packages share: load, and request → PNG.
@InferenceActor
enum MossImageServing {
    static func load(_ c: MossImageConfiguration) throws -> MossImageGenerator {
        guard let snapshot = c.resolvedSnapshotDirectory(storeRoot: c.modelsRootDirectory),
              FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("transformer").path)
        else {
            throw MossImagePackageError.unreadableSnapshot(c.snapshotPath.isEmpty ? c.variant.repo : c.snapshotPath)
        }
        guard let shared = c.resolvedSharedDirectory(storeRoot: c.modelsRootDirectory),
              FileManager.default.fileExists(atPath: shared.appendingPathComponent("text_encoder").path),
              FileManager.default.fileExists(atPath: shared.appendingPathComponent("vae").path)
        else {
            throw MossImagePackageError.unreadableSnapshot(
                c.sharedSnapshotPath.isEmpty ? MossImageConfiguration.sharedRepo : c.sharedSnapshotPath)
        }
        return try MossImageGenerator.load(configuration: c, snapshot: snapshot, shared: shared)
    }

    static func run(_ request: any CapabilityRequest, generator: MossImageGenerator,
                    configuration c: MossImageConfiguration) async throws -> any CapabilityResponse {
        guard let t2i = request as? T2IRequest else { throw PackageError.unsupportedCapability(request.capability) }
        let (width, height) = try MossImageEnvelope.outputSize(t2i.width, t2i.height)
        var p = MossImageGenerationParameters()
        p.width = width
        p.height = height
        p.steps = t2i.steps ?? c.defaultSteps
        p.guidanceScale = Float(t2i.guidanceScale ?? c.defaultGuidanceScale)
        p.negativePrompt = t2i.negativePrompt ?? ""
        p.seed = t2i.seed ?? 0
        let rgb = try await generator.generate(prompt: t2i.prompt, parameters: p)
        try Task.checkCancellation()
        let png = try encodePNG(rgb)
        return T2IResponse(image: Image(format: .png, data: png, width: rgb.width, height: rgb.height))
    }

    /// Interleaved RGB8 → PNG (canonical serialized artifact form, C3).
    nonisolated static func encodePNG(_ rgb: (pixels: [UInt8], width: Int, height: Int)) throws -> Data {
        let (pixels, width, height) = rgb
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw MossImagePackageError.pngEncode }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0 ..< (width * height) {
            buf[i * 4] = pixels[i * 3]
            buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage() else { throw MossImagePackageError.pngEncode }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)
        else { throw MossImagePackageError.pngEncode }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw MossImagePackageError.pngEncode }
        return out as Data
    }

    /// Manifest shared by both variants; only the name, summary and provenance differ.
    nonisolated static func manifest(variant: MossImageVariant, name: String, summary: String) -> PackageManifest {
        PackageManifest(
            // C7: DiT, Qwen3.5-9B encoder and FLUX.2 small decoder are all Apache-2.0. C8: port MIT.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: variant.repo, revision: variant.revision, tier: 1),
            requirements: RequirementsManifest(
                footprints: MossImageFootprints.declared,
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil  // memory admission gates
            ),
            specialties: [],
            surfaces: [T2IContract.descriptor(name: name, summary: summary, modes: [])]
        )
    }
}

@InferenceActor
public final class MossImageSFTPackage: ModelPackage {
    public typealias Configuration = MossImageConfiguration

    public nonisolated static var manifest: PackageManifest {
        MossImageServing.manifest(
            variant: .sft, name: "moss-image-sft",
            summary: "MOSS-Image-SFT text-to-image (OpenMOSS, Apache-2.0): 5B single-stream DiT on a "
                + "Qwen3.5-9B encoder. Strongest open-weight long-text rendering in English and Chinese "
                + "(line breaks in the prompt are kept), and JSON layout prompts that place text and "
                + "objects by bounding box (summary.caption + items, yxyx centred pixels, same canvas "
                + "size as the request). 50 steps, CFG 3, 256²–1456² by area, sides multiples of 16.")
    }

    let configuration: Configuration
    private var generator: MossImageGenerator?

    public nonisolated init(configuration: Configuration) {
        var c = configuration
        c.variant = .sft
        self.configuration = c
    }

    public func load() async throws {
        guard generator == nil else { return }
        generator = try MossImageServing.load(configuration)
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()  // CAN-1: first act of run()
        guard let generator else { throw PackageError.notLoaded }
        return try await MossImageServing.run(request, generator: generator, configuration: configuration)
    }
}

@InferenceActor
public final class MossImageBasePackage: ModelPackage {
    public typealias Configuration = MossImageConfiguration

    public nonisolated static var manifest: PackageManifest {
        MossImageServing.manifest(
            variant: .base, name: "moss-image-base",
            summary: "MOSS-Image-Base text-to-image (OpenMOSS, Apache-2.0): the pretrained checkpoint "
                + "MOSS-Image-SFT is fine-tuned from — same architecture, prompts and layout JSON, less "
                + "curated composition; the starting point for LoRA training. 50 steps, CFG 3, "
                + "256²–1456² by area.")
    }

    let configuration: Configuration
    private var generator: MossImageGenerator?

    public nonisolated init(configuration: Configuration) {
        var c = configuration
        c.variant = .base
        self.configuration = c
    }

    public func load() async throws {
        guard generator == nil else { return }
        generator = try MossImageServing.load(configuration)
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()
        guard let generator else { throw PackageError.notLoaded }
        return try await MossImageServing.run(request, generator: generator, configuration: configuration)
    }
}

extension MossImageSFTPackage {
    public nonisolated static var registration: PackageRegistration { .of(MossImageSFTPackage.self) }
}

extension MossImageBasePackage {
    public nonisolated static var registration: PackageRegistration { .of(MossImageBasePackage.self) }
}
