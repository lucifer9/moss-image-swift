// Offline manifest / configuration / materialization tests for the MOSS-Image engine packages.
import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXMossImage
import MossImage

final class MossImagePackageTests: XCTestCase {
    static let weights = "/Volumes/Satechi/Development/mlxengine-image/weights"

    func testManifestsAreApacheAndT2IOnly() {
        for m in [MossImageSFTPackage.manifest, MossImageBasePackage.manifest] {
            XCTAssertEqual(m.license.weightLicense, .apache2)
            XCTAssertTrue(SPDXLicense.permissiveAllowlist.contains(m.license.weightLicense))
            XCTAssertEqual(m.license.portCodeLicense, .mit)
            XCTAssertEqual(m.contractVersion, ContractVersion.current)
            XCTAssertEqual(m.surfaces.map(\.capability), [.textToImage])
            XCTAssertEqual(Set(m.requirements.footprints.map(\.quant)), [.bf16, .int8])
            XCTAssertTrue(m.requirements.footprints.allSatisfy { $0.residentBytes > 0 && $0.peakActivationBytes > 0 })
        }
        XCTAssertEqual(MossImageSFTPackage.manifest.surfaces.map(\.name), ["moss-image-sft"])
        XCTAssertEqual(MossImageBasePackage.manifest.surfaces.map(\.name), ["moss-image-base"])
        XCTAssertEqual(MossImageBasePackage.manifest.provenance.sourceRepo, "mlx-community/MOSS-Image-Base-bf16")
        XCTAssertEqual(MossImageVariant.base.upstreamRepo, "OpenMOSS-Team/MOSS-Image-Base")
    }

    /// The declared split: bf16 10.5 / int8 5.8 GB resident, 17.5 GB activation (the in-app 1456² decode, AB-R-0575).
    func testManifestCarriesTheMeasuredSplit() {
        let fp = MossImageSFTPackage.manifest.requirements.footprints
        XCTAssertEqual(fp.first { $0.quant == .bf16 }?.residentBytes, 10_500_000_000)
        XCTAssertEqual(fp.first { $0.quant == .int8 }?.residentBytes, 5_800_000_000)
        XCTAssertTrue(fp.allSatisfy { $0.peakActivationBytes == 17_500_000_000 })
    }

    func testEnvelope() throws {
        XCTAssertEqual(try MossImageEnvelope.outputSize(nil, nil).0, 1024)
        XCTAssertNoThrow(try MossImageEnvelope.outputSize(1456, 1456))
        XCTAssertNoThrow(try MossImageEnvelope.outputSize(1920, 1088))  // 16:9 inside the area cap
        XCTAssertThrowsError(try MossImageEnvelope.outputSize(1536, 1536))
        XCTAssertThrowsError(try MossImageEnvelope.outputSize(128, 1024))
        let (w, h) = try MossImageEnvelope.outputSize(1000, 1000)
        XCTAssertEqual([w, h], [992, 992])  // floored to multiples of 16
    }

    func testBaseSharesTheSFTEncoderAndVAE() {
        let base = MossImageConfiguration(variant: .base)
        XCTAssertEqual(base.weightSources.first { $0.role == "transformer" }?.repo, "mlx-community/MOSS-Image-Base-bf16")
        XCTAssertEqual(base.weightSources.first { $0.role == "shared" }?.repo, "mlx-community/MOSS-Image-SFT-bf16")
        // Pinned mirror commits, never a moving "main".
        XCTAssertTrue(base.weightSources.allSatisfy { ($0.revision ?? "").count == 40 })
        XCTAssertEqual(base.weightSources.first { $0.role == "shared" }?.matching,
                       ["text_encoder/*", "tokenizer/*", "vae/*", "LICENSE"])
    }

    func testMATGate() {
        for variant in MossImageVariant.allCases {
            let local = "\(Self.weights)/MOSS-Image-\(variant == .sft ? "SFT" : "Base")"
            let shared = "\(Self.weights)/MOSS-Image-SFT"
            let haveLocal = FileManager.default.fileExists(atPath: local + "/transformer")
                && FileManager.default.fileExists(atPath: shared + "/text_encoder")
            let satisfied = MossImageConfiguration(variant: variant, snapshotPath: local, sharedSnapshotPath: shared)
            let report = MaterializationConformance.check(
                freshConfiguration: MossImageConfiguration(variant: variant),
                satisfiedConfiguration: haveLocal ? satisfied : nil)
            XCTAssertTrue(report.passed, "\(variant): \(report.summary)")
        }
    }

    func testConfigurationRoundTrips() throws {
        let cfg = MossImageConfiguration(variant: .base, quant: .int8, snapshotPath: "/tmp/x", defaultSteps: 30)
        let back = try JSONDecoder().decode(MossImageConfiguration.self, from: JSONEncoder().encode(cfg))
        XCTAssertEqual(back.variant, .base)
        XCTAssertEqual(back.quant, .int8)
        XCTAssertEqual(back.defaultSteps, 30)
        XCTAssertEqual(back.defaultGuidanceScale, 3.0)
        XCTAssertFalse(back.keepEncoderResident)
        XCTAssertEqual(back.textEncoderQuant, .bf16)
        var te8 = cfg
        te8.textEncoderQuant = .int8
        let back8 = try JSONDecoder().decode(MossImageConfiguration.self, from: JSONEncoder().encode(te8))
        XCTAssertEqual(back8.textEncoderQuant, .int8)
    }

    /// Configurations saved before `textEncoderQuant` existed decode as the bf16 encoder.
    func testLegacyConfigurationDecodesWithBF16Encoder() throws {
        let legacy = #"{"variant":"sft","quant":"int8","snapshotPath":"","sharedSnapshotPath":"","defaultSteps":50,"#
            + #""defaultGuidanceScale":3,"keepEncoderResident":false}"#
        let c = try JSONDecoder().decode(MossImageConfiguration.self, from: Data(legacy.utf8))
        XCTAssertEqual(c.quant, .int8)
        XCTAssertEqual(c.textEncoderQuant, .bf16)
        XCTAssertNil(c.peakActivationBytesHint)
    }

    /// The int8 encoder is a lane over either DiT tier, and the cap peak (the VAE decode) doesn't depend on it,
    /// so every lane falls through to the quant-keyed footprint (AB-T-0317).
    func testInt8EncoderLaneUsesTheQuantKeyedFootprint() {
        for quant in [Quant.bf16, .int8] {
            let c = MossImageConfiguration(quant: quant, textEncoderQuant: .int8)
            XCTAssertNil(c.residentBytesHint)
            XCTAssertNil(c.peakActivationBytesHint)
        }
        XCTAssertNil(MossImageConfiguration().peakActivationBytesHint)
    }

    /// Declared resident + activation covers the in-app phys peak at 1024² and the 1456² cap (AB-R-0570, AB-R-0575).
    func testDeclaredFootprintCoversInAppPeaks() {
        let measuredPeak: [Quant: UInt64] = [.bf16: 27_280_000_000, .int8: 22_610_000_000]
        for f in MossImageFootprints.declared {
            XCTAssertGreaterThanOrEqual(f.residentBytes + f.peakActivationBytes, measuredPeak[f.quant]!)
            XCTAssertGreaterThanOrEqual(f.peakActivationBytes, 16_840_000_000)
        }
    }
}
