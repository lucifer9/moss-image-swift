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

    /// The declared split: bf16 10.5 / int8 5.8 GB resident, 16.7 GB activation (the in-app bf16-encoder encode, 0.3.0).
    func testManifestCarriesTheMeasuredSplit() {
        let fp = MossImageSFTPackage.manifest.requirements.footprints
        XCTAssertEqual(fp.first { $0.quant == .bf16 }?.residentBytes, 10_500_000_000)
        XCTAssertEqual(fp.first { $0.quant == .int8 }?.residentBytes, 5_800_000_000)
        XCTAssertTrue(fp.allSatisfy { $0.peakActivationBytes == 16_700_000_000 })
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

    /// The int8 encoder is a lane over either DiT tier: resident stays quant-keyed, activation is hinted (0.3.0).
    func testInt8EncoderLaneDeclaresItsActivation() {
        for quant in [Quant.bf16, .int8] {
            let c = MossImageConfiguration(quant: quant, textEncoderQuant: .int8)
            XCTAssertNil(c.residentBytesHint)
            XCTAssertEqual(c.peakActivationBytesHint, MossImageFootprints.int8EncoderPeakActivationBytes)
        }
        XCTAssertLessThan(MossImageFootprints.int8EncoderPeakActivationBytes,
                          MossImageFootprints.declared.map(\.peakActivationBytes).min()!)
        XCTAssertNil(MossImageConfiguration().peakActivationBytesHint)
    }

    /// Declared resident + activation covers each lane's in-app phys peak at 1024² and the 1456² cap (0.3.0, the
    /// row-band decode: the encode is the peak at both sizes), and each activation covers its measured lane.
    func testDeclaredFootprintCoversInAppPeaks() {
        // (DiT, encoder): (peak, resident between jobs), phys_footprint bytes, TAGP v030.
        let measured: [(Quant, MossImageTextEncoderQuant, UInt64, UInt64)] = [
            (.bf16, .bf16, 26_454_003_128, 10_440_057_376), (.bf16, .int8, 20_392_000_608, 10_441_482_688),
            (.int8, .bf16, 21_798_484_488, 5_768_613_896), (.int8, .int8, 15_732_025_544, 5_769_613_344),
        ]
        for (quant, te, peak, resident) in measured {
            let f = MossImageFootprints.declared.first { $0.quant == quant }!
            let activation = MossImageConfiguration(quant: quant, textEncoderQuant: te).peakActivationBytesHint
                ?? f.peakActivationBytes
            XCTAssertGreaterThanOrEqual(f.residentBytes + activation, peak, "\(quant)/\(te)")
            XCTAssertGreaterThanOrEqual(activation, peak - resident, "\(quant)/\(te)")
        }
    }
}
