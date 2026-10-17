// Offline manifest / configuration / materialization tests for the MOSS-Image engine packages.
import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXMossImage

final class MossImagePackageTests: XCTestCase {
    static let weights = "/Volumes/Satechi/Development/mlxengine-image/weights"

    func testManifestsAreApacheAndT2IOnly() {
        for m in [MossImageSFTPackage.manifest, MossImageBasePackage.manifest] {
            XCTAssertEqual(m.license.weightLicense, .apache2)
            XCTAssertTrue(SPDXLicense.permissiveAllowlist.contains(m.license.weightLicense))
            XCTAssertEqual(m.license.portCodeLicense, .mit)
            XCTAssertEqual(m.contractVersion, ContractVersion.current)
            XCTAssertEqual(m.surfaces.map(\.capability), [.textToImage])
            XCTAssertEqual(Set(m.requirements.footprints.map(\.quant)), [.bf16, .int8, .int4])
            XCTAssertTrue(m.requirements.footprints.allSatisfy { $0.residentBytes > 0 && $0.peakActivationBytes > 0 })
        }
        XCTAssertEqual(MossImageSFTPackage.manifest.surfaces.map(\.name), ["moss-image-sft"])
        XCTAssertEqual(MossImageBasePackage.manifest.surfaces.map(\.name), ["moss-image-base"])
        XCTAssertEqual(MossImageBasePackage.manifest.provenance.sourceRepo, "OpenMOSS-Team/MOSS-Image-Base")
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
        XCTAssertEqual(base.weightSources.first { $0.role == "transformer" }?.repo, "OpenMOSS-Team/MOSS-Image-Base")
        XCTAssertEqual(base.weightSources.first { $0.role == "shared" }?.repo, "OpenMOSS-Team/MOSS-Image-SFT")
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
    }
}
