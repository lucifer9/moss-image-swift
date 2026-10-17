// S0 key contract: the weight-free modules' parameter paths == the snapshot safetensors headers
// (read as raw JSON — no MLX eval, so this runs under plain `swift test`). Skips when the weights
// are not on disk.

import Foundation
import MLX
import MLXNN
import Testing
@testable import MossImage

let weightsRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().appendingPathComponent("../../weights").standardized

func headerKeys(_ directory: URL) throws -> Set<String> {
    var keys = Set<String>()
    for f in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    where f.pathExtension == "safetensors" {
        let h = try FileHandle(forReadingFrom: f)
        let n = h.readData(ofLength: 8).withUnsafeBytes { $0.load(as: UInt64.self) }
        let json = try JSONSerialization.jsonObject(with: h.readData(ofLength: Int(n))) as! [String: Any]
        keys.formUnion(json.keys.filter { $0 != "__metadata__" })
    }
    return keys
}

@Suite struct KeyContractTests {
    @Test(arguments: ["SFT", "Base"])
    func transformerKeys(variant: String) throws {
        let dir = weightsRoot.appendingPathComponent("MOSS-Image-\(variant)/transformer")
        guard FileManager.default.fileExists(atPath: dir.path) else { return }
        let model = MossImageTransformer(try MossImageTransformerConfiguration.load(transformerDirectory: dir))
        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(try headerKeys(dir).subtracting(["rope.bands"]).map(MossImageWeights.transformerKey))
        #expect(moduleKeys.subtracting(fileKeys).isEmpty, "missing: \(moduleKeys.subtracting(fileKeys).sorted().prefix(5))")
        #expect(fileKeys.subtracting(moduleKeys).isEmpty, "unused: \(fileKeys.subtracting(moduleKeys).sorted().prefix(5))")
    }

    @Test func textEncoderKeys() throws {
        let dir = weightsRoot.appendingPathComponent("MOSS-Image-SFT/text_encoder")
        guard FileManager.default.fileExists(atPath: dir.path) else { return }
        let model = Qwen35Encoder(try Qwen35EncoderConfiguration.load(textEncoderDirectory: dir))
        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = try headerKeys(dir).filter { $0.hasPrefix("language_model.") && !$0.contains("mtp.") }
        #expect(moduleKeys.subtracting(fileKeys).isEmpty, "missing: \(moduleKeys.subtracting(fileKeys).sorted().prefix(5))")
        #expect(fileKeys.subtracting(moduleKeys).isEmpty, "unused: \(fileKeys.subtracting(moduleKeys).sorted().prefix(5))")
    }
}
