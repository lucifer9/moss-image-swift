// Prompt → conditioning for MOSS-Image: the reference `encode_prompt`, batch 1.
//   tokenizer.apply_chat_template([{role: user, content: p}], add_generation_prompt=True,
//   enable_thinking=True) renders to
//     "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n"
//   (pinned by oracle goldens/shared/tok.json, including the empty negative prompt), truncated at
//   max_sequence_length (4096) tokens, then Qwen3.5-9B's last_hidden_state — unpadded, the whole
//   sequence including the template tokens.

import Foundation
import MLX
import Tokenizers

public final class MossImagePromptEncoder {
    public let model: Qwen35Encoder
    public let tokenizer: any Tokenizers.Tokenizer
    public static let maxSequenceLength = 4096

    public static func render(_ prompt: String) -> String {
        "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n"
    }

    public init(model: Qwen35Encoder, tokenizer: any Tokenizers.Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    /// `snapshot` = a MOSS-Image snapshot root (text_encoder/ + tokenizer/).
    public static func load(snapshot: URL, dtype: DType = .bfloat16) async throws -> MossImagePromptEncoder {
        let model = try MossImageWeights.loadTextEncoder(
            directory: snapshot.appendingPathComponent("text_encoder"), dtype: dtype)
        return MossImagePromptEncoder(model: model, tokenizer: try await loadTokenizer(snapshot: snapshot))
    }

    public static func loadTokenizer(snapshot: URL) async throws -> any Tokenizers.Tokenizer {
        try await AutoTokenizer.from(modelFolder: snapshot.appendingPathComponent("tokenizer"))
    }

    public func tokenize(_ prompt: String, maxLength: Int = maxSequenceLength) -> [Int] {
        let ids = tokenizer.encode(text: Self.render(prompt), addSpecialTokens: false)
        return Array(ids.prefix(maxLength))
    }

    /// [1, S, 4096] in the encoder's dtype.
    public func encode(_ prompt: String, maxLength: Int = maxSequenceLength) -> MLXArray {
        let ids = tokenize(prompt, maxLength: maxLength)
        let input = MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count)
        let (hidden, _) = model.lastHiddenState(input)
        eval(hidden)
        return hidden
    }
}
