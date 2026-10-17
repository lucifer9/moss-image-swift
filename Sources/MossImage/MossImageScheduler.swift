// MOSSImageScheduler — Euler flow matching on the training project's "V2, nearest-grid" schedule:
// the inference grid (shifted by `shift`) is snapped to the nearest point of the 1000-step training
// grid (shifted by `training_shift`), consecutive duplicates dropped, then fed to diffusers'
// FlowMatchEulerDiscreteScheduler with no further shift (timesteps = sigma · 1000, terminal 0).
// Everything is fp32 like the reference; torch.linspace's fp32 algorithm is reproduced so the
// snapped sigmas are bit-identical (gate: oracle goldens/shared/sched).

import Foundation

public struct MossImageScheduler: Sendable {
    public var numTrainTimesteps = 1000
    public var shift: Float = 5.0
    public var trainingShift: Float = 1.0

    public init(shift: Float = 5.0, trainingShift: Float = 1.0) {
        self.shift = shift
        self.trainingShift = trainingShift
    }

    /// torch.linspace(start, end, steps) for float32 on CPU: step = (end - start) / (steps - 1) in
    /// fp32; the first half counts up from start, the second half down from end, each a FUSED
    /// multiply-add (the vectorized kernel's fmadd — plain mul-then-add differs in 239 of 1000
    /// training-grid points, oracle-checked).
    static func linspace(_ start: Float, _ end: Float, _ steps: Int) -> [Float] {
        guard steps > 1 else { return [start] }
        let step = (end - start) / Float(steps - 1)
        let halfway = steps / 2
        return (0 ..< steps).map { i in
            i < halfway ? start.addingProduct(step, Float(i)) : end.addingProduct(-step, Float(steps - i - 1))
        }
    }

    /// `grid(count, shift)`: shift · v / (1 + (shift - 1) · v) over linspace(1, lower, count).
    func grid(_ count: Int, _ shift: Float) -> [Float] {
        // `lower` is a Python float (double); linspace converts both ends to fp32.
        let lower = Float(Double(shift) / (Double(numTrainTimesteps) + Double(shift) - 1))
        return Self.linspace(1, lower, count).map { v in shift * v / (1 + (shift - 1) * v) }
    }

    /// Sigmas (with the terminal 0) for `steps` inference steps.
    public func sigmas(steps: Int) -> [Float] {
        precondition(steps >= 1 && shift > 0 && trainingShift > 0)
        let inference = grid(steps, shift)
        let training = Array(grid(numTrainTimesteps, trainingShift).reversed())  // ascending
        let total = numTrainTimesteps
        var snapped: [Float] = []
        for s in inference {
            // torch.searchsorted (left): first index with training[i] >= s, clamped to total - 1.
            var lo = 0, hi = training.count
            while lo < hi {
                let mid = (lo + hi) / 2
                if training[mid] < s { lo = mid + 1 } else { hi = mid }
            }
            let upper = min(lo, total - 1)
            let lower = max(upper - 1, 0)
            let pick = abs(s - training[lower]) <= abs(training[upper] - s) ? lower : upper
            let value = training[pick]
            if snapped.last != value { snapped.append(value) }
        }
        return snapped + [0]
    }

    /// Model timesteps (sigma · num_train_timesteps, fp32), one per step.
    public func timesteps(sigmas: [Float]) -> [Float] {
        sigmas.dropLast().map { $0 * Float(numTrainTimesteps) }
    }
}
