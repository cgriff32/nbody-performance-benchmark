# N-Body Gravity Simulator Optimization Report (Iteration 4)

We successfully combined the top findings from all three prior agents and designed a highly optimized AVX2 SIMD single-precision float solver with fused Velocity Verlet updates and low-overhead C buffered I/O.

## Optimization Strategy

1. **Single-Precision Float & SoA**:
   - Converted all inner loop storage and math to `float` to double SIMD width.
   - Padded the array size to a multiple of 16 (`n_pad = 1008`) and set padding masses to `0.0f` to avoid loop tails or conditional branching inside the inner kernel.
   - Memory allocated on 64-byte alignments using `posix_memalign` and pointer alignment hints (`__builtin_assume_aligned`).

2. **AVX2 SIMD Inner Loop (Unrolled by 2)**:
   - Replaced division and `sqrt` with fast reciprocal square root (`_mm256_rsqrt_ps`) plus one Newton-Raphson step to meet precision requirements.
   - Fused Multiply-Adds (FMA) used for distance squaring and force accumulation.
   - Unrolled the inner loop by 2 (`j += 16`) to interleave instruction chains and hide dependency latencies without register spilling.

3. **Verlet Loop Fusion & Memory Reduction**:
   - Re-formulated Velocity Verlet to fuse position updates and velocity updates. Removed temporary `old_ax/ay/az` buffers completely.
   - Inside the main loop, we have exactly one simple O(N) fused position/velocity update loop and one O(N^2) acceleration loop.

4. **Low-Overhead Buffered C I/O**:
   - Replaced slow C++ streams with `fscanf` and large-buffered `fprintf` (`setvbuf`), reducing file I/O time to <5ms.

## Performance Comparison

| Metric | Baseline | Optimized | Speedup |
| :--- | :--- | :--- | :--- |
| **Elapsed Time** | 1.79s | 0.02s | **89.5x** |
| **CPU Time** | 1.79s | 0.02s | **89.5x** |
| **Energy Drift** | 3.75e-14 | 2.17e-11 | *Passed (Limit <= 1e-5)* |
| **Max RSS** | 3.93 MB | 3.97 MB | *Negligible overhead* |

## Optimization Iterations & Attempts

- **Total Compilation/Run Attempts**: 6
- **Failed Builds**: 1 (typo in `static_cast`)
- **Success Rate**: 83.3%
- **Final Score**: **74.58**
