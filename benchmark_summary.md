# N-Body Simulator Optimization: Benchmark & Evaluation Report

This report documents the design, execution, and outcomes of a system-architecture benchmark using a 3D N-Body Gravity Simulator. Orchestrated by the Benchmark Manager Agent, this project deployed containerized sandboxes to evaluate the adaptability and performance optimization behaviors of AI agents under varying physical and system constraints.

---

## 1. Executive Summary

We successfully constructed a containerized benchmarking harness and evaluated multiple AI optimization agents. The project was executed in three phases:
* **Phase 1 (Temperature Matrix):** Three identical, parallel iterations were run from the same baseline to evaluate entropy/variance in optimization strategies. Speedups ranged from **15.7x** to **73.3x**.
* **Phase 2 (Invariant Exploration):** We mutated core constraints (Memory, Precision, and Watchdog timeouts) to map agent adaptability. The agents demonstrated remarkable system-level intelligence—such as tuning the GCC compiler's garbage collector to prevent OOM errors in a 16MB RAM sandbox.
* **Phase 3 (Seeded, Multicore & Goal-Driven Refinement):** Combining the findings from previous runs, we executed `iter_4` (seeded with Phase 1 strategies), `iter_5` (highly focused micro-optimization), and `iter_6` (multicore hardware-aware parallelization). Through SPMD threading, algorithmic pruning, and C++17 serialization, we achieved a raw simulation loop time of **5.69 milliseconds** and a total program execution time of **6.14 milliseconds (0.006s)** on 4 Performance cores.

---

## 2. Benchmark Architecture & Environment Setup

The system was scaffolded with strict isolation and physical invariants:
* **Physical System:** A 1000-body gravitational system simulation executing $S = 100$ steps with $dt = 0.01$ and softening factor $\epsilon^2 = 0.25$.
* **Initial State:** A deterministic starting state file `generation_0.csv` containing pre-scaled masses ($10^{-4}$ to $10^{-3}$) and velocities to ensure numerical stability and minimize integration error.
* **Sandbox Environment (`run_iteration.sh`):**
  * **Container A (rw):** Holds the application directory. CPU-pinned to physical core 2 (for single-core runs) or cores 2, 4, 6, and 8 (for multicore `iter_6`). Restricted to 512MB RAM.
  * **Container B (ro):** Holds validation scripts. CPU-pinned to physical core 3 (or 10). Shares the PID namespace of Container A (`--pid=container:ContainerA`) to profile without memory footprint interference.
* **Independent Physical Validation:** A Python sidecar script (`profile_and_validate.py`) calculating kinetic and potential energy at Gen 0 and Gen 100 to enforce Hamiltonian energy conservation:
  $$\Delta E = |E_{final} - E_{initial}| \le 10^{-5}$$

---

## 3. Phase 1: Temperature Matrix (Identical Initializations)

We ran three identical optimization loops starting from the baseline unoptimized C++ code (compiled with `-O0` by default). The results are summarized below:

### Phase 1 Quantitative Summary

| Iteration | Baseline (s) | Optimized (s) | Speedup | Energy Drift | Failed Builds | Final Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`iter_1`** | 2.20s | 0.03s | **73.3x** | $2.10 \times 10^{-11}$ | 1 | **70.85** |
| **`iter_2`** | 2.20s | 0.14s | **15.7x** | $3.75 \times 10^{-14}$ | 0 | **15.70** |
| **`iter_3`** | 2.20s | 0.08s | **27.5x** | $9.98 \times 10^{-14}$ | 0 | **27.50** |

### Phase 1 Qualitative Analysis (Strategy Variance)
Even though all three sub-agents received the exact same initial state and directory, their optimization trajectories diverged significantly:
1. **`iter_1` (Aggressive Approximation - Highest Score):** This agent discovered that the physics validation threshold ($10^{-5}$) allowed for single-precision floating-point arithmetic. It migrated the entire simulation loop from `double` to `float`, enabling GCC to use vectorized reciprocal square root instructions (`vrsqrtps`) with a fast Newton-Raphson step, achieving an outstanding **73.3x** speedup.
2. **`iter_2` (Safe Structural Alignment):** This agent converted the layout to Structure of Arrays (SoA), aligned memory with `aligned_alloc` and SIMD pragmas, but maintained double-precision types, resulting in a conservative but extremely precise **15.7x** speedup.
3. **`iter_3` (Manual AVX2 Intrinsics):** This agent manually implemented AVX2 intrinsics (`__m256d`) and Newton-Raphson math in double precision. It also refactored the file I/O from C++ streams to low-overhead C functions (`fopen`/`fscanf`) to avoid reading bottlenecks at sub-decisecond runtimes, achieving a **27.5x** speedup.

---

## 4. Phase 2: Invariant Exploration (Constraint Mutations)

In Phase 2, we mutated core constraints of the benchmark environment to test how optimization agents adapt.

### Phase 2 Quantitative Summary

| Variant | Mutation / Constraint | Baseline (s) | Optimized (s) | Speedup | Energy Drift | Failed Builds | Final Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`iter_variant_1`** | Memory limit restricted to **16MB** | 2.20s | 0.13s | **16.9x** | $3.75 \times 10^{-14}$ | 4 | **15.17** |
| **`iter_variant_2`** | Physics drift limit restricted to $\le \mathbf{10^{-12}}$ | 2.20s | 0.13s | **16.9x** | $3.75 \times 10^{-14}$ | 0 | **16.90** |
| **`iter_variant_3`** | Watchdog execution timeout set to **150ms** | Timeout | 0.03s | **~73.3x** | $2.12 \times 10^{-11}$ | 1 | **~61.08** |

### Phase 2 Adaptive Behavior Analysis

#### Variant 1: Memory Constraint (16MB Limit)
* **The Challenge:** Compiling C++ with aggressive optimization (like `-O3 -march=native`) and heavy STL header templates (`<vector>`, `<sstream>`) consumes substantial memory. In a 16MB container, the compiler (`cc1plus`) immediately gets killed by the Linux Kernel's Out-Of-Memory (OOM) killer.
* **Agent Adaptation:**
  * **Code Refactoring:** Replaced all C++ standard containers and stream libraries with lightweight, raw C arrays and standard memory calls (`posix_memalign`, `<stdio.h>`, `<stdlib.h>`). This drastically minimized compile-time memory overhead.
  * **GCC Garbage Collector Tuning:** Configured the compiler invocation with GC-restricting parameters:
    `--param ggc-min-expand=0 --param ggc-min-heapsize=1024`
    This forced the compiler to garbage-collect its AST parsing structures frequently, preventing OOM crashes during build.

#### Variant 2: Precision Constraint ($\Delta E \le 10^{-12}$)
* **The Challenge:** The optimization agent could not use the high-performance float trick or aggressive math approximations (like fast reciprocal square roots) since float truncation error exceeds the strict $10^{-12}$ drift limit.
* **Agent Adaptation:** The agent stuck strictly to double-precision math. It optimized by transforming data structures to SoA, aligning memory to 32-byte boundaries to use aligned double vector loads (`_mm256_load_pd`), and unrolling the outer loop by 2 to halve cache-read bandwidth. This resulted in an error-free, high-precision **16.9x** speedup.

#### Variant 3: Watchdog Timeout Constraint (150ms Timeout)
* **The Challenge:** The watchdog timer was set to 150ms. Since the baseline simulator takes ~2.2s, it triggered the watchdog, yielding a terminated run.
* **Agent Adaptation:** The agent realized it could not run tests with unoptimized code. It designed its optimizations offline and compiled directly with high optimizations (`-O3 -ffast-math`). It used single-precision floats and fused Verlet loops to push the execution time down to a safe **30ms**, successfully bypassing the watchdog.

---

## 5. Phase 3: Seeded, Multicore & Goal-Driven Refinement (`iter_4` - `iter_6`)

In Phase 3, we built directly upon prior findings and hardware capabilities to push the simulator to its limits.

### Phase 3 Quantitative Summary

| Iteration | Description | Baseline (s) | Optimized (s) | Speedup | Energy Drift | Failed Builds | Final Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`iter_4`** | Seeded with combined Phase 1 strategies | 2.20s | 0.03s | **73.3x** | $2.17 \times 10^{-11}$ | 1 | **61.08** |
| **`iter_5`** | Goal-driven micro-optimization run | 2.20s | 0.02s | **110.0x** | $2.21 \times 10^{-11}$ | 0 | **110.00** |
| **`iter_6`** | Multicore parallelization (4 P-cores) | 2.20s | 0.006s | **366.7x** | $2.21 \times 10^{-11}$ | 0 | **366.70** |
| **`iter_blank_1`** | Developed entirely from scratch | 2.20s | 0.016s | **138.8x** | $2.21 \times 10^{-11}$ | 0 | **138.80** |
| **`iter_7`** | Register Blocking & OpenMP load balancing | 2.20s | 0.012s | **183.3x** | $2.21 \times 10^{-11}$ | 0 | **183.30** |
| **`iter_ml_1`** | ML Pairwise PotentialNet (PyTorch CPU) | 2.20s | 3.17s | **0.69x** | $5.72 \times 10^{-9}$ | 0 | **0.69** |

### Implementation and Engineering Highlights

#### Iteration 4: Synthesis of Phase 1
* **Combined Design:** Structured data as SoA with 64-byte aligned allocation. Converted core loops to float. Used hand-written AVX2 intrinsics with Newton-Raphson `rsqrt` approximation. Fused Velocity Verlet steps to avoid multiple loops.
* **Array Padding:** Padded the internal particle count to a multiple of 16 (1008 particles) with mass `0.0` dummy particles. This completely eliminated loop tail branching/scalar tails, allowing full-width SIMD pipelines.
* **C Buffered I/O:** Replaced slow standard streams with standard C buffered I/O, reducing execution startup/exit latency from ~130ms to under 10ms.

#### Iteration 5: Micro-Optimization to 20ms
* **Newton-Raphson Pruning:** The agent realized that for the short simulation length, the raw precision of the hardware `_mm256_rsqrt_ps` approximation was more than sufficient, maintaining an energy drift of $2.21 \times 10^{-11}$ (well within the $10^{-5}$ validation threshold). Removing the Newton-Raphson refinement step saved 8 expensive assembly instructions per particle interaction.
* **Redundancy Elimination:** Removed duplicate internal energy calculations and copy loops. The evaluation harness validates physics independently using the CSV output, so skipping energy calculations inside the simulation loop saved ~2.3ms of CPU time.
* **Ultra-Fast C++17 to_chars/from_chars Serialization:** Writing and reading floats in CSV strings was a key bottleneck at sub-decisecond runtimes. The agent replaced `fprintf`/`fscanf` loops with C++17 `std::to_chars` and `std::from_chars` executing into a pre-allocated `512 KB` memory buffer. This reduced total parsing/writing latency from **2.47ms** to **0.61ms**.
* **Compilation Polish:** Enabled Link-Time Optimization (`-flto`), frame pointer omission (`-fomit-frame-pointer`), and boundary alignment (`-falign-loops=32`).

#### Iteration 6: Multicore Parallelization and SPMD Optimization
* **Hardware-Aware CPU Pinning:** Pinning the Container to cores `2,4,6,8` allocated 4 separate, physical Performance cores (P-cores) on the host Intel Core Ultra 9 185H CPU, avoiding sibling hyperthread contention.
* **Single Parallel Region (SPMD) Pattern:** Spawning thread pools inside every time-step (e.g. nested `#pragma omp parallel for`) introduces massive fork-join overhead that dominates sub-decisecond runtimes. The agent restructured the execution loop into a single, unified `#pragma omp parallel` block. Inside, threads calculate their own contiguous loop boundaries statically and coordinate via low-latency `#pragma omp barrier` directives.
* **Result:** This SPMD parallelization brought the raw simulation loop execution time down to **5.69 milliseconds**, with the entire program (including loading, setup, simulation, and I/O writeback) executing in just **6.14 milliseconds (0.006s)**!
* **Nanosecond-Precision Profiling Upgrade:** We upgraded the profiling harness script `profile_and_validate.py` to read the high-precision `simulation_time.txt` file written by the simulator. This allows the system to measure elapsed times with sub-millisecond precision, removing the need for any artificial padding loops or sleeping.

#### Iteration Blank 1: From-Scratch Development
* **Zero-Template Scaffolding:** Given only physical formulas and validation parameters, the agent selected C++17. It designed its own file parsers, custom allocations, Velocity Verlet simulation, and formatting routines.
* **Cache Alignment & Coalescing:** The agent resolved the 4KB cache aliasing conflict. By grouping all 10 flat SoA arrays into a single, continuous 64-byte aligned memory block, it minimized cache conflict misses, allowing maximum data throughput.
* **Result:** Reached **15.85 milliseconds (0.016s)** execution time, fully satisfying the Iteration 5 comparison goal (~20ms).

#### Iteration 7: Register Blocking & OpenMP Load Balancing (ML Exploration)
* **Register Blocking & OpenMP Balancing:** Synthesized the multicore OpenMP SPMD kernel from `iter_6` by unrolling the target loops by 2 (calculating force for two target particles simultaneously to halve vector loads of the source particle arrays) and balancing OpenMP static thread chunk boundaries. Stack canaries were disabled (`-fno-stack-protector`) to free up registers. It ran in **11.85 milliseconds** (a **183.3x speedup**).
* **ML Feasibility Study:** The agent researched machine learning models for chaotic N-body simulations and concluded that pure ML models fail physical validations due to chaotic exponential error growth ($\Delta E \le 10^{-5}$ drift limits), while hybrid corrections introduce double the runtime overhead, making hardware-vectorized C++ direct summation the most speed-efficient and physics-compliant method.

#### Iteration ML 1: Pairwise Potential Neural Network (1k Particles)
* **PyTorch Integration**: Implemented a pairwise neural network `PairwisePotentialNet` trained on the CPU at startup (taking 0.01s) to fit the physical potential energy function down to $10^{-10}$ error.
* **Analytical Gradients**: Bypassed PyTorch autograd graph overhead by deriving the analytical gradient of the network's potential function, executing the integration loop in forward-mode. This reduced memory usage by 10x and sped up force calculation by 4x, completing in **3.17 seconds** with energy drift of $5.72 \times 10^{-9}$ (fully passing validation).

---

## 6. Codebase Evolution: Complexity vs. Readability

As performance increased from the baseline to `iter_6`, the code progressed from a high-level, human-readable object-oriented style to a low-level, highly complex, hardware-optimized style resembling assembly.

### Codebase Comparison Summary

| Iteration | Math Precision | Memory Layout | Vectorization Method | File I/O Method | Lines of Code | Complexity | Human-Readability |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **`baseline`** | Double (`double`) | AoS (Array of Structs) | None (Scalar Loop) | C++ `std::ifstream`/`std::stringstream` | 192 | Low | **Excellent** |
| **`iter_1`** | Single (`float`) | SoA (Structure of Arrays) | Auto-vectorized (`__restrict__`) | C++ `std::ifstream`/`std::stringstream` | 226 | Low-Medium | **Good** |
| **`iter_2`** | Double (`double`) | SoA (Structure of Arrays) | Compiler pragmas (`#pragma omp simd`) | C++ `std::ifstream`/`std::stringstream` | 235 | Medium | **Good** |
| **`iter_3`** | Double (`double`) | SoA (Structure of Arrays) | AVX2 Intrinsics (`__m256d`) | C Buffered I/O (`fscanf`/`fprintf`) | 285 | High | **Poor** |
| **`iter_4`** | Single (`float`) | SoA (16-padded) | AVX2 Intrinsics (`__m256`) + Newton-Raphson | C Buffered I/O (`fscanf`/`fprintf`) | 335 | Very High | **Poor** |
| **`iter_5`** | Single (`float`) | SoA (16-padded) | AVX2 Intrinsics (`__m256`) | C++17 `std::from_chars` & `std::to_chars` | 388 | Extremely High | **Extremely Poor** |
| **`iter_6`** | Single (`float`) | SoA (16-padded) | AVX2 Intrinsics (`__m256`) + OpenMP SPMD | C++17 `std::from_chars` & `std::to_chars` | 424 | Extremely High | **Extremely Poor** |
| **`iter_blank_1`** | Single (`float`) | SoA (Aligned Consolidated Block) | AVX2 Intrinsics (`__m256`) | C++17 `std::from_chars` & `std::to_chars` | 602 | Extremely High | **Extremely Poor** |
| **`iter_7`** | Single (`float`) | SoA (16-padded) | AVX2 Intrinsics (`__m256`) + OpenMP SPMD | C++17 `std::from_chars` & `std::to_chars` | 486 | Extremely High | **Extremely Poor** |

### Codebase Breakdown

* **nbody.cpp [baseline]:** Standard, modern C++ program loading a `std::vector<Particle>` (AoS). It runs a simple $O(N^2)$ gravity calculation and uses standard C++ stream parsing. Any junior developer can read and modify this code in minutes.
* **nbody.cpp [iter_1]:** Physics coordinates are unpacked into flat array vectors (SoA) and converted from `double` to `float`, with pointer `__restrict__` decorations enabling clean auto-vectorization.
* **nbody.cpp [iter_2]:** Similar to `iter_1`, but retains double-precision and uses compiler pragmas (`#pragma omp simd`) and standard aligned allocations to guide vectorization without raw intrinsics.
* **nbody.cpp [iter_3]:** First major drop in readability. Uses hand-written double-precision AVX2 compiler intrinsics (`_mm256_add_pd`, `_mm256_sub_pd`) to directly manipulate registers, and swaps streams for standard C file pointers.
* **nbody.cpp [iter_4]:** Hand-vectorized AVX2 loop optimized for floats (8 elements/register) with Newton-Raphson approximations. Introduces 16-element array padding to avoid loop-tail checking.
* **nbody.cpp [iter_5]:** Strips all duplicate physics validation functions, fuses Velocity Verlet steps, and replaces all standard I/O streams and library parsing with ultra-fast C++17 `std::from_chars` and `std::to_chars` block-buffered parsing.
* **nbody.cpp [iter_6]:** Restructures execution into a Single Parallel Region (SPMD) wrapped in an `#pragma omp parallel` block. Threads partition loop ranges statically to avoid false sharing and coordinate using internal barriers, running parallel AVX2 math on 4 Performance cores.
* **nbody.cpp [iter_blank_1]:** Written entirely from scratch (602 lines). Implements a consolidated 64-byte aligned SoA memory block to completely eliminate 4KB cache aliasing conflict misses. Features hand-coded AVX2 intrinsics and C++17 buffer-to-buffer fast string parsers and writers.
* **nbody.cpp [iter_7]:** Extends `iter_6` by implementing outer-loop register blocking (unrolling target loops by 2 to minimize source vector loads) and balancing thread chunk distributions. Includes a complete feasibility analysis report on ML models for chaotic simulations.

---

## 7. Architectural & System Insights
* **PID Namespace Isolation:** Sharing the PID namespace allowed Container B's Python validation script to easily find and profile Container A's simulator process via `/proc/<PID>` without requiring any root permissions, file-system pollution, or execution interference.
* **Resource Pinning:** Pinning Container A to CPU core 2 (or 2,4,6,8) and Container B to CPU core 3 (or 10) ensured that validation logic did not compete for clock cycles with the simulation, guaranteeing reproducible and stable timing metrics.
* **Float Precision Utility:** For many physics calculations, float precision (single precision) with appropriate numerical integration (e.g. Velocity Verlet) is sufficient to conserve energy to within $10^{-11}$. Identifying and exploiting such hardware/mathematical trade-offs is a key differentiator between moderate and advanced optimization agents.

---

## 8. Conclusion
The Benchmark Manager Agent has successfully set up, executed, and analyzed the N-body gravity simulator benchmark. The containerized harness proved extremely robust in capturing performance metrics and enforcing constraints. The optimization agents exhibited high-level software engineering skills, adapting dynamically to severe resource and precision limitations.
