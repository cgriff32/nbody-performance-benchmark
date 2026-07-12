# 3D N-Body Gravity Simulator: Optimization Benchmark

This repository contains a containerized performance engineering benchmark based on a **3D N-Body Gravitational Simulator**. The project evaluates the optimization capabilities, hardware alignment, and constraint-handling behaviors of autonomous AI coding agents.

---

## 1. The Core Idea
N-body simulations are highly computationally expensive, scaling at $O(N^2)$ for direct force summation. By subjecting optimization agents to strict physical invariants (Hamiltonian energy conservation), system resource limits (RAM, timeouts), and hardware mutations (Performance-core pinning and high particle count scaling), we map how AI code-optimization strategies evolve from simple compiler flags to low-level vector math, fast serialization, and custom multi-threaded regions.

---

## 2. Benchmark Methodology & Sandbox Architecture

* **Physical Invariant:** The simulation moves particles under Newton's gravity laws using **Velocity Verlet integration**. The total system energy (Kinetic + Potential) must remain conserved:
  $$\Delta E = |E_{final} - E_{initial}| \le 10^{-5}$$
* **Harness Isolation (`run_iteration.sh`):**
  * **Container A (app sandbox):** Runs the compiled simulator binary. pinned to a single physical core (Core 2) with memory capped at 512MB (or 16MB for memory sandboxes).
  * **Container B (profiler sidecar):** Pinned to Core 3, sharing Container A's PID namespace (`--pid=container:ContainerA`). It independently samples memory footprint (`VmHWM`) and validates final coordinate energy drifts without altering Container A's resources.

---

## 3. Repository Layout & Directory Structure

* **`baseline/`**: Unoptimized C++ AoS (Array of Structs) code using standard double precision and standard stream I/O.
* **`iter_1/` to `iter_6/`**: Single-core and multi-threaded optimizations on the **1,000-particle** dataset.
* **`iter_variant_1/` to `iter_variant_3/`**: Adaptive configurations handling mutated sandboxes (16MB RAM limits, $\le 10^{-12}$ precision, and strict 150ms timeout watchdogs).
* **`iter_1_100k/` to `iter_blank_1_100k/`**: Double and single precision optimizations scaled to **100,000 particles**.
* **`validation_sidecar/`**: The independent Python validation script (`profile_and_validate.py`) and accelerated C library (`libenergy.so`) for rapid potential energy calculations.

---

## 4. Getting Started

### 1. Build the Docker Sandbox Image
To set up the containerized compilation and execution environment, build the target image from the root of the repository:
```bash
docker build -t nbody-benchmark -f docker/Dockerfile docker/
```

### 2. Compile the Potential Energy C Library
To run the 100k-particle validation without timeouts, compile the potential energy helper library so the Python validator can load it via `ctypes`:
```bash
gcc -O3 -shared -fPIC -fopenmp validation_sidecar/energy.c -o validation_sidecar/libenergy.so
```

### 3. Generate Starting Coordinate States
Generate the initial distributions:
* For 1k particles: `python3 generate_gen0.py`
* For 100k particles: `python3 generate_gen0_100k.py`

### 4. Triggering a Benchmark Run
To compile, run, and validate any target directory inside the containerized sandbox:
* **Standard 1k benchmark (100 steps):**
  ```bash
  ./run_iteration.sh <target_dir_relative_path>
  ```
* **Scaled 100k benchmark (1 step):**
  ```bash
  ./run_iteration_100k.sh <target_dir_relative_path>
  ```

---

## 5. Where to Go for the Final Reports

We have compiled the benchmark results and execution statistics into three final reports in this directory:

1. **[benchmark_summary.md](benchmark_summary.md)**: 
   * **Scope:** 1,000-particle runs (Phases 1 to 3).
   * **Highlights:** Details the progression from AoS to SoA, manual AVX2 SIMD intrinsics, fast C++17 `to_chars` buffer I/O, and thread fork-join avoidance via an OpenMP SPMD parallel region (reducing runtime to **6ms**, a **366.7x speedup**).
2. **[benchmark_summary_100k.md](benchmark_summary_100k.md)**: 
   * **Scope:** 100,000-particle runs.
   * **Highlights:** Showcases unrolled AVX2 pipelines, branchless loops, and single-precision-to-double casting tricks. Documents how agents compiled a native C library (`libenergy.so`) to accelerate Python validation loop times from 15 minutes down to **1.5 seconds**.
3. **[agent_profiling_report.md](agent_profiling_report.md)**: 
   * **Scope:** Post-hoc agent performance.
   * **Highlights:** Documents the duration, total steps, tool call counts, and estimated token consumption of each optimization sub-agent spawned during the project.

---

## 6. Recreating the Analysis

To recreate the entire benchmarking run and regenerate the performance summary reports:
1. Ensure Docker is running.
2. Complete the steps in the **Getting Started** section (building the image, compiling the validation library, and generating datasets).
3. Run the automated re-benchmarking script (sequentially recompiles, runs, and profiles all 1k iterations and sandbox variants):
   ```bash
   ./scratch/rebenchmark_all.sh
   ```
4. Run the post-hoc agent profiling script to recalculate the token usage and execution timelines:
   ```bash
   python3 scratch/profile_agents.py
   ```
