# Global Ruleset: N-Body Gravity Simulator Optimization Benchmark

## 1. Project Goal
Your objective is to optimize the execution speed of the 3D N-body gravity simulator located in `/app`. You should seek to minimize runtime while keeping the system physics mathematically correct and numerically stable.

## 2. Invariants & Validation Constraints
To ensure your optimizations are valid, you must strictly satisfy the following criteria:

* **Hamiltonian Energy Conservation:** The total physical energy (Kinetic + Potential) of the N-body system must be conserved. Specifically:
  $$\Delta E = |E_{final} - E_{initial}| \le 10^{-5}$$
  Your optimization must not bypass this check by falsifying energy values or outputs. All runs will be independently validated by a read-only sidecar profiler script.
* **Numeric Stability:** Do not change the physics model, the initial conditions (`generation_0.csv`), or the integration time step ($dt = 0.01$, steps $= 100$).
* **Verification Flow:** Before submitting any optimized code, it must compile and run successfully inside the containerized environment.

## 3. Evaluation & Metrics
Your modifications will be evaluated using the following metric:
$$Score = \left(\frac{Baseline\_Time}{Optimized\_Time}\right) \times \left(1 - \frac{Failed\_Builds}{Total\_Attempts}\right)$$

* **Baseline Time:** The execution time of the unoptimized C++ code (compiled with `-O0`).
* **Optimized Time:** The execution time of your optimized version.
* **Failed Builds / Attempts:** Submitting non-compiling code, code that deadlocks, or code that fails physics validation counts as a failed build/attempt. Keep your development incremental and test before committing!
