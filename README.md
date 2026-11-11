# GPU Operator Fusion Decision Engine

C++17/CUDA analytical cost model and hardware micro-benchmark suite for evaluating operator fusion boundaries in deep learning compute graphs. Combines an exact O(N^2) **Dynamic Programming (DP)** solver with a **Quantum-Inspired Evolutionary Optimizer (QIEO)** to optimize fusion layouts under register allocation granularity, SM warp occupancy, and HBM memory bandwidth constraints.

## Key Features
* **Hardware-Calibrated Cost Model (`src/fusion.cpp`):** Models NVIDIA Turing (`sm_75`, Tesla T4) and Ampere (`sm_86`, RTX 3090) SM limits, including 8-register warp allocation granularity, shared memory capacity, and the 40% latency-hiding occupancy cliff. Base operator register counts are calibrated directly against `nvcc -Xptxas=-v` SASS compilation metrics (`39` regs for tiled GEMM, `18` regs for warp-shuffle LayerNorm, `14` regs for scalar fused elementwise, `24` regs for `float4` vectorized elementwise).
* **Dual Optimization Solvers:**
  * **DP Solver:** Exact O(N^2) prefix partition search.
  * **QIEO Solver:** Population-based metaheuristic using quantum rotation gates (`theta` probability amplitudes) for scaling to multi-layer graphs (131,072 candidate states in Case C).
* **Custom CUDA Benchmark Suite (`benchmarks/benchmark.cu`):**
  * **Fused GEMM Epilogue (`matmul_bias_epilogue_fused_kernel`):** Fuses `BiasAdd` and `ReLU` directly into GEMM tile accumulators prior to global memory writeback (`0` extra registers required).
  * **Warp-Shuffle LayerNorm (`layernorm_warp_shfl_kernel`):** Uses `__shfl_down_sync` register reductions to avoid shared memory bank conflicts.
  * **Vectorized Elementwise Fusion (`elem_fused_vec4_kernel`):** Uses 128-bit `float4` loads/stores (`__ldg`) to maximize HBM bus utilization while keeping intermediate activations in registers.
  * **Runtime Hardware Introspection:** Queries compiled SASS register counts (`numRegs`) and achieved SM occupancy via `cudaFuncGetAttributes` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

## Workloads & Hardware Benchmark Results (NVIDIA Tesla T4, SM 7.5)
1. **Case A (6-Op Pipeline: `MatMul -> BiasAdd -> ReLU -> LayerNorm -> MatMul -> BiasAdd`):**
   * **Optimal Plan:** `[0-2] -> [3-3] -> [4-5]`
   * Fuses `BiasAdd + ReLU` into `MatMul_0` (`[0-2]`) and `BiasAdd` into `MatMul_1` (`[4-5]`), eliminating **40 MB** of intermediate HBM traffic while keeping `LayerNorm` isolated (`[3-3]`) due to row-reduction barrier and register pressure constraints.
   * **Tesla T4 Benchmark:** `3.3005 ms` (Unfused 6 kernels) vs. `2.0176 ms` (Fused 3 kernels) — **1.64x speedup** (`L_inf` error `0.00e+00`).
2. **Case B (3-Op Memory-Bound Chain: `ElemAdd -> ReLU -> BiasAdd`):**
   * **Optimal Plan:** `[0-2]` (Full Fusion)
   * Eliminates **32 MB** of intermediate HBM read/write traffic by forwarding values in 32-bit registers.
   * **Tesla T4 Benchmark:** `0.2389 ms` (Unfused 3 kernels) vs. `0.0996 ms` (Fused Scalar, **2.40x speedup**) and `0.1057 ms` (Fused `float4`, **2.26x speedup**, `L_inf` error `0.00e+00`).
3. **Case C (18-Op 3-Layer Transformer Pipeline):**
   * Evaluates solver scaling across 131,072 candidate fusion plans; QIEO matches the exact DP optimum (`[0-2] -> [3-3] -> [4-5]` per layer, saving **120 MB** HBM traffic).

## Build & Run

    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j$(nproc)
    ./fusion_cli
    ./benchmark_runner

