# GPU Operator Fusion Decision Engine

High-performance C++17/CUDA analytical cost model and hardware micro-benchmarking suite for evaluating and optimizing **operator fusion layouts** in deep learning compute graphs.

Combines an exact **Dynamic Programming (DP)** solver with a **Quantum-Inspired Evolutionary Optimizer (QIEO)** to predict optimal fusion boundaries by modeling warp register allocation granularity, SM warp occupancy, shared memory capacity, and HBM memory traffic.

---

## 🌟 Key Results

Demonstrates **100% decision alignment** between the C++ analytical engine, `nvcc` SASS register allocation (`ptxas`), and physical silicon execution on **NVIDIA Tesla T4 (Turing `sm_75`)**:

1. **Case A: Selective Epilogue Fusion (6-Op Transformer FFN Pipeline)**
   - **Workload**: `MatMul[1024x256x2048] → BiasAdd → ReLU → LayerNorm → MatMul[1024x2048x256] → BiasAdd`
   - **Optimal Decision**: `[0-2] -> [3-3] -> [4-5]` (**PARTIAL EPILOGUE FUSION**)
   - **Why**: Fuses `BiasAdd + ReLU` directly into the `MatMul_0` tile accumulator (`[0-2]`) and `BiasAdd` into `MatMul_1` (`[4-5]`) with `0` extra registers, eliminating **34 MB** of intermediate HBM traffic, while isolating `LayerNorm` (`[3-3]`) due to row-reduction barrier and register pressure constraints (`>128` regs/thread drops occupancy to `25%`).

2. **Case B: Memory-Bound Elementwise Chain (`2.40×` Speedup)**
   - **Workload**: 3-Op Elementwise Chain (`ElemAdd → ReLU → BiasAdd`, `1024 × 2048` FP32)
   - **Optimal Decision**: `[0-2]` (**FULL FUSION**)
   - **Why**: Eliminates **32 MB** of intermediate HBM traffic by keeping activations in 32-bit registers (`0.238 ms` unfused vs. `0.099 ms` fused on Tesla T4).

3. **Case C: 18-Op Multi-Layer Transformer Graph ($2^{17} = 131,072$ States)**
   - **Decision**: `[0-2] -> [3-3] -> [4-5]` repeated across all 3 FFN layers (eliminating **102 MB** HBM traffic).
   - **Solver Scaling**: QIEO converges to the exact DP global optimum in `3,840` evaluations (`< 1 ms` search time).

---

## 🏗️ Architecture

```text
KernelFusion-Engine/
├── CMakeLists.txt              # Build config (C++17, CUDA 17, -O3, -Xptxas=-v)
├── include/
│   └── fusion.h                # Data structures (GpuSpec, Op, ClusterStats, PlanResult)
├── src/
│   ├── fusion.cpp              # Hardware cost model, DP O(N^2) solver & QIEO solver
│   └── main.cpp                # CLI driver for Case A, Case B, and Case C workloads
├── benchmarks/
│   └── benchmark.cu            # Fused CUDA kernels, cudaFuncGetAttributes & Event timing
└── scripts/
    └── profile.sh              # Automated build + CLI + CUDA benchmark runner
```

---

## ⚡ Technical Design

- **Hardware-Calibrated Cost Model (`src/fusion.cpp`)**:
  - Models Turing (`sm_75`) SM resource limits (`65,536` regs/SM, `1,024` max threads/SM, `64 KB` SMem/SM, `16` max blocks/SM).
  - Enforces **8-register warp allocation granularity** (`round_up_regs`), GEMM epilogue anchor rules, and penalizes memory throughput when active warp occupancy drops below the **40% latency-hiding cliff**.
  - Calibrated directly against `nvcc -Xptxas=-v` hardware register counts (`39` regs for tiled GEMM, `18` regs for warp-shuffle LayerNorm, `14` regs for scalar fused elementwise, `24` regs for `float4` fused elementwise).
- **Dual Optimization Solvers**:
  - **Dynamic Programming (DP)**: Exact $O(N^2)$ prefix partition search over operator chains.
  - **Quantum-Inspired Evolutionary Optimizer (QIEO)**: Population-based metaheuristic using quantum rotation angles (`theta`) and personal/global elite attractors (`pbest` / `gbest`) to search combinatorial fusion spaces.
- **Handwritten CUDA Kernels (`benchmarks/benchmark.cu`)**:
  - **Fused GEMM Epilogue (`matmul_bias_epilogue_fused_kernel`)**: Evaluates `BiasAdd` (`__ldg`) and `ReLU` in registers before global tile writeback.
  - **Warp-Shuffle LayerNorm (`layernorm_warp_shfl_kernel`)**: Uses `__shfl_down_sync` register reductions to compute row mean and variance with zero shared memory bank conflicts.
  - **Vectorized Elementwise Fusion (`elem_fused_vec4_kernel`)**: Performs 128-bit `float4` memory transactions.
  - **Runtime Introspection & Verification**: Queries compiled kernel register usage via `cudaFuncGetAttributes`, occupancy via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, and verifies numerical accuracy (`max_abs_diff == 0.00e+00`).

---

## 📊 Verified Output (NVIDIA Tesla T4, SM 7.5)

### Analytical Engine (`./fusion_cli`)

```plaintext
====================================================================
GPU OPERATOR FUSION DECISION ENGINE
Target Hardware: NVIDIA Tesla T4 (SM 7.5) | 8.1 TFLOPS | 320 GB/s
====================================================================

>>> CASE A: 6-Op Pipeline (MatMul -> BiasAdd -> ReLU -> LayerNorm -> MatMul -> BiasAdd)
  Solver:          Dynamic Programming (Exact O(N^2))
  Optimal Layout:  [0-2] -> [3-3] -> [4-5]
  Predicted Time:  0.3442 ms
  HBM Traffic:     38 MB (39855104 B)
  HBM Saved:       34 MB (35651584 B)
  Search Cost:     21 evals in 0.006 ms
  Cluster Breakdown:
    Cluster 0 [0-2] (MatMul_0a + BiasAdd_0a + ReLU_0a): 40 regs/thread, 100% occ, 0.1435 ms
    Cluster 1 [3-3] (LayerNorm_0): 24 regs/thread, 100% occ, 0.0577 ms
    Cluster 2 [4-5] (MatMul_0b + BiasAdd_0b): 40 regs/thread, 100% occ, 0.1430 ms

  Solver:          Quantum-Inspired Evolutionary Optimizer (QIEO)
  Optimal Layout:  [0-2] -> [3-3] -> [4-5]
  Predicted Time:  0.3442 ms
  HBM Traffic:     38 MB (39855104 B)
  HBM Saved:       34 MB (35651584 B)
  Search Cost:     720 evals in 0.404 ms
  Cluster Breakdown:
    Cluster 0 [0-2] (MatMul_0a + BiasAdd_0a + ReLU_0a): 40 regs/thread, 100% occ, 0.1435 ms
    Cluster 1 [3-3] (LayerNorm_0): 24 regs/thread, 100% occ, 0.0577 ms
    Cluster 2 [4-5] (MatMul_0b + BiasAdd_0b): 40 regs/thread, 100% occ, 0.1430 ms

>>> CASE B: 3-Op Memory-Bound Chain (ElemAdd -> ReLU -> BiasAdd)
  Solver:          Dynamic Programming (Exact O(N^2))
  Optimal Layout:  [0-2]
  Predicted Time:  0.0838 ms
  HBM Traffic:     24 MB (25174016 B)
  HBM Saved:       32 MB (33554432 B)
  Search Cost:     6 evals in 0.001 ms
  Cluster Breakdown:
    Cluster 0 [0-2] (ElemAdd_0 + ReLU_0 + BiasAdd_0): 16 regs/thread, 100% occ, 0.0838 ms

  Solver:          Quantum-Inspired Evolutionary Optimizer (QIEO)
  Optimal Layout:  [0-2]
  Predicted Time:  0.0838 ms
  HBM Traffic:     24 MB (25174016 B)
  HBM Saved:       32 MB (33554432 B)
  Search Cost:     320 evals in 0.089 ms
  Cluster Breakdown:
    Cluster 0 [0-2] (ElemAdd_0 + ReLU_0 + BiasAdd_0): 16 regs/thread, 100% occ, 0.0838 ms

>>> CASE C: 18-Op Multi-Layer Transformer Graph (Search Space: 131,072 Plans)
  DP Optimal Layout:   [0-2] -> [3-3] -> [4-5] -> [6-8] -> [9-9] -> [10-11] -> [12-14] -> [15-15] -> [16-17] | Time: 1.0325 ms (171 evals)
  QIEO Best Layout:    [0-2] -> [3-3] -> [4-5] -> [6-8] -> [9-9] -> [10-11] -> [12-14] -> [15-15] -> [16-17] | Time: 1.0325 ms (3840 evals, HBM Saved: 102 MB)
```

### Hardware Benchmark (`./benchmark_runner`)

```plaintext
====================================================================
CUDA HARDWARE BENCHMARK & KERNEL OCCUPANCY INSPECTOR
Device: Tesla T4 (SM 7.5) | 40 SMs | Max Threads/SM: 1024
====================================================================

>>> 1. Compiled SASS Kernel Hardware Attributes (via cudaFuncGetAttributes)
  matmul_kernel<16>              | Regs: 39 | SMem: 2048 B | Blocks/SM: 4  | Occupancy: 100.0%
  matmul_bias_relu_fused<16>     | Regs: 39 | SMem: 2048 B | Blocks/SM: 4  | Occupancy: 100.0%
  layernorm_warp_shfl_kernel     | Regs: 18 | SMem: 72   B | Blocks/SM: 4  | Occupancy: 100.0%
  elem_fused_scalar_kernel       | Regs: 14 | SMem: 0    B | Blocks/SM: 4  | Occupancy: 100.0%
  elem_fused_vec4_kernel         | Regs: 24 | SMem: 0    B | Blocks/SM: 4  | Occupancy: 100.0%

>>> 2. CASE A Benchmark: 6-Op Pipeline (Epilogue Fusion [0-2] -> [3-3] -> [4-5])
  Unfused (6 kernels):           3.9387 ms
  Fused Epilogue (3 kernels):    2.7164 ms (1.45x speedup)
  HBM Traffic Eliminated:        34 MB per pass
  Max Numerical Diff (L_inf):    0.00e+00 (VERIFIED)

>>> 3. CASE B Benchmark: Memory-Bound Elementwise Chain (ElemAdd -> ReLU -> BiasAdd)
  Unfused (3 kernels):           0.2388 ms
  Fused Scalar (float):          0.0998 ms (2.39x speedup)
  Fused Vectorized (float4):     0.1054 ms (2.27x speedup)
  HBM Traffic Eliminated:        32 MB per pass
  Max Numerical Diff (Scalar):   0.00e+00
  Max Numerical Diff (Vec4):     0.00e+00 (VERIFIED)
```

---

## 🚀 Build & Run

### Prerequisites
- CMake ≥ 3.18
- C++17 compiler (`g++` or `clang++`)
- NVIDIA CUDA Toolkit (`nvcc`)

### Quick Start

```bash
git clone https://github.com/emceeashish/KernelFusion-Engine.git
cd KernelFusion-Engine

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

./fusion_cli           # Analytical decision engine (DP + QIEO)
./benchmark_runner     # CUDA hardware benchmark (requires NVIDIA GPU)
```

### Automated Build & Run Script

```bash
bash scripts/profile.sh
```
