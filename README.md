# GPU Operator Fusion Decision Engine

High-performance C++/CUDA analytical engine and hardware benchmarking suite for evaluating and optimizing **operator fusion layouts** in deep learning compute graphs.

Combines an exact **Dynamic Programming (DP)** solver with a **Quantum-Inspired Evolutionary Optimizer (QIEO)** to predict optimal fusion boundary points by modeling register allocation, SM warp occupancy, cache residency, and HBM memory traffic.

---

## 🌟 Key Results

Demonstrates **100% decision alignment** between the C++ analytical model and physical silicon execution (Tesla T4 + RTX 3090 specs):

1. **Case A: Fusion Rejected (Occupancy Cliff)**
   - **Workload**: 6-Op Pipeline (`MatMul → BiasAdd → ReLU → LayerNorm → MatMul → BiasAdd`)
   - **Decision**: `[0-0] -> [1-1] -> [2-2] -> [3-3] -> [4-4] -> [5-5]` (**NO FUSION**)
   - **Why**: LayerNorm reduction adds +28 regs/thread, dropping warp occupancy below the 40% latency-hiding threshold. The analytical engine correctly rejects fusion.

2. **Case B: Fusion Wins (2.08× Speedup)**
   - **Workload**: 3-Op Memory-Bound Chain (`ElemAdd → ReLU → BiasAdd`)
   - **Decision**: `[0-2]` (**FULL FUSION**)
   - **Analytical prediction**: Eliminates 32 MB of intermediate HBM traffic.
   - **Physical benchmark (Tesla T4)**:
     - Unfused (3 kernels): `0.270 ms` — 3 HBM round-trips
     - Fused (1 kernel): `0.129 ms` — intermediates stay in 32-bit registers
     - **Speedup: 2.08×** 🚀

---

## 🏗️ Architecture

```
gpu-operator-fusion-engine/
├── CMakeLists.txt              # Build config (C++17, CUDA 17, -O3)
├── main.cpp                    # CLI driver for analytical simulation
├── include/
│   └── fusion_engine.h         # Public API & CalibrationSummary
├── src/
│   └── fusion_engine.cpp       # Cost model, DP & QIEO solvers
├── benchmarks/
│   └── benchmark_runner.cu     # CUDA hardware micro-benchmarks
└── scripts/
    └── profile.sh              # Build + execute + NCU/NSYS profiling
```

---

## ⚡ Technical Design

- **DAG Validation**: Tensor continuity, rank compatibility, producer-consumer ordering enforced before solver execution.
- **Op-Type-Aware Cost Model**:
  - Elementwise epilogues: +2 to +4 regs/thread (trivial RF forwarding).
  - Heavy reductions (LayerNorm): +28 regs/thread (shared-memory accumulators + spill).
  - Coalescing penalties modeled per 128-byte cache line transaction bounds.
  - SM block limits computed across threads, registers, shared memory, and arch caps.
- **Dual Solvers**:
  - **DP**: Exact O(N²) optimal partition search via prefix subproblems.
  - **QIEO**: Population-based metaheuristic with Q-bit state vectors and annealed rotation gates.
- **Mock CUDA Generator**: Emits structural `__global__` kernel shells for winning cluster schedules.
- **Hardware Benchmark**: Handwritten CUDA kernels timed with `cudaEventRecord()` — no host-side overhead contamination.

---

## 📊 Output Samples

### Analytical Engine (`fusion_cli`)

```plaintext
Case A (6-Op Pipeline):
  Plan layout:    [0-0] -> [1-1] -> [2-2] -> [3-3] -> [4-4] -> [5-5]
  Runtime:        0.37312 ms
  HBM saved:      0 bytes
  Verdict:        NO FUSION (occupancy cliff from LayerNorm reduction)

Case B (Elementwise Chain):
  Plan layout:    [0-2]
  Runtime:        0.03585 ms
  HBM saved:      33,554,432 bytes (32 MB)
  Verdict:        FULL FUSION (intermediates stay in registers)
```

### Hardware Benchmark (`benchmark_runner`, Tesla T4)

```plaintext
Unfused (3 kernels):    0.269688 ms
Fused   (1 kernel):     0.129393 ms
Speedup:                2.08×
HBM eliminated:         ~32 MB
Verdict:                FUSION WINS
```

---

## 🚀 Build & Run

### Prerequisites
- CMake ≥ 3.24
- C++17 compiler (gcc 9+, clang 10+, MSVC 2019+)
- NVIDIA CUDA Toolkit (for hardware benchmarks)

### Quick Start

```bash
git clone https://github.com/emceeashish/KernelFusion-Engine.git
cd KernelFusion-Engine

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)

./fusion_cli           # Analytical decision engine
./benchmark_runner     # CUDA hardware benchmark (requires GPU)
```

### Automated Profiling

```bash
bash scripts/profile.sh
```

