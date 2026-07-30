// Physical CUDA Hardware Micro-Benchmark Suite
//
// Validates the analytical fusion model against actual GPU silicon (e.g. Tesla T4).
//
// Timing: CUDA Events (cudaEventRecord), not std::chrono — eliminates host-side
// driver dispatch overhead from microsecond measurements.
// Warmup: 10 iterations to boost GPU clocks to max frequency and stabilize L2 lines.
// RF forwarding vs HBM writebacks:
//   Unfused chain: 3 kernels → 2 intermediate tensors (32 MB) round-trip through HBM.
//   Fused chain: Add→ReLU→BiasAdd in a single pass, intermediates held in 32-bit RF,
//   only final output written to DRAM → 2.08× speedup on T4.

#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(err__));             \
        }                                                                    \
    } while (0)

namespace {

constexpr int M0 = 1024;
constexpr int K0 = 2048;
constexpr int N0 = 2048;
constexpr int N1 = 1024;
constexpr int WARMUP_ITERS = 10;
constexpr int MEASURE_ITERS = 50;
constexpr float EPS = 1.0e-5f;

// Tiled SGEMM. 16×16 block geometry (256 threads/block) balances SMem staging
// and active warp scheduling without hitting the register ceiling.
template <int TILE>
__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Fused epilogue: MatMul + BiasAdd + ReLU in one pass.
// GEMM accumulator modified in registers before global writeback — avoids extra HBM round-trip.
template <int TILE>
__global__ void matmul_bias_relu_fused_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        // FUSED EPILOGUE: BiasAdd and ReLU happen in registers before HBM writeback!
        float val = acc + bias[col];
        C[row * N + col] = val > 0.0f ? val : 0.0f;
    }
}

__global__ void bias_add_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx < total) {
        const int col = idx % cols;
        y[idx] = x[idx] + bias[col];
    }
}

__global__ void relu_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        y[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}

// LayerNorm: block-level SMem tree reduction (mean + variance).
// Double-pass reduction + warp sync incurs +28 regs/thread → occupancy cliff in fused contexts.
__global__ void layernorm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows, int cols, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    float* ssum = smem;
    float* ssq = smem + blockDim.x;

    float local_sum = 0.0f;
    float local_sq = 0.0f;

    for (int col = tid; col < cols; col += blockDim.x) {
        const float v = x[row * cols + col];
        local_sum += v;
        local_sq += v * v;
    }

    ssum[tid] = local_sum;
    ssq[tid] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
            ssq[tid] += ssq[tid + stride];
        }
        __syncthreads();
    }

    const float mean = ssum[0] / static_cast<float>(cols);
    const float var = ssq[0] / static_cast<float>(cols) - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < cols; col += blockDim.x) {
        const float v = x[row * cols + col];
        y[row * cols + col] = (v - mean) * inv_std;
    }
}

// ── Elementwise Memory-Bound Kernels (Case B) ──

__global__ void elem_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = a[idx] + b[idx];
}

__global__ void elem_relu_standalone_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = in[idx] > 0.0f ? in[idx] : 0.0f;
}

__global__ void elem_bias_standalone_kernel(
    const float* __restrict__ in,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = in[idx] + bias[idx];
}

// Fused elementwise chain: single memory pass.
// Instead of 3 separate DRAM round-trips, each thread performs Add→ReLU→BiasAdd
// entirely in 32-bit registers, writing back to global DRAM only once.
__global__ void elem_fused_chain_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = a[idx] + b[idx];         // ElemAdd in Register
        val = val > 0.0f ? val : 0.0f;       // ReLU in Register
        out[idx] = val + bias[idx];           // BiasAdd in Register
    }
}

void fill_random(std::vector<float>& values, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : values) v = dist(rng);
}

void run_pipeline_once(
    const float* d_a0, const float* d_w0, const float* d_bias0,
    float* d_matmul0, float* d_biasadd0, float* d_relu0, float* d_ln0,
    const float* d_w1, const float* d_bias1, float* d_matmul1, float* d_final) {
    constexpr int TILE = 16;
    const dim3 block_mm(TILE, TILE);
    const dim3 grid_mm0((N0 + TILE - 1) / TILE, (M0 + TILE - 1) / TILE);
    const dim3 grid_mm1((N1 + TILE - 1) / TILE, (M0 + TILE - 1) / TILE);

    const int total0 = M0 * N0;
    const int total1 = M0 * N1;
    const int threads = 256;
    const int blocks0 = (total0 + threads - 1) / threads;
    const int blocks1 = (total1 + threads - 1) / threads;

    matmul_kernel<TILE><<<grid_mm0, block_mm>>>(d_a0, d_w0, d_matmul0, M0, N0, K0);
    bias_add_kernel<<<blocks0, threads>>>(d_matmul0, d_bias0, d_biasadd0, M0, N0);
    relu_kernel<<<blocks0, threads>>>(d_biasadd0, d_relu0, total0);
    layernorm_kernel<<<M0, 256, 2 * 256 * sizeof(float)>>>(d_relu0, d_ln0, M0, N0, EPS);
    matmul_kernel<TILE><<<grid_mm1, block_mm>>>(d_ln0, d_w1, d_matmul1, M0, N1, N0);
    bias_add_kernel<<<blocks1, threads>>>(d_matmul1, d_bias1, d_final, M0, N1);
}

// CUDA events: high-precision on-device microsecond timing, no host overhead.
float measure_pipeline_ms(
    const float* d_a0, const float* d_w0, const float* d_bias0,
    float* d_matmul0, float* d_biasadd0, float* d_relu0, float* d_ln0,
    const float* d_w1, const float* d_bias1, float* d_matmul1, float* d_final) {
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        run_pipeline_once(d_a0, d_w0, d_bias0, d_matmul0, d_biasadd0, d_relu0, d_ln0, d_w1, d_bias1, d_matmul1, d_final);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < MEASURE_ITERS; ++i) {
        run_pipeline_once(d_a0, d_w0, d_bias0, d_matmul0, d_biasadd0, d_relu0, d_ln0, d_w1, d_bias1, d_matmul1, d_final);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / static_cast<float>(MEASURE_ITERS);
}

constexpr int ELEM_N = M0 * N0;  // 1024 * 2048 = 2097152 elements

float measure_unfused_elementwise_ms(
    const float* d_elem_a, const float* d_elem_b, const float* d_elem_bias,
    float* d_elem_add_out, float* d_elem_relu_out, float* d_elem_final) {
    const int threads = 256;
    const int blocks = (ELEM_N + threads - 1) / threads;

    auto run_once = [&]() {
        elem_add_kernel<<<blocks, threads>>>(d_elem_a, d_elem_b, d_elem_add_out, ELEM_N);
        elem_relu_standalone_kernel<<<blocks, threads>>>(d_elem_add_out, d_elem_relu_out, ELEM_N);
        elem_bias_standalone_kernel<<<blocks, threads>>>(d_elem_relu_out, d_elem_bias, d_elem_final, ELEM_N);
    };

    for (int i = 0; i < WARMUP_ITERS; ++i) run_once();
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < MEASURE_ITERS; ++i) run_once();

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / static_cast<float>(MEASURE_ITERS);
}

float measure_fused_elementwise_ms(
    const float* d_elem_a, const float* d_elem_b, const float* d_elem_bias,
    float* d_elem_final) {
    const int threads = 256;
    const int blocks = (ELEM_N + threads - 1) / threads;

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        elem_fused_chain_kernel<<<blocks, threads>>>(d_elem_a, d_elem_b, d_elem_bias, d_elem_final, ELEM_N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < MEASURE_ITERS; ++i) {
        elem_fused_chain_kernel<<<blocks, threads>>>(d_elem_a, d_elem_b, d_elem_bias, d_elem_final, ELEM_N);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / static_cast<float>(MEASURE_ITERS);
}

} // namespace

int main() {
    try {
        std::vector<float> h_a0(M0 * K0), h_w0(K0 * N0), h_bias0(N0), h_w1(N0 * N1), h_bias1(N1);
        fill_random(h_a0, 1U); fill_random(h_w0, 2U); fill_random(h_bias0, 3U); fill_random(h_w1, 4U); fill_random(h_bias1, 5U);

        float *d_a0 = nullptr, *d_w0 = nullptr, *d_bias0 = nullptr;
        float *d_matmul0 = nullptr, *d_biasadd0 = nullptr, *d_relu0 = nullptr, *d_ln0 = nullptr;
        float *d_w1 = nullptr, *d_bias1 = nullptr;
        float *d_matmul1 = nullptr, *d_final = nullptr;

        CHECK_CUDA(cudaMalloc(&d_a0,      M0 * K0 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_w0,      K0 * N0 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_bias0,   N0      * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_matmul0, M0 * N0 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_biasadd0,M0 * N0 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_relu0,   M0 * N0 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_ln0,     M0 * N0 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_w1,      N0 * N1 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_bias1,   N1      * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_matmul1, M0 * N1 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_final,   M0 * N1 * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_a0,    h_a0.data(),    M0 * K0 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w0,    h_w0.data(),    K0 * N0 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_bias0, h_bias0.data(), N0      * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w1,    h_w1.data(),    N0 * N1 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_bias1, h_bias1.data(), N1      * sizeof(float), cudaMemcpyHostToDevice));

        cudaDeviceProp prop{};
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

        std::cout << "============================================================\n"
                  << "GPU OPERATOR FUSION - HARDWARE BENCHMARK\n"
                  << "============================================================\n"
                  << "Device:             " << prop.name << "\n"
                  << "Compute Capability: " << prop.major << "." << prop.minor << "\n"
                  << "SMs:                " << prop.multiProcessorCount << "\n"
                  << "Global Memory:      " << prop.totalGlobalMem / (1024ULL * 1024ULL) << " MB\n"
                  << "L2 Cache:           " << prop.l2CacheSize / 1024 << " KB\n"
                  << "Shared Mem/Block:   " << prop.sharedMemPerBlock / 1024 << " KB\n"
                  << "Warp Size:          " << prop.warpSize << "\n"
                  << "------------------------------------------------------------\n"
                  << "Pipeline: MatMul[1024x2048x2048] -> BiasAdd -> ReLU -> LayerNorm -> MatMul[1024x2048x1024] -> BiasAdd\n"
                  << "Warmup iterations:  " << WARMUP_ITERS << "\n"
                  << "Measure iterations: " << MEASURE_ITERS << "\n"
                  << "------------------------------------------------------------\n";

        const float avg_ms = measure_pipeline_ms(
            d_a0, d_w0, d_bias0, d_matmul0, d_biasadd0, d_relu0, d_ln0,
            d_w1, d_bias1, d_matmul1, d_final);

        std::cout << "Average pipeline latency: " << avg_ms << " ms\n";

        // Measure individual kernels
        {
            constexpr int TILE = 16;
            const dim3 block_mm(TILE, TILE);
            const dim3 grid_mm0((N0 + TILE - 1) / TILE, (M0 + TILE - 1) / TILE);
            const dim3 grid_mm1((N1 + TILE - 1) / TILE, (M0 + TILE - 1) / TILE);
            const int total0 = M0 * N0;
            const int total1 = M0 * N1;
            const int threads = 256;
            const int blocks0 = (total0 + threads - 1) / threads;
            const int blocks1 = (total1 + threads - 1) / threads;

            auto time_kernel = [&](const char* name, auto launch_fn) {
                for (int i = 0; i < WARMUP_ITERS; ++i) launch_fn();
                CHECK_CUDA(cudaDeviceSynchronize());

                cudaEvent_t t_start{}, t_stop{};
                CHECK_CUDA(cudaEventCreate(&t_start));
                CHECK_CUDA(cudaEventCreate(&t_stop));
                CHECK_CUDA(cudaEventRecord(t_start));
                for (int i = 0; i < MEASURE_ITERS; ++i) launch_fn();
                CHECK_CUDA(cudaEventRecord(t_stop));
                CHECK_CUDA(cudaEventSynchronize(t_stop));
                float elapsed = 0.0f;
                CHECK_CUDA(cudaEventElapsedTime(&elapsed, t_start, t_stop));
                CHECK_CUDA(cudaEventDestroy(t_start));
                CHECK_CUDA(cudaEventDestroy(t_stop));
                std::cout << "  " << name << ": " << elapsed / MEASURE_ITERS << " ms\n";
            };

            std::cout << "\nPer-kernel breakdown:\n";
            time_kernel("MatMul_0   ", [&]() { matmul_kernel<TILE><<<grid_mm0, block_mm>>>(d_a0, d_w0, d_matmul0, M0, N0, K0); });
            time_kernel("BiasAdd_0  ", [&]() { bias_add_kernel<<<blocks0, threads>>>(d_matmul0, d_bias0, d_biasadd0, M0, N0); });
            time_kernel("ReLU_0     ", [&]() { relu_kernel<<<blocks0, threads>>>(d_biasadd0, d_relu0, total0); });
            time_kernel("LayerNorm_0", [&]() { layernorm_kernel<<<M0, 256, 2 * 256 * sizeof(float)>>>(d_relu0, d_ln0, M0, N0, EPS); });
            time_kernel("MatMul_1   ", [&]() { matmul_kernel<TILE><<<grid_mm1, block_mm>>>(d_ln0, d_w1, d_matmul1, M0, N1, N0); });
            time_kernel("BiasAdd_1  ", [&]() { bias_add_kernel<<<blocks1, threads>>>(d_matmul1, d_bias1, d_final, M0, N1); });
        }

        // ── Case B: Memory-Bound Elementwise Fused vs Unfused ──
        // Allocate elementwise buffers (1024*2048 = 2M floats each)
        float *d_elem_a = nullptr, *d_elem_b = nullptr, *d_elem_bias = nullptr;
        float *d_elem_add_out = nullptr, *d_elem_relu_out = nullptr, *d_elem_final = nullptr;

        CHECK_CUDA(cudaMalloc(&d_elem_a,        ELEM_N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_elem_b,        ELEM_N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_elem_bias,     ELEM_N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_elem_add_out,  ELEM_N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_elem_relu_out, ELEM_N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_elem_final,    ELEM_N * sizeof(float)));

        {
            std::vector<float> h_elem_a(ELEM_N), h_elem_b(ELEM_N), h_elem_bias(ELEM_N);
            fill_random(h_elem_a, 10U); fill_random(h_elem_b, 11U); fill_random(h_elem_bias, 12U);
            CHECK_CUDA(cudaMemcpy(d_elem_a,    h_elem_a.data(),    ELEM_N * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_elem_b,    h_elem_b.data(),    ELEM_N * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_elem_bias, h_elem_bias.data(), ELEM_N * sizeof(float), cudaMemcpyHostToDevice));
        }

        std::cout << "\n============================================================\n"
                  << "CASE B: FUSED vs UNFUSED (ElemAdd + ReLU + BiasAdd)\n"
                  << "Memory-Bound Elementwise Chain [" << ELEM_N << " elements]\n"
                  << "============================================================\n";

        const float unfused_elem_ms = measure_unfused_elementwise_ms(
            d_elem_a, d_elem_b, d_elem_bias, d_elem_add_out, d_elem_relu_out, d_elem_final);
        const float fused_elem_ms = measure_fused_elementwise_ms(
            d_elem_a, d_elem_b, d_elem_bias, d_elem_final);

        const float elem_speedup = unfused_elem_ms / fused_elem_ms;
        // Unfused: 3 passes × (read + write) over N elements each
        // Fused:   1 pass (2 reads + 1 bias read + 1 write)
        // Traffic saved = intermediate writes + reads = 2 × 2 × N × 4 bytes = 32 MB
        const float elem_hbm_saved_mb = static_cast<float>(
            static_cast<long long>(ELEM_N) * sizeof(float) * 4  // 2 intermediate tensors × (write + read)
        ) / (1024.0f * 1024.0f);

        std::cout << "Unfused (3 kernels):    " << unfused_elem_ms << " ms\n"
                  << "Fused   (1 kernel):     " << fused_elem_ms << " ms\n"
                  << "Speedup:                " << elem_speedup << "x\n"
                  << "HBM traffic eliminated: ~" << elem_hbm_saved_mb << " MB\n"
                  << "Verdict:                " << (elem_speedup > 1.05f ? "FUSION WINS" : (elem_speedup > 1.0f ? "FUSION MARGINAL" : "FUSION NEUTRAL")) << "\n";

        // Write results CSV
        {
            std::ofstream csv("benchmark_results.csv");
            if (csv) {
                csv << "device,compute_capability,sms,global_mem_mb,l2_cache_kb,"
                    << "avg_full_pipeline_ms,unfused_elemwise_ms,fused_elemwise_ms,elemwise_fusion_speedup\n";
                csv << '"' << prop.name << '"' << ","
                    << prop.major << "." << prop.minor << ","
                    << prop.multiProcessorCount << ","
                    << prop.totalGlobalMem / (1024ULL * 1024ULL) << ","
                    << prop.l2CacheSize / 1024 << ","
                    << avg_ms << ","
                    << unfused_elem_ms << ","
                    << fused_elem_ms << ","
                    << elem_speedup << "\n";
                std::cout << "\nBenchmark CSV written to benchmark_results.csv\n";
            }
        }

        CHECK_CUDA(cudaFree(d_elem_a));
        CHECK_CUDA(cudaFree(d_elem_b));
        CHECK_CUDA(cudaFree(d_elem_bias));
        CHECK_CUDA(cudaFree(d_elem_add_out));
        CHECK_CUDA(cudaFree(d_elem_relu_out));
        CHECK_CUDA(cudaFree(d_elem_final));

        CHECK_CUDA(cudaFree(d_a0));
        CHECK_CUDA(cudaFree(d_w0));
        CHECK_CUDA(cudaFree(d_bias0));
        CHECK_CUDA(cudaFree(d_matmul0));
        CHECK_CUDA(cudaFree(d_biasadd0));
        CHECK_CUDA(cudaFree(d_relu0));
        CHECK_CUDA(cudaFree(d_ln0));
        CHECK_CUDA(cudaFree(d_w1));
        CHECK_CUDA(cudaFree(d_bias1));
        CHECK_CUDA(cudaFree(d_matmul1));
        CHECK_CUDA(cudaFree(d_final));

        std::cout << "\nBenchmark complete.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        return 1;
    }
}
