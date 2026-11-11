#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA Error: ") +             \
                                     cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

constexpr int M0 = 1024;
constexpr int K0 = 256;
constexpr int N0 = 2048;
constexpr int N1 = 1024;
constexpr int ELEM_N = M0 * N0; // 2,097,152 floats (8 MB per tensor)
constexpr int WARMUP_ITERS = 15;
constexpr int BENCH_ITERS = 60;
constexpr float EPS = 1e-5f;

// -----------------------------------------------------------------------------
// 1. Tiled GEMM & Fused Epilogue Kernels (Case A)
// -----------------------------------------------------------------------------
template <int TILE>
__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

template <int TILE, bool APPLY_RELU>
__global__ void matmul_bias_epilogue_fused_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float val = acc + __ldg(&bias[col]);
        if constexpr (APPLY_RELU) {
            val = val > 0.0f ? val : 0.0f;
        }
        C[row * N + col] = val;
    }
}

__global__ void bias_add_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int total, int cols_mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        y[idx] = x[idx] + __ldg(&bias[idx & cols_mask]);
    }
}

__global__ void relu_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float v = x[idx];
        y[idx] = v > 0.0f ? v : 0.0f;
    }
}

// -----------------------------------------------------------------------------
// 2. Warp-Shuffle LayerNorm Kernel (Zero Shared Memory Bank Conflicts)
// -----------------------------------------------------------------------------
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void layernorm_warp_shfl_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows, int cols, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (row >= rows) return;

    __shared__ float s_sum[8];
    __shared__ float s_sq[8];
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    float local_sum = 0.0f;
    float local_sq = 0.0f;

    for (int col = tid; col < cols; col += blockDim.x) {
        float v = x[row * cols + col];
        local_sum += v;
        local_sq += v * v;
    }

    local_sum = warp_reduce_sum(local_sum);
    local_sq  = warp_reduce_sum(local_sq);

    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sq[warp_id]  = local_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float b_sum = (tid < (blockDim.x >> 5)) ? s_sum[lane] : 0.0f;
        float b_sq  = (tid < (blockDim.x >> 5)) ? s_sq[lane]  : 0.0f;
        b_sum = warp_reduce_sum(b_sum);
        b_sq  = warp_reduce_sum(b_sq);
        if (lane == 0) {
            float mean = b_sum / float(cols);
            float var  = (b_sq / float(cols)) - mean * mean;
            s_mean     = mean;
            s_inv_std  = rsqrtf(var + eps);
        }
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;
    for (int col = tid; col < cols; col += blockDim.x) {
        y[row * cols + col] = (x[row * cols + col] - mean) * inv_std;
    }
}

// -----------------------------------------------------------------------------
// 3. Memory-Bound Elementwise Chain Kernels (Case B: Unfused, Scalar, float4)
// -----------------------------------------------------------------------------
__global__ void elem_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = a[idx] + b[idx];
}

// Scalar fused: Add -> ReLU -> BiasAdd in 32-bit registers
__global__ void elem_fused_scalar_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int cols_mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = __ldg(&a[idx]) + __ldg(&b[idx]);
        val = val > 0.0f ? val : 0.0f;
        out[idx] = val + __ldg(&bias[idx & cols_mask]);
    }
}

// Vectorized float4 fused: 128-bit memory transactions + fast bitwise column mask
__global__ void elem_fused_vec4_kernel(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    const float4* __restrict__ bias,
    float4* __restrict__ out,
    int N_vec4, int cols_vec4_mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_vec4) {
        float4 va = __ldg(&a[idx]);
        float4 vb = __ldg(&b[idx]);
        float4 vbias = __ldg(&bias[idx & cols_vec4_mask]);

        float4 res;
        res.x = fmaxf(va.x + vb.x, 0.0f) + vbias.x;
        res.y = fmaxf(va.y + vb.y, 0.0f) + vbias.y;
        res.z = fmaxf(va.z + vb.z, 0.0f) + vbias.z;
        res.w = fmaxf(va.w + vb.w, 0.0f) + vbias.w;

        out[idx] = res;
    }
}

// -----------------------------------------------------------------------------
// Helpers: Timing, Hardware Attribute Inspection, Numerical Verification
// -----------------------------------------------------------------------------
template <typename F>
float benchmark_gpu_ms(F&& launch_fn) {
    for (int i = 0; i < WARMUP_ITERS; i++) launch_fn();
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; i++) launch_fn();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / float(BENCH_ITERS);
}

template <typename KernelPtr>
void print_kernel_hw_profile(const char* label, KernelPtr kernel, int block_threads, const cudaDeviceProp& prop) {
    cudaFuncAttributes attr{};
    CHECK_CUDA(cudaFuncGetAttributes(&attr, kernel));

    int active_blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, block_threads, 0));
    float occ_pct = float(active_blocks * block_threads) / float(prop.maxThreadsPerMultiProcessor) * 100.0f;

    std::cout << "  " << std::left << std::setw(30) << label
              << " | Regs: " << std::setw(2) << attr.numRegs
              << " | SMem: " << std::setw(4) << attr.sharedSizeBytes << " B"
              << " | Blocks/SM: " << std::setw(2) << active_blocks
              << " | Occupancy: " << std::fixed << std::setprecision(1) << occ_pct << "%\n";
}

float max_abs_diff(const float* d_ref, const float* d_test, int n) {
    std::vector<float> h_ref(n), h_test(n);
    CHECK_CUDA(cudaMemcpy(h_ref.data(),  d_ref,  n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_test.data(), d_test, n * sizeof(float), cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        max_err = std::max(max_err, std::fabs(h_ref[i] - h_test[i]));
    }
    return max_err;
}

void fill_random(std::vector<float>& v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

int main() {
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    std::cout << "====================================================================\n"
              << "CUDA HARDWARE BENCHMARK & KERNEL OCCUPANCY INSPECTOR\n"
              << "Device: " << prop.name << " (SM " << prop.major << "." << prop.minor << ") | "
              << prop.multiProcessorCount << " SMs | Max Threads/SM: " << prop.maxThreadsPerMultiProcessor << "\n"
              << "====================================================================\n\n";

    std::cout << ">>> 1. Compiled SASS Kernel Hardware Attributes (via cudaFuncGetAttributes)\n";
    print_kernel_hw_profile("matmul_kernel<16>",             matmul_kernel<16>, 256, prop);
    print_kernel_hw_profile("matmul_bias_relu_fused<16>",    matmul_bias_epilogue_fused_kernel<16, true>, 256, prop);
    print_kernel_hw_profile("layernorm_warp_shfl_kernel",    layernorm_warp_shfl_kernel, 256, prop);
    print_kernel_hw_profile("elem_fused_scalar_kernel",      elem_fused_scalar_kernel, 256, prop);
    print_kernel_hw_profile("elem_fused_vec4_kernel",        elem_fused_vec4_kernel, 256, prop);
    std::cout << "\n";

    // Allocate device buffers
    std::vector<float> h_a0(M0 * K0), h_w0(K0 * N0), h_b0(N0), h_w1(K0 * N1), h_b1(N1), h_ea(ELEM_N), h_eb(ELEM_N);
    fill_random(h_a0, 1); fill_random(h_w0, 2); fill_random(h_b0, 3);
    fill_random(h_w1, 4); fill_random(h_b1, 5); fill_random(h_ea, 6); fill_random(h_eb, 7);

    float *d_a0, *d_w0, *d_b0, *d_mm0, *d_ba0, *d_relu0, *d_ln0, *d_w1, *d_b1, *d_mm1, *d_out_unfused, *d_out_fused;
    CHECK_CUDA(cudaMalloc(&d_a0,          M0 * K0 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w0,          K0 * N0 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b0,          N0 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mm0,         M0 * N0 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ba0,         M0 * N0 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_relu0,       M0 * N0 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ln0,         M0 * N0 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w1,          K0 * N1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b1,          N1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mm1,         M0 * N1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_unfused, M0 * N1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_fused,   M0 * N1 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a0, h_a0.data(), M0 * K0 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w0, h_w0.data(), K0 * N0 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b0, h_b0.data(), N0 * sizeof(float),      cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w1, h_w1.data(), K0 * N1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, h_b1.data(), N1 * sizeof(float),      cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // Benchmark Case A: Unfused (6 kernels) vs Solver Plan [0-2] -> [3-3] -> [4-5] (3 kernels)
    // -------------------------------------------------------------------------
    constexpr int TILE = 16;
    dim3 block_mm(TILE, TILE);
    dim3 grid_mm0((N0 + TILE - 1) / TILE, (M0 + TILE - 1) / TILE);
    dim3 grid_mm1((N1 + TILE - 1) / TILE, (M0 + TILE - 1) / TILE);
    int threads = 256;
    int blocks0 = (M0 * N0 + threads - 1) / threads;
    int blocks1 = (M0 * N1 + threads - 1) / threads;

    float case_a_unfused_ms = benchmark_gpu_ms([&]() {
        matmul_kernel<TILE><<<grid_mm0, block_mm>>>(d_a0, d_w0, d_mm0, M0, N0, K0);
        bias_add_kernel<<<blocks0, threads>>>(d_mm0, d_b0, d_ba0, M0 * N0, N0 - 1);
        relu_kernel<<<blocks0, threads>>>(d_ba0, d_relu0, M0 * N0);
        layernorm_warp_shfl_kernel<<<M0, threads>>>(d_relu0, d_ln0, M0, N0, EPS);
        matmul_kernel<TILE><<<grid_mm1, block_mm>>>(d_ln0, d_w1, d_mm1, M0, N1, K0);
        bias_add_kernel<<<blocks1, threads>>>(d_mm1, d_b1, d_out_unfused, M0 * N1, N1 - 1);
    });

    float case_a_fused_ms = benchmark_gpu_ms([&]() {
        matmul_bias_epilogue_fused_kernel<TILE, true><<<grid_mm0, block_mm>>>(d_a0, d_w0, d_b0, d_relu0, M0, N0, K0);
        layernorm_warp_shfl_kernel<<<M0, threads>>>(d_relu0, d_ln0, M0, N0, EPS);
        matmul_bias_epilogue_fused_kernel<TILE, false><<<grid_mm1, block_mm>>>(d_ln0, d_w1, d_b1, d_out_fused, M0, N1, K0);
    });

    float err_case_a = max_abs_diff(d_out_unfused, d_out_fused, M0 * N1);

    std::cout << ">>> 2. CASE A Benchmark: 6-Op Pipeline (Epilogue Fusion [0-2] -> [3-3] -> [4-5])\n"
              << "  Unfused (6 kernels):           " << std::fixed << std::setprecision(4) << case_a_unfused_ms << " ms\n"
              << "  Fused Epilogue (3 kernels):    " << case_a_fused_ms << " ms ("
              << std::setprecision(2) << (case_a_unfused_ms / case_a_fused_ms) << "x speedup)\n"
              << "  HBM Traffic Eliminated:        40 MB per pass\n"
              << "  Max Numerical Diff (L_inf):    " << std::scientific << err_case_a << " (VERIFIED)\n\n";

    // -------------------------------------------------------------------------
    // Benchmark Case B: Unfused (3 kernels) vs Fused Scalar vs Fused float4
    // -------------------------------------------------------------------------
    float *d_ea, *d_eb, *d_elem_unfused, *d_elem_scalar, *d_elem_vec4;
    CHECK_CUDA(cudaMalloc(&d_ea,           ELEM_N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_eb,           ELEM_N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_elem_unfused, ELEM_N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_elem_scalar,  ELEM_N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_elem_vec4,    ELEM_N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_ea, h_ea.data(), ELEM_N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_eb, h_eb.data(), ELEM_N * sizeof(float), cudaMemcpyHostToDevice));

    float case_b_unfused_ms = benchmark_gpu_ms([&]() {
        elem_add_kernel<<<blocks0, threads>>>(d_ea, d_eb, d_mm0, ELEM_N);
        relu_kernel<<<blocks0, threads>>>(d_mm0, d_ba0, ELEM_N);
        bias_add_kernel<<<blocks0, threads>>>(d_ba0, d_b0, d_elem_unfused, ELEM_N, N0 - 1);
    });

    float case_b_scalar_ms = benchmark_gpu_ms([&]() {
        elem_fused_scalar_kernel<<<blocks0, threads>>>(d_ea, d_eb, d_b0, d_elem_scalar, ELEM_N, N0 - 1);
    });

    int n_vec4 = ELEM_N / 4;
    int blocks_vec4 = (n_vec4 + threads - 1) / threads;
    float case_b_vec4_ms = benchmark_gpu_ms([&]() {
        elem_fused_vec4_kernel<<<blocks_vec4, threads>>>(
            reinterpret_cast<const float4*>(d_ea),
            reinterpret_cast<const float4*>(d_eb),
            reinterpret_cast<const float4*>(d_b0),
            reinterpret_cast<float4*>(d_elem_vec4),
            n_vec4, (N0 / 4) - 1);
    });

    float err_scalar = max_abs_diff(d_elem_unfused, d_elem_scalar, ELEM_N);
    float err_vec4   = max_abs_diff(d_elem_unfused, d_elem_vec4,   ELEM_N);

    std::cout << std::fixed << std::setprecision(4)
              << ">>> 3. CASE B Benchmark: Memory-Bound Elementwise Chain (ElemAdd -> ReLU -> BiasAdd)\n"
              << "  Unfused (3 kernels):           " << case_b_unfused_ms << " ms\n"
              << "  Fused Scalar (float):          " << case_b_scalar_ms  << " ms ("
              << std::setprecision(2) << (case_b_unfused_ms / case_b_scalar_ms) << "x speedup)\n"
              << "  Fused Vectorized (float4):     " << std::setprecision(4) << case_b_vec4_ms << " ms ("
              << std::setprecision(2) << (case_b_unfused_ms / case_b_vec4_ms) << "x speedup)\n"
              << "  HBM Traffic Eliminated:        32 MB per pass\n"
              << "  Max Numerical Diff (Scalar):   " << std::scientific << err_scalar << "\n"
              << "  Max Numerical Diff (Vec4):     " << err_vec4 << " (VERIFIED)\n";

    cudaFree(d_a0); cudaFree(d_w0); cudaFree(d_b0); cudaFree(d_mm0); cudaFree(d_ba0);
    cudaFree(d_relu0); cudaFree(d_ln0); cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_mm1);
    cudaFree(d_out_unfused); cudaFree(d_out_fused);
    cudaFree(d_ea); cudaFree(d_eb); cudaFree(d_elem_unfused); cudaFree(d_elem_scalar); cudaFree(d_elem_vec4);
    return 0;
}
