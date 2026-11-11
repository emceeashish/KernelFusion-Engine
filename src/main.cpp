#include "fusion.h"
#include <iostream>
#include <vector>

// Builds the 6-op pipeline using exact base_regs measured from nvcc ptxas on Tesla T4:
// matmul=39 regs, bias_add=12 regs, relu=10 regs, layernorm=18 regs
static std::vector<Op> build_case_a_pipeline(int layer_id = 0) {
    const size_t M = 1024, K = 256, N = 2048, N1 = 1024;
    const size_t elem_sz = sizeof(float);

    const size_t bytes_2048 = M * N * elem_sz;  // 8 MB activation
    const size_t bytes_1024 = M * N1 * elem_sz; // 4 MB activation
    std::string s = std::to_string(layer_id);

    return {
        {"MatMul_" + s + "a",    MATMUL,    2.0 * M * K * N,  M * K * elem_sz, K * N * elem_sz,  bytes_2048, 256, 39, 2048},
        {"BiasAdd_" + s + "a",   BIAS_ADD,  double(M * N),    bytes_2048,      N * elem_sz,      bytes_2048, 256, 12, 0},
        {"ReLU_" + s + "a",      RELU,      double(M * N),    bytes_2048,      0,                bytes_2048, 256, 10, 0},
        {"LayerNorm_" + s,       LAYERNORM, 6.0 * M * N,      bytes_2048,      0,                bytes_2048, 256, 18, 72},
        {"MatMul_" + s + "b",    MATMUL,    2.0 * M * K * N1, bytes_2048,      K * N1 * elem_sz, bytes_1024, 256, 39, 2048},
        {"BiasAdd_" + s + "b",   BIAS_ADD,  double(M * N1),   bytes_1024,      N1 * elem_sz,     bytes_1024, 256, 12, 0}
    };
}

int main() {
    GpuSpec gpu = get_tesla_t4_spec();

    std::cout << "====================================================================\n"
              << "GPU OPERATOR FUSION DECISION ENGINE\n"
              << "Target Hardware: " << gpu.name << " | " << gpu.peak_tflops << " TFLOPS | "
              << gpu.bw_gbps << " GB/s\n"
              << "====================================================================\n\n";

    // -------------------------------------------------------------------------
    // CASE A: 6-Op Pipeline (Epilogue Fusion [0-2] -> [3-3] -> [4-5])
    // -------------------------------------------------------------------------
    std::cout << ">>> CASE A: 6-Op Pipeline (MatMul -> BiasAdd -> ReLU -> LayerNorm -> MatMul -> BiasAdd)\n";
    std::vector<Op> case_a = build_case_a_pipeline(0);

    PlanResult dp_a   = solve_dp(case_a, gpu);
    PlanResult qieo_a = solve_qieo(case_a, gpu, 24, 30);

    print_plan_report(case_a, dp_a);
    print_plan_report(case_a, qieo_a);

    // -------------------------------------------------------------------------
    // CASE B: 3-Op Memory-Bound Elementwise Chain (Full Fusion [0-2])
    // -------------------------------------------------------------------------
    std::cout << ">>> CASE B: 3-Op Memory-Bound Chain (ElemAdd -> ReLU -> BiasAdd)\n";
    const size_t N_elem = 1024 * 2048;
    const size_t tensor_bytes = N_elem * sizeof(float); // 8 MB per tensor

    std::vector<Op> case_b = {
        {"ElemAdd_0", ELEM_ADD, double(N_elem), tensor_bytes, tensor_bytes, tensor_bytes, 256, 12, 0},
        {"ReLU_0",    RELU,     double(N_elem), tensor_bytes, 0,            tensor_bytes, 256, 10, 0},
        {"BiasAdd_0", BIAS_ADD, double(N_elem), tensor_bytes, 2048 * 4,     tensor_bytes, 256, 12, 0}
    };

    PlanResult dp_b   = solve_dp(case_b, gpu);
    PlanResult qieo_b = solve_qieo(case_b, gpu, 16, 20);

    print_plan_report(case_b, dp_b);
    print_plan_report(case_b, qieo_b);

    // -------------------------------------------------------------------------
    // CASE C: 18-Op 3-Layer Transformer Pipeline (2^17 = 131,072 States)
    // -------------------------------------------------------------------------
    std::cout << ">>> CASE C: 18-Op Multi-Layer Transformer Graph (Search Space: 131,072 Plans)\n";
    std::vector<Op> case_c;
    for (int layer = 0; layer < 3; layer++) {
        auto blk = build_case_a_pipeline(layer);
        case_c.insert(case_c.end(), blk.begin(), blk.end());
    }

    PlanResult dp_c   = solve_dp(case_c, gpu);
    PlanResult qieo_c = solve_qieo(case_c, gpu, 48, 80);

    std::cout << "  DP Optimal Layout:   " << format_layout(dp_c)
              << " | Time: " << dp_c.total_time_ms << " ms (" << dp_c.evals << " evals)\n";
    std::cout << "  QIEO Best Layout:    " << format_layout(qieo_c)
              << " | Time: " << qieo_c.total_time_ms << " ms (" << qieo_c.evals << " evals, "
              << "HBM Saved: " << (qieo_c.saved_hbm_bytes / (1024 * 1024)) << " MB)\n";

    return 0;
}
