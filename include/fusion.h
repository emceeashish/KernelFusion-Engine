#pragma once
#include <cstddef>
#include <string>
#include <vector>

enum OpType {
    MATMUL,
    BIAS_ADD,
    RELU,
    LAYERNORM,
    ELEM_ADD
};

struct GpuSpec {
    std::string name;
    double peak_tflops;
    double bw_gbps;
    int max_regs_per_sm;
    int max_threads_per_sm;
    int max_blocks_per_sm;
    size_t max_smem_per_sm;
};

struct Op {
    std::string name;
    OpType type;
    double flops;
    size_t in_bytes;      // Primary activation read from HBM
    size_t weight_bytes;  // Secondary inputs (weights, bias, residual)
    size_t out_bytes;     // Activation write to HBM
    int threads;          // Thread block size (e.g., 256)
    int base_regs;        // Standalone registers/thread (calibrated via ptxas)
    size_t smem_bytes;    // Standalone shared memory/block
};

struct ClusterStats {
    int l = 0;
    int r = 0;
    int regs_per_thread = 0;
    int active_blocks = 0;
    double occupancy = 0.0;
    size_t hbm_bytes = 0;
    double time_ms = 0.0;
};

struct PlanResult {
    std::string solver_name;
    std::vector<int> edges; // edges[i] == 1 fuses Op[i] with Op[i+1]
    std::vector<ClusterStats> clusters;
    double total_time_ms = 0.0;
    size_t total_hbm_bytes = 0;
    size_t saved_hbm_bytes = 0;
    int evals = 0;
    double search_ms = 0.0;
};

GpuSpec get_tesla_t4_spec();
GpuSpec get_rtx_3090_spec();

ClusterStats eval_cluster(const std::vector<Op>& ops, int l, int r, const GpuSpec& gpu);
PlanResult evaluate_plan(const std::vector<Op>& ops, const std::vector<int>& edges, const GpuSpec& gpu);

PlanResult solve_dp(const std::vector<Op>& ops, const GpuSpec& gpu);
PlanResult solve_qieo(const std::vector<Op>& ops, const GpuSpec& gpu, int pop_size = 48, int gens = 80);

std::string format_layout(const PlanResult& plan);
void print_plan_report(const std::vector<Op>& ops, const PlanResult& plan);
