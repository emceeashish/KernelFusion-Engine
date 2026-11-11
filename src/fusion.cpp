#include "fusion.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

// Turing SMs allocate registers per warp in granularity steps of 8 regs/thread
static int round_up_regs(int regs) {
    return ((regs + 7) / 8) * 8;
}

GpuSpec get_tesla_t4_spec() {
    // NVIDIA Tesla T4 (Turing SM 7.5): 40 SMs, 8.1 TFLOPS FP32, 320 GB/s GDDR6
    return {"NVIDIA Tesla T4 (SM 7.5)", 8.1, 320.0, 65536, 1024, 16, 64 * 1024};
}

ClusterStats eval_cluster(const std::vector<Op>& ops, int l, int r, const GpuSpec& gpu) {
    ClusterStats c{};
    c.l = l;
    c.r = r;

    int threads = ops[l].threads;
    int regs = ops[l].base_regs;
    size_t smem = ops[l].smem_bytes;
    double total_flops = 0.0;

    int matmul_count = 0;
    bool has_layernorm = false;

    for (int i = l; i <= r; i++) {
        // A tiled GEMM must be the anchor (first op) of a fused cluster;
        // upstream elementwise ops cannot be fused into a downstream GEMM prologue
        // without redundant shared-memory tile recomputation.
        if (i > l && ops[i].type == MATMUL) {
            c.time_ms = 1e9;
            return c;
        }
        if (ops[i].type == MATMUL) matmul_count++;
        if (ops[i].type == LAYERNORM) has_layernorm = true;
        total_flops += ops[i].flops;
    }

    // Cannot fuse GEMM + LayerNorm into one kernel without a grid-wide sync barrier
    // because a 16x16 GEMM tile only holds 16 columns of a 2048-column row reduction.
    if (matmul_count > 1 || (matmul_count > 0 && has_layernorm)) {
        c.time_ms = 1e9;
        return c;
    }

    // Estimate register pressure and shared memory for fused clusters
    bool pure_elemwise = (matmul_count == 0 && !has_layernorm);
    for (int i = l + 1; i <= r; i++) {
        regs = std::max(regs, ops[i].base_regs);
        smem = std::max(smem, ops[i].smem_bytes);

        if (has_layernorm) {
            // Fusing operators across a 2-pass row reduction (mean + variance)
            // forces each thread (256 threads/row -> 8 floats/thread) to buffer
            // intermediate activations live across warp barriers (>128 regs/thread),
            // dropping SM occupancy from 100% (4 blocks/SM) to 25% (1 block/SM).
            regs += 112;
            smem += threads * sizeof(float) * 8;
        } else if (pure_elemwise) {
            regs += 2; // In-register scalar/vec4 forwarding (Add -> ReLU -> Bias)
        } else {
            regs += 0; // Epilogue fused into GEMM accumulator at tile writeback
        }
    }

    c.regs_per_thread = round_up_regs(regs);
    if (c.regs_per_thread > 255) {
        c.time_ms = 1e9;
        return c;
    }

    // SM Occupancy calculation across hardware limits
    int blocks_by_threads = gpu.max_threads_per_sm / threads;
    int blocks_by_regs    = gpu.max_regs_per_sm / (threads * c.regs_per_thread);
    int blocks_by_smem    = smem > 0 ? int(gpu.max_smem_per_sm / smem) : gpu.max_blocks_per_sm;

    c.active_blocks = std::min({blocks_by_threads, blocks_by_regs, blocks_by_smem, gpu.max_blocks_per_sm});
    if (c.active_blocks <= 0) {
        c.time_ms = 1e9;
        return c;
    }

    c.occupancy = double(c.active_blocks * threads) / double(gpu.max_threads_per_sm);

    // HBM Traffic: Only read first op's input + external weights/biases, and write last op's output.
    // All intermediate tensors inside [l..r] remain in registers (0 HBM bytes).
    c.hbm_bytes = ops[l].in_bytes + ops[r].out_bytes;
    for (int i = l; i <= r; i++) {
        c.hbm_bytes += ops[i].weight_bytes;
    }

    double compute_ms = (total_flops / (gpu.peak_tflops * 1e12)) * 1e3;
    double mem_ms     = (double(c.hbm_bytes) / (gpu.bw_gbps * 1e9)) * 1e3;

    // Occupancy cliff: below 40% active warps, SM schedulers cannot hide HBM latency
    if (c.occupancy < 0.40) {
        mem_ms *= (0.40 / c.occupancy);
    }

    // Roofline latency with 15% imperfectly overlapped memory tail + 5 us launch overhead
    constexpr double kLaunchOverheadMs = 0.005;
    c.time_ms = std::max(compute_ms, mem_ms) + 0.15 * std::min(compute_ms, mem_ms) + kLaunchOverheadMs;
    return c;
}

PlanResult evaluate_plan(const std::vector<Op>& ops, const std::vector<int>& edges, const GpuSpec& gpu) {
    PlanResult res{};
    res.edges = edges;
    int n = int(ops.size());

    size_t unfused_hbm = 0;
    for (int i = 0; i < n; i++) {
        unfused_hbm += ops[i].in_bytes + ops[i].weight_bytes + ops[i].out_bytes;
    }

    int start = 0;
    for (int i = 0; i < n - 1; i++) {
        if (edges[i] == 0) {
            ClusterStats c = eval_cluster(ops, start, i, gpu);
            res.clusters.push_back(c);
            res.total_time_ms += c.time_ms;
            res.total_hbm_bytes += c.hbm_bytes;
            start = i + 1;
        }
    }
    ClusterStats last_c = eval_cluster(ops, start, n - 1, gpu);
    res.clusters.push_back(last_c);
    res.total_time_ms += last_c.time_ms;
    res.total_hbm_bytes += last_c.hbm_bytes;

    res.saved_hbm_bytes = (unfused_hbm > res.total_hbm_bytes) ? (unfused_hbm - res.total_hbm_bytes) : 0;
    return res;
}

PlanResult solve_dp(const std::vector<Op>& ops, const GpuSpec& gpu) {
    auto t0 = std::chrono::steady_clock::now();
    int n = int(ops.size());

    std::vector<double> dp(n, 1e18);
    std::vector<int> parent(n, -1);
    int evals = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double prefix = (j == 0) ? 0.0 : dp[j - 1];
            double cost = eval_cluster(ops, j, i, gpu).time_ms;
            evals++;
            if (prefix + cost < dp[i]) {
                dp[i] = prefix + cost;
                parent[i] = j;
            }
        }
    }

    std::vector<int> edges(n - 1, 0);
    int cur = n - 1;
    while (cur >= 0) {
        int start = parent[cur];
        for (int e = start; e < cur; e++) {
            edges[e] = 1;
        }
        cur = start - 1;
    }

    PlanResult res = evaluate_plan(ops, edges, gpu);
    res.solver_name = "Dynamic Programming (Exact O(N^2))";
    res.evals = evals;
    res.search_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

PlanResult solve_qieo(const std::vector<Op>& ops, const GpuSpec& gpu, int pop_size, int gens) {
    auto t0 = std::chrono::steady_clock::now();
    int num_edges = int(ops.size()) - 1;

    // Q-bit angle theta[p][e] initialized to pi/4 (sin^2(pi/4) = 0.50 probability)
    std::vector<std::vector<double>> theta(pop_size, std::vector<double>(num_edges, 3.1415926535 / 4.0));
    std::vector<PlanResult> pbest(pop_size);
    for (int p = 0; p < pop_size; p++) pbest[p].total_time_ms = 1e18;

    PlanResult gbest{};
    gbest.total_time_ms = 1e18;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    int evals = 0;

    for (int g = 0; g < gens; g++) {
        double rot_step = 0.06 * (1.0 - double(g) / double(gens)) + 0.005;

        for (int p = 0; p < pop_size; p++) {
            std::vector<int> cand_edges(num_edges);
            for (int e = 0; e < num_edges; e++) {
                double s = std::sin(theta[p][e]);
                cand_edges[e] = (dist(rng) < s * s) ? 1 : 0;
            }

            PlanResult cand = evaluate_plan(ops, cand_edges, gpu);
            evals++;

            if (cand.total_time_ms < pbest[p].total_time_ms) pbest[p] = cand;
            if (cand.total_time_ms < gbest.total_time_ms)    gbest = cand;

            // Rotate Q-bits toward personal best and global elite
            for (int e = 0; e < num_edges; e++) {
                int target_bit = (g % 2 == 0) ? gbest.edges[e] : pbest[p].edges[e];
                if (cand_edges[e] != target_bit) {
                    double dir = (target_bit == 1) ? 1.0 : -1.0;
                    theta[p][e] = std::clamp(theta[p][e] + dir * rot_step, 0.05, 1.52);
                }
            }
        }
    }

    gbest.solver_name = "Quantum-Inspired Evolutionary Optimizer (QIEO)";
    gbest.evals = evals;
    gbest.search_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return gbest;
}

std::string format_layout(const PlanResult& plan) {
    std::ostringstream oss;
    for (size_t i = 0; i < plan.clusters.size(); i++) {
        oss << "[" << plan.clusters[i].l << "-" << plan.clusters[i].r << "]";
        if (i + 1 < plan.clusters.size()) oss << " -> ";
    }
    return oss.str();
}

void print_plan_report(const std::vector<Op>& ops, const PlanResult& plan) {
    std::cout << "  Solver:          " << plan.solver_name << "\n"
              << "  Optimal Layout:  " << format_layout(plan) << "\n"
              << "  Predicted Time:  " << std::fixed << std::setprecision(4) << plan.total_time_ms << " ms\n"
              << "  HBM Traffic:     " << (plan.total_hbm_bytes / (1024 * 1024)) << " MB (" << plan.total_hbm_bytes << " B)\n"
              << "  HBM Saved:       " << (plan.saved_hbm_bytes / (1024 * 1024)) << " MB (" << plan.saved_hbm_bytes << " B)\n"
              << "  Search Cost:     " << plan.evals << " evals in " << std::setprecision(3) << plan.search_ms << " ms\n";

    std::cout << "  Cluster Breakdown:\n";
    for (size_t i = 0; i < plan.clusters.size(); i++) {
        const auto& c = plan.clusters[i];
        std::cout << "    Cluster " << i << " [" << c.l << "-" << c.r << "] (";
        for (int idx = c.l; idx <= c.r; idx++) {
            std::cout << ops[idx].name << (idx < c.r ? " + " : "");
        }
        std::cout << "): " << c.regs_per_thread << " regs/thread, "
                  << int(c.occupancy * 100.0) << "% occ, "
                  << std::setprecision(4) << c.time_ms << " ms\n";
    }
    std::cout << "\n";
}
