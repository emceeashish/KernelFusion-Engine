#pragma once

#include <cstddef>
#include <string>

// Minimal public API for the analytical fusion engine.
// Designed to be host-side-only, thread-safe, and consumable by CLI drivers
// or external profiling harnesses without pulling in internal graph types.

namespace gpu_fusion {

// Encapsulates all decision metrics produced by the optimization solvers.
// Captures winning layout, predicted latency, raw vs. modeled HBM traffic,
// and byte savings — enough to dump structured CSV reports and validate
// memory-bound vs compute-bound heuristics.
struct CalibrationSummary {
    std::string winning_optimizer;             // DP vs QIEO solver identifier
    std::string winning_layout;                // Partitioning layout (e.g. "[0-0] -> [1-1]" or "[0-2]")
    double predicted_runtime_ms{0.0};           // Analytical kernel execution latency prediction
    std::size_t nominal_hbm_bytes{0};          // Raw un-coalesced DRAM byte count
    double modeled_hbm_bytes{0.0};              // Hardware-adjusted traffic (coalescing + L2 caching)
    std::size_t intermediate_hbm_saved_bytes{0}; // Bytes eliminated by keeping intermediates in RF/SMem
};

// Main entry point. Runs graph analysis, DP solver, QIEO solver, and comparative output.
// verbose=true prints full DAG validity, lifetime allocations, and per-optimizer decisions.
CalibrationSummary run_demo_engine(
    bool verbose = true,
    const std::string& generated_cuda_file = "generated_fusion_layout_mock.cu"
);

// Persists prediction metrics to CSV for automated plotting and regression testing.
void write_prediction_csv(
    const CalibrationSummary& summary,
    const std::string& csv_path
);

} // namespace gpu_fusion
