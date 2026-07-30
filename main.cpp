#include "fusion_engine.h"
#include <exception>
#include <iostream>

// Lightweight CLI driver for the analytical fusion simulator.
// Two contrasting workloads run inside run_demo_engine():
//   Case A (6-Op w/ LayerNorm): demonstrates "Fusion Rejected" — occupancy cliff at +28 regs/thread.
//   Case B (3-Op elementwise):  demonstrates "Fusion Wins" — intermediates stay in RF, saves 32 MB HBM.

int main() {
    try {
        // Verbose mode: full DAG validation, buffer lifetime analysis, DP + QIEO decisions to stdout.
        const gpu_fusion::CalibrationSummary summary =
            gpu_fusion::run_demo_engine(true);

        // Serialize winning plan metrics to CSV for automated benchmarking pipelines.
        gpu_fusion::write_prediction_csv(
            summary,
            "prediction_results.csv"
        );

        std::cout << "\nPrediction CSV written to prediction_results.csv\n";
        return 0;
    } catch (const std::exception& error) {
        // Top-level catch: clean error reports if DAG validation fails or file I/O errors occur.
        std::cerr << "Fatal error: " << error.what() << '\n';
        return 1;
    }
}
