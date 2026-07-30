#!/usr/bin/env bash
# profile.sh — Build and profile the GPU Operator Fusion Engine
#
# Automates: CMake configure → build → analytical CLI → CUDA benchmark → NCU/NSYS profiling.
# NCU captures per-kernel register pressure, SMem occupancy, and HBM throughput.
# NSYS records CUDA stream timelines, driver launch delays, and device sync gaps.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

echo "============================================================"
echo "GPU Operator Fusion Engine — Build & Profile"
echo "============================================================"

# ── 1. Build ─────────────────────────────────────────────────────
echo ""
echo "[1/4] Configuring CMake..."
# Release mode (-O3) to prevent host solver overhead from skewing latency measurements.
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release

echo ""
echo "[2/4] Building..."
# -j $(nproc): utilize all host CPU cores for parallel compilation.
cmake --build "${BUILD_DIR}" --config Release -j "$(nproc)"

# ── 2. Run analytical CLI ────────────────────────────────────────
echo ""
echo "[3/4] Running analytical fusion CLI..."
# Generates prediction CSV and console decision report for both workloads.
"${BUILD_DIR}/fusion_cli"

# ── 3. Run CUDA benchmark (if GPU available) ─────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "[4/4] Running CUDA hardware benchmark..."
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
    "${BUILD_DIR}/benchmark_runner"

    # Optional: Nsight Compute — inspects register pressure per warp, verifies HBM reduction in Case B.
    if command -v ncu &>/dev/null; then
        echo ""
        echo "[bonus] Nsight Compute kernel profiling..."
        ncu --set full \
            --target-processes all \
            --export "${BUILD_DIR}/ncu_report" \
            "${BUILD_DIR}/benchmark_runner"
        echo "Nsight Compute report saved to ${BUILD_DIR}/ncu_report.ncu-rep"
    fi

    # Optional: Nsight Systems — captures system-wide timeline traces and kernel launch queueing.
    if command -v nsys &>/dev/null; then
        echo ""
        echo "[bonus] Nsight Systems timeline profiling..."
        nsys profile \
            --output "${BUILD_DIR}/nsys_report" \
            --force-overwrite true \
            "${BUILD_DIR}/benchmark_runner"
        echo "Nsight Systems report saved to ${BUILD_DIR}/nsys_report.nsys-rep"
    fi
else
    echo ""
    echo "[4/4] Skipping CUDA benchmark — no GPU detected."
fi

echo ""
echo "============================================================"
echo "Profile complete."
echo "============================================================"
