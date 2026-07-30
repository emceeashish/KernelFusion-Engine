# GPU Operator Fusion Decision Engine — Build Configuration
# CMake 3.24+ required for first-class CUDA architecture targeting and C++17 interop.
cmake_minimum_required(VERSION 3.24)
project(gpu_operator_fusion_engine LANGUAGES CXX CUDA)

# C++17 / CUDA 17: needed for constexpr if, std::optional, and modern template metaprogramming
# in the analytical cost model while keeping direct CUDA runtime API compatibility.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default Release. -O3 must be on by default so host-side graph analysis overhead
# doesn't skew analytical solver timing benchmarks.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Decoupled graph engine into a static library so it links into both the CLI driver
# and future unit-test / Python-binding targets without recompilation.
add_library(fusion_core STATIC
    src/fusion_engine.cpp
)

target_include_directories(fusion_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# -Wall -Wextra -pedantic: catch type mismatches, alignment issues, and implicit
# conversions early — critical when modeling byte-level memory traffic.
target_compile_options(fusion_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -pedantic>
)

# Analytical Decision CLI — runs DP and QIEO solvers, outputs decision profiles.
add_executable(fusion_cli
    main.cpp
)
target_link_libraries(fusion_cli PRIVATE fusion_core)

# CUDA hardware micro-benchmark executable. Compiles handwritten kernels for on-device execution.
add_executable(benchmark_runner
    benchmarks/benchmark_runner.cu
)

# Separable compilation allows device-side helper functions to link across translation
# units cleanly if the benchmark suite expands.
set_target_properties(benchmark_runner PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# -O3 to NVCC: enables max loop unrolling, register allocation optimization,
# and dead-code elimination in device kernels.
target_compile_options(benchmark_runner PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
