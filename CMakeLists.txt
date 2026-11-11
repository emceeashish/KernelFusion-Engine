cmake_minimum_required(VERSION 3.18)
project(gpu_operator_fusion_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_library(fusion_core STATIC src/fusion.cpp)
target_include_directories(fusion_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(fusion_core PRIVATE -O3 -Wall -Wextra)

add_executable(fusion_cli src/main.cpp)
target_link_libraries(fusion_cli PRIVATE fusion_core)

add_executable(benchmark_runner benchmarks/benchmark.cu)
target_compile_options(benchmark_runner PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -Xptxas=-v>
)
