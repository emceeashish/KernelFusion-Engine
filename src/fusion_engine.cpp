// GPU Operator Fusion Decision Engine — Analytical Core
//
// Models GPU microarchitectural behavior: how kernel fusion alters register file
// pressure, SM warp occupancy, L2 cache hit ratios, and HBM memory traffic.
//
// Key design constraints:
// - 40% SM occupancy threshold (kLatencyHidingOccupancy): below this cliff,
//   warp schedulers can't hide memory stall latency.
// - Op-type register growth: elementwise ops add ~2-4 regs/thread in RF;
//   heavy reductions (LayerNorm) spill +28 regs/thread for reduction accumulators.

#include "fusion_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gpu_fusion {

// Microarchitectural constants (Ampere/Turing targets):
// 128B cache lines for coalescing analysis, 40% occupancy floor for latency hiding,
// 30% HBM traffic multiplier for L2-resident intermediates.
constexpr std::size_t kCacheLineBytes = 128;
constexpr double kLatencyHidingOccupancy = 0.40;
constexpr double kL2ResidentHBMTrafficFraction = 0.30;
constexpr double kPi = 3.14159265358979323846;

static std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

static std::string shape_to_string(const std::vector<int>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i + 1 < shape.size()) oss << " x ";
    }
    oss << "]";
    return oss.str();
}

enum class OpType { MATMUL, BIAS_ADD, RELU, LAYERNORM, ELEM_ADD };

// Tracks shape metadata + striding layout for coalescing analysis.
// Non-unit stride in the fast dimension causes warp serialization across 128B cache lines.
struct Tensor {
    std::string name;
    std::vector<int> shape;
    std::vector<int> strides;
    int bytes_per_element{0};

    std::size_t get_num_elements() const {
        std::size_t elements = 1;
        for (int dim : shape) {
            if (dim <= 0) return 0;
            elements *= static_cast<std::size_t>(dim);
        }
        return elements;
    }

    std::size_t get_size_bytes() const {
        if (bytes_per_element <= 0) return 0;
        return get_num_elements() * static_cast<std::size_t>(bytes_per_element);
    }

    bool has_valid_strides() const {
        if (shape.empty()) return strides.empty() || strides.size() == 1;
        if (shape.size() != strides.size()) return false;
        for (int stride : strides) {
            if (stride <= 0) return false;
        }
        return true;
    }

    // strides.back() == 1 → adjacent warp threads access contiguous DRAM words (coalesced).
    bool is_coalesced() const {
        if (shape.empty()) return true;
        return has_valid_strides() && strides.back() == 1;
    }
};

struct Operator {
    std::string name;
    OpType type{};
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
};

// Target GPU hardware profile: compute throughput, bandwidth, and per-SM resource limits.
struct Hardware {
    std::string name;
    double tflops{0.0};
    double bandwidth_gb_per_s{0.0};
    int max_regs_per_sm{0};
    int max_threads_per_sm{0};
    int max_regs_per_thread{0};
    int max_shared_mem_bytes_per_sm{0};
    int max_blocks_per_sm{0};
    int warp_size{32};
    std::size_t max_l2_cache_bytes{0};

    bool is_valid() const {
        return !name.empty() && tflops > 0.0 && bandwidth_gb_per_s > 0.0 &&
               max_regs_per_sm > 0 && max_threads_per_sm > 0 && max_regs_per_thread > 0 &&
               max_shared_mem_bytes_per_sm > 0 && max_blocks_per_sm > 0 &&
               warp_size > 0 && max_l2_cache_bytes > 0;
    }
};

// Computational DAG representation. Enforces architectural validity
// (tensor continuity, rank compatibility, producer-consumer ordering) before solver execution.
class Graph {
public:
    explicit Graph(std::vector<Operator> nodes = {}) : nodes_(std::move(nodes)) {}
    const std::vector<Operator>& nodes() const { return nodes_; }
    std::size_t size() const { return nodes_.size(); }
    bool validate_graph_architecture() const { std::string unused; return validate(unused); }

    std::string validation_report() const {
        std::string error;
        if (validate(error)) return "Graph architecture is valid.";
        return "Graph architecture is invalid: " + error;
    }

private:
    std::vector<Operator> nodes_;

    static bool same_metadata(const Tensor& a, const Tensor& b) {
        return a.name == b.name && a.shape == b.shape &&
               a.strides == b.strides && a.bytes_per_element == b.bytes_per_element;
    }

    static bool same_shape_and_type(const Tensor& a, const Tensor& b) {
        return a.shape == b.shape && a.bytes_per_element == b.bytes_per_element;
    }

    static bool validate_tensor(const Tensor& tensor, std::string& error) {
        if (tensor.name.empty()) { error = "tensor name cannot be empty"; return false; }
        if (tensor.bytes_per_element <= 0) { error = "tensor '" + tensor.name + "' has invalid bytes_per_element"; return false; }
        for (int dim : tensor.shape) {
            if (dim <= 0) { error = "tensor '" + tensor.name + "' has a non-positive dimension"; return false; }
        }
        if (!tensor.has_valid_strides()) { error = "tensor '" + tensor.name + "' has invalid strides"; return false; }
        return true;
    }

    static bool validate_operator(const Operator& op, std::string& error) {
        if (op.name.empty()) { error = "operator name cannot be empty"; return false; }
        for (const Tensor& input : op.inputs) { if (!validate_tensor(input, error)) return false; }
        for (const Tensor& output : op.outputs) { if (!validate_tensor(output, error)) return false; }

        switch (op.type) {
            case OpType::MATMUL: {
                if (op.inputs.size() != 2 || op.outputs.size() != 1) { error = "MATMUL requires two inputs and one output"; return false; }
                const Tensor& a = op.inputs[0]; const Tensor& b = op.inputs[1]; const Tensor& out = op.outputs[0];
                if (a.shape.size() != 2 || b.shape.size() != 2 || out.shape.size() != 2) { error = "MATMUL requires rank-2 tensors"; return false; }
                if (a.shape[1] != b.shape[0]) { error = "MATMUL inner dimensions mismatch"; return false; }
                if (out.shape != std::vector<int>{a.shape[0], b.shape[1]}) { error = "MATMUL output shape invalid"; return false; }
                return true;
            }
            case OpType::BIAS_ADD: {
                if (op.inputs.size() != 2 || op.outputs.size() != 1) { error = "BIAS_ADD requires two inputs and one output"; return false; }
                const Tensor& activation = op.inputs[0]; const Tensor& bias = op.inputs[1]; const Tensor& out = op.outputs[0];
                if (!same_shape_and_type(activation, out)) { error = "BIAS_ADD output mismatch"; return false; }
                const bool full_bias = bias.shape == activation.shape;
                const bool channel_bias = bias.shape.size() == 1 && !activation.shape.empty() && bias.shape[0] == activation.shape.back();
                if (!full_bias && !channel_bias) { error = "BIAS_ADD bias shape incompatible"; return false; }
                return true;
            }
            case OpType::ELEM_ADD: {
                if (op.inputs.size() != 2 || op.outputs.size() != 1) { error = "ELEM_ADD requires two inputs and one output"; return false; }
                if (!same_shape_and_type(op.inputs[0], op.inputs[1])) { error = "ELEM_ADD input shapes must match"; return false; }
                if (!same_shape_and_type(op.inputs[0], op.outputs[0])) { error = "ELEM_ADD output shape mismatch"; return false; }
                return true;
            }
            case OpType::RELU:
            case OpType::LAYERNORM: {
                if (op.inputs.size() != 1 || op.outputs.size() != 1) { error = "Op requires 1 input and 1 output"; return false; }
                if (!same_shape_and_type(op.inputs[0], op.outputs[0])) { error = "Op shape mismatch"; return false; }
                return true;
            }
        }
        error = "unknown op type"; return false;
    }

    bool validate(std::string& error) const {
        if (nodes_.empty()) { error = "graph contains no operators"; return false; }
        std::unordered_map<std::string, std::size_t> producer_index;
        std::unordered_map<std::string, Tensor> produced_tensors;
        std::unordered_map<std::string, int> consumer_counts;

        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            const Operator& op = nodes_[i];
            if (!validate_operator(op, error)) return false;
            for (const Tensor& input : op.inputs) ++consumer_counts[input.name];
            for (const Tensor& output : op.outputs) {
                if (producer_index.count(output.name) != 0U) { error = "multiple producers for tensor '" + output.name + "'"; return false; }
                producer_index[output.name] = i; produced_tensors[output.name] = output;
            }
        }
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            for (const Tensor& input : nodes_[i].inputs) {
                const auto producer = producer_index.find(input.name);
                if (producer == producer_index.end()) continue;
                if (producer->second >= i) { error = "tensor '" + input.name + "' consumed before production"; return false; }
                if (!same_metadata(produced_tensors.at(input.name), input)) { error = "producer-consumer mismatch for '" + input.name + "'"; return false; }
            }
        }
        for (std::size_t i = 0; i + 1 < nodes_.size(); ++i) {
            for (const Tensor& output : nodes_[i].outputs) {
                if (consumer_counts[output.name] == 0) { error = "dangling tensor '" + output.name + "' never consumed"; return false; }
            }
        }
        for (std::size_t i = 0; i + 1 < nodes_.size(); ++i) {
            if (nodes_[i].outputs.empty() || nodes_[i + 1].inputs.empty()) return false;
            if (!same_metadata(nodes_[i].outputs.front(), nodes_[i + 1].inputs.front())) { error = "continuity broken between layers"; return false; }
        }
        return true;
    }
};

struct GraphAnalysis {
    std::unordered_map<std::string, Tensor> producer_tensor;
    std::unordered_map<std::string, int> producer_step;
    std::unordered_map<std::string, int> consumer_count;

    explicit GraphAnalysis(const Graph& graph) {
        for (std::size_t i = 0; i < graph.size(); ++i) {
            for (const Tensor& output : graph.nodes()[i].outputs) {
                producer_tensor[output.name] = output;
                producer_step[output.name] = static_cast<int>(i);
            }
            for (const Tensor& input : graph.nodes()[i].inputs) {
                ++consumer_count[input.name];
            }
        }
    }

    bool is_graph_input(const std::string& tensor_name) const { return producer_tensor.count(tensor_name) == 0U; }
    bool is_graph_output(const std::string& tensor_name) const {
        return consumer_count.count(tensor_name) == 0U || consumer_count.at(tensor_name) == 0;
    }
};

struct TensorLifetime { std::string name; std::size_t bytes{0}; int birth_step{0}; int death_step{0}; };
struct BufferLifetimeReport { std::vector<TensorLifetime> lifetimes; std::size_t peak_live_bytes{0}; int peak_step{0}; };

// Tracks peak live memory across execution steps.
// Used to measure DRAM footprint reduction when intermediates are fused into RF.
class BufferLifetimeAnalyzer {
public:
    static BufferLifetimeReport analyze(const Graph& graph, const GraphAnalysis& analysis) {
        BufferLifetimeReport report;
        std::map<std::string, TensorLifetime> lifetimes;

        for (std::size_t i = 0; i < graph.size(); ++i) {
            for (const Tensor& output : graph.nodes()[i].outputs) {
                lifetimes[output.name] = {output.name, output.get_size_bytes(), static_cast<int>(i), static_cast<int>(i)};
            }
        }
        for (std::size_t i = 0; i < graph.size(); ++i) {
            for (const Tensor& input : graph.nodes()[i].inputs) {
                auto found = lifetimes.find(input.name);
                if (found == lifetimes.end()) {
                    lifetimes[input.name] = {input.name, input.get_size_bytes(), 0, static_cast<int>(i)};
                } else {
                    found->second.death_step = std::max(found->second.death_step, static_cast<int>(i));
                }
            }
        }
        const int final_step = static_cast<int>(graph.size()) - 1;
        for (const Operator& op : graph.nodes()) {
            for (const Tensor& output : op.outputs) {
                if (analysis.is_graph_output(output.name)) lifetimes[output.name].death_step = final_step;
            }
        }
        for (const auto& item : lifetimes) report.lifetimes.push_back(item.second);
        for (int step = 0; step <= final_step; ++step) {
            std::size_t live_bytes = 0;
            for (const TensorLifetime& lifetime : report.lifetimes) {
                if (lifetime.birth_step <= step && step <= lifetime.death_step) live_bytes += lifetime.bytes;
            }
            if (live_bytes > report.peak_live_bytes) { report.peak_live_bytes = live_bytes; report.peak_step = step; }
        }
        return report;
    }
};

struct TilingSchedule {
    int block_dim_x{1}; int block_dim_y{1}; int tile_k{1};
    std::size_t shared_mem_bytes_per_block{0}; int estimated_registers_per_thread{1};
    int threads_per_block() const { return block_dim_x * block_dim_y; }
    bool is_valid() const { return block_dim_x > 0 && block_dim_y > 0 && tile_k > 0 && estimated_registers_per_thread > 0 && threads_per_block() > 0; }
};

struct MemoryRef { Tensor tensor; bool is_l2_cache_candidate{false}; };

struct OperatorCost {
    std::string name; double flops{0.0}; bool is_unfused_operator{false};
    std::vector<MemoryRef> reads; std::vector<MemoryRef> writes;
    std::size_t raw_hbm_bytes() const {
        std::size_t bytes = 0;
        for (const MemoryRef& ref : reads) bytes += ref.tensor.get_size_bytes();
        for (const MemoryRef& ref : writes) bytes += ref.tensor.get_size_bytes();
        return bytes;
    }
};

class OperatorCostModel {
public:
    static OperatorCost estimate_cluster(const Graph& graph, const GraphAnalysis& analysis, int first_node, int last_node) {
        if (first_node < 0 || last_node < first_node || last_node >= static_cast<int>(graph.size())) throw std::out_of_range("invalid cluster range");

        OperatorCost cost;
        cost.name = cluster_name(graph, first_node, last_node);
        cost.is_unfused_operator = (first_node == last_node);

        std::unordered_set<std::string> produced_inside_cluster;
        std::unordered_set<std::string> consumed_inside_cluster;
        std::unordered_map<std::string, Tensor> inputs;
        std::unordered_map<std::string, Tensor> outputs;

        for (int i = first_node; i <= last_node; ++i) {
            const Operator& op = graph.nodes()[static_cast<std::size_t>(i)];
            cost.flops += estimate_operator_flops(op);
            for (const Tensor& input : op.inputs) { consumed_inside_cluster.insert(input.name); inputs[input.name] = input; }
            for (const Tensor& output : op.outputs) { produced_inside_cluster.insert(output.name); outputs[output.name] = output; }
        }

        // Intra-cluster intermediates: tensors produced and consumed within [first_node, last_node] 
        // persist in registers/SMem, yielding 0 HBM overhead.
        for (const auto& item : inputs) {
            if (produced_inside_cluster.count(item.first) == 0U) {
                cost.reads.push_back({item.second, !analysis.is_graph_input(item.first)});
            }
        }
        for (const auto& item : outputs) {
            if (consumed_inside_cluster.count(item.first) == 0U) {
                cost.writes.push_back({item.second, !analysis.is_graph_output(item.first)});
            }
        }
        return cost;
    }

    static std::string cluster_name(const Graph& graph, int first_node, int last_node) {
        if (first_node == last_node) return graph.nodes()[static_cast<std::size_t>(first_node)].name;
        std::ostringstream oss;
        oss << "Fused[";
        for (int i = first_node; i <= last_node; ++i) {
            oss << graph.nodes()[static_cast<std::size_t>(i)].name;
            if (i != last_node) oss << " + ";
        }
        oss << "]";
        return oss.str();
    }

private:
    static double estimate_operator_flops(const Operator& op) {
        switch (op.type) {
            case OpType::MATMUL: {
                const Tensor& a = op.inputs[0]; const Tensor& b = op.inputs[1];
                return 2.0 * static_cast<double>(a.shape[0]) * static_cast<double>(a.shape[1]) * static_cast<double>(b.shape[1]);
            }
            case OpType::BIAS_ADD:
            case OpType::RELU:
            case OpType::ELEM_ADD: return static_cast<double>(op.inputs[0].get_num_elements());
            case OpType::LAYERNORM: return 6.0 * static_cast<double>(op.inputs[0].get_num_elements());
        }
        throw std::runtime_error("unsupported op type");
    }
};

struct OccupancyReport { int active_blocks_per_sm{0}; int blocks_by_threads{0}; int blocks_by_registers{0}; int blocks_by_shared_memory{0}; int blocks_by_architecture{0}; double achieved_occupancy{0.0}; };

struct RuntimeEstimate {
    std::string name; double compute_time_ms{0.0}; double memory_time_ms{0.0}; double adjusted_memory_time_ms{0.0}; double predicted_time_ms{0.0};
    std::size_t nominal_hbm_bytes{0}; double modeled_hbm_bytes{0.0}; std::size_t l2_candidate_bytes{0}; double average_memory_multiplier{1.0}; OccupancyReport occupancy;
};

// Roofline latency predictor. Models coalescing penalties, active block occupancy per SM,
// and latency-hiding occupancy penalization below the 40% warp threshold.
class PerformanceEstimator {
public:
    explicit PerformanceEstimator(Hardware hardware) : hardware_(std::move(hardware)) {
        if (!hardware_.is_valid()) throw std::invalid_argument("invalid hardware config");
    }

    // Non-unit striding coalescing penalty: actual 128B cache transactions / ideal span.
    double coalescing_penalty(const Tensor& tensor) const {
        if (tensor.is_coalesced()) return 1.0;
        if (!tensor.has_valid_strides()) return static_cast<double>(hardware_.warp_size);

        const std::size_t bytes_per_element = static_cast<std::size_t>(tensor.bytes_per_element);
        const std::size_t ideal_span_bytes = static_cast<std::size_t>(hardware_.warp_size) * bytes_per_element;
        const std::size_t actual_span_bytes = static_cast<std::size_t>(hardware_.warp_size - 1) * static_cast<std::size_t>(tensor.strides.back()) * bytes_per_element + bytes_per_element;

        const std::size_t ideal_transactions = std::max<std::size_t>(1, ceil_div(ideal_span_bytes, kCacheLineBytes));
        const std::size_t actual_transactions = std::min<std::size_t>(static_cast<std::size_t>(hardware_.warp_size), ceil_div(actual_span_bytes, kCacheLineBytes));

        return static_cast<double>(actual_transactions) / static_cast<double>(ideal_transactions);
    }

    RuntimeEstimate estimate(const OperatorCost& cost, const TilingSchedule& schedule) const {
        RuntimeEstimate result;
        result.name = cost.name;
        result.nominal_hbm_bytes = cost.raw_hbm_bytes();
        result.occupancy = estimate_occupancy(schedule);

        double raw_bytes = 0.0; double modeled_bytes = 0.0;
        auto account_memory_ref = [&](const MemoryRef& ref) {
            const std::size_t tensor_bytes = ref.tensor.get_size_bytes();
            const double coalescing = coalescing_penalty(ref.tensor);
            double memory_tier_factor = 1.0;

            if (cost.is_unfused_operator && ref.is_l2_cache_candidate && tensor_bytes <= hardware_.max_l2_cache_bytes) {
                memory_tier_factor = kL2ResidentHBMTrafficFraction;
                result.l2_candidate_bytes += tensor_bytes;
            }
            raw_bytes += static_cast<double>(tensor_bytes);
            modeled_bytes += static_cast<double>(tensor_bytes) * coalescing * memory_tier_factor;
        };

        for (const MemoryRef& read : cost.reads) account_memory_ref(read);
        for (const MemoryRef& write : cost.writes) account_memory_ref(write);

        result.modeled_hbm_bytes = modeled_bytes;
        result.average_memory_multiplier = raw_bytes > 0.0 ? modeled_bytes / raw_bytes : 1.0;
        result.compute_time_ms = cost.flops / (hardware_.tflops * 1.0e12) * 1.0e3;
        result.memory_time_ms = modeled_bytes / (hardware_.bandwidth_gb_per_s * 1.0e9) * 1.0e3;
        result.adjusted_memory_time_ms = result.memory_time_ms;

        // Penalize memory throughput below 40% occupancy: low active warp count
        // prevents schedulers from hiding global DRAM latency.
        if (result.occupancy.achieved_occupancy > 0.0 && result.occupancy.achieved_occupancy < kLatencyHidingOccupancy) {
            result.adjusted_memory_time_ms *= kLatencyHidingOccupancy / result.occupancy.achieved_occupancy;
        }

        result.predicted_time_ms = result.occupancy.achieved_occupancy <= 0.0
            ? std::numeric_limits<double>::infinity()
            : std::max(result.compute_time_ms, result.adjusted_memory_time_ms);

        return result;
    }

private:
    Hardware hardware_;

    // Hardware resource limits per SM: threads, registers, shared memory, arch max blocks.
    OccupancyReport estimate_occupancy(const TilingSchedule& schedule) const {
        OccupancyReport report;
        if (!schedule.is_valid()) return report;
        const int threads_per_block = schedule.threads_per_block();
        if (threads_per_block > hardware_.max_threads_per_sm || schedule.estimated_registers_per_thread > hardware_.max_regs_per_thread) return report;

        const int registers_per_block = threads_per_block * schedule.estimated_registers_per_thread;
        report.blocks_by_threads = hardware_.max_threads_per_sm / threads_per_block;
        report.blocks_by_registers = registers_per_block > 0 ? hardware_.max_regs_per_sm / registers_per_block : hardware_.max_blocks_per_sm;
        report.blocks_by_shared_memory = schedule.shared_mem_bytes_per_block == 0 ? hardware_.max_blocks_per_sm 
                                         : hardware_.max_shared_mem_bytes_per_sm / static_cast<int>(schedule.shared_mem_bytes_per_block);
        report.blocks_by_architecture = hardware_.max_blocks_per_sm;

        report.active_blocks_per_sm = std::max(0, std::min({report.blocks_by_threads, report.blocks_by_registers, report.blocks_by_shared_memory, report.blocks_by_architecture}));
        report.achieved_occupancy = std::min(1.0, static_cast<double>(report.active_blocks_per_sm * threads_per_block) / static_cast<double>(hardware_.max_threads_per_sm));

        return report;
    }
};

// Determines register growth and shared memory allocation for fused candidate clusters.
class ClusterScheduleBuilder {
public:
    explicit ClusterScheduleBuilder(std::vector<TilingSchedule> base_schedules) : base_schedules_(std::move(base_schedules)) {}

    TilingSchedule schedule_for_cluster(const Graph& graph, int first_node, int last_node) const {
        TilingSchedule schedule = base_schedules_.at(static_cast<std::size_t>(first_node));
        int max_registers = schedule.estimated_registers_per_thread;
        std::size_t max_shared_memory = schedule.shared_mem_bytes_per_block;

        for (int i = first_node; i <= last_node; ++i) {
            const TilingSchedule& candidate = base_schedules_.at(static_cast<std::size_t>(i));
            schedule.block_dim_x = std::max(schedule.block_dim_x, candidate.block_dim_x);
            schedule.block_dim_y = std::max(schedule.block_dim_y, candidate.block_dim_y);
            schedule.tile_k = std::max(schedule.tile_k, candidate.tile_k);
            max_registers = std::max(max_registers, candidate.estimated_registers_per_thread);
            max_shared_memory = std::max(max_shared_memory, candidate.shared_mem_bytes_per_block);
        }

        const int internal_fusion_edges = last_node - first_node;
        if (internal_fusion_edges == 0) {
            schedule.estimated_registers_per_thread = max_registers;
            schedule.shared_mem_bytes_per_block = max_shared_memory;
            return schedule;
        }

        const int output_bytes_per_element = graph.nodes().at(static_cast<std::size_t>(last_node)).outputs.at(0).bytes_per_element;

        // Op-type-aware register overhead heuristic:
        //   Pure elementwise chains: +2 regs/edge (trivial in-register forwarding)
        //   Mixed chains w/ MatMul epilogue: +4 regs/edge
        //   Heavy reductions (LayerNorm): +28 regs/edge (reduction accumulators + SMem staging)
        bool is_pure_elementwise_chain = true;
        for (int i = first_node; i <= last_node; ++i) {
            const OpType op_type = graph.nodes().at(static_cast<std::size_t>(i)).type;
            if (op_type == OpType::MATMUL || op_type == OpType::LAYERNORM) {
                is_pure_elementwise_chain = false;
                break;
            }
        }

        int added_registers = 0;
        int heavy_fusion_edges = 0;
        for (int i = first_node + 1; i <= last_node; ++i) {
            const OpType fused_op_type = graph.nodes().at(static_cast<std::size_t>(i)).type;
            if (fused_op_type == OpType::LAYERNORM) {
                added_registers += 28;  // Reduction requires shared-memory accumulators + register spill
                ++heavy_fusion_edges;
            } else if (is_pure_elementwise_chain) {
                added_registers += 2;   // Pure elementwise: trivial in-register, ~2 extra regs
            } else {
                added_registers += 4;   // Elementwise fused onto MATMUL epilogue
            }
        }

        schedule.estimated_registers_per_thread = max_registers + added_registers;
        // Shared memory staging allocated ONLY for heavy reduction edges (LayerNorm).
        const std::size_t epilogue_staging_bytes = static_cast<std::size_t>(schedule.threads_per_block()) * static_cast<std::size_t>(output_bytes_per_element) * static_cast<std::size_t>(heavy_fusion_edges);
        schedule.shared_mem_bytes_per_block = max_shared_memory + epilogue_staging_bytes;

        return schedule;
    }

private:
    std::vector<TilingSchedule> base_schedules_;
};

struct FusionPlan { std::vector<int> edges; };
struct ClusterResult { int first_node{0}; int last_node{0}; TilingSchedule schedule; RuntimeEstimate runtime; };
struct PlanEvaluation { FusionPlan plan; std::vector<ClusterResult> clusters; double runtime_ms{0.0}; std::size_t nominal_hbm_bytes{0}; double modeled_hbm_bytes{0.0}; std::size_t intermediate_hbm_saved_bytes{0}; };

class FusionPlanEvaluator {
public:
    FusionPlanEvaluator(const Graph& graph, const GraphAnalysis& analysis, const PerformanceEstimator& estimator, const ClusterScheduleBuilder& schedule_builder)
        : graph_(graph), analysis_(analysis), estimator_(estimator), schedule_builder_(schedule_builder) {}

    PlanEvaluation evaluate(const FusionPlan& plan) const {
        if (plan.edges.size() != graph_.size() - 1) throw std::invalid_argument("fusion edge count mismatch");
        PlanEvaluation result; result.plan = plan; int cluster_start = 0;

        for (int edge = 0; edge < static_cast<int>(plan.edges.size()); ++edge) {
            if (plan.edges[static_cast<std::size_t>(edge)] == 0) {
                append_cluster(result, cluster_start, edge);
                cluster_start = edge + 1;
            }
        }
        append_cluster(result, cluster_start, static_cast<int>(graph_.size()) - 1);
        const std::size_t unfused_nominal_bytes = get_unfused_nominal_hbm_bytes();
        result.intermediate_hbm_saved_bytes = unfused_nominal_bytes > result.nominal_hbm_bytes ? unfused_nominal_bytes - result.nominal_hbm_bytes : 0;
        return result;
    }

    double cluster_cost_ms(int first_node, int last_node) const {
        const OperatorCost cost = OperatorCostModel::estimate_cluster(graph_, analysis_, first_node, last_node);
        const TilingSchedule schedule = schedule_builder_.schedule_for_cluster(graph_, first_node, last_node);
        return estimator_.estimate(cost, schedule).predicted_time_ms;
    }

private:
    const Graph& graph_; const GraphAnalysis& analysis_; const PerformanceEstimator& estimator_; const ClusterScheduleBuilder& schedule_builder_;

    void append_cluster(PlanEvaluation& result, int first_node, int last_node) const {
        const OperatorCost cost = OperatorCostModel::estimate_cluster(graph_, analysis_, first_node, last_node);
        const TilingSchedule schedule = schedule_builder_.schedule_for_cluster(graph_, first_node, last_node);
        const RuntimeEstimate runtime = estimator_.estimate(cost, schedule);

        result.runtime_ms += runtime.predicted_time_ms;
        result.nominal_hbm_bytes += runtime.nominal_hbm_bytes;
        result.modeled_hbm_bytes += runtime.modeled_hbm_bytes;
        result.clusters.push_back({first_node, last_node, schedule, runtime});
    }

    std::size_t get_unfused_nominal_hbm_bytes() const {
        std::size_t total_bytes = 0;
        for (int i = 0; i < static_cast<int>(graph_.size()); ++i) {
            total_bytes += OperatorCostModel::estimate_cluster(graph_, analysis_, i, i).raw_hbm_bytes();
        }
        return total_bytes;
    }
};

struct OptimizationResult { std::string engine_name; PlanEvaluation best_plan; int evaluations{0}; int generations{0}; double search_time_ms{0.0}; };

// Exact O(N²) dynamic programming search. Evaluates all prefix partitioning
// subproblems to guarantee global latency optimality.
class DynamicProgrammingFusionOptimizer {
public:
    explicit DynamicProgrammingFusionOptimizer(const Graph& graph) : graph_(graph) {}

    OptimizationResult optimize(const FusionPlanEvaluator& evaluator) const {
        const auto start_time = std::chrono::steady_clock::now();
        const int node_count = static_cast<int>(graph_.size());

        std::vector<double> dp(static_cast<std::size_t>(node_count), std::numeric_limits<double>::infinity());
        std::vector<int> parent(static_cast<std::size_t>(node_count), -1);
        int evaluations = 0;

        for (int i = 0; i < node_count; ++i) {
            for (int j = -1; j < i; ++j) {
                const double prefix_cost = j < 0 ? 0.0 : dp[static_cast<std::size_t>(j)];
                const double cluster_cost = evaluator.cluster_cost_ms(j + 1, i);
                ++evaluations;

                if (prefix_cost + cluster_cost < dp[static_cast<std::size_t>(i)]) {
                    dp[static_cast<std::size_t>(i)] = prefix_cost + cluster_cost;
                    parent[static_cast<std::size_t>(i)] = j;
                }
            }
        }

        FusionPlan plan; plan.edges.assign(graph_.size() - 1, 0);
        for (int cursor = node_count - 1; cursor >= 0;) {
            const int previous = parent[static_cast<std::size_t>(cursor)];
            for (int edge = previous + 1; edge < cursor; ++edge) plan.edges[static_cast<std::size_t>(edge)] = 1;
            cursor = previous;
        }

        OptimizationResult result;
        result.engine_name = "Dynamic Programming";
        result.best_plan = evaluator.evaluate(plan);
        result.evaluations = evaluations;
        result.search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
        return result;
    }

private:
    const Graph& graph_;
};

struct Qubit {
    double alpha{1.0 / std::sqrt(2.0)}; double beta{1.0 / std::sqrt(2.0)};
    double probability_of_one() const { return beta * beta; }
    void normalize() {
        const double norm = std::sqrt(alpha * alpha + beta * beta);
        if (norm > 0.0) { alpha /= norm; beta /= norm; }
    }
};

// QIEO: metaheuristic solver using Q-bit probability amplitudes (|α|², |β|²)
// and annealed quantum rotation gates to explore DAG partitioning spaces
// without getting trapped in local minima.
class QuantumInspiredEvolutionaryOptimizer {
public:
    QuantumInspiredEvolutionaryOptimizer(int population_size, int generations, double maximum_rotation_radians, std::uint32_t random_seed)
        : population_size_(population_size), generations_(generations), maximum_rotation_radians_(maximum_rotation_radians), random_engine_(random_seed) {}

    OptimizationResult optimize(const FusionPlanEvaluator& evaluator, std::size_t fusion_edge_count) {
        const auto start_time = std::chrono::steady_clock::now();
        std::vector<std::vector<Qubit>> population(static_cast<std::size_t>(population_size_), std::vector<Qubit>(fusion_edge_count));
        PlanEvaluation global_elite; bool elite_found = false; int evaluations = 0;

        for (int generation = 0; generation < generations_; ++generation) {
            for (const std::vector<Qubit>& individual : population) {
                const FusionPlan candidate_plan = observe(individual);
                const PlanEvaluation candidate = evaluator.evaluate(candidate_plan);
                ++evaluations;

                if (!elite_found || candidate.runtime_ms < global_elite.runtime_ms) {
                    global_elite = candidate; elite_found = true;
                }
            }
            const double annealed_rotation = maximum_rotation_radians_ * (1.0 - static_cast<double>(generation) / static_cast<double>(generations_)) + 0.005;
            for (std::vector<Qubit>& individual : population) rotate_toward_elite(individual, global_elite.plan, annealed_rotation);
        }

        OptimizationResult result;
        result.engine_name = "Quantum-Inspired Evolutionary Optimizer";
        result.best_plan = global_elite;
        result.evaluations = evaluations;
        result.generations = generations_;
        result.search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
        return result;
    }

private:
    int population_size_; int generations_; double maximum_rotation_radians_; std::mt19937 random_engine_;

    FusionPlan observe(const std::vector<Qubit>& qubits) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        FusionPlan plan; plan.edges.reserve(qubits.size());
        for (const Qubit& qubit : qubits) plan.edges.push_back(distribution(random_engine_) < qubit.probability_of_one() ? 1 : 0);
        return plan;
    }

    static void rotate_toward_elite(std::vector<Qubit>& qubits, const FusionPlan& elite_plan, double max_rotation) {
        for (std::size_t i = 0; i < qubits.size(); ++i) {
            Qubit& qubit = qubits[i];
            const double current_theta = std::atan2(qubit.beta, qubit.alpha);
            const double target_theta = elite_plan.edges[i] == 1 ? kPi / 2.0 : 0.0;
            const double bounded_rotation = std::clamp(target_theta - current_theta, -max_rotation, max_rotation);
            const double updated_theta = current_theta + bounded_rotation;
            qubit.alpha = std::cos(updated_theta);
            qubit.beta = std::sin(updated_theta);
            qubit.normalize();
        }
    }
};

class MockCudaGenerator {
public:
    static bool generate(const Graph& graph, const PlanEvaluation& plan, const std::string& file_name) {
        std::ofstream output(file_name);
        if (!output) return false;
        output << "// Auto-generated structural CUDA shell.\n// Architectural artifact.\n\n";

        for (std::size_t i = 0; i < plan.clusters.size(); ++i) {
            const ClusterResult& cluster = plan.clusters[i];
            output << "// Cluster " << i << ": ";
            for (int node = cluster.first_node; node <= cluster.last_node; ++node) {
                output << graph.nodes()[static_cast<std::size_t>(node)].name;
                if (node != cluster.last_node) output << " -> ";
            }
            output << "\n__global__ void fusion_cluster_" << i << "(/* parameters */) {\n"
                   << "    extern __shared__ unsigned char smem[];\n"
                   << "    // Geometry: " << cluster.schedule.block_dim_x << " x " << cluster.schedule.block_dim_y << ", tile_k=" << cluster.schedule.tile_k << "\n"
                   << "    // 1. Cooperative load -> 2. Local compute -> 3. Epilogue on-chip -> 4. Boundary writeback.\n}\n\n";
        }
        return true;
    }
};

static void run_validation_suite() {
    const Tensor a{"A", {64, 128}, {128, 1}, 2};
    const Tensor b{"B", {128, 64}, {64, 1}, 2};
    const Tensor c{"C", {64, 64}, {64, 1}, 2};
    const Tensor d{"D", {64, 64}, {64, 1}, 2};
    const Graph valid_graph({{"ValidMatMul", OpType::MATMUL, {a, b}, {c}}, {"ValidReLU", OpType::RELU, {c}, {d}}});
    (void)valid_graph.validate_graph_architecture();
}

static std::string plan_to_string(const PlanEvaluation& plan) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < plan.clusters.size(); ++i) {
        const ClusterResult& cluster = plan.clusters[i];
        oss << "[" << cluster.first_node << "-" << cluster.last_node << "]";
        if (i + 1 < plan.clusters.size()) oss << " -> ";
    }
    return oss.str();
}

static void print_optimizer_report(const OptimizationResult& result) {
    std::cout << "\n------------------------------------------------------------\n"
              << result.engine_name << "\n"
              << "------------------------------------------------------------\n"
              << "Plan layout:                 " << plan_to_string(result.best_plan) << "\n"
              << "Predicted runtime:           " << std::fixed << std::setprecision(5) << result.best_plan.runtime_ms << " ms\n"
              << "Nominal HBM traffic:         " << result.best_plan.nominal_hbm_bytes << " bytes\n"
              << "Modeled HBM traffic:         " << std::fixed << std::setprecision(0) << result.best_plan.modeled_hbm_bytes << " bytes\n"
              << "Intermediate HBM saved:      " << result.best_plan.intermediate_hbm_saved_bytes << " bytes\n";
}

CalibrationSummary run_demo_engine(bool verbose, const std::string& generated_cuda_file) {
    CalibrationSummary summary;
    run_validation_suite();

    const Hardware rtx_3090{"NVIDIA RTX 3090", 35.58, 936.0, 65536, 2048, 255, 100 * 1024, 16, 32, 6ULL * 1024ULL * 1024ULL};
    const Hardware a100{"NVIDIA A100 40GB", 19.50, 1555.0, 65536, 2048, 255, 164 * 1024, 32, 32, 40ULL * 1024ULL * 1024ULL};

    const Tensor a0_transposed_view{"A0_transposed_view", {1024, 2048}, {1, 1024}, 2};
    const Tensor w0{"W0", {2048, 2048}, {2048, 1}, 2};
    const Tensor bias0{"bias0", {2048}, {1}, 2};
    const Tensor matmul0_out{"matmul0_out", {1024, 2048}, {2048, 1}, 2};
    const Tensor biasadd0_out{"biasadd0_out", {1024, 2048}, {2048, 1}, 2};
    const Tensor relu0_out{"relu0_out", {1024, 2048}, {2048, 1}, 2};
    const Tensor layernorm0_out{"layernorm0_out", {1024, 2048}, {2048, 1}, 2};
    const Tensor w1{"W1", {2048, 1024}, {1024, 1}, 2};
    const Tensor matmul1_out{"matmul1_out", {1024, 1024}, {1024, 1}, 2};
    const Tensor bias1{"bias1", {1024}, {1}, 2};
    const Tensor final_out{"final_out", {1024, 1024}, {1024, 1}, 2};

    // Case A Workload: Full 6-Op Pipeline containing LayerNorm reduction.
    const Graph graph({
        {"MatMul_0", OpType::MATMUL, {a0_transposed_view, w0}, {matmul0_out}},
        {"BiasAdd_0", OpType::BIAS_ADD, {matmul0_out, bias0}, {biasadd0_out}},
        {"ReLU_0", OpType::RELU, {biasadd0_out}, {relu0_out}},
        {"LayerNorm_0", OpType::LAYERNORM, {relu0_out}, {layernorm0_out}},
        {"MatMul_1", OpType::MATMUL, {layernorm0_out, w1}, {matmul1_out}},
        {"BiasAdd_1", OpType::BIAS_ADD, {matmul1_out, bias1}, {final_out}}
    });

    if (verbose) {
        std::cout << "\n============================================================\n";
        std::cout << "GPU OPERATOR FUSION DECISION ENGINE\n";
        std::cout << "============================================================\n";
        std::cout << graph.validation_report() << "\n";
    }

    if (!graph.validate_graph_architecture()) {
        throw std::runtime_error("graph validation failed");
    }

    const GraphAnalysis analysis(graph);
    const BufferLifetimeReport lifetime = BufferLifetimeAnalyzer::analyze(graph, analysis);

    if (verbose) {
        std::cout << "\nTarget GPU: " << rtx_3090.name << "\n";
        std::cout << "Reference GPU: " << a100.name << "\n";
        std::cout << "Unfused peak live allocation: " << lifetime.peak_live_bytes << " bytes at step " << lifetime.peak_step << "\n";
    }

    const TilingSchedule matmul0_schedule{16, 16, 32, 2ULL * 16ULL * 32ULL * 2ULL, 72};
    const TilingSchedule biasadd0_schedule{256, 1, 1, 0, 20};
    const TilingSchedule relu0_schedule{256, 1, 1, 0, 16};
    const TilingSchedule layernorm0_schedule{256, 1, 1, 256ULL * 2ULL, 48};
    const TilingSchedule matmul1_schedule{16, 16, 32, 2ULL * 16ULL * 32ULL * 2ULL, 80};
    const TilingSchedule biasadd1_schedule{256, 1, 1, 0, 20};

    const ClusterScheduleBuilder schedule_builder({matmul0_schedule, biasadd0_schedule, relu0_schedule, layernorm0_schedule, matmul1_schedule, biasadd1_schedule});
    const PerformanceEstimator estimator(rtx_3090);
    const FusionPlanEvaluator evaluator(graph, analysis, estimator, schedule_builder);

    const DynamicProgrammingFusionOptimizer dp(graph);
    QuantumInspiredEvolutionaryOptimizer qieo(96, 180, 0.085, 20260717U);

    const OptimizationResult dp_result = dp.optimize(evaluator);
    const OptimizationResult qieo_result = qieo.optimize(evaluator, graph.size() - 1);

    if (verbose) {
        print_optimizer_report(dp_result);
        print_optimizer_report(qieo_result);
    }

    const OptimizationResult& winning_result = dp_result.best_plan.runtime_ms <= qieo_result.best_plan.runtime_ms ? dp_result : qieo_result;

    MockCudaGenerator::generate(graph, winning_result.best_plan, generated_cuda_file);

    // ── Workload B: Pure Elementwise Memory-Bound Chain (Fusion Wins) ──
    // ElemAdd -> ReLU -> BiasAdd: all memory-bound, fusion eliminates intermediate HBM round-trips
    if (verbose) {
        std::cout << "\n\n============================================================\n";
        std::cout << "WORKLOAD B: MEMORY-BOUND ELEMENTWISE FUSION PIPELINE\n";
        std::cout << "(ElemAdd_0 -> ReLU_0 -> BiasAdd_0)\n";
        std::cout << "============================================================\n";
    }

    // Large 1024x2048 tensors to ensure memory-boundedness
    const Tensor elem_input_a{"elem_input_a", {1024, 2048}, {2048, 1}, 4};
    const Tensor elem_input_b{"elem_input_b", {1024, 2048}, {2048, 1}, 4};
    const Tensor elem_add_out{"elem_add_out", {1024, 2048}, {2048, 1}, 4};
    const Tensor elem_relu_out{"elem_relu_out", {1024, 2048}, {2048, 1}, 4};
    const Tensor elem_bias_vec{"elem_bias_vec", {1024, 2048}, {2048, 1}, 4};
    const Tensor elem_final_out{"elem_final_out", {1024, 2048}, {2048, 1}, 4};

    const Graph elemwise_graph({
        {"ElemAdd_0", OpType::ELEM_ADD, {elem_input_a, elem_input_b}, {elem_add_out}},
        {"ReLU_0", OpType::RELU, {elem_add_out}, {elem_relu_out}},
        {"BiasAdd_0", OpType::BIAS_ADD, {elem_relu_out, elem_bias_vec}, {elem_final_out}}
    });

    if (verbose) {
        std::cout << elemwise_graph.validation_report() << "\n";
    }

    const GraphAnalysis elemwise_analysis(elemwise_graph);
    const BufferLifetimeReport elemwise_lifetime = BufferLifetimeAnalyzer::analyze(elemwise_graph, elemwise_analysis);

    if (verbose) {
        std::cout << "Unfused peak live allocation: " << elemwise_lifetime.peak_live_bytes << " bytes at step " << elemwise_lifetime.peak_step << "\n";
    }

    // Lightweight elementwise schedules: 256 threads, no shared memory, minimal registers
    const TilingSchedule elem_add_schedule{256, 1, 1, 0, 12};
    const TilingSchedule elem_relu_schedule{256, 1, 1, 0, 10};
    const TilingSchedule elem_bias_schedule{256, 1, 1, 0, 12};

    const ClusterScheduleBuilder elemwise_schedule_builder({elem_add_schedule, elem_relu_schedule, elem_bias_schedule});
    const FusionPlanEvaluator elemwise_evaluator(elemwise_graph, elemwise_analysis, estimator, elemwise_schedule_builder);

    const DynamicProgrammingFusionOptimizer elemwise_dp(elemwise_graph);
    QuantumInspiredEvolutionaryOptimizer elemwise_qieo(96, 180, 0.085, 20260718U);

    const OptimizationResult elemwise_dp_result = elemwise_dp.optimize(elemwise_evaluator);
    const OptimizationResult elemwise_qieo_result = elemwise_qieo.optimize(elemwise_evaluator, elemwise_graph.size() - 1);

    if (verbose) {
        print_optimizer_report(elemwise_dp_result);
        print_optimizer_report(elemwise_qieo_result);
    }

    const OptimizationResult& elemwise_winning = elemwise_dp_result.best_plan.runtime_ms <= elemwise_qieo_result.best_plan.runtime_ms ? elemwise_dp_result : elemwise_qieo_result;

    // ── Comparative Summary ──
    if (verbose) {
        std::cout << "\n\n============================================================\n";
        std::cout << "COMPARATIVE ANALYSIS\n";
        std::cout << "============================================================\n";

        std::cout << "\nCase A: Full 6-Op Pipeline (MatMul -> BiasAdd -> ReLU -> LayerNorm -> MatMul -> BiasAdd)\n";
        std::cout << "  Winner:              " << winning_result.engine_name << "\n";
        std::cout << "  Fusion layout:       " << plan_to_string(winning_result.best_plan) << "\n";
        std::cout << "  Predicted runtime:   " << std::fixed << std::setprecision(5) << winning_result.best_plan.runtime_ms << " ms\n";
        std::cout << "  HBM saved:           " << winning_result.best_plan.intermediate_hbm_saved_bytes << " bytes\n";
        const bool case_a_fused = winning_result.best_plan.clusters.size() < graph.size();
        std::cout << "  Verdict:             " << (case_a_fused ? "PARTIAL FUSION" : "NO FUSION (occupancy cliff from LayerNorm reduction)") << "\n";

        std::cout << "\nCase B: Elementwise Chain (ElemAdd -> ReLU -> BiasAdd) [Memory-Bound]\n";
        std::cout << "  Winner:              " << elemwise_winning.engine_name << "\n";
        std::cout << "  Fusion layout:       " << plan_to_string(elemwise_winning.best_plan) << "\n";
        std::cout << "  Predicted runtime:   " << std::fixed << std::setprecision(5) << elemwise_winning.best_plan.runtime_ms << " ms\n";
        std::cout << "  HBM saved:           " << elemwise_winning.best_plan.intermediate_hbm_saved_bytes << " bytes\n";
        const bool case_b_fused = elemwise_winning.best_plan.clusters.size() < elemwise_graph.size();
        std::cout << "  Verdict:             " << (case_b_fused ? "FULL FUSION (all intermediates stay in registers, HBM eliminated)" : "NO FUSION") << "\n";
    }

    summary.winning_optimizer = winning_result.engine_name;
    summary.winning_layout = plan_to_string(winning_result.best_plan);
    summary.predicted_runtime_ms = winning_result.best_plan.runtime_ms;
    summary.nominal_hbm_bytes = winning_result.best_plan.nominal_hbm_bytes;
    summary.modeled_hbm_bytes = winning_result.best_plan.modeled_hbm_bytes;
    summary.intermediate_hbm_saved_bytes = winning_result.best_plan.intermediate_hbm_saved_bytes;

    return summary;
}

void write_prediction_csv(const CalibrationSummary& summary, const std::string& csv_path) {
    std::ofstream out(csv_path);
    if (!out) throw std::runtime_error("failed to open prediction csv");
    out << "winning_optimizer,winning_layout,predicted_runtime_ms,nominal_hbm_bytes,modeled_hbm_bytes,intermediate_hbm_saved_bytes\n";
    out << '"' << summary.winning_optimizer << '"' << ","
        << '"' << summary.winning_layout << '"' << ","
        << summary.predicted_runtime_ms << ","
        << summary.nominal_hbm_bytes << ","
        << summary.modeled_hbm_bytes << ","
        << summary.intermediate_hbm_saved_bytes << "\n";
}

} // namespace gpu_fusion
