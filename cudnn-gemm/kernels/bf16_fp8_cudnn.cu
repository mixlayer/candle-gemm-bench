// Version using tensor handles directly instead of UIDs
#include <cstdint>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

// Custom hash for tuple
struct TupleHash {
    std::size_t operator()(const std::tuple<int64_t, int64_t, int64_t, int, int>& t) const {
        auto h1 = std::hash<int64_t>{}(std::get<0>(t));
        auto h2 = std::hash<int64_t>{}(std::get<1>(t));
        auto h3 = std::hash<int64_t>{}(std::get<2>(t));
        auto h4 = std::hash<int>{}(std::get<3>(t));
        auto h5 = std::hash<int>{}(std::get<4>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
    }
};

// Structure to hold a cached graph and its tensor handles
struct CachedGraph {
    std::unique_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> A_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> B_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> C_tensor;
    int64_t workspace_size;
    
    CachedGraph() : workspace_size(0) {}
};

// Cache key includes all parameters that affect graph structure
using CacheKey = std::tuple<int64_t, int64_t, int64_t, int, int>; // M, N, K, fp8_kind, out_is_fp32

// Global cache
static std::unordered_map<CacheKey, std::unique_ptr<CachedGraph>, TupleHash> graph_cache;

extern "C" int bf16_fp8_matmul_cudnn(
    cudnnHandle_t handle,
    cudaStream_t stream,
    int64_t M, int64_t N, int64_t K,
    const uint16_t* A_bf16, int64_t lda,
    const uint8_t*  B_fp8,  int64_t ldb,
    void*           C_out,  int64_t ldc,
    const float*    descale_B,
    int             fp8_kind,
    int             out_is_fp32
) {
    if (cudnnSetStream(handle, stream) != CUDNN_STATUS_SUCCESS) { 
        return 2; 
    }

    auto cache_key = std::make_tuple(M, N, K, fp8_kind, out_is_fp32);
    
    auto it = graph_cache.find(cache_key);
    CachedGraph* cached = nullptr;
    
    if (it == graph_cache.end()) {
        // std::cerr << "Building new graph for shape (" << M << ", " << N << ", " << K << ")" << std::endl;
        
        // Build new graph and cache it
        auto new_cached = std::make_unique<CachedGraph>();
        new_cached->graph = std::make_unique<fe::graph::Graph>();
        auto& graph = *new_cached->graph;
        
        int64_t bsz = 1;

        // Store tensor handles as shared_ptr
        new_cached->A_tensor = graph.tensor(
            fe::graph::Tensor_attributes()
                .set_name("A")
                .set_dim({bsz, M, K})
                .set_stride({M * K, K, 1})
                .set_data_type(fe::DataType_t::BFLOAT16));

        // Convert to FP8
        auto A_as_fp8 = graph.pointwise(
            new_cached->A_tensor,
            fe::graph::Pointwise_attributes()
                .set_mode(fe::PointwiseMode_t::IDENTITY)
                .set_compute_data_type(fe::DataType_t::FLOAT));
        A_as_fp8->set_data_type(fp8_kind == 0 ? fe::DataType_t::FP8_E4M3 : fe::DataType_t::FP8_E5M2);

        new_cached->B_tensor = graph.tensor(
            fe::graph::Tensor_attributes()
                .set_name("B")
                .set_dim({bsz, K, N})
                .set_stride({K * N, N, 1})
                .set_data_type(fp8_kind == 0 ? fe::DataType_t::FP8_E4M3 : fe::DataType_t::FP8_E5M2));

        // Matmul
        auto mm_attr = fe::graph::Matmul_attributes()
                .set_name("GEMM")
                .set_compute_data_type(fe::DataType_t::FLOAT);
        
        auto Cfinal = graph.matmul(A_as_fp8, new_cached->B_tensor, mm_attr);
        
        // Set output
        Cfinal->set_output(true)
              .set_dim({bsz, M, N})
              .set_stride({M * N, N, 1})
              .set_data_type(out_is_fp32 ? fe::DataType_t::FLOAT : fe::DataType_t::BFLOAT16);
        
        new_cached->C_tensor = Cfinal;

        // Build & plan
        auto status = graph.validate();
        if (!status.is_good()) {
            std::cerr << "cudnn validate failed: " << status.get_message() << std::endl;
            return 3;
        }
        
        status = graph.build_operation_graph(handle);
        if (!status.is_good()) {
            std::cerr << "cudnn build_operation_graph failed: " << status.get_message() << std::endl;
            return 4;
        }
        
        status = graph.create_execution_plans({fe::HeurMode_t::A});
        if (!status.is_good()) {
            std::cerr << "cudnn create_execution_plans failed: " << status.get_message() << std::endl;
            return 5;
        }
        
        status = graph.check_support(handle);
        if (!status.is_good()) {
            std::cerr << "cudnncheck_support failed: " << status.get_message() << std::endl;
            return 6;
        }
        
        status = graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
        if (!status.is_good()) {
            std::cerr << "cudnn build_plans failed: " << status.get_message() << std::endl;
            return 7;
        }
        
        // Get workspace size
        status = graph.get_workspace_size(new_cached->workspace_size);
        if (!status.is_good()) {
            std::cerr << "cudnn get_workspace_size failed: " << status.get_message() << std::endl;
            return 8;
        }
        // std::cerr << "Workspace size: " << new_cached->workspace_size << " bytes" << std::endl;
        
        // Insert into cache
        cached = new_cached.get();
        graph_cache[cache_key] = std::move(new_cached);
    } else {
        // std::cerr << "Using cached graph for shape (" << M << ", " << N << ", " << K << ")" << std::endl;
        cached = it->second.get();
    }

    // Allocate workspace
    void* workspace = nullptr;
    if (cached->workspace_size > 0) {
        if (cudaMalloc(&workspace, cached->workspace_size) != cudaSuccess) {
            std::cerr << "Failed to allocate workspace of size " << cached->workspace_size << std::endl;
            return 9;
        }
    }

    // Build variant pack using tensor handles directly
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
    variant_pack[cached->A_tensor] = (void*)A_bf16;
    variant_pack[cached->B_tensor] = (void*)B_fp8;
    variant_pack[cached->C_tensor] = C_out;
    
    // std::cerr << "Variant pack has " << variant_pack.size() << " entries" << std::endl;
    
    // Execute using the cached graph
    auto status = cached->graph->execute(handle, variant_pack, workspace);

    if (!status.is_good()) {
        std::cerr << "Execute failed!" << std::endl;
        std::cerr << "Error details:" << std::endl;
        std::cerr << "  Code: " << status.get_code() << std::endl;
        std::cerr << "  Message: " << status.get_message() << std::endl;
        
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(cuda_err) << std::endl;
        }
    }
    
    if (workspace) {
        cudaFree(workspace);
    }
    
    return status.is_good() ? 0 : 10;
}

// Clear cache function
extern "C" void clear_graph_cache() {
    graph_cache.clear();
}