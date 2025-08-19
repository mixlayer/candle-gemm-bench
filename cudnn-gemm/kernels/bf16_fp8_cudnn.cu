// Build:
//   nvcc -std=c++17 -O3 -Xcompiler -fPIC \
//        -I/path/to/cudnn_frontend/include \
//        bf16_fp8_matmul_cudnn.cu -lcudnn -shared -o libbf16_fp8_cudnn.so
//
// Requires: cuDNN >= 9.0 (FP8 matmul) and Hopper+
// API style matches NVIDIA's v1.14.0 fp8 matmul sample.

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

static std::unordered_map<std::tuple<int64_t, int64_t, int64_t>, fe::graph::Graph*> graph_cache;

extern "C" int bf16_fp8_matmul_cudnn(
    cudnnHandle_t handle,
    cudaStream_t stream,
    int64_t M, int64_t N, int64_t K,
    const uint16_t* A_bf16, int64_t lda,     // row-major A[M,K], lda >= K
    const uint8_t*  B_fp8,  int64_t ldb,     // row-major B[K,N], ldb >= N (scaled FP8 bytes)
    void*           C_out,  int64_t ldc,     // row-major C[M,N], dtype controlled by out_is_fp32
    const float*    descale_B,               // device ptr to 1 float (often 1/scale_B), or nullptr
    int             fp8_kind,                // 0 = FP8_E4M3, 1 = FP8_E5M2
    int             out_is_fp32              // 1 => FP32 out, 0 => BF16 out
) {
    // cudnnHandle_t handle = *handle_ptr;
    // if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) return 1;
    if (cudnnSetStream(handle, stream) != CUDNN_STATUS_SUCCESS) { 
      return 2; 
    }

    // fe::graph::Graph graph{};

    // Check cache first (simplified, should include all params for real use)
    auto cache_key = std::make_tuple(M, N, K);
    fe::graph::Graph* graph_ptr = nullptr;

    int64_t bsz = 1;

    if (graph_cache.find(cache_key) == graph_cache.end()) {
      auto graph = new fe::graph::Graph{};

      // A: [bsz, M, K]  strides = {M*K, K, 1}  (K contiguous)
      auto A = graph.tensor(
          fe::graph::Tensor_attributes()
              .set_name("A_bf16")
              .set_dim({bsz, M, K})
              .set_stride({M * K, K, 1})
              .set_data_type(fe::DataType_t::BFLOAT16));

      auto A_as_fp8 = graph.pointwise(
          A,
          fe::graph::Pointwise_attributes()
              .set_mode(fe::PointwiseMode_t::IDENTITY)
              .set_compute_data_type(fe::DataType_t::FLOAT));
      A_as_fp8->set_data_type(fp8_kind == 0 ? fe::DataType_t::FP8_E4M3 : fe::DataType_t::FP8_E5M2);
            

      // B: [bsz, K, N]  strides: choose ONE that matches your memory:
      //   row-major    -> {K*N, N, 1}  (N contiguous)
      //   K-contiguous -> {K*N, 1, K}  (matches NVIDIA sample)
      auto B = graph.tensor(
          fe::graph::Tensor_attributes()
              .set_name("B_fp8")
              .set_dim({bsz, K, N})
              .set_stride({K * N, /*either*/ 1 /*or N*/, /*either*/ K /*or 1*/})
              .set_data_type(fp8_kind == 0 ? fe::DataType_t::FP8_E4M3 : fe::DataType_t::FP8_E5M2));

      // Matmul (compute = FP32 or FAST_FLOAT_FOR_FP8, see “B” below)
      auto mm_attr = fe::graph::Matmul_attributes()
          .set_name("GEMM")
          .set_compute_data_type(fe::DataType_t::FLOAT);

      auto Ctmp = graph.matmul(A_as_fp8, B, mm_attr);
      Ctmp->set_data_type(fe::DataType_t::FLOAT);

      // Optional descale_B (broadcast scalar)
      auto Cfinal = Ctmp;
    
      // Mark the real output (C: [bsz, M, N])
      Cfinal->set_output(true)
            .set_data_type(out_is_fp32 ? fe::DataType_t::FLOAT : fe::DataType_t::BFLOAT16);

      // Build & plan (matches sample sequence)
      if (!graph.validate().is_good()) { return 3; }
      if (!graph.build_operation_graph(handle).is_good()) { return 4; }
      if (!graph.create_execution_plans({fe::HeurMode_t::A}).is_good()) { return 5; }
      if (!graph.check_support(handle).is_good()) { return 6; }
      if (!graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good()) { return 7; }

      graph_cache[cache_key] = graph;
      graph_ptr = graph;
    } else {
      graph_ptr = graph_cache[cache_key];
    }

    // Workspace
    int64_t workspace_size = 0;
    if (!graph_ptr->get_workspace_size(workspace_size).is_good()) { return 8; }
    void* workspace = nullptr;
    if (workspace_size > 0 && cudaMalloc(&workspace, workspace_size) != cudaSuccess) {
        return 9;
    }

    // Variant pack: use the exact handle type via decltype(A) to avoid Tensor alias issues.
  
    std::unordered_map<decltype(A), void*> vp;
    vp.emplace(A, (void*)A_bf16);
    vp.emplace(B, (void*)B_fp8);
    vp.emplace(Cfinal, C_out);
    if (Bdesc) vp.emplace(*Bdesc, (void*)descale_B);


    // Rebuild descale handle correctly (capture from above):
    // The above "search by name" comment can be ignored since we kept Bdesc local earlier.
    // To keep code simple, fold descale binding into the earlier branch:
    // (see final version below)
    // Execute
    auto ok = graph_ptr->execute(handle, vp, workspace).is_good();

    if (workspace) cudaFree(workspace);
    
    return ok ? 0 : 10;
}
