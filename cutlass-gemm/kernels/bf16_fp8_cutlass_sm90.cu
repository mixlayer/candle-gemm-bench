// Only supports a single scalar scale for the FP8 (B) operand.
// Preserves: explicit A/B swap+transpose to enable TMA epilogue.

#include <cstdint>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

using namespace cute;

// ---------------- Types & kernel config (unchanged core) --------------------
using MmaType   = cutlass::bfloat16_t;              // wide type used by tensor cores
using QuantType = cutlass::float_e4m3_t;        // narrow FP8 on B
constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

// A
using ElementA   = MmaType;
using LayoutA    = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

// B (FP8)
using ElementB   = QuantType;
using LayoutB    = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

// Transposed for explicit swap path
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

// C/D
using ElementC = cutlass::bfloat16_t;
using ElementD   = ElementC;
using LayoutC    = cutlass::layout::RowMajor;
using LayoutD    = LayoutC;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core kernel params
using ElementAccumulator = float;
using ElementCompute     = float;
using ArchTag            = cutlass::arch::Sm90;
// using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassTensorOp;
using TileShape          = Shape<_128,_128,_128>;
using ClusterShape       = Shape<_2,_1,_1>;

// using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput;
using KernelSchedule     = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
// using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;

using EpilogueSchedule   = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileType   = cutlass::epilogue::collective::EpilogueTileAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  ArchTag, cutlass::arch::OpClassTensorOp,
  TileShape, ClusterShape,
  EpilogueTileType,
  ElementAccumulator, ElementAccumulator,
  // explicit swap+transpose: epilogue sees transposed C/D
  ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type, AlignmentC,
  ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
  EpilogueSchedule
>::CollectiveOp;

// Mainloop: ConvertOnly (no per-tile scales/zeros)
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag, OperatorClass,
  ElementB, LayoutB_Transpose, AlignmentB,   // B first (swapped)
  ElementA, LayoutA_Transpose, AlignmentA,
  ElementAccumulator,
  TileShape, ClusterShape,
  cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
  KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  Shape<int,int,int,int>,  // (M,N,K,L) placeholder
  CollectiveMainloop,
  CollectiveEpilogue
>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Strides (packed, derived from layouts)
using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;

extern "C" {

// 1 = supported on current device (Hopper / SM90*), else 0.
int cutlass_hopper_mixed_dtype_is_supported() {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  int dev = 0;
  cudaDeviceProp props{};
  if (cudaGetDevice(&dev) != cudaSuccess) return 0;
  if (cudaGetDeviceProperties(&props, dev) != cudaSuccess) return 0;
  return (props.major == 9) ? 1 : 0;
#else
  return 0;
#endif
}

const char* cutlass_status_string(int status_code) {
  return cutlassGetStatusString(static_cast<cutlass::Status>(status_code));
}

// Single entry point: tensor-wide scale for FP8 B.
// All pointers are *device* pointers. If beta==0, C may be null.
// Layouts must match the original example (A: [M,K,L] row-major, B: [K,N,L] col-major, C/D: [M,N,L] row-major).
int cutlass_hopper_fp8_gemm_run_scalar(
    int m, int n, int k, int batch_l,
    float scale_b,      // tensor-wide scale for FP8 B
    float alpha, float beta,
    const void* A,      // half
    const void* B,      // e4m3
    const void* C,      // half (nullable when beta==0)
    void* D,            // half (output)
    void* stream_void   // cudaStream_t or CUstream (opaque). null => default stream
) {
#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  (void)m;(void)n;(void)k;(void)batch_l;(void)scale_b;(void)alpha;(void)beta;
  (void)A;(void)B;(void)C;(void)D;(void)stream_void;
  return static_cast<int>(cutlass::Status::kErrorNotSupported);
#else
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);

  // Shapes and packed strides (note C/D reversed due to explicit swap+transpose)
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, batch_l));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, batch_l));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, batch_l));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, batch_l));

  // Fold the FP8 tensor-wide scale into alpha (A*(sB) == s*(A*B)).
  float alpha_eff = alpha * scale_b;

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {n, m, k, batch_l},   // swapped problem (N, M, K, L)
    {reinterpret_cast<const ElementB*>(B), stride_B,
     reinterpret_cast<const ElementA*>(A), stride_A},
    {{alpha_eff, beta},
     reinterpret_cast<const ElementC*>(C), stride_C,
     reinterpret_cast<ElementD*>(D), stride_D}
  };

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(ws);

  cutlass::Status st;
  st = gemm.can_implement(args);                      if (st != cutlass::Status::kSuccess) return static_cast<int>(st);
  st = gemm.initialize(args, workspace.get(), stream);if (st != cutlass::Status::kSuccess) return static_cast<int>(st);
  st = gemm.run(stream);                              return static_cast<int>(st);
#endif
}

} // extern "C"
