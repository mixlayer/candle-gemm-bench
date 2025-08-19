use candle::{DType, Device, Shape};
use candle_gemm_bench_util::{LLAMA_DIMS, MatMul, tensor_rand};
use criterion::{Criterion, criterion_group, criterion_main};
use float8::F8E4M3;
use half::bf16;

fn cuda_device() -> Device {
    candle::Device::new_cuda(0).expect("error creating cuda device")
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = cuda_device();

    let dummy_scale: f32 = 1.232;

    for matmul in LLAMA_DIMS.bench_problems(1) {
        let MatMul {
            weight_dims,
            input_dims,
        } = matmul;

        // let (am, an) = weight_dims;
        // let (bn, bm) = input_dims;

        let W = tensor_rand::<f32>(Shape::from(weight_dims), &device);
        let X = tensor_rand::<f32>(Shape::from(input_dims), &device);

        let mut group = c.benchmark_group(format!("{:?}@{:?}", weight_dims, input_dims));

        let weight = W.to_dtype(DType::BF16).unwrap();
        let x = X.to_dtype(DType::BF16).unwrap().t().unwrap();

        group.bench_function("candle_builtin_cublas_gemm_bf16", |b| {
            b.iter(|| {
                let _y = weight.matmul(&x).unwrap();
                device.synchronize().unwrap();
            });
        });

        let x = X.to_dtype(DType::BF16).unwrap();
        let weight = W.to_dtype(DType::F8E4M3).unwrap().t().unwrap();

        group.bench_function("cutlass_gemm_bf16_fp8", |b| {
            let op = candle_gemm_bench_cutlass::Bf16Fp8CutlassMatmul {
                alpha: 1.0,
                beta: 0.0,
                fp8_b_scale: dummy_scale,
            };

            b.iter(|| {
                let _y = x.apply_op2_no_bwd(&weight, &op).unwrap();
                device.synchronize().unwrap();
            });
        });

        let weight = W.to_dtype(DType::F8E4M3).unwrap();
        let x = X.to_dtype(DType::F8E4M3).unwrap();

        group.bench_function("cutlass_gemm_fp8_fp8", |b| {
            let op = candle_gemm_bench_cutlass::Fp8Fp8CutlassMatmul {
                alpha: 1.0,
                beta: 0.0,
                fp8_a_scale: dummy_scale,
                fp8_b_scale: dummy_scale,
            };

            b.iter(|| {
                let _y = x.apply_op2_no_bwd(&weight, &op).unwrap();
                device.synchronize().unwrap();
            });
        });

        let weight = W.to_dtype(DType::F8E4M3).unwrap();
        let x = X.to_dtype(DType::BF16).unwrap();

        group.bench_function("cudnn_gemm_bf16_fp8", |b| {
            let op = candle_gemm_bench_cudnn::Bf16Fp8CudnnMatmul::new(&device, dummy_scale);
            b.iter(|| {
                let _y = x.apply_op2_no_bwd(&weight, &op).unwrap();
                device.synchronize().unwrap();
            });
        });

        group.finish()
    }
}

// fn builtin_cublas_bench_fn<A: WithDType + FloatDType, B: WithDType + FloatDType>(
//     device: &Device,
// ) -> impl Fn(&mut criterion::Bencher)
// where
//     StandardUniform: Distribution<A>,
//     StandardUniform: Distribution<B>,
// {
//     move |b| {
//         let m_a = LLAMA_70B_DIMS.rand_q::<A>(device);
//         let m_b = LLAMA_70B_DIMS.rand_q::<B>(device);

//         b.iter(|| {
//             let _y = m_a.matmul(&m_b).unwrap();
//             device.synchronize().unwrap();
//         });
//     }
// }

// fn cutlass_bench_fn(device: &Device) -> impl Fn(&mut criterion::Bencher) {
//     move |b| {
//         let m_a = LLAMA_70B_DIMS.rand_q::<bf16>(device);
//         let m_b = LLAMA_70B_DIMS.rand_q::<F8E4M3>(device);

//         let dummy_scale: f32 = 1.232;

//         b.iter(|| {
//             let op = candle_gemm_bench_cutlass::Bf16Fp8CutlassMatmul {
//                 alpha: 1.0,
//                 beta: 0.0,
//                 fp8_b_scale: dummy_scale,
//             };

//             let _y = m_a.apply_op2_no_bwd(&m_b, &op).unwrap();
//             device.synchronize().unwrap();
//         });
//     }
// }

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
