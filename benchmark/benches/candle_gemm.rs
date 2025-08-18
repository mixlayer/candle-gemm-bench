use candle::{Device, FloatDType, Shape, Tensor, WithDType};
use candle_gemm_bench_util::{LLAMA_70B_DIMS, LLAMA_DIMS, tensor_rand};
use criterion::{Criterion, criterion_group, criterion_main};
use float8::F8E4M3;
use half::{bf16, f16};
use rand_distr::{Distribution, StandardUniform};

fn cuda_device() -> Device {
    candle::Device::new_cuda(0).expect("error creating cuda device")
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = cuda_device();

    let mut group = c.benchmark_group("candle_builtin_cublas_gemm_bf16");
    for ((am, an), (bm, bn)) in LLAMA_DIMS.bench_problems(1) {
        let m_a = tensor_rand::<bf16>(Shape::from((am, an)), &device);
        let m_b = tensor_rand::<bf16>(Shape::from((bm, bn)), &device);

        group.bench_function(format!("{}x{}@{}x{}", am, an, bm, bn), |b| {
            b.iter(|| {
                let _y = m_a.matmul(&m_b).unwrap();
                device.synchronize().unwrap();
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("cutlass_gemm_bf16_fp8");
    let dummy_scale: f32 = 1.232;
    for ((am, an), (bm, bn)) in LLAMA_DIMS.bench_problems(1) {
        let m_a = tensor_rand::<F8E4M3>(Shape::from((am, an)), &device);
        let m_b_t = tensor_rand::<bf16>(Shape::from((bn, bm)), &device);

        group.bench_function(format!("{}x{}@{}x{}", am, an, bm, bn), |b| {
            let op = candle_gemm_bench_cutlass::Bf16Fp8CutlassMatmul {
                alpha: 1.0,
                beta: 0.0,
                fp8_b_scale: dummy_scale,
            };

            b.iter(|| {
                let _y = m_b_t.apply_op2_no_bwd(&m_a, &op).unwrap();
                device.synchronize().unwrap();
            });
        });
    }

    group.finish();
}

fn builtin_cublas_bench_fn<A: WithDType + FloatDType, B: WithDType + FloatDType>(
    device: &Device,
) -> impl Fn(&mut criterion::Bencher)
where
    StandardUniform: Distribution<A>,
    StandardUniform: Distribution<B>,
{
    move |b| {
        let m_a = LLAMA_70B_DIMS.rand_q::<A>(device);
        let m_b = LLAMA_70B_DIMS.rand_q::<B>(device);

        b.iter(|| {
            let _y = m_a.matmul(&m_b).unwrap();
            device.synchronize().unwrap();
        });
    }
}

fn cutlass_bench_fn(device: &Device) -> impl Fn(&mut criterion::Bencher) {
    move |b| {
        let m_a = LLAMA_70B_DIMS.rand_q::<bf16>(device);
        let m_b = LLAMA_70B_DIMS.rand_q::<F8E4M3>(device);

        let dummy_scale: f32 = 1.232;

        b.iter(|| {
            let op = candle_gemm_bench_cutlass::Bf16Fp8CutlassMatmul {
                alpha: 1.0,
                beta: 0.0,
                fp8_b_scale: dummy_scale,
            };

            let _y = m_a.apply_op2_no_bwd(&m_b, &op).unwrap();
            device.synchronize().unwrap();
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
