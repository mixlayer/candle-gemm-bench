use anyhow::Result;
use candle::{Shape, Tensor};
use candle_gemm_bench_util::{LLAMA_DIMS, MatMul, tensor_rand};
use float8::F8E4M3;
use half::bf16;

fn main() -> Result<()> {
    let device = candle::Device::new_cuda(0).unwrap();

    let problems = LLAMA_DIMS.bench_problems(1);
    let dummy_scale: f32 = 1.232;

    for problem in problems {
        let MatMul {
            input_dims,
            weight_dims,
        } = problem;

        // let m_input = Tensor::rand(Shape::from(input_dims), &device);
        // let m_weights = Tensor::rand(Shape::from(weight_dims), &device)
        //     .t()?
        //     .contiguous()?;

        let m_input = tensor_rand::<bf16>(Shape::from(input_dims), &device);
        let m_weights = tensor_rand::<F8E4M3>(Shape::from(weight_dims), &device)
            .t()?
            .contiguous()?;

        let op = candle_gemm_bench_cudnn::Bf16Fp8CudnnMatmul::new(&device, dummy_scale);
        let y = m_input.apply_op2_no_bwd(&m_weights, &op).unwrap();

        println!("{:?} @ {:?} = {:?}", m_input, m_weights, y);
        device.synchronize().unwrap();
    }

    Ok(())
}
