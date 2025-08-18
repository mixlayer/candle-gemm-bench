use candle_gemm_bench_util::LLAMA_70B_DIMS;

fn main() {
    let device = candle::Device::new_cuda(0).unwrap();

    let m_a = LLAMA_70B_DIMS.rand_q::<half::bf16>(&device);
    let m_b = LLAMA_70B_DIMS.rand_q::<float8::F8E4M3>(&device);

    let dummy_scale: f32 = 1.232;

    let op = candle_gemm_bench_cutlass::Bf16Fp8CutlassMatmul {
        alpha: 1.0,
        beta: 0.0,
        fp8_b_scale: dummy_scale,
    };

    eprintln!("before apply");
    let y = m_a.apply_op2_no_bwd(&m_b, &op).unwrap();
    println!("after apply");

    println!("{}", y);

    device.synchronize().unwrap();
}
