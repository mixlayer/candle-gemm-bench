use candle::{
    DType, Layout, Shape,
    backend::BackendStorage,
    cuda::{
        CudaDType,
        cudarc::driver::{DevicePtr, DeviceRepr},
    },
};

unsafe extern "C" {
    fn cutlass_hopper_mixed_dtype_is_supported() -> ::std::os::raw::c_int;
    fn cutlass_status_string(code: ::std::os::raw::c_int) -> *const ::std::os::raw::c_char;

    fn cutlass_hopper_fp8_gemm_run_scalar(
        m: i32,
        n: i32,
        k: i32,
        batch_l: i32,
        scale_b: f32, // tensor-wide FP8 scale
        alpha: f32,
        beta: f32,                       // epilogue scalars
        a: *const ::std::ffi::c_void,    // half
        b: *const ::std::ffi::c_void,    // e4m3
        c: *const ::std::ffi::c_void,    // half (nullable if beta==0)
        d: *mut ::std::ffi::c_void,      // half
        stream: *mut ::std::ffi::c_void, // CUstream/cudaStream_t or null
    ) -> ::std::os::raw::c_int;

    fn cutlass_hopper_fp8_fp8_gemm_run_scalar(
        m: i32,
        n: i32,
        k: i32,
        batch_l: i32,
        scale_a: f32,
        scale_b: f32,
        alpha: f32,
        beta: f32,
        a: *const ::std::ffi::c_void,
        b: *const ::std::ffi::c_void,
        c: *const ::std::ffi::c_void,
        d: *mut ::std::ffi::c_void,
        stream: *mut ::std::ffi::c_void,
    ) -> ::std::os::raw::c_int;
}

pub fn run_cutlass_fp8_scalar(
    m: i32,
    n: i32,
    k: i32,
    batch_l: i32,
    scale_b: f32,
    alpha: f32,
    beta: f32,
    a_dev: *const std::ffi::c_void,
    b_dev: *const std::ffi::c_void,
    c_dev: *const std::ffi::c_void, // pass null if beta == 0.0
    d_dev: *mut std::ffi::c_void,
    stream: *mut std::ffi::c_void, // pass null for default stream
) -> candle::Result<()> {
    unsafe {
        // if cutlass_hopper_mixed_dtype_is_supported() == 0 {
        //     candle::bail!("CUTLASS/Hopper path not supported on this GPU/runtime");
        // }
        let code = cutlass_hopper_fp8_gemm_run_scalar(
            m, n, k, batch_l, scale_b, alpha, beta, a_dev, b_dev, c_dev, d_dev, stream,
        );
        if code != 0 {
            let msg = std::ffi::CStr::from_ptr(cutlass_status_string(code))
                .to_string_lossy()
                .into_owned();
            candle::bail!("cutlass failed: {} (code {})", msg, code);
        }
        Ok(())
    }
}

pub fn run_cutlass_fp8_fp8_scalar(
    m: i32,
    n: i32,
    k: i32,
    batch_l: i32,
    scale_a: f32,
    scale_b: f32,
    alpha: f32,
    beta: f32,
    a_dev: *const std::ffi::c_void,
    b_dev: *const std::ffi::c_void,
    c_dev: *const std::ffi::c_void, // pass null if beta == 0.0
    d_dev: *mut std::ffi::c_void,
    stream: *mut std::ffi::c_void, // pass null for default stream
) -> candle::Result<()> {
    unsafe {
        let code = cutlass_hopper_fp8_fp8_gemm_run_scalar(
            m, n, k, batch_l, scale_a, scale_b, alpha, beta, a_dev, b_dev, c_dev, d_dev, stream,
        );

        if code != 0 {
            let msg = std::ffi::CStr::from_ptr(cutlass_status_string(code))
                .to_string_lossy()
                .into_owned();
            candle::bail!("cutlass failed: {} (code {})", msg, code);
        }
    }

    Ok(())
}

/// A (fp16) @ B (fp8) -> C (fp16)
pub struct Bf16Fp8CutlassMatmul {
    pub alpha: f32,
    pub beta: f32,
    pub fp8_b_scale: f32,
}

impl Bf16Fp8CutlassMatmul {
    pub fn fwd_f8e4m3(
        &self,
        a: &candle::CudaStorage,
        a_l: &Layout,
        b: &candle::CudaStorage,
        b_l: &Layout,
        _bias: Option<&candle::CudaStorage>,
        _bias_l: Option<&Layout>,
    ) -> candle::Result<(candle::CudaStorage, Shape)> {
        let dev = a.device();
        let stream = dev.cuda_stream();

        let a_slice = a.as_cuda_slice::<half::bf16>()?;
        let b_slice = b.as_cuda_slice::<float8::F8E4M3>()?;

        let (a_ptr, _a_guard) = a_slice.device_ptr(&stream);
        let (b_ptr, _b_guard) = b_slice.device_ptr(&stream);

        let a_ptr = a_ptr as *const u16;
        let b_ptr = b_ptr as *const u8;

        let (m, k) = a_l.shape().dims2()?;
        let (n, b_1) = b_l.shape().dims2()?;

        if b_1 != k {
            candle::bail!("This layer only supports TN layout");
        }

        let out_shape = Shape::from((n, m));
        let out = unsafe { dev.alloc::<half::bf16>(out_shape.elem_count())? };

        let (d_dev, _d_guard) = out.device_ptr(&stream);

        run_cutlass_fp8_scalar(
            m as i32,
            n as i32,
            k as i32,
            1,
            self.fp8_b_scale,
            1.0f32, //self.alpha,
            0.0f32, //self.beta,
            a_ptr as *const std::ffi::c_void,
            b_ptr as *const std::ffi::c_void,
            std::ptr::null(),
            d_dev as *mut std::ffi::c_void,
            std::ptr::null::<std::ffi::c_void>() as *mut std::ffi::c_void,
            // stream.cu_stream() as *mut std::ffi::c_void,
        )?;

        drop(_d_guard);

        let out = candle::CudaStorage::wrap_cuda_slice(out, dev.clone());

        Ok((out, out_shape))
    }
}

impl candle::CustomOp2 for Bf16Fp8CutlassMatmul {
    fn name(&self) -> &'static str {
        "cutlass-matmul-bf16-fp8"
    }

    fn cpu_fwd(
        &self,
        _: &candle::CpuStorage,
        _: &Layout,
        _: &candle::CpuStorage,
        _: &Layout,
    ) -> candle::Result<(candle::CpuStorage, Shape)> {
        candle::bail!("no cpu support for cutlass-matmul-bf16-fp8")
    }

    fn cuda_fwd(
        &self,
        a: &candle::CudaStorage,
        a_l: &Layout,
        b: &candle::CudaStorage,
        b_l: &Layout,
    ) -> candle::Result<(candle::CudaStorage, Shape)> {
        match (a.dtype(), b.dtype()) {
            (DType::BF16, DType::F8E4M3) => self.fwd_f8e4m3(a, a_l, b, b_l, None, None),
            // (DType::F16, DType::F8E4M3) => self.fwd_f8e4m3(a, a_l, b, b_l, None, None),
            (dt, dt2) => candle::bail!(
                "cutlass-matmul-bf16-fp8 is only supported for bf16/f16 input dtypes: ({dt:?}, {dt2:?})"
            ),
        }
    }
}

pub struct Fp8Fp8CutlassMatmul {
    pub alpha: f32,
    pub beta: f32,
    pub fp8_a_scale: f32,
    pub fp8_b_scale: f32,
}

impl Fp8Fp8CutlassMatmul {
    pub fn fwd_f8e4m3(
        &self,
        a: &candle::CudaStorage,
        a_l: &Layout,
        b: &candle::CudaStorage,
        b_l: &Layout,
        _bias: Option<&candle::CudaStorage>,
        _bias_l: Option<&Layout>,
    ) -> candle::Result<(candle::CudaStorage, Shape)> {
        let dev = a.device();
        let stream = dev.cuda_stream();

        let a_slice = a.as_cuda_slice::<float8::F8E4M3>()?;
        let b_slice = b.as_cuda_slice::<float8::F8E4M3>()?;

        let (a_ptr, _a_guard) = a_slice.device_ptr(&stream);
        let (b_ptr, _b_guard) = b_slice.device_ptr(&stream);

        let a_ptr = a_ptr as *const u16;
        let b_ptr = b_ptr as *const u8;

        let (m, k) = a_l.shape().dims2()?;
        let (n, b_1) = b_l.shape().dims2()?;

        if b_1 != k {
            candle::bail!("This layer only supports TN layout");
        }

        let out_shape = Shape::from((n, m));
        let out = unsafe { dev.alloc::<half::bf16>(out_shape.elem_count())? };

        let (d_dev, _d_guard) = out.device_ptr(&stream);

        run_cutlass_fp8_fp8_scalar(
            m as i32,
            n as i32,
            k as i32,
            1,
            self.fp8_a_scale,
            self.fp8_b_scale,
            1.0f32, //self.alpha,
            0.0f32, //self.beta,
            a_ptr as *const std::ffi::c_void,
            b_ptr as *const std::ffi::c_void,
            std::ptr::null(),
            d_dev as *mut std::ffi::c_void,
            std::ptr::null::<std::ffi::c_void>() as *mut std::ffi::c_void,
            // stream.cu_stream() as *mut std::ffi::c_void,
        )?;

        drop(_d_guard);

        let out = candle::CudaStorage::wrap_cuda_slice(out, dev.clone());

        Ok((out, out_shape))
    }
}

impl candle::CustomOp2 for Fp8Fp8CutlassMatmul {
    fn name(&self) -> &'static str {
        "cutlass-matmul-fp8-fp8"
    }

    fn cpu_fwd(
        &self,
        _: &candle::CpuStorage,
        _: &Layout,
        _: &candle::CpuStorage,
        _: &Layout,
    ) -> candle::Result<(candle::CpuStorage, Shape)> {
        candle::bail!("no cpu support for cutlass-matmul-bf16-fp8")
    }

    fn cuda_fwd(
        &self,
        a: &candle::CudaStorage,
        a_l: &Layout,
        b: &candle::CudaStorage,
        b_l: &Layout,
    ) -> candle::Result<(candle::CudaStorage, Shape)> {
        match (a.dtype(), b.dtype()) {
            (DType::F8E4M3, DType::F8E4M3) => self.fwd_f8e4m3(a, a_l, b, b_l, None, None),
            // (DType::F16, DType::F8E4M3) => self.fwd_f8e4m3(a, a_l, b, b_l, None, None),
            (dt, dt2) => candle::bail!(
                "cutlass-matmul-fp8-fp8 is only supported for fp8/fp8 input dtypes: ({dt:?}, {dt2:?})"
            ),
        }
    }
}
