use candle::backend::BackendStorage;
use candle::{DType, Layout, Shape};
use candle::{Device, cuda::DeviceId};
use cudarc::cudnn::{result, sys};
use cudarc::driver::DevicePtr;
use std::{cell::RefCell, collections::HashMap, sync::Arc};

// let handle = result::create_handle()?;
//         unsafe { result::set_stream(handle, stream.cu_stream as *mut _) }?;
thread_local! {
    // static CUDNN: RefCell<HashMap<DeviceId, Arc<Cudnn>>> = HashMap::new().into();
    static CUDNN: RefCell<HashMap<DeviceId, Arc<CudnnHandle>>> = HashMap::new().into();
}

pub struct CudnnHandle {
    handle: sys::cudnnHandle_t,
}

pub(crate) fn cudnn(device: &Device) -> Arc<CudnnHandle> {
    let cuda_device = device.as_cuda_device().unwrap();
    let device_id = cuda_device.id();
    let cudnn = CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return cudnn.clone();
        }

        let c = Arc::new(CudnnHandle {
            handle: result::create_handle().expect("error creating cudnn handle"),
        });

        cudnn.borrow_mut().insert(device_id, c.clone());

        c
    });

    cudnn
}

use std::ffi::c_void;
use std::os::raw::{c_int, c_longlong};

unsafe extern "C" {
    /// Returns 0 on success.
    pub fn bf16_fp8_matmul_cudnn(
        handle: sys::cudnnHandle_t,
        stream: *mut c_void,
        m: c_longlong,
        n: c_longlong,
        k: c_longlong,
        a_bf16: *const u16,
        lda: c_longlong,
        b_fp8: *const u8,
        ldb: c_longlong,
        c_out: *mut c_void,
        ldc: c_longlong,       // f32* if out_is_fp32 else u16*
        descale_b: *const f32, // can be null
        fp8_kind: c_int,       // 0=E4M3, 1=E5M2
        out_is_fp32: c_int,
    ) -> c_int;
}

pub enum Fp8Kind {
    E4M3 = 0,
    E5M2 = 1,
}

/// Candle-friendly wrapper.
/// Safety: pointers must be device pointers on the same CUDA device & stream.
pub fn cudnn_gemm_bf16_fp8(
    cudnn: sys::cudnnHandle_t,
    stream: *mut c_void,
    m: i64,
    n: i64,
    k: i64,
    a_bf16: *const u16,
    lda: i64,
    b_fp8: *const u8,
    ldb: i64,
    c_out: *mut c_void,
    ldc: i64,
    descale_b: Option<*const f32>,
) -> Result<(), i32> {
    unsafe {
        let code = bf16_fp8_matmul_cudnn(
            cudnn,
            stream,
            m,
            n,
            k,
            a_bf16,
            lda,
            b_fp8,
            ldb,
            c_out,
            ldc,
            descale_b.unwrap_or(std::ptr::null()),
            Fp8Kind::E4M3 as i32,
            0,
        );

        if code == 0 { Ok(()) } else { Err(code) }
    }
}

/// A (fp16) @ B (fp8) -> C (fp16)
pub struct Bf16Fp8CudnnMatmul {
    device: Device,
    fp8_b_scale: f32,
}

impl Bf16Fp8CudnnMatmul {
    pub fn new(device: &Device, fp8_b_scale: f32) -> Self {
        Self {
            device: device.clone(),
            fp8_b_scale,
        }
    }

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

        let d_a @ (m, k) = a_l.shape().dims2()?;
        let d_b @ (k_b, n) = b_l.shape().dims2()?;

        if k_b != k {
            candle::bail!(
                "This layer only supports NN, row major layout, input dims: {:?}, {:?}",
                d_a,
                d_b
            );
        }

        let out_shape = Shape::from((m, n));
        let out = unsafe { dev.alloc::<half::bf16>(out_shape.elem_count())? };

        let (d_dev, _d_guard) = out.device_ptr(&stream);

        let cudnn = cudnn(&self.device);

        cudnn_gemm_bf16_fp8(
            cudnn.handle,
            stream.cu_stream() as *mut c_void,
            m as i64,
            n as i64,
            k as i64,
            a_ptr as *const u16,
            k as i64,
            b_ptr as *const u8,
            n as i64,
            d_dev as *mut c_void,
            m as i64,
            None,
        )
        .map_err(|e| candle::Error::msg(format!("cudnn error: {}", e)))?;

        drop(_d_guard);

        let out = candle::CudaStorage::wrap_cuda_slice(out, dev.clone());

        Ok((out, out_shape))
    }
}

impl candle::CustomOp2 for Bf16Fp8CudnnMatmul {
    fn name(&self) -> &'static str {
        "cudnn-matmul-bf16-fp8"
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
                "cudnn-matmul-bf16-fp8 is only supported for bf16/f16 input dtypes: ({dt:?}, {dt2:?})"
            ),
        }
    }
}
