use candle::{Device, FloatDType, Shape, Tensor, WithDType};
use rand::{distr::StandardUniform, prelude::*};

pub fn tensor_rand<D: WithDType + FloatDType>(shape: Shape, device: &Device) -> Tensor
where
    StandardUniform: Distribution<D>,
{
    let t = Tensor::rand(-1.0f32, 1.0f32, shape, device).expect("tensor::rand error");
    t.to_dtype(D::DTYPE).expect("to_dtype error")
    // let elems = shape.elem_count();
    // let mut rng = rand::rng();
    // let es = (0..elems).map(|_| rng.random::<D>()).collect::<Vec<_>>();
    // let buf = device
    //     .as_cuda_device()
    //     .unwrap()
    //     .rand_uniform(&shape, DType::F32, -1.0, 1.0)
    //     .expect("rand_uniform error");
    // let layout = Layout::contiguous(shape);
    // let buf = buf.to_dtype(&layout, D::DTYPE).expect("to_dtype error");
    // Tensor::from_vec(es, shape, device).unwrap()
}

/// A collection of some sample tensor dimensions to
/// benchmark against.
pub struct ModelDims {
    q_proj_dims: (usize, usize),
    kv_proj_dims: (usize, usize),
    mlp_dims: (usize, usize),
}

#[derive(Debug, Clone)]
pub struct MatMul {
    pub input_dims: (usize, usize),
    pub weight_dims: (usize, usize),
}

impl ModelDims {
    pub fn bench_problems(&self, batch_sz: usize) -> Vec<MatMul> {
        let ps = &[
            MatMul {
                input_dims: (batch_sz, self.q_proj_dims.1),
                weight_dims: (self.q_proj_dims.0, self.q_proj_dims.1),
            },
            MatMul {
                input_dims: (batch_sz, self.kv_proj_dims.1),
                weight_dims: (self.kv_proj_dims.0, self.kv_proj_dims.1),
            },
            MatMul {
                input_dims: (batch_sz, self.mlp_dims.1),
                weight_dims: (self.mlp_dims.0, self.mlp_dims.1),
            },
        ];

        ps.to_vec()
    }
}

pub const LLAMA_DIMS: ModelDims = ModelDims {
    q_proj_dims: (8192, 8192),
    kv_proj_dims: (1024, 8192),
    mlp_dims: (28672, 8192),
};
