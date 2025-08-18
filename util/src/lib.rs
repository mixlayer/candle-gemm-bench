use candle::{Device, FloatDType, Shape, Tensor, WithDType};
use rand::{distr::StandardUniform, prelude::*};
pub struct ModelDims<const Q_PROJ_M: usize, const Q_PROJ_N: usize, const HIDDEN_SZ: usize> {}

impl<const Q_PROJ_M: usize, const Q_PROJ_N: usize, const HIDDEN_SZ: usize>
    ModelDims<Q_PROJ_M, Q_PROJ_N, HIDDEN_SZ>
{
    pub fn shape_q() -> Shape {
        Shape::from((Q_PROJ_M, Q_PROJ_N))
    }

    pub fn shape_act() -> Shape {
        Shape::from((HIDDEN_SZ, HIDDEN_SZ))
    }

    pub fn random_act<D: WithDType + FloatDType>(&self, device: &Device) -> Tensor
    where
        StandardUniform: Distribution<D>,
    {
        let shape = Self::shape_act();
        let elems = shape.elem_count();
        let mut rng = rand::rng();
        let es = (0..elems).map(|_| rng.random::<D>()).collect::<Vec<_>>();
        Tensor::from_vec(es, shape, device).unwrap()
    }

    pub fn rand_q<D: WithDType + FloatDType>(&self, device: &Device) -> Tensor
    where
        StandardUniform: Distribution<D>,
    {
        let shape = Self::shape_q();
        let elems = shape.elem_count();
        let mut rng = rand::rng();
        let es = (0..elems).map(|_| rng.random::<D>()).collect::<Vec<_>>();
        Tensor::from_vec(es, shape, device).unwrap()
    }
}

pub const LLAMA_70B_DIMS: ModelDims<1024, 1024, 1024> = ModelDims {};

pub fn tensor_rand<D: WithDType + FloatDType>(shape: Shape, device: &Device) -> Tensor
where
    StandardUniform: Distribution<D>,
{
    let elems = shape.elem_count();
    let mut rng = rand::rng();
    let es = (0..elems).map(|_| rng.random::<D>()).collect::<Vec<_>>();
    Tensor::from_vec(es, shape, device).unwrap()
}

pub struct ModelDims2 {
    q_proj_dims: (usize, usize),
    kv_proj_dims: (usize, usize),
    mlp_dims: (usize, usize),
}

impl ModelDims2 {
    pub fn bench_problems(&self, batch_sz: usize) -> Vec<((usize, usize), (usize, usize))> {
        let ps = &[
            (
                (self.q_proj_dims.0, self.q_proj_dims.1),
                (self.q_proj_dims.1, batch_sz),
            ),
            (
                (self.kv_proj_dims.0, self.kv_proj_dims.1),
                (self.kv_proj_dims.1, batch_sz),
            ),
            (
                (self.mlp_dims.0, self.mlp_dims.1),
                (self.mlp_dims.1, batch_sz),
            ),
        ];

        ps.to_vec()
    }
}

pub const LLAMA_DIMS: ModelDims2 = ModelDims2 {
    q_proj_dims: (8192, 8192),
    kv_proj_dims: (1024, 8192),
    mlp_dims: (28672, 8192),
};
