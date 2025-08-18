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
