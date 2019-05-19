use pairing::bls12_381::Fr;
use super::error::{GPUResult, GPUError};

pub struct FFTKernel;

impl FFTKernel {

    pub fn create(_: u32) -> GPUResult<FFTKernel> {
        return Err(GPUError {msg: "GPU accelerator is not enabled!".to_string()});
    }

    pub fn radix_fft(&mut self, _: &mut [Fr], _: &Fr, _: u32) -> GPUResult<()> {
        return Err(GPUError {msg: "GPU accelerator is not enabled!".to_string()});
    }
}
