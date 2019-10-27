use log::info;
use ocl::{Device, Platform, ProQue};
use std::panic;
use super::error::{GPUResult, GPUError};
use super::sources;

pub const GPU_NVIDIA_PLATFORM_NAME : &str = "NVIDIA CUDA";
pub const CPU_INTEL_PLATFORM_NAME : &str = "Intel(R) CPU Runtime for OpenCL(TM) Applications";

pub fn get_devices(platform_name: &str) -> GPUResult<Vec<Device>> {
    match panic::catch_unwind(|| {
        let platform = Platform::list().into_iter().find(|&p|
            match p.name() {
                Ok(p) => p == platform_name,
                Err(_) => false
            });
        Device::list_all(platform.unwrap()).unwrap()
    }) {
        Ok(devs) => Ok(devs),
        Err(_) => Err(GPUError {msg: "GPU platform not found!".to_string()})
    }
}

use paired::bls12_381::Bls12;
lazy_static! {
    pub static ref BLS12_KERNELS: Vec<ProQue> = {
        get_devices(GPU_NVIDIA_PLATFORM_NAME)
            .unwrap_or(Vec::new())
            .iter()
            .map(|d| {
                let src = sources::kernel::<Bls12>();
                ProQue::builder().device(d).src(src).build()
            })
            .filter(|res| res.is_ok())
            .map(|res| {
                let pq = res.unwrap();
                info!("Kernel initialized for device: {}", pq.device().name().unwrap_or("Unknown".to_string()));
                pq
            })
            .collect()
    };
}
