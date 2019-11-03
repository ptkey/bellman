mod error;
pub use self::error::*;

#[cfg(feature = "gpu")]
mod sources;
#[cfg(feature = "gpu")]
pub use self::sources::*;

#[cfg(feature = "gpu")]
mod utils;
#[cfg(feature = "gpu")]
pub use self::utils::*;

#[cfg(feature = "gpu")]
mod structs;
#[cfg(feature = "gpu")]
pub use self::structs::*;

#[cfg(feature = "gpu")]
mod fft;
#[cfg(feature = "gpu")]
pub use self::fft::*;

#[cfg(feature = "gpu")]
mod multiexp;
#[cfg(feature = "gpu")]
pub use self::multiexp::*;

#[cfg(not (feature = "gpu"))]
mod nogpu;
#[cfg(not (feature = "gpu"))]
pub use self::nogpu::*;

#[cfg(feature = "gpu")]
use paired::bls12_381::Bls12;
#[cfg(feature = "gpu")]
use ocl::ProQue;
#[cfg(feature = "gpu")]
use log::info;
#[cfg(feature = "gpu")]
lazy_static! {
    pub static ref BLS12_KERNELS: Vec<ProQue> = {
        get_devices(GPU_NVIDIA_PLATFORM_NAME)
            .unwrap_or_default()
            .iter()
            .map(|d| {
                let src = sources::kernel::<Bls12>();
                (d, ProQue::builder().device(d).src(src).build())
            })
            .filter_map(|(d, res)| {
                if res.is_err() {
                    info!("Cannot compile kernel for device: {}", d.name().unwrap_or("Unknown".to_string()));
                    return None;
                }
                let pq = res.unwrap();
                info!("Kernel compiled for device: {}", d.name().unwrap_or("Unknown".to_string()));
                Some(pq)
            })
            .collect()
    };
}
