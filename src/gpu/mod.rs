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

use paired::bls12_381::Bls12;
use ocl::ProQue;
use log::info;
lazy_static! {
    pub static ref BLS12_KERNELS: Vec<ProQue> = {
        get_devices(CPU_INTEL_PLATFORM_NAME)
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
