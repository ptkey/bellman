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

use ocl::{Result as OclResult, Platform, Context, Queue, Buffer, Program, Kernel, EventList, Device};

type Kparts = (Program, Context, Device);

lazy_static! {
    pub static ref BLS12_KERNELS: Vec<Kparts> = {
        let mut kernels = Vec::new();
        let platform = Platform::list().into_iter().find(|&p|
            match p.name() {
                Ok(p) => p == GPU_NVIDIA_PLATFORM_NAME,
                Err(_) => false
            });
        
        let context = Context::builder().platform(platform.unwrap()).build().unwrap();

        let devices = Device::list_all(platform.unwrap()).unwrap();
        for d in devices {
        //for d in get_devices(platform.unwrap()).unwrap() {
            println!("device {:?}", d.name());
            let src = sources::kernel::<Bls12>();

            //let context = Context::builder().platform(platform.unwrap()).build().unwrap();
            let program = Program::builder().src(src).devices(d).build(&context).unwrap();
            //let queue = Queue::new(&context, d, None);

            //(d, ProQue::builder().device(d).src(src).build())
            kernels.push((program.clone(), context.clone(), d));
        }
        return kernels
        // get_devices(GPU_NVIDIA_PLATFORM_NAME)
        //     .unwrap_or_default()
        //     .iter()
        //     .map(|d| {
        //         println!("device {:?}", d.name());
        //         let src = sources::kernel::<Bls12>();

        //         let platform = Platform::list().into_iter().find(|&p|
        //             match p.name() {
        //                 Ok(p) => p == GPU_NVIDIA_PLATFORM_NAME,
        //                 Err(_) => false
        //             });

        //         let context = Context::builder().platform(platform.unwrap()).build().unwrap();
        //         let program = Program::builder().src(src).devices(d).build(&context);
        //         let queue = Queue::new(&context.unwrap(), *d, None);

        //         //(d, ProQue::builder().device(d).src(src).build())
        //         (d, program, queue)
        //     })
        //     .filter_map(|(d, res, res2)| {
        //         if res.is_err() {
        //             info!("Cannot compile kernel for device: {}", d.name().unwrap_or("Unknown".to_string()));
        //             return None;
        //         }
        //         //let pq = res.unwrap();
        //         let p = res.unwrap();
        //         info!("Kernel compiled for device: {}", d.name().unwrap_or("Unknown".to_string()));
        //         return (p, res2)
        //     })
        //     .collect()
    };
}
