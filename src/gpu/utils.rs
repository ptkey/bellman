use crate::gpu::error::{GPUError, GPUResult};
use ocl::{Device, Platform};

use fs2::FileExt;
use log::info;
use std::collections::HashMap;
use std::fs::File;
use std::{env, io};

pub const GPU_NVIDIA_PLATFORM_NAME: &str = "NVIDIA CUDA";
// pub const CPU_INTEL_PLATFORM_NAME: &str = "Intel(R) CPU Runtime for OpenCL(TM) Applications";

pub fn get_devices(platform_name: &str) -> GPUResult<Vec<Device>> {
    if env::var("BELLMAN_NO_GPU").is_ok() {
        return Err(GPUError {
            msg: "GPU accelerator is disabled!".to_string(),
        });
    }

    let platform = Platform::list()?.into_iter().find(|&p| match p.name() {
        Ok(p) => p == platform_name,
        Err(_) => false,
    });
    match platform {
        Some(p) => Ok(Device::list_all(p)?),
        None => Err(GPUError {
            msg: "GPU platform not found!".to_string(),
        }),
    }
}

lazy_static::lazy_static! {
    static ref CORE_COUNTS: HashMap<String, usize> = {
        let mut core_counts : HashMap<String, usize> = vec![
            ("GeForce RTX 2080 Ti".to_string(), 4352),
            ("GeForce RTX 2080 SUPER".to_string(), 3072),
            ("GeForce RTX 2080".to_string(), 2944),
            ("GeForce GTX 1080 Ti".to_string(), 3584),
            ("GeForce GTX 1080".to_string(), 2560),
            ("GeForce GTX 1060".to_string(), 1280),
        ].into_iter().collect();

        match env::var("BELLMAN_CUSTOM_GPU").and_then(|var| {
            for card in var.split(",") {
                let splitted = card.split(":").collect::<Vec<_>>();
                if splitted.len() != 2 { panic!("Invalid BELLMAN_CUSTOM_GPU!"); }
                let name = splitted[0].trim().to_string();
                let cores : usize = splitted[1].trim().parse().expect("Invalid BELLMAN_CUSTOM_GPU!");
                info!("Adding \"{}\" to GPU list with {} CUDA cores.", name, cores);
                core_counts.insert(name, cores);
            }
            Ok(())
        }) { Err(_) => { }, Ok(_) => { } }

        core_counts
    };
}

pub fn get_core_count(d: Device) -> GPUResult<usize> {
    match CORE_COUNTS.get(&d.name()?[..]) {
        Some(&cores) => Ok(cores),
        None => Err(GPUError {
            msg: "Device unknown!".to_string(),
        }),
    }
}

pub fn get_memory(d: Device) -> GPUResult<u64> {
    match d.info(ocl::enums::DeviceInfo::GlobalMemSize)? {
        ocl::enums::DeviceInfoResult::GlobalMemSize(sz) => Ok(sz),
        _ => Err(GPUError {
            msg: "Cannot extract GPU memory!".to_string(),
        }),
    }
}

const GPU_LOCK_NAME: &str = "/tmp/bellman.gpu.lock";
#[derive(Debug)]
pub struct GPULock(File);
impl GPULock {
    pub fn new() -> io::Result<GPULock> {
        let file = File::create(GPU_LOCK_NAME)?;
        Ok(GPULock(file))
    }
    pub fn lock(&mut self) -> io::Result<()> {
        info!("Acquiring GPU lock...");
        self.0.lock_exclusive()?;
        info!("GPU lock acquired!");
        Ok(())
    }
    pub fn unlock(&mut self) -> io::Result<()> {
        self.0.unlock()?;
        info!("GPU lock released!");
        Ok(())
    }
}

const PRIORITY_LOCK_NAME: &str = "/tmp/bellman.priority.lock";
#[derive(Debug)]
pub struct PriorityLock(File);
use std::sync::Mutex;
lazy_static::lazy_static! {
    static ref IS_ME : Mutex<bool> = Mutex::new(false);
}
impl PriorityLock {
    pub fn new() -> io::Result<PriorityLock> {
        let file = File::create(PRIORITY_LOCK_NAME)?;
        Ok(PriorityLock(file))
    }
    pub fn lock(&mut self) -> io::Result<()> {
        let mut is_me = IS_ME.lock().unwrap();
        info!("Acquiring priority lock...");
        self.0.lock_exclusive()?;
        *is_me = true;
        info!("Priority lock acquired!");
        Ok(())
    }
    pub fn unlock(&mut self) -> io::Result<()> {
        let mut is_me = IS_ME.lock().unwrap();
        self.0.unlock()?;
        *is_me = false;
        info!("Priority lock released!");
        Ok(())
    }
    pub fn can_lock() -> io::Result<bool> {
        // Either taken by me or not taken by somebody else
        Ok(*IS_ME.lock().unwrap()
            || File::create(PRIORITY_LOCK_NAME)?
                .try_lock_exclusive()
                .is_ok())
    }
}

pub struct LockedKernel<'a, K, F>
where
    F: Fn() -> Option<K>,
{
    creator: F,
    name: &'static str, // Name of the kernel, for logging purposes
    supported: bool,
    kernel: Option<K>,
    lock: &'a mut GPULock,
}

use log::warn;
impl<'a, K, F> LockedKernel<'a, K, F>
where
    F: Fn() -> Option<K>,
{
    pub fn new(lock: &'a mut GPULock, name: &'static str, f: F) -> LockedKernel<'a, K, F> {
        lock.lock().unwrap();
        let kern = f();
        if kern.is_some() {
            info!("GPU {} is supported!", name);
        } else {
            warn!("GPU {} is NOT supported!", name);
            lock.unlock().unwrap();
        }
        LockedKernel::<K, F> {
            supported: kern.is_some(),
            creator: f,
            name: name,
            kernel: kern,
            lock: lock,
        }
    }
    pub fn get(&mut self) -> &mut Option<K> {
        if !PriorityLock::can_lock().unwrap_or(false) {
            warn!("GPU acquired by some other process! Freeing up kernels...");
            self.kernel = None; // This would drop kernel and free up the GPU
            self.lock.unlock().unwrap();
        } else if self.supported && self.kernel.is_none() {
            warn!("GPU is free again! Trying to reacquire GPU...");
            self.lock.lock().unwrap();
            self.kernel = (self.creator)();
            if self.kernel.is_none() {
                self.lock.unlock().unwrap();
            }
        }
        &mut self.kernel
    }
}

impl<'a, K, F> Drop for LockedKernel<'a, K, F>
where
    F: Fn() -> Option<K>,
{
    fn drop(&mut self) {
        self.lock.unlock().unwrap();
    }
}
