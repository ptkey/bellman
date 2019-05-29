use ocl::{ProQue, Buffer, MemFlags};
use ocl::prm::Ulong4;
use paired::bls12_381::Fr;
use std::cmp;
use ff::Field;
use super::error::GPUResult;

static DEFS_SRC : &str = include_str!("multiexp/defs.cl");
static FIELD_SRC : &str = include_str!("field.cl");
static EC_SRC : &str = include_str!("multiexp/ec.cl");
static KERNEL_SRC : &str = include_str!("multiexp/multiexp.cl");

pub struct MultiexpKernel {
    proque: ProQue
}

impl MultiexpKernel {

    pub fn create(n: u32) -> GPUResult<MultiexpKernel> {
        let src = format!("{}\n{}\n{}\n{}", DEFS_SRC, FIELD_SRC, EC_SRC, KERNEL_SRC);
        let pq = ProQue::builder().src(src).dims(n).build()?;
        Ok(MultiexpKernel {proque: pq})
    }

    pub fn radix_fft(&mut self) -> GPUResult<()> {
        Ok(())
    }
}
