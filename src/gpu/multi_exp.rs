use ocl::{ProQue, Buffer, MemFlags};
use ocl::prm::Ulong4;
use paired::bls12_381::Fr;
use std::cmp;
use ff::Field;
use super::error::GPUResult;

static UINT256_SRC : &str = include_str!("uint256.cl");
static KERNEL_SRC : &str = include_str!("multi_exp.cl");

pub struct Multi_Exp_Kernel {
    proque: ProQue,
    fft_src_buffer: Buffer<Ulong4>,
    fft_dst_buffer: Buffer<Ulong4>,
    fft_omg_buffer: Buffer<Ulong4>
}

impl Multi_Exp_Kernel {

    pub fn create(n: u32) -> GPUResult<Multi_Exp_Kernel> {
        let src = format!("{}\n{}", UINT256_SRC, KERNEL_SRC);
        let pq = ProQue::builder().src(src).dims(n).build()?;
        let srcbuff = Buffer::builder().queue(pq.queue().clone())
            .flags(MemFlags::new().read_write()).len(n)
            .build()?;
        let dstbuff = Buffer::builder().queue(pq.queue().clone())
            .flags(MemFlags::new().read_write()).len(n)
            .build()?;
        let omgbuff = Buffer::builder().queue(pq.queue().clone())
            .flags(MemFlags::new().read_write()).len(32)
            .build()?;

        Ok(Multi_Exp_Kernel {proque: pq,
                      fft_src_buffer: srcbuff,
                      fft_dst_buffer: dstbuff,
                      fft_omg_buffer: omgbuff})
    }

    fn multi_exp(&mut self, a: &mut [Ulong4], lgn: u32, lgp: u32, deg: u32, max_deg: u32, in_src: bool) -> ocl::Result<()> {
        let kernel = self.proque.kernel_builder("multi_exp")
            .global_work_size(1)
            .local_work_size(1)
            .arg(1)
            .build()?;
        unsafe { kernel.enq()?; }
        Ok(())
    }

}
