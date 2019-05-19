use ocl::{ProQue, Buffer, MemFlags};
use ocl::prm::Ulong4;
use pairing::bls12_381::Fr;
use std::cmp;
use ff::Field;
use super::error::GPUResult;

static UINT256_SRC : &str = include_str!("uint256.cl");
static KERNEL_SRC : &str = include_str!("fft.cl");
const MAX_RADIX_DEGREE : u32 = 8; // Radix2
const MAX_LOCAL_WORK_SIZE_DEGREE : u32 = 7; // 1

pub struct FFTKernel {
    proque: ProQue,
    fft_src_buffer: Buffer<Ulong4>,
    fft_dst_buffer: Buffer<Ulong4>,
    fft_pq_buffer: Buffer<Ulong4>,
    fft_omg_buffer: Buffer<Ulong4>
}

pub fn initialize(n: u32) -> ocl::Result<FFTKernel> {
    let src = format!("{}\n{}", UINT256_SRC, KERNEL_SRC);
    let pq = ProQue::builder().src(src).dims(n).build()?;
    let srcbuff = Buffer::builder().queue(pq.queue().clone())
        .flags(MemFlags::new().read_write()).len(n)
        .build()?;
    let dstbuff = Buffer::builder().queue(pq.queue().clone())
        .flags(MemFlags::new().read_write()).len(n)
        .build()?;
    let pqbuff = Buffer::builder().queue(pq.queue().clone())
        .flags(MemFlags::new().read_write()).len(1 << MAX_RADIX_DEGREE >> 1)
        .build()?;
    let omgbuff = Buffer::builder().queue(pq.queue().clone())
        .flags(MemFlags::new().read_write()).len(32)
        .build()?;

    Ok(FFTKernel {proque: pq,
                  fft_src_buffer: srcbuff,
                  fft_dst_buffer: dstbuff,
                  fft_pq_buffer: pqbuff,
                  fft_omg_buffer: omgbuff})
}

impl FFTKernel {

    fn radix_fft_round(&mut self, a: &mut [Ulong4], lgn: u32, lgp: u32, deg: u32, max_deg: u32, in_src: bool) -> ocl::Result<()> {
        let n = 1 << lgn;
        let lwsd = cmp::min(deg - 1, MAX_LOCAL_WORK_SIZE_DEGREE);
        let kernel = self.proque.kernel_builder("radix_fft")
            .global_work_size([n >> deg << lwsd])
            .local_work_size(1 << lwsd)
            .arg(if in_src { &self.fft_src_buffer } else { &self.fft_dst_buffer })
            .arg(if in_src { &self.fft_dst_buffer } else { &self.fft_src_buffer })
            .arg(&self.fft_pq_buffer)
            .arg(&self.fft_omg_buffer)
            .arg_local::<Ulong4>(1 << deg)
            .arg(a.len() as u32)
            .arg(lgp)
            .arg(deg)
            .arg(max_deg)
            .build()?;
        unsafe { kernel.enq()?; }
        Ok(())
    }

    fn setup_pq(&mut self, omega: &Fr, n: usize, max_deg: u32) -> ocl::Result<()>  {
        let mut tpq = vec![Ulong4::zero(); 1 << max_deg >> 1];
        let mut pq = unsafe { std::mem::transmute::<&mut [Ulong4], &mut [Fr]>(&mut tpq) };
        let tw = omega.pow([(n >> max_deg) as u64]);
        pq[0] = Fr::one();
        if max_deg > 1 {
            pq[1] = tw;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i-1];
                pq[i].mul_assign(&tw);
            }
        }
        self.fft_pq_buffer.write(&tpq).enq()?;

        let mut tom = vec![Ulong4::zero(); 32];
        let mut om = unsafe { std::mem::transmute::<&mut [Ulong4], &mut [Fr]>(&mut tom) };
        om[0] = *omega;
        for i in 1..32 { om[i] = om[i-1].pow([2u64]); }
        self.fft_omg_buffer.write(&tom).enq()?;

        Ok(())
    }

    pub fn radix_fft(&mut self, a: &mut [Fr], omega: &Fr, lgn: u32) -> GPUResult<()> {
        let n = 1 << lgn;

        let ta = unsafe { std::mem::transmute::<&mut [Fr], &mut [Ulong4]>(a) };

        let max_deg = cmp::min(MAX_RADIX_DEGREE, lgn);
        self.setup_pq(omega, n, max_deg)?;

        self.fft_src_buffer.write(&*ta).enq()?;
        let mut in_src = true;
        let mut lgp = 0u32;
        while lgp < lgn {
            let deg = cmp::min(max_deg, lgn - lgp);
            self.radix_fft_round(ta, lgn, lgp, deg, max_deg, in_src)?;
            lgp += deg;
            in_src = !in_src;
        }
        if in_src { self.fft_src_buffer.read(ta).enq()?; }
        else { self.fft_dst_buffer.read(ta).enq()?; }
        self.proque.finish()?; // Wait for all commands in the queue (Including read command)

        Ok(())
    }
}
