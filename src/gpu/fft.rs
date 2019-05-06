use ocl::{ProQue, Buffer, MemFlags};
use ocl::prm::Ulong4;
use pairing::bls12_381::Fr;
use std::cmp;

static UINT256_SRC : &str = include_str!("uint256.cl");
static KERNEL_SRC : &str = include_str!("fft.cl");
const MAX_RADIX_DEGREE : u32 = 7; // Radix256

pub struct FFTKernel {
    proque: ProQue,
    fft_src_buffer: Buffer<Ulong4>,
    fft_dst_buffer: Buffer<Ulong4>
}

pub fn initialize(n: u32) -> FFTKernel {
    let src = format!("{}\n{}", UINT256_SRC, KERNEL_SRC);
    let pq = ProQue::builder().src(src).dims(n).build().expect("Cannot create kernel!");
    let src = Buffer::builder().queue(pq.queue().clone())
        .flags(MemFlags::new().read_write()).len(n)
        .build().expect("Cannot allocate buffer!");
    let dst = Buffer::builder().queue(pq.queue().clone())
        .flags(MemFlags::new().read_write()).len(n)
        .build().expect("Cannot allocate buffer!");
    FFTKernel {proque: pq, fft_src_buffer: src, fft_dst_buffer: dst}
}

impl FFTKernel {

    fn radix_fft_round(&mut self, a: &mut [Ulong4], omega: &Ulong4, lgn: u32, lgp: u32, deg: u32, in_src: bool) -> ocl::Result<()> {
        let n = 1 << lgn;
        let kernel = self.proque.kernel_builder("radix_fft")
            .global_work_size([n >> deg])
            .arg(if in_src { &self.fft_src_buffer } else { &self.fft_dst_buffer })
            .arg(if in_src { &self.fft_dst_buffer } else { &self.fft_src_buffer })
            .arg(a.len() as u32)
            .arg(omega)
            .arg(lgp)
            .arg(deg)
            .build()?;
        unsafe { kernel.enq()?; }
        Ok(())
    }

    pub fn radix_fft(&mut self, a: &mut [Fr], omega: &Fr, lgn: u32) -> ocl::Result<()> {

        let ta = unsafe { std::mem::transmute::<&mut [Fr], &mut [Ulong4]>(a) };
        let tomega = unsafe { std::mem::transmute::<&Fr, &Ulong4>(omega) };

        self.fft_src_buffer.write(&*ta).enq()?;

        let mut in_src = true;
        let mut lgp = 0u32;
        while lgp < lgn {
            let deg = cmp::min(MAX_RADIX_DEGREE, lgn - lgp);
            match self.radix_fft_round(ta, tomega, lgn, lgp, deg, in_src) {
                Ok(_) => (), Err(e) => return Err(e),
            }
            lgp += deg;
            in_src = !in_src;
        }

        if in_src { self.fft_src_buffer.read(ta).enq()?; }
        else { self.fft_dst_buffer.read(ta).enq()?; }

        Ok(())
    }
}
