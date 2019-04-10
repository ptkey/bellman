extern crate ocl;

use self::ocl::{ProQue, Buffer};
use self::ocl::prm::Ulong4;
use pairing::bls12_381::Fr;

static KERNEL_SRC : &str = include_str!("fft.cl");

pub struct Kernel {
    proque: ProQue,
    fft_src_buffer: Buffer<Ulong4>,
    fft_dst_buffer: Buffer<Ulong4>
}

pub fn create_kernel(n: u32) -> Kernel {
    let pq = ProQue::builder().src(KERNEL_SRC).dims(n).build().expect("Cannot create kernel!");
    let src = pq.create_buffer::<Ulong4>().expect("Cannot allocate buffer!");
    let dst = pq.create_buffer::<Ulong4>().expect("Cannot allocate buffer!");
    Kernel { proque: pq, fft_src_buffer: src, fft_dst_buffer: dst }
}

impl Kernel {
    pub fn regular_fft(&mut self, a: &mut [Fr], omega: &Fr, lgn: u32) -> ocl::Result<()> {
        fn bitreverse(mut n: u32, l: u32) -> u32 {
            let mut r = 0;
            for _ in 0..l {
                r = (r << 1) | (n & 1);
                n >>= 1;
            }
            r
        }


        let ta = unsafe { std::mem::transmute::<&mut [Fr], &mut [Ulong4]>(a) };
        for k in 0..(a.len() as u32) {
            let rk = bitreverse(k, lgn);
            if k < rk {
                ta.swap(rk as usize, k as usize);
            }
        }

        let tomega = *(unsafe { std::mem::transmute::<&Fr, &Ulong4>(omega) });

        let mut vec = vec![Ulong4::zero(); self.fft_src_buffer.len()];
        for i in 0..self.fft_src_buffer.len() { vec[i] = ta[i]; }

        self.fft_src_buffer.write(&vec).enq()?;

        for lgm in 0..lgn {

            let kernel = self.proque.kernel_builder("regular_fft")
                .arg(&self.fft_src_buffer)
                .arg(ta.len() as u32)
                .arg(lgn as u32)
                .arg(tomega)
                .arg(lgm)
                .build()?;

            unsafe { kernel.enq()?; }
        }

        self.fft_src_buffer.read(&mut vec).enq()?;

        for i in 0..a.len() { ta[i] = vec[i]; }
        Ok(())
    }

    pub fn radix2_fft(&mut self, a: &mut [Fr], omega: &Fr, lgn: u32) -> ocl::Result<()> {

        let ta = unsafe { std::mem::transmute::<&mut [Fr], &mut [Ulong4]>(a) };
        let tomega = *(unsafe { std::mem::transmute::<&Fr, &Ulong4>(omega) });

        let mut vec = vec![Ulong4::zero(); self.fft_src_buffer.len()];
        for i in 0..self.fft_src_buffer.len() { vec[i] = ta[i]; }

        self.fft_src_buffer.write(&vec).enq()?;

        let mut in_src = true;

        for lgm in 0..lgn {

            let kernel = self.proque.kernel_builder("radix2_fft")
                .arg(if in_src { &self.fft_src_buffer } else { &self.fft_dst_buffer })
                .arg(if in_src { &self.fft_dst_buffer } else { &self.fft_src_buffer })
                .arg(ta.len() as u32)
                .arg(lgn as u32)
                .arg(tomega)
                .arg(1u32 << lgm)
                .build()?;

            unsafe { kernel.enq()?; }

            in_src = !in_src;
        }

        if in_src {
            self.fft_src_buffer.read(&mut vec).enq()?;
        } else {
            self.fft_dst_buffer.read(&mut vec).enq()?;
        }

        for i in 0..a.len() { ta[i] = vec[i]; }
        Ok(())
    }

    // pub fn newKern(&mut self, a: &mut [Fr], omega: &Fr, lgn: u32) -> ocl::Result<()> {

    //     Ok(())
    // }
}
