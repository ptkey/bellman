extern crate ocl;

use self::ocl::{ProQue, Buffer};
use self::ocl::prm::Ulong4;
use pairing::bls12_381::Fr;

static KERNEL_SRC : &str = include_str!("fft.cl");

pub struct Kernel {
    proque: ProQue,
    fft_buffer: Buffer<Ulong4>
}

pub fn create_kernel(n: u32) -> Kernel {
    let pq = ProQue::builder().src(KERNEL_SRC).dims(n).build().expect("Cannot create kernel!");
    let buffer = pq.create_buffer::<Ulong4>().expect("Cannot allocate buffer!");
    Kernel { proque: pq, fft_buffer: buffer }
}

impl Kernel {
    pub fn fft(&mut self, a: &mut [Fr], omega: &Fr, lgn: u32) -> ocl::Result<()> {
        let ta = unsafe { std::mem::transmute::<&mut [Fr], &mut [Ulong4]>(a) };
        let tomega = *(unsafe { std::mem::transmute::<&Fr, &Ulong4>(omega) });

        let mut vec = vec![Ulong4::zero(); self.fft_buffer.len()];
        for i in 0..self.fft_buffer.len() { vec[i] = ta[i]; }

        self.fft_buffer.write(&vec).enq()?;

        let kernel = self.proque.kernel_builder("fft")
            .arg(&self.fft_buffer)
            .arg(ta.len() as u32)
            .arg(lgn as u32)
            .arg(tomega)
            .build()?;

        unsafe { kernel.enq()?; }

        self.fft_buffer.read(&mut vec).enq()?;

        for i in 0..a.len() { ta[i] = vec[i]; }
        Ok(())
    }
}
