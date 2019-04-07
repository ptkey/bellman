extern crate ocl;

use self::ocl::ProQue;
use self::ocl::prm::Ulong4;
use pairing::bls12_381::{Bls12, Fr};

static KERNEL_SRC : &str = include_str!("fft.cl");
static mut PRO_QUE : Option<ProQue> = None;

pub fn fft(a: &mut [Fr], omega: &Fr, lgn: u32) -> ocl::Result<()> {
    // WARNING: This is a big mess!
    unsafe {
        let ta = unsafe { std::mem::transmute::<&mut [Fr], &mut [Ulong4]>(a) };
        let tomega = *(unsafe { std::mem::transmute::<&Fr, &Ulong4>(omega) });

        if let Some(ref mut value) = PRO_QUE { }
        else { PRO_QUE = Some(ProQue::builder().src(KERNEL_SRC).dims(a.len()).build()?); }
        let pro_que : &mut ProQue =
            match PRO_QUE {
                Some(ref mut x) => &mut *x,
                None => panic!(),
            };

        // WARNING: It is creating a new buffer every time!
        let buffer = pro_que.create_buffer::<Ulong4>()?;

        let mut vec = vec![Ulong4::zero(); buffer.len()];
        for i in 0..buffer.len() { vec[i] = ta[i]; }

        buffer.write(&vec).enq()?;

        let kernel = pro_que.kernel_builder("fft")
            .arg(&buffer)
            .arg(ta.len() as u32)
            .arg(lgn as u32)
            .arg(tomega)
            .build()?;

        unsafe { kernel.enq()?; }

        buffer.read(&mut vec).enq()?;

        for i in 0..a.len() { ta[i] = vec[i]; }
    }
    Ok(())
}
