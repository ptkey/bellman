extern crate ocl;

use self::ocl::ProQue;
use self::ocl::prm::Ulong4;
use pairing::bls12_381::{Bls12, Fr};

pub fn fft(a: &mut [Fr], omega: &Fr, lgn: u32) -> ocl::Result<()> {
    let ta = unsafe { std::mem::transmute::<&mut [Fr], &mut [Ulong4]>(a) };
    let tomega = *(unsafe { std::mem::transmute::<&Fr, &Ulong4>(omega) });

    // This is getting recompiled again and again. Make it a global variable!
    let src = include_str!("fft.cl");
    let pro_que = ProQue::builder()
        .src(src)
        .dims(ta.len())
        .build()?;

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
    Ok(())
}
