extern crate ocl;

use self::ocl::{ProQue, Platform, Device, Context, Queue, Buffer, Program, Kernel, Event, EventList, flags};
use self::ocl::prm::Ulong4;
use self::ocl::{core};
use pairing::bls12_381::Fr;
use std::io::Read;
use std::fs::File;

static UINT256_SRC : &str = include_str!("uint256.cl");
static KERNEL_SRC : &str = include_str!("fft.cl");

pub struct FFT_Kernel {
    proque: ProQue,
    fft_buffer: Buffer<Ulong4>,
    fft_dst_buffer: Buffer<Ulong4>
}

pub fn initialize(n: u32) -> FFT_Kernel {
    let src = format!("{}\n{}", UINT256_SRC, KERNEL_SRC);
    let pq = ProQue::builder().src(src).dims(n).build().expect("Cannot create kernel!");
    let src = pq.create_buffer::<Ulong4>().expect("Cannot allocate buffer!");
    let dst = pq.create_buffer::<Ulong4>().expect("Cannot allocate buffer!");
    FFT_Kernel {proque: pq, fft_buffer: src, fft_dst_buffer: dst }
}

pub fn find_gpu() -> bool {
    let platforms = Platform::list();
    let mut test = false;
    println!("Looping through avaliable platforms ({}):", platforms.len());

    for p_idx in 0..platforms.len() {
        let platform = &platforms[p_idx];

        let devices = Device::list(platform, Some(flags::DEVICE_TYPE_GPU)).unwrap();

        if devices.is_empty() { continue; }

        test = true;
        // for device in devices.iter() {
        //     println!("Device Name: {:?}, Vendor: {:?}", device.name().unwrap(),
        //         device.vendor().unwrap());
        // }
    }
    test
}

impl FFT_Kernel {

    pub fn bealto_radix2_fft(&mut self, a: &mut [Fr], omega: &Fr, lgn: u32) -> ocl::Result<()> {

        let ta = unsafe { std::mem::transmute::<&mut [Fr], &mut [Ulong4]>(a) };
        let tomega = *(unsafe { std::mem::transmute::<&Fr, &Ulong4>(omega) });

        let mut vec = vec![Ulong4::zero(); self.fft_buffer.len()];
        for i in 0..self.fft_buffer.len() { vec[i] = ta[i]; }

        self.fft_buffer.write(&vec).enq()?;

        let mut in_src = true;
        let n = 1 << lgn;

        for lg2p in 0..lgn {
            let kernel = self.proque.kernel_builder("bealto_radix2_fft")
                .global_work_size([n >> 1])
                .arg(if in_src { &self.fft_buffer } else { &self.fft_dst_buffer })
                .arg(if in_src { &self.fft_dst_buffer } else { &self.fft_buffer })
                .arg(ta.len() as u32)
                .arg(tomega)
                .arg(lg2p as u32)
                .build()?;

            unsafe { kernel.enq()?; }

            in_src = !in_src;
        }

        if in_src {
            self.fft_buffer.read(&mut vec).enq()?;
        } else {
            self.fft_dst_buffer.read(&mut vec).enq()?;
        }

        for i in 0..a.len() { ta[i] = vec[i]; }
        Ok(())
    }

    pub fn bealto_radix4_fft(&mut self, a: &mut [Fr], omega: &Fr, lgn: u32) -> ocl::Result<()> {

        let ta = unsafe { std::mem::transmute::<&mut [Fr], &mut [Ulong4]>(a) };
        let tomega = *(unsafe { std::mem::transmute::<&Fr, &Ulong4>(omega) });

        let mut vec = vec![Ulong4::zero(); self.fft_buffer.len()];
        for i in 0..self.fft_buffer.len() { vec[i] = ta[i]; }

        self.fft_buffer.write(&vec).enq()?;

        let mut in_src = true;
        let n = 1 << lgn;

        for lg4p in 0..lgn/2 {
            let kernel = self.proque.kernel_builder("bealto_radix4_fft")
                .global_work_size([n >> 2])
                .arg(if in_src { &self.fft_buffer } else { &self.fft_dst_buffer })
                .arg(if in_src { &self.fft_dst_buffer } else { &self.fft_buffer })
                .arg(ta.len() as u32)
                .arg(tomega)
                .arg(lg4p * 2 as u32)
                .build()?;

            unsafe { kernel.enq()?; }

            in_src = !in_src;
        }

        if in_src {
            self.fft_buffer.read(&mut vec).enq()?;
        } else {
            self.fft_dst_buffer.read(&mut vec).enq()?;
        }

        for i in 0..a.len() { ta[i] = vec[i]; }

        Ok(())
    }
}
