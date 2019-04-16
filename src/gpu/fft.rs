extern crate ocl;

use self::ocl::{ProQue, Platform, Device, Context, Queue, Buffer, Program, Kernel, Event, EventList, flags};
use self::ocl::prm::Ulong4;
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

// pub fn find_platform() -> Option<Platform> {
//     let platform_name = "Experimental OpenCL 2.1 CPU Only Platform";

//     for platform in Platform::list() {
//         if platform.name() == platform_name {
//             return Some(platform);
//         }
//     }

//     None
// }

impl FFT_Kernel {

    pub fn custom_radix2_fft(&mut self, a: &mut [Fr], omega: &Fr, lgn: u32) -> ocl::Result<()> {
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

        let mut vec = vec![Ulong4::zero(); self.fft_buffer.len()];
        for i in 0..self.fft_buffer.len() { vec[i] = ta[i]; }

        self.fft_buffer.write(&vec).enq()?;
        let n = 1 << lgn;

        for lgm in 0..lgn {

            let kernel = self.proque.kernel_builder("custom_radix2_fft")
                .global_work_size([n >> 1])
                .arg(&self.fft_buffer)
                .arg(ta.len() as u32)
                .arg(lgn as u32)
                .arg(tomega)
                .arg(lgm)
                .build()?;

            unsafe { kernel.enq()?; }
        }

        self.fft_buffer.read(&mut vec).enq()?;

        for i in 0..a.len() { ta[i] = vec[i]; }
        Ok(())
    }

    pub fn bealto_radix2_fft(&mut self, a: &mut [Fr], omega: &Fr, lgn: u32) -> ocl::Result<()> {

        let ta = unsafe { std::mem::transmute::<&mut [Fr], &mut [Ulong4]>(a) };
        let tomega = *(unsafe { std::mem::transmute::<&Fr, &Ulong4>(omega) });

        let mut vec = vec![Ulong4::zero(); self.fft_buffer.len()];
        for i in 0..self.fft_buffer.len() { vec[i] = ta[i]; }

        self.fft_buffer.write(&vec).enq()?;

        let mut in_src = true;
        let n = 1 << lgn;
        let base: i32 = 2;

        for lgm in 0..lgn {
            let kernel = self.proque.kernel_builder("bealto_radix2_fft")
                .global_work_size([n >> 1])
                .arg(if in_src { &self.fft_buffer } else { &self.fft_dst_buffer })
                .arg(if in_src { &self.fft_dst_buffer } else { &self.fft_buffer })
                .arg(ta.len() as u32)
                .arg(tomega)
                .arg(lgm as u32)
                .arg(base.pow(lgm) as u32)
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
       // let platform = Platform::first().unwrap();
       // println!("{:?}", p.name());

       // let device = Device::first(platform).unwrap();
       // println!("{:?}", device.name());
       // let context = Context::builder()
       //     .platform(platform)
       //     .devices(device.clone())
       //     .build().expect("Failed to create context");

       // let queue = Queue::new(&context, device, None).unwrap();

       // let buffer = Buffer::<u32>::builder()
       //     .queue(queue.clone())
       //     .flags(flags::MEM_READ_WRITE)
       //     .len(1)
       //     .fill_val(0)
       //     .build().unwrap();

        let ta = unsafe { std::mem::transmute::<&mut [Fr], &mut [Ulong4]>(a) };
        let tomega = *(unsafe { std::mem::transmute::<&Fr, &Ulong4>(omega) });

        let mut vec = vec![Ulong4::zero(); self.fft_buffer.len()];
        for i in 0..self.fft_buffer.len() { vec[i] = ta[i]; }

        self.fft_buffer.write(&vec).enq()?;

        let mut in_src = true;
        let n = 1 << lgn;
        let base: i32 = 4;

        for lgm in 0..lgn/2 {
            let kernel = self.proque.kernel_builder("bealto_radix2_fft")
                .global_work_size([(n >> 1)/2])
                .arg(if in_src { &self.fft_buffer } else { &self.fft_dst_buffer })
                .arg(if in_src { &self.fft_dst_buffer } else { &self.fft_buffer })
                .arg(ta.len() as u32)
                .arg(tomega)
                .arg(lgm as u32)
                .arg(base.pow(lgm) as u32)
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
