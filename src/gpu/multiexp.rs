use ocl::{ProQue, Buffer, MemFlags};
use ocl::traits::OclPrm;
use paired::bls12_381::{FrRepr, G1Affine, G1, G2Affine, G2};
use std::cmp;
use std::sync::Arc;
use ocl::prm::{Ulong4, Uchar};
use ff::{Field, PrimeField, PrimeFieldRepr, ScalarEngine};
use paired::{CurveAffine, CurveProjective};
use super::error::{GPUResult, GPUError};
use super::sources;

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct FqStruct { vals: [u64; 6] }
unsafe impl OclPrm for FqStruct { }

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct G1AffineStruct { x: FqStruct, y: FqStruct, inf: bool }
unsafe impl OclPrm for G1AffineStruct { }

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct G1ProjectiveStruct { x: FqStruct, y: FqStruct, z: FqStruct }
unsafe impl OclPrm for G1ProjectiveStruct { }

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct Fq2Struct { c0: FqStruct, c1: FqStruct }
unsafe impl OclPrm for Fq2Struct { }

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct G2AffineStruct { x: Fq2Struct, y: Fq2Struct, inf: bool }
unsafe impl OclPrm for G2AffineStruct { }

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct G2ProjectiveStruct { x: Fq2Struct, y: Fq2Struct, z: Fq2Struct }
unsafe impl OclPrm for G2ProjectiveStruct { }

pub struct MultiexpKernel {
    proque: ProQue,
    g1_base_buffer: Buffer<G1AffineStruct>,
    g1_result_buffer: Buffer<G1ProjectiveStruct>,
    g2_base_buffer: Buffer<G2AffineStruct>,
    g2_result_buffer: Buffer<G2ProjectiveStruct>,
    exp_buffer: Buffer<Ulong4>,
    dm_buffer: Buffer<Uchar>
}

impl MultiexpKernel {

    pub fn create(n: u32) -> GPUResult<MultiexpKernel> {
        let src = sources::multiexp_kernel();
        let pq = ProQue::builder().src(src).dims(n).build()?;
        let g1basebuff = Buffer::<G1AffineStruct>::builder().queue(pq.queue().clone())
            .flags(MemFlags::new().read_write()).len(n)
            .build()?;
        let expbuff = Buffer::<Ulong4>::builder().queue(pq.queue().clone())
            .flags(MemFlags::new().read_write()).len(n)
            .build()?;
        let g1resbuff = Buffer::<G1ProjectiveStruct>::builder().queue(pq.queue().clone())
            .flags(MemFlags::new().read_write()).len(n)
            .build()?;
        let g2basebuff = Buffer::<G2AffineStruct>::builder().queue(pq.queue().clone())
            .flags(MemFlags::new().read_write()).len(n)
            .build()?;
        let g2resbuff = Buffer::<G2ProjectiveStruct>::builder().queue(pq.queue().clone())
            .flags(MemFlags::new().read_write()).len(n)
            .build()?;
        let dmbuff = Buffer::<Uchar>::builder().queue(pq.queue().clone())
            .flags(MemFlags::new().read_write()).len(n)
            .build()?;
        Ok(MultiexpKernel {proque: pq,
            g1_base_buffer: g1basebuff, g1_result_buffer: g1resbuff,
            g2_base_buffer: g2basebuff, g2_result_buffer: g2resbuff,
            exp_buffer: expbuff, dm_buffer: dmbuff})
    }

    pub fn multiexp<G>(&mut self,
            bases: Arc<Vec<G>>,
            exps: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
            dm: Vec<bool>,
            skip: usize)
            -> GPUResult<(<G as CurveAffine>::Projective)>
            where G: CurveAffine {

        let sz = std::mem::size_of::<G>(); // Trick, used for dispatching between G1 and G2!
        let exps = unsafe { std::mem::transmute::<Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,Arc<Vec<FrRepr>>>(exps) }.to_vec();
        let texps = unsafe { std::mem::transmute::<&[FrRepr], &[Ulong4]>(&exps[..]) };
        let tdm = unsafe { std::mem::transmute::<&[bool], &[Uchar]>(&dm[..]) };
        if sz == 104 {
            let bases = unsafe { std::mem::transmute::<Arc<Vec<G>>,Arc<Vec<G1Affine>>>(bases) }.to_vec();
            let tbases = unsafe { std::mem::transmute::<&[G1Affine], &[G1AffineStruct]>(&bases[..]) };
            self.g1_base_buffer.write(tbases).enq()?;
            self.exp_buffer.write(texps).enq()?;
            self.dm_buffer.write(tdm).enq()?;
            let kernel = self.proque.kernel_builder("G1_batched_multiexp")
                .global_work_size([1])
                .arg(&self.g1_base_buffer)
                .arg(&self.g1_result_buffer)
                .arg(&self.exp_buffer)
                .arg(&self.dm_buffer)
                .arg(skip as u32)
                .arg(texps.len() as u32)
                .build()?;
            unsafe { kernel.enq()?; }
            let mut res = [<G as CurveAffine>::Projective::zero()];
            let mut tres = unsafe { std::mem::transmute::<&mut [<G as CurveAffine>::Projective], &mut [G1ProjectiveStruct]>(&mut res) };
            self.g1_result_buffer.read(tres).enq()?;
            return Ok((res[0]))
        } else if sz == 200 {
            let bases = unsafe { std::mem::transmute::<Arc<Vec<G>>,Arc<Vec<G2Affine>>>(bases) }.to_vec();
            let tbases = unsafe { std::mem::transmute::<&[G2Affine], &[G2AffineStruct]>(&bases[..]) };
            self.g2_base_buffer.write(tbases).enq()?;
            self.exp_buffer.write(texps).enq()?;
            self.dm_buffer.write(tdm).enq()?;
            let kernel = self.proque.kernel_builder("G2_batched_multiexp")
                .global_work_size([1])
                .arg(&self.g2_base_buffer)
                .arg(&self.g2_result_buffer)
                .arg(&self.exp_buffer)
                .arg(&self.dm_buffer)
                .arg(skip as u32)
                .arg(texps.len() as u32)
                .build()?;
            unsafe { kernel.enq()?; }
            let mut res = [<G as CurveAffine>::Projective::zero()];
            let mut tres = unsafe { std::mem::transmute::<&mut [<G as CurveAffine>::Projective], &mut [G2ProjectiveStruct]>(&mut res) };
            self.g2_result_buffer.read(tres).enq()?;
            return Ok((res[0]))
        }
        else {
            Err(GPUError {msg: "Only Bls12-381 G1 is supported!".to_string()} )
        }
    }
}
