use ocl::{ProQue, Buffer, MemFlags};
use ocl::traits::OclPrm;
use ocl::builders::ProgramBuilder;
use paired::Engine;
use std::cmp;
use std::sync::Arc;
use ocl::prm::{Ulong4, Uchar};
use ff::{Field, PrimeField, PrimeFieldRepr, ScalarEngine};
use paired::{CurveAffine, CurveProjective};
use std::marker::PhantomData;
use super::error::{GPUResult, GPUError};
use super::sources;

const NUM_WORKS : usize = 1024;

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

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct PTable { table: [G1ProjectiveStruct;7] }
unsafe impl OclPrm for PTable { }

pub struct MultiexpKernel<E> where E: Engine {
    proque: ProQue,
    g1_base_buffer: Buffer<G1AffineStruct>,
    g1_result_buffer: Buffer<G1ProjectiveStruct>,
    g2_base_buffer: Buffer<G2AffineStruct>,
    g2_result_buffer: Buffer<G2ProjectiveStruct>,
    exp_buffer: Buffer<Ulong4>,
    dm_buffer: Buffer<Uchar>,
    field_type: PhantomData<E>
}

impl<E> MultiexpKernel<E> where E: Engine {

    pub fn create(n: u32) -> GPUResult<MultiexpKernel<E>> {
        let src = sources::multiexp_kernel::<E::Fq>();
        let mut bldr = ProgramBuilder::new();
        bldr.src(src);
        let pq = ProQue::builder().prog_bldr(bldr).dims(n).build()?;
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
        let ptablebuff = Buffer::<PTable>::builder().queue(pq.queue().clone())
            .flags(MemFlags::new().read_write()).len(n)
            .build()?;
        Ok(MultiexpKernel {proque: pq,
            g1_base_buffer: g1basebuff, g1_result_buffer: g1resbuff,
            g2_base_buffer: g2basebuff, g2_result_buffer: g2resbuff,
            exp_buffer: expbuff, dm_buffer: dmbuff,
            field_type: PhantomData})
    }

    pub fn multiexp<G>(&mut self,
            bases: Arc<Vec<G>>,
            exps: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
            dm: Vec<bool>,
            skip: usize)
            -> GPUResult<(<G as CurveAffine>::Projective)>
            where G: CurveAffine {


        // Calculate P Lookup tabel for window size [3]
        let mut _s = 0;
        for (&base, dm) in bases.iter().zip(dm.iter()) {
            let mut pvec: Vec<PTable> = Vec::new();
            let mut tmp0 = base.into_projective();
            if (*dm) {
                for i in 0..7 {
                  let mut acc = G::Projective::zero();
                  let add = tmp0.add_assign_mixed(&acc);
                  pvec[_s].table[i] = unsafe { std::mem::transmute::<&[E::G1Projective], &[G1ProjectiveStruct]>(add) };
                }
            }
            _s += 1;
        }

        // TODO: write pvec to self.ptablebuff

        let sz = std::mem::size_of::<G>(); // Trick, used for dispatching between G1 and G2!
        let exps = unsafe { std::mem::transmute::<Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,Arc<Vec<<E::Fr as PrimeField>::Repr>>>(exps) }.to_vec();
        let texps = unsafe { std::mem::transmute::<&[<E::Fr as PrimeField>::Repr], &[Ulong4]>(&exps[..]) };
        let tdm = unsafe { std::mem::transmute::<&[bool], &[Uchar]>(&dm[..]) };
        let n = texps.len();
        if sz == 104 {
            let bases = unsafe { std::mem::transmute::<Arc<Vec<G>>,Arc<Vec<E::G1Affine>>>(bases) }.to_vec();
            let tbases = unsafe { std::mem::transmute::<&[E::G1Affine], &[G1AffineStruct]>(&bases[..]) };
            self.g1_base_buffer.write(tbases).enq()?;
            self.exp_buffer.write(texps).enq()?;
            self.dm_buffer.write(tdm).enq()?;
            let kernel = self.proque.kernel_builder("G1_batched_multiexp")
                .global_work_size([n])
                .arg(&self.g1_base_buffer)
                .arg(&self.g1_result_buffer)
                .arg(&self.exp_buffer)
                .arg(&self.dm_buffer)
                .arg(&self.ptablebuff)
                .arg(skip as u32)
                .arg(texps.len() as u32)
                .build()?;
            unsafe { kernel.enq()?; }
            let mut res = vec![<G as CurveAffine>::Projective::zero(); n];
            let mut tres = unsafe { std::mem::transmute::<&mut [<G as CurveAffine>::Projective], &mut [G1ProjectiveStruct]>(&mut res) };
            self.g1_result_buffer.read(tres).enq()?;
            let mut acc = <G as CurveAffine>::Projective::zero();
            for i in 0..n { acc.add_assign(&res[i]); }
            return Ok((acc))
        } else if sz == 200 {
            let mut acc = <G as CurveAffine>::Projective::zero();
            return Ok((acc))
        }
        else {
            Err(GPUError {msg: "Only Bls12-381 G1 is supported!".to_string()} )
        }
    }
}
