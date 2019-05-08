while p < n {
    while (p * r * 2 > n ) { r >>= 1; }
    //println!("p: {}, r: {}", p, r);
    let kernel = self.proque.kernel_builder(kernel_name.clone())
        .global_work_size(n / 2)
        .local_work_size(r)
        .arg(if in_src { &self.fft_src_buffer } else { &self.fft_dst_buffer })
        .arg(if in_src { &self.fft_dst_buffer } else { &self.fft_src_buffer })
        .arg(a.len() as u32)
        .arg(tomega)
        .arg(p)
        .arg_local::<Ulong4>((2 * r) as usize)
        .build()?;
    unsafe { kernel.enq()?; }
    in_src = !in_src;
    p = p * r * 2;
}

__kernel void radix_r_fft(__global ulong4* src,
                  __global ulong4* dst,
                  uint n,
                  ulong4 om,
                  uint p,
                  __local ulong4* tmp) {

  __global uint256 *x = src;
  __global uint256 *y = dst;
  __local uint256 *u = tmp;
  uint256 omega = *(uint256*)&om;

  uint32 gid = get_group_id(0);
  uint32 numg = get_num_groups(0);
  uint32 r = get_local_size(0);
  uint32 i = get_local_id(0);
  uint32 k = gid & (p - 1);
  uint32 j = (gid - k) * 2 * r + k;

  uint256 twiddle = powmod(omega, (n / 2 / p / r) * k);
  u[i] = mulmod(powmod(twiddle, i), x[gid + i * numg]);
  u[i+r] = mulmod(powmod(twiddle, i + r), x[gid + (i + r) * numg]);

  barrier(CLK_LOCAL_MEM_FENCE);

  uint256 a,b;
  for(uint32 bit = r; bit > 0; bit >>= 1) {
    uint32 di = i & (bit - 1);
    uint32 i0 = (i << 1) - di;
    uint32 i1 = i0 + bit;
    uint256 w = powmod(omega, n / 2 * di / bit);
    a = u[i0];
    b = u[i1];
    u[i0] = addmod(a, b);
    u[i1] = mulmod(w, submod(a, b));
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  uint32 lgr = (uint32)log2((float)r);
  y[j + i*p] = u[bitreverse(i, lgr + 1)];
  y[j + (i+r)*p] = u[bitreverse(i + r, lgr + 1)];
}
