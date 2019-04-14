__kernel void custom_radix2_fft(__global ulong4* buffer,
                  uint n,
                  uint lgn,
                  ulong4 om,
                  uint lgm) {

  int index = get_global_id(0);

  __global uint256 *elems = buffer;
  uint256 omega = *(uint256*)&om;

  uint works = n >> (lgm + 1);
  uint m = 1 << lgm;

  uint32 k = index >> lgm << (lgm + 1);
  uint32 j = index & (m - 1);
  uint256 w = powmod(omega, (n >> lgm >> 1) * j);

  uint256 t = elems[k+j+m];
  t = mulmod(t, w);
  uint256 tmp = elems[k+j];
  tmp = submod(tmp, t);
  elems[k+j+m] = tmp;
  elems[k+j] = addmod(elems[k+j], t);
}

__kernel void bealto_radix2_fft(__global ulong4* src,
                  __global ulong4* dst,
                  uint n,
                  uint lgn,
                  ulong4 om,
                  uint lgm) {

  __global uint256 *x = src;
  __global uint256 *y = dst;
  uint256 omega = *(uint256*)&om;

  uint i = get_global_id(0);
  uint t = n >> 1;
  uint m = 1 << lgm;

  uint k = i & (m - 1);

  uint256 u0;
  u0 = x[i];
  uint256 u1;
  u1 = x[i+t];

  uint256 twiddle = powmod(omega, (n >> lgm >> 1) * k);
  u1 = mulmod(u1, twiddle);

  uint256 tmp = submod(u0, u1);
  u0 = addmod(u0, u1);
  u1 = tmp;

  uint j = (i<<1) - k;
  y[j] = u0;
  y[j+m] = u1;
}