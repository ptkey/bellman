__kernel void radix2_fft(__global ulong4* src,
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
