__kernel void radix2_fft(__global ulong4* src,
                  __global ulong4* dst,
                  uint n,
                  ulong4 om,
                  uint lgp) {

  __global uint256 *x = src;
  __global uint256 *y = dst;
  uint256 omega = *(uint256*)&om;

  uint32 i = get_global_id(0);
  uint32 t = n >> 1;
  uint32 p = 1 << lgp;

  uint32 k = i & (p - 1);
  x += i;

  uint256 u0;
  u0 = x[0];
  uint256 u1;
  u1 = x[t];

  uint256 twiddle = powmod(omega, (n >> lgp >> 1) * k);
  u1 = mulmod(u1, twiddle);

  uint256 tmp = submod(u0, u1);
  u0 = addmod(u0, u1);
  u1 = tmp;

  uint j = ((i-k)<<1) + k;
  y += j;
  y[0] = u0;
  y[p] = u1;
}

__kernel void radix4_fft(__global ulong4* src,
                  __global ulong4* dst,
                  uint n,
                  ulong4 om,
                  uint lgp) {

  __global uint256 *x = src;
  __global uint256 *y = dst;
  uint256 omega = *(uint256*)&om;
  uint32 i = get_global_id(0);
  uint32 t = n >> 2;
  uint32 p = 1 << lgp;
  uint32 k = i & (p - 1);

  x += i;
  y += ((i - k) << 2) + k;

  uint256 twiddle = powmod(omega, (n >> lgp >> 2) * k);

  uint256 u0 = x[0];
  uint256 u1 = mulmod(twiddle, x[1*t]);
  uint256 u2 = x[2*t];
  uint256 u3 = mulmod(twiddle, x[3*t]);

  twiddle = mulmod(twiddle, twiddle);
  u2 = mulmod(twiddle, u2);
  u3 = mulmod(twiddle, u3);

  uint256 v0 = addmod(u0, u2);
  uint256 v1 = submod(u0, u2);
  uint256 v2 = addmod(u1, u3);
  uint256 v3 = mulmod(submod(u1, u3), powmod(omega, n >> 2));

  y[0] = addmod(v0, v2);
  y[p] = addmod(v1, v3);
  y[2*p] = submod(v0, v2);
  y[3*p] = submod(v1, v3);
}
