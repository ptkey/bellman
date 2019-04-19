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

__kernel void radix8_fft(__global ulong4* src,
                  __global ulong4* dst,
                  uint n,
                  ulong4 om,
                  uint lgp) {

  __global uint256 *x = src;
  __global uint256 *y = dst;
  uint256 omega = *(uint256*)&om;
  uint32 i = get_global_id(0);
  uint32 t = n >> 3;
  uint32 p = 1 << lgp;
  uint32 k = i & (p - 1);

  x += i;
  y += ((i - k) << 3) + k;

  uint256 twiddle = powmod(omega, (n >> lgp >> 3) * k);
  uint256 u0 = x[0];
  uint256 u1 = mulmod(twiddle, x[t]);
  uint256 u2 = x[2*t];
  uint256 u3 = mulmod(twiddle, x[3*t]);
  uint256 u4 = x[4*t];
  uint256 u5 = mulmod(twiddle, x[5*t]);
  uint256 u6 = x[6*t];
  uint256 u7 = mulmod(twiddle, x[7*t]);
  twiddle = mulmod(twiddle, twiddle);
  u2 = mulmod(twiddle, u2);
  u3 = mulmod(twiddle, u3);
  u6 = mulmod(twiddle, u6);
  u7 = mulmod(twiddle, u7);
  twiddle = mulmod(twiddle, twiddle);
  u4 = mulmod(twiddle, u4);
  u5 = mulmod(twiddle, u5);
  u6 = mulmod(twiddle, u6);
  u7 = mulmod(twiddle, u7);

  uint256 p0q4 = powmod(omega, (n >> 3) * 0); // or p0q2
  uint256 p1q4 = powmod(omega, (n >> 3) * 1);
  uint256 p2q4 = powmod(omega, (n >> 3) * 2); // or p1q2
  uint256 p3q4 = powmod(omega, (n >> 3) * 3);

  uint256 v0 = addmod(u0, u4);
  uint256 v4 = submod(u0, u4); v4 = mulmod(v4, p0q4);
  uint256 v1 = addmod(u1, u5);
  uint256 v5 = submod(u1, u5); v5 = mulmod(v5, p1q4);
  uint256 v2 = addmod(u2, u6);
  uint256 v6 = submod(u2, u6); v6 = mulmod(v6, p2q4);
  uint256 v3 = addmod(u3, u7);
  uint256 v7 = submod(u3, u7); v7 = mulmod(v7, p3q4);

  u0 = addmod(v0, v2);
  u2 = submod(v0, v2); u2 = mulmod(u2, p0q4);
  u1 = addmod(v1, v3);
  u3 = submod(v1, v3); u3 = mulmod(u3, p2q4);
  u4 = addmod(v4, v6);
  u6 = submod(v4, v6); u6 = mulmod(u6, p0q4);
  u5 = addmod(v5, v7);
  u7 = submod(v5, v7); u7 = mulmod(u7, p2q4);

  y[0] = addmod(u0, u1);
  y[p] = addmod(u4, u5);
  y[2*p] = addmod(u2, u3);
  y[3*p] = addmod(u6, u7);
  y[4*p] = submod(u0, u1);
  y[5*p] = submod(u4, u5);
  y[6*p] = submod(u2, u3);
  y[7*p] = submod(u6, u7);
}
