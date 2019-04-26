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
  y += ((i - k) << 1) + k;

  uint256 twiddle = powmod(omega, (n >> lgp >> 1) * k);

  uint256 u0 = x[0];
  uint256 u1 = mulmod(twiddle, x[1*t]);

  y[0] = addmod(u0, u1);
  y[p] = submod(u0, u1);
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

  uint256 p1q2 = powmod(omega, n >> 2);

  uint256 v0 = addmod(u0, u2);
  uint256 v1 = submod(u0, u2);
  uint256 v2 = addmod(u1, u3);
  uint256 v3 = submod(u1, u3); v3 = mulmod(v3, p1q2);

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

  uint256 p1q4 = powmod(omega, n >> 3);
  uint256 p2q4 = mulmod(p1q4, p1q4); // or p1q2
  uint256 p3q4 = mulmod(p2q4, p1q4);

  uint256 v0 = addmod(u0, u4);
  uint256 v4 = submod(u0, u4);
  uint256 v1 = addmod(u1, u5);
  uint256 v5 = submod(u1, u5); v5 = mulmod(v5, p1q4);
  uint256 v2 = addmod(u2, u6);
  uint256 v6 = submod(u2, u6); v6 = mulmod(v6, p2q4);
  uint256 v3 = addmod(u3, u7);
  uint256 v7 = submod(u3, u7); v7 = mulmod(v7, p3q4);

  u0 = addmod(v0, v2);
  u2 = submod(v0, v2);
  u1 = addmod(v1, v3);
  u3 = submod(v1, v3); u3 = mulmod(u3, p2q4);
  u4 = addmod(v4, v6);
  u6 = submod(v4, v6);
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

__kernel void radix16_fft(__global ulong4* src,
                  __global ulong4* dst,
                  uint n,
                  ulong4 om,
                  uint lgp) {

  __global uint256 *x = src;
  __global uint256 *y = dst;
  uint256 omega = *(uint256*)&om;
  uint32 i = get_global_id(0);
  uint32 t = n >> 4;
  uint32 p = 1 << lgp;
  uint32 k = i & (p - 1);

  x += i;
  y += ((i - k) << 4) + k;

  uint256 twiddle = powmod(omega, (n >> lgp >> 4) * k);
  uint256 u0 = x[0];
  uint256 u1 = mulmod(twiddle, x[t]);
  uint256 u2 = x[2*t];
  uint256 u3 = mulmod(twiddle, x[3*t]);
  uint256 u4 = x[4*t];
  uint256 u5 = mulmod(twiddle, x[5*t]);
  uint256 u6 = x[6*t];
  uint256 u7 = mulmod(twiddle, x[7*t]);
  uint256 u8 = x[8*t];
  uint256 u9 = mulmod(twiddle, x[9*t]);
  uint256 u10 = x[10*t];
  uint256 u11 = mulmod(twiddle, x[11*t]);
  uint256 u12 = x[12*t];
  uint256 u13 = mulmod(twiddle, x[13*t]);
  uint256 u14 = x[14*t];
  uint256 u15 = mulmod(twiddle, x[15*t]);

  twiddle = mulmod(twiddle, twiddle);
  u2 = mulmod(twiddle, u2);
  u3 = mulmod(twiddle, u3);
  u6 = mulmod(twiddle, u6);
  u7 = mulmod(twiddle, u7);
  u10 = mulmod(twiddle, u10);
  u11 = mulmod(twiddle, u11);
  u14 = mulmod(twiddle, u14);
  u15 = mulmod(twiddle, u15);
  twiddle = mulmod(twiddle, twiddle);
  u4 = mulmod(twiddle, u4);
  u5 = mulmod(twiddle, u5);
  u6 = mulmod(twiddle, u6);
  u7 = mulmod(twiddle, u7);
  u12 = mulmod(twiddle, u12);
  u13 = mulmod(twiddle, u13);
  u14 = mulmod(twiddle, u14);
  u15 = mulmod(twiddle, u15);
  twiddle = mulmod(twiddle, twiddle);
  u8 = mulmod(twiddle, u8);
  u9 = mulmod(twiddle, u9);
  u10 = mulmod(twiddle, u10);
  u11 = mulmod(twiddle, u11);
  u12 = mulmod(twiddle, u12);
  u13 = mulmod(twiddle, u13);
  u14 = mulmod(twiddle, u14);
  u15 = mulmod(twiddle, u15);

  uint256 p1q8 = powmod(omega, n >> 4);
  uint256 p2q8 = mulmod(p1q8, p1q8); // or p1q4
  uint256 p3q8 = mulmod(p2q8, p1q8);
  uint256 p4q8 = mulmod(p3q8, p1q8); // or p1q2 or p2q4
  uint256 p5q8 = mulmod(p4q8, p1q8);
  uint256 p6q8 = mulmod(p5q8, p1q8); // or p3q4
  uint256 p7q8 = mulmod(p6q8, p1q8);

  uint256 v0 = addmod(u0, u8);
  uint256 v8 = submod(u0, u8);
  uint256 v1 = addmod(u1, u9);
  uint256 v9 = submod(u1, u9); v9 = mulmod(v9, p1q8);
  uint256 v2 = addmod(u2, u10);
  uint256 v10 = submod(u2, u10); v10 = mulmod(v10, p2q8);
  uint256 v3 = addmod(u3, u11);
  uint256 v11 = submod(u3, u11); v11 = mulmod(v11, p3q8);
  uint256 v4 = addmod(u4, u12);
  uint256 v12 = submod(u4, u12); v12 = mulmod(v12, p4q8);
  uint256 v5 = addmod(u5, u13);
  uint256 v13 = submod(u5, u13); v13 = mulmod(v13, p5q8);
  uint256 v6 = addmod(u6, u14);
  uint256 v14 = submod(u6, u14); v14 = mulmod(v14, p6q8);
  uint256 v7 = addmod(u7, u15);
  uint256 v15 = submod(u7, u15); v15 = mulmod(v15, p7q8);

  u0 = addmod(v0, v4);
  u4 = submod(v0, v4);
  u1 = addmod(v1, v5);
  u5 = submod(v1, v5); u5 = mulmod(u5, p2q8);
  u2 = addmod(v2, v6);
  u6 = submod(v2, v6); u6 = mulmod(u6, p4q8);
  u3 = addmod(v3, v7);
  u7 = submod(v3, v7); u7 = mulmod(u7, p6q8);
  u8 = addmod(v8, v12);
  u12 = submod(v8, v12);
  u9 = addmod(v9, v13);
  u13 = submod(v9, v13); u13 = mulmod(u13, p2q8);
  u10 = addmod(v10, v14);
  u14 = submod(v10, v14); u14 = mulmod(u14, p4q8);
  u11 = addmod(v11, v15);
  u15 = submod(v11, v15); u15 = mulmod(u15, p6q8);

  v0 = addmod(u0, u2);
  v2 = submod(u0, u2);
  v1 = addmod(u1, u3);
  v3 = submod(u1, u3); v3 = mulmod(v3, p4q8);
  v4 = addmod(u4, u6);
  v6 = submod(u4, u6);
  v5 = addmod(u5, u7);
  v7 = submod(u5, u7); v7 = mulmod(v7, p4q8);
  v8 = addmod(u8, u10);
  v10 = submod(u8, u10);
  v9 = addmod(u9, u11);
  v11 = submod(u9, u11); v11 = mulmod(v11, p4q8);
  v12 = addmod(u12, u14);
  v14 = submod(u12, u14);
  v13 = addmod(u13, u15);
  v15 = submod(u13, u15); v15 = mulmod(v15, p4q8);

  y[0] = addmod(v0, v1);
  y[p] = addmod(v8, v9);
  y[2*p] = addmod(v4, v5);
  y[3*p] = addmod(v12, v13);
  y[4*p] = addmod(v2, v3);
  y[5*p] = addmod(v10, v11);
  y[6*p] = addmod(v6, v7);
  y[7*p] = addmod(v14, v15);
  y[8*p] = submod(v0, v1);
  y[9*p] = submod(v8, v9);
  y[10*p] = submod(v4, v5);
  y[11*p] = submod(v12, v13);
  y[12*p] = submod(v2, v3);
  y[13*p] = submod(v10, v11);
  y[14*p] = submod(v6, v7);
  y[15*p] = submod(v14, v15);
}

__kernel void radix_r_fft(__global ulong4* src,
                  __global ulong4* dst,
                  uint n,
                  ulong4 om,
                  uint lgp,
                  __local ulong4* tmp) {

  __global uint256 *x = src;
  __global uint256 *y = dst;
  __global uint256 *u = tmp;

  uint256 omega = *(uint256*)&om;
  uint32 p = 1 << lgp;

  uint32 r = get_local_size(0);
  uint32 i = get_local_id(0);
  // uint32 k = i & (p - 1);  
  uint32 k = get_group_id(0)&(p-1);
  uint32 j = (get_group_id(0)-k)*2*r+k;

  uint256 twiddle = powmod(omega, (n >> lgp >> r) * k);

  uint256 sn = powmod(i*twiddle, (n >> lgp >> r) * k);
  u[i] = mulmod(sn, x[get_group_id(0) + i * get_num_groups(0)]);
  sn = powmod((i+r)*twiddle, (n >> lgp >> r) * k);
  u[i+r] = mulmod(sn, x[get_group_id(0) + (i+r) * get_num_groups(0)]);

  uint256 a,b;

  for(uint32 bit = r; bit > 0; bit >>= 1) {
    uint32 di = i&(bit-1);
    uint32 i0 = (i<<1)-di;
    uint32 i1 = i0 + bit;
    sn = powmod(omega, (n >> lgp >> bit) * (di*k));
    a = u[i0];
    b = u[i1];
    u[i0] = addmod(a, b);
    u[i1] = mulmod(sn, submod(a, b));
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  y[j+i*p] = u[bitreverse(2*r,i)];
  y[j+(i+r)*p] = u[bitreverse(2*r, i+r)];
}