#define MAX_RADIX_DEGREE (8)

__kernel void radix_fft(__global ulong4* src,
                        __global ulong4* dst,
                        uint n,
                        ulong4 om,
                        uint lgp,
                        uint deg) // 1=>radix2, 2=>radix4, 3=>radix8, ...
{
  __global uint256 *x = src;
  __global uint256 *y = dst;
  uint256 omega = *(uint256*)&om;
  uint32 index = get_global_id(0);
  uint32 t = n >> deg;
  uint32 p = 1 << lgp;
  uint32 k = index & (p - 1);

  x += index;
  y += ((index - k) << deg) + k;

  uint256 uu[1<<MAX_RADIX_DEGREE]; uint256 *u = uu;
  uint256 vv[1<<MAX_RADIX_DEGREE]; uint256 *v = vv;
  uint256 pq[1<<MAX_RADIX_DEGREE>>1];

  uint32 count = 1 << deg; // 2^deg
  uint32 counth = count >> 1; // Half of count

  uint256 twiddle = powmod(omega, (n >> lgp >> deg) * k);
  uint256 curr = ONE;
  for(uint32 i = 0; i < count; i++) {
    u[i] = mulmod(curr, x[i*t]);
    curr = mulmod(curr, twiddle);
  }

  pq[0] = ONE;
  pq[1] = powmod(omega, n >> deg);
  for(uint32 i = 2; i < counth; i++)
    pq[i] = mulmod(pq[i - 1], pq[1]);

  for(uint32 rnd = 0; rnd < deg - 1; rnd++) {
    uint32 lg = 1 << rnd;
    uint32 st = counth >> rnd;
    for(uint32 j = 0; j < lg; j++) {
      uint32 offset = j * st << 1;
      for(uint32 i = 0; i < st; i++) {
        uint32 a = offset + i, b = offset + i + st;
        v[a] = addmod(u[a], u[b]);
        v[b] = submod(u[a], u[b]);
        if(i > 0)
          v[b] = mulmod(v[b], pq[i * lg]);
      }
    }
    uint256 *tmp = u; u = v; v = tmp; // Now result is in v, swap!
  }

  for(uint32 i = 0; i < counth; i++) {
    uint32 rev = bitreverse(i, deg);
    y[i*p] = addmod(u[rev], u[rev+1]);
    y[(i+counth)*p] = submod(u[rev], u[rev+1]);
  }
}

__kernel void radix_ifft(__global ulong4* src,
                        __global ulong4* dst,
                        uint n,
                        ulong4 mv,
                        uint lgp,
                        uint deg) // 1=>radix2, 2=>radix4, 3=>radix8, ...
{
  __global uint256 *x = src;
  __global uint256 *y = dst;
  uint256 minv = *(uint256*)&mv;
  uint32 index = get_global_id(0);
  uint32 t = n >> deg;
  uint32 p = 1 << lgp;
  uint32 k = index & (p - 1);

  x += index;
  y += ((index - k) << deg) + k;

  uint256 uu[1 << MAX_RADIX_DEGREE]; uint256 *u = uu;
  uint256 vv[1 << MAX_RADIX_DEGREE]; uint256 *v = vv;
  uint256 pq[1 << MAX_RADIX_DEGREE >> 1];

  uint32 count = 1 << deg; // 2^deg
  uint32 counth = count >> 1; // Half of count

  uint256 twiddle = powmod(minv, (n >> lgp >> deg) * k);
  uint256 curr = ONE;
  for(uint32 i = 0; i < count; i++) {
    u[i] = mulmod(curr, x[i*t]);
    curr = mulmod(curr, twiddle);
  }

  pq[0] = ONE;
  pq[1] = powmod(minv, n >> deg);
  for(uint32 i = 2; i < counth; i++)
    pq[i] = mulmod(pq[i - 1], pq[1]);

  for(uint32 rnd = 0; rnd < deg - 1; rnd++) {
    uint32 lg = 1 << rnd;
    uint32 st = counth >> rnd;
    for(uint32 j = 0; j < lg; j++) {
      uint32 offset = j * st << 1;
      for(uint32 i = 0; i < st; i++) {
        uint32 a = offset + i, b = offset + i + st;
        v[a] = addmod(u[a], u[b]);
        v[b] = submod(u[a], u[b]);
        if(i > 0)
          v[b] = mulmod(v[b], pq[i * lg]);
      }
    }
    uint256 *tmp = u; u = v; v = tmp; // Now result is in v, swap!
  }

  for(uint32 i = 0; i < counth; i++) {
    uint32 rev = bitreverse(i, deg);
    y[i*p] = submod(u[rev], u[rev+1]);
    y[(i+counth)*p] = addmod(u[rev], u[rev+1]);
  }
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

  uint32 r = get_local_size(0);
  uint32 i = get_local_id(0);
  // uint32 k = i & (p - 1);
  uint32 k = get_group_id(0)&(p-1);
  uint32 j = (get_group_id(0)-k)*2*r+k;

  uint256 twiddle = powmod(omega, (n >> p >> r) * k);

  uint256 sn = powmod(mulmod(twiddle, *(uint256 *)i), (n >> p >> r) * k);
  u[i] = mulmod(sn, x[get_group_id(0) + i * get_num_groups(0)]);
  sn = powmod(mulmod(*(uint256 *)(i+r), twiddle), (n >> p >> r) * k);
  u[i+r] = mulmod(sn, x[get_group_id(0) + (i+r) * get_num_groups(0)]);

  uint256 a,b;

  for(uint32 bit = r; bit > 0; bit >>= 1) {
    uint32 di = i&(bit-1);
    uint32 i0 = (i<<1)-di;
    uint32 i1 = i0 + bit;
    sn = powmod(omega, (n >> p >> bit) * (di*k));
    a = u[i0];
    b = u[i1];
    u[i0] = addmod(a, b);
    u[i1] = mulmod(sn, submod(a, b));
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  y[j+i*p] = u[bitreverse(2*r,i)];
  y[j+(i+r)*p] = u[bitreverse(2*r, i+r)];
}
