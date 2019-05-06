#define MAX_RADIX_DEGREE (7)

__kernel void radix_fft(__global ulong4* src,
                        __global ulong4* dst,
                        __global ulong4* tpq,
                        uint n,
                        ulong4 om,
                        uint lgp,
                        uint deg) // 1=>radix2, 2=>radix4, 3=>radix8, ...
{
  __global uint256 *x = src;
  __global uint256 *y = dst;
  __global uint256 *pq = tpq;
  uint256 omega = *(uint256*)&om;
  uint32 index = get_global_id(0);
  uint32 t = n >> deg;
  uint32 p = 1 << lgp;
  uint32 k = index & (p - 1);

  x += index;
  y += ((index - k) << deg) + k;

  uint256 uu[1<<MAX_RADIX_DEGREE]; uint256 *u = uu;
  uint256 vv[1<<MAX_RADIX_DEGREE]; uint256 *v = vv;

  uint32 count = 1 << deg; // 2^deg
  uint32 counth = count >> 1; // Half of count

  uint256 twiddle = powmod(omega, (n >> lgp >> deg) * k);
  uint256 curr = ONE;
  for(uint32 i = 0; i < count; i++) {
    u[i] = mulmod(curr, x[i*t]);
    curr = mulmod(curr, twiddle);
  }

  uint32 pqshift = MAX_RADIX_DEGREE - deg;

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
          v[b] = mulmod(v[b], pq[i * lg << pqshift]);
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
