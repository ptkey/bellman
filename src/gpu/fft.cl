__kernel void radix2_fft(__global ulong4* buffer,
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
