// Naive multiexp

__kernel void naive_multiexp(__global affine *bases,
    __global projective *results,
    __global ulong4 *exps,
    uint nbases, uint nexps) {
  projective p = {ZERO, ONE, ZERO};
  for(uint i = 0; i < nexps; i++)
    p = ec_add(p, ec_mul(bases[i], exps[i]));
  results[0] = p;
}

/* Batched multiexp */

bool get_bit(ulong4 l, uint i) {
  if(i < 64)
    return (l.s0 >> i) & 1;
  else if(i < 128)
    return (l.s1 >> (i - 64)) & 1;
  else if(i < 192)
    return (l.s2 >> (i - 128)) & 1;
  else
    return (l.s3 >> (i - 192)) & 1;
}

__kernel void batched_multiexp(__global affine *bases,
    __global projective *results,
    __global ulong4 *exps,
    __global bool *dm,
    uint nbases, uint nexps) {

  projective p = {ZERO, ONE, ZERO};
  for(int i = 255; i >= 0; i--) {
    p = ec_double(p);
    for(uint j = 0; j < nexps; j++) {
      if(get_bit(exps[j], i))
        p = ec_add_mixed(p, bases[j]);
    }
  }
  results[0] = p;
}
