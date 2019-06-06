// Naive multiexp

__kernel void naive_multiexp(__global G1_affine *bases,
    __global G1_projective *results,
    __global ulong4 *exps,
    __global bool *dm,
    uint skip,
    uint nbases, uint nexps) {
  bases += skip;
  G1_projective p = G1_ZERO;
  for(uint i = 0; i < nexps; i++)
    if(dm[i])
      p = G1_add(p, G1_mul(bases[i], exps[i]));
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

__kernel void batched_multiexp(__global G1_affine *bases,
    __global G1_projective *results,
    __global ulong4 *exps,
    __global bool *dm,
    uint skip,
    uint nbases, uint nexps) {
  bases += skip;

  G1_projective p = G1_ZERO;
  for(int i = 255; i >= 0; i--) {
    p = G1_double(p);
    for(uint j = 0; j < nexps; j++) {
      if(dm[j])
        if(get_bit(exps[j], i))
          p = G1_add_mixed(p, bases[j]);
    }
  }
  results[0] = p;
}
