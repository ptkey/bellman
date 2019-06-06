// Naive multiexp

__kernel void POINT_naive_multiexp(__global POINT_affine *bases,
    __global POINT_projective *results,
    __global ulong4 *exps,
    __global bool *dm,
    uint skip,
    uint nbases, uint nexps) {
  bases += skip;
  POINT_projective p = POINT_ZERO;
  for(uint i = 0; i < nexps; i++)
    if(dm[i])
      p = POINT_add(p, POINT_mul(bases[i], exps[i]));
  results[0] = p;
}

/* Batched multiexp */

__kernel void POINT_batched_multiexp(__global POINT_affine *bases,
    __global POINT_projective *results,
    __global ulong4 *exps,
    __global bool *dm,
    uint skip,
    uint nbases, uint nexps) {
  bases += skip;

  POINT_projective p = POINT_ZERO;
  for(int i = 255; i >= 0; i--) {
    p = POINT_double(p);
    for(uint j = 0; j < nexps; j++) {
      if(dm[j])
        if(get_bit(exps[j], i))
          p = POINT_add_mixed(p, bases[j]);
    }
  }
  results[0] = p;
}
