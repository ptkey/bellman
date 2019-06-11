/* Batched multiexp */

__kernel void POINT_batched_multiexp(__global POINT_affine *bases,
    __global POINT_projective *results,
    __global ulong4 *exps,
    __global bool *dm,
    uint skip,
    uint n) {
  uint32 work = get_global_id(0);

  bases += skip;
  POINT_projective p = POINT_ZERO;
  for(int i = 255; i >= 0; i--) {
    p = POINT_double(p);
    if(dm[work])
      p = POINT_add_mixed(p, bases[work]);

  }
  results[work] = p;
}
