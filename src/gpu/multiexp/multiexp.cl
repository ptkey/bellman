__kernel void multiexp(__global affine *bases, __global ulong4 *exps,
    uint nbases, uint nexps) {
  projective p = {ZERO, ONE, ZERO};
  for(uint i = 0; i < nexps; i++)
    p = ec_add(p, ec_mul(bases[i], exps[i]));
}
