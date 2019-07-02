/* Batched multiexp */

#define WINDOW_SIZE 4
#define TABLE_SIZE ((1 << WINDOW_SIZE) - 1)
#define NUM_BITS 255

typedef struct {
  POINT_affine table[7];
} PTABLE;


__kernel void POINT_batched_multiexp(__global POINT_affine *bases,
    __global POINT_projective *results,
    __global ulong4 *exps,
    __global bool *dm,
    uint skip,
    uint n) {

  uint32 work = get_global_id(0);
  __local POINT_affine base; 
  base = bases[work];
  __local ulong4 exp; 
  exp = exps[work];

  if(work == 420) {
    print(base.x);
    // printf("%u", exps[work]);
    printf("------");
    print(bases[work].x);
  }

  bases += skip;
  POINT_projective p = POINT_ZERO;
  if(dm[work]) {
    for(int i = 255; i >= 0; i--) {
      p = POINT_double(p);
      if(get_bit(exps[work], i))
        p = POINT_add_mixed(p, bases[work]);
    }
  }
  results[work] = p;
}


__kernel void POINT_lookup_multiexp(__global POINT_projective *bases,
    __global POINT_projective *results,
    __global ulong4 *exps,
    __global bool *dm,
    uint skip,
    uint n) {

  // uint32 p_table[7] = {0, 1, 2, 3, 4, 5, 6, 7}; 
  // a 2^window_size lookup table

  uint32 work = get_global_id(0);
  // uint32 num_windows = NUM_BITS/WINDOW_SIZE;

  bases += skip;
  // ptable += skip;

  for(uint j = 1; j < TABLE_SIZE; j++) {
    bases[work + j * n] = POINT_add(bases[work + (j - 1) * n], bases[work]);
  }

  POINT_projective res = POINT_ZERO;

  for(uint i = 0; i < 256 / WINDOW_SIZE; i++) {
    for(int j = 0; j < WINDOW_SIZE; j++) {
      res = POINT_double(res);
    }

    if(dm[work]) {
      ulong ind = shr(&exps[work], WINDOW_SIZE);
      if(ind)
        res = POINT_add(res, bases[work + (ind - 1) * n]);      
    }
  }
  results[work] = res;
}
