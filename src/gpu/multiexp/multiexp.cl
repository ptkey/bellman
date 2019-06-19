/* Batched multiexp */

#define WINDOW_SIZE 3
#define NUM_BITS 255

typedef struct {
  POINT_affine table[7];
} PTABLE;


__kernel void POINT_batched_multiexp2(__global POINT_affine *bases,
    __global POINT_projective *results,
    __global ulong4 *exps,
    __global bool *dm,
    uint skip,
    uint n) {

  uint32 work = get_global_id(0);
  uint32 num_windows = get_global_size(0)/WINDOW_SIZE;

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


__kernel void POINT_lookup_multiexp(__global POINT_projective *results,
    __global ulong4 *exps,
    __global bool *dm,
    __global PTABLE *ptable,
    uint skip,
    uint n) {

  // uint32 p_table[7] = {0, 1, 2, 3, 4, 5, 6, 7}; 
  // a 2^window_size lookup table

  uint32 work = get_global_id(0);
  uint32 num_windows = NUM_BITS/WINDOW_SIZE;

  // bases += skip;
  POINT_projective res = POINT_ZERO;
  if(dm[work]) {
    for(int i = 0; i < num_windows; ++i) {

      for(int j = 0; j < WINDOW_SIZE; ++j) {
        res = POINT_double(res);
      }

      uint32 window_end = 254 - (WINDOW_SIZE * i);

      uint32 window_bits = 
        4 * get_bit(exps[work], window_end) +
        2 * get_bit(exps[work], window_end - 1) +
        get_bit(exps[work], window_end - 2);

      if(window_bits != 0) {
        res = POINT_add_mixed(res, ptable[work].table[window_bits]);
        // res = POINT_double(ptable[work].table[window_bits]);
      }
    }
  }
  results[work] = res;
}
