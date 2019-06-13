#pragma OPENCL EXTENSION cl_nv_compiler_options : enable

typedef uint uint32;
typedef ulong uint64;
typedef uint64 limb;

// Adds `num` to `i`th digit of `res` and propagates carry in case of overflow
void add_digit(limb *res, limb num) {
  limb old = *res;
  *res += num;
  if(*res < old) {
    res++;
    while(++(*(res++)) == 0);
  }
}

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

limb mac_with_carry(limb a, limb b, limb c, limb *carry) {
  limb lo = a * b;
  limb hi = mul_hi(a, b);
  hi += lo + c < lo; lo += c;
  hi += lo + *carry < lo; lo += *carry;
  *carry = hi;
  return lo;
}