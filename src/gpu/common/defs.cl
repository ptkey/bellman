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
  uint64 res;
  if(i < 64)
    res = (l.s0 >> i);
  else if(i < 128)
    res = (l.s1 >> (i - 64));
  else if(i < 192)
    res = (l.s2 >> (i - 128));
  else
    res = (l.s3 >> (i - 192));
  return res & 1;
}

uint get_bits(ulong4 l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= get_bit(l, skip + i);
  }
  return ret;
}

ulong shr(__global ulong4 *l, uint i) {
  uint shift = 64 - i;
  ulong and = ((1 << i) - 1) << shift;
  ulong anded = (l->s3 & and) >> shift;
  l->s3 = (l->s3 << i) + ((l->s2 & and) >> shift);
  l->s2 = (l->s2 << i) + ((l->s1 & and) >> shift);
  l->s1 = (l->s1 << i) + ((l->s0 & and) >> shift);
  l->s0 = (l->s0 << i);
  return anded;
}

limb mac_with_carry(limb a, limb b, limb c, limb *carry) {
  limb lo = a * b;
  limb hi = mul_hi(a, b);
  hi += lo + c < lo; lo += c;
  hi += lo + *carry < lo; lo += *carry;
  *carry = hi;
  return lo;
}