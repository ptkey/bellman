// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)
// Montgomery reduction parameters:
// B = 2^32 (Because our digits are uint32)

typedef struct { limb val[FIELD_LIMBS]; } FIELD;

void print(FIELD v) {
  printf("%u %u %u %u %u %u %u %u %u %u %u %u\n",
    v.val[11],v.val[10],v.val[9],v.val[8],v.val[7],v.val[6],v.val[5],v.val[4],v.val[3],v.val[2],v.val[1],v.val[0]);
}

// Greater than or equal
bool FIELD_gte(FIELD a, FIELD b) {
  for(char i = FIELD_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
bool FIELD_eq(FIELD a, FIELD b) {
  for(uchar i = 0; i < FIELD_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
FIELD FIELD_add_(FIELD a, FIELD b) {
  bool borrow = 0;
  for(uchar i = 0; i < FIELD_LIMBS; i++) {
    limb old = a.val[i];
    a.val[i] -= b.val[i] + borrow;
    borrow = borrow ? old <= a.val[i] : old < a.val[i];
  }
  return a;
}

// Normal subtraction
FIELD FIELD_sub_(FIELD a, FIELD b) {
  uint32 borrow = 0;
  for(int i = 0; i < FIELD_LIMBS; i++) {
    limb old = a.val[i];
    a.val[i] -= b.val[i] + borrow;
    borrow = borrow ? old <= a.val[i] : old < a.val[i];
  }
  return a;
}

uint64 hi(uint64 x) {
    return x >> 32;
}

uint64 lo(uint64 x) {
    return ((1L << 32) - 1) & x;
}

// Modular multiplication
FIELD FIELD_mul(FIELD a, FIELD b) {
  FIELD p = FIELD_P; // TODO: Find a solution for this
  // Long multiplication
  limb res[FIELD_LIMBS * 2] = {0};
  for(uchar i = 0; i < FIELD_LIMBS; i++) {
    limb carry = 0;
    for(uchar j = 0; j < FIELD_LIMBS; j++) {
      limb2 product = (limb2)a.val[i] * b.val[j] + res[i + j] + carry;
      res[i + j] = product & LIMB_MAX;
      carry = product >> LIMB_BITS;
    }
    res[i + FIELD_LIMBS] = carry;
  }

  // Montgomery reduction
  for(uchar i = 0; i < FIELD_LIMBS; i++) {
    limb u = FIELD_INV * res[i];
    limb carry = 0;
    for(uchar j = 0; j < FIELD_LIMBS; j++) {
      limb2 product = (limb2)u * p.val[j] + res[i + j] + carry;
      res[i + j] = product & LIMB_MAX;
      carry = product >> LIMB_BITS;
    }
    add_digit(res + i + FIELD_LIMBS, carry);
  }

  // Divide by R
  FIELD result;
  for(uchar i = 0; i < FIELD_LIMBS; i++) result.val[i] = res[i+FIELD_LIMBS];

  if(FIELD_gte(result, FIELD_P))
    result = FIELD_sub_(result, FIELD_P);

  return result;
}

// Modular negation
FIELD FIELD_neg(FIELD a) {
  return FIELD_sub_(FIELD_P, a);
}

// Modular subtraction
FIELD FIELD_sub(FIELD a, FIELD b) {
  FIELD res = FIELD_sub_(a, b);
  if(!FIELD_gte(a, b)) res = FIELD_add_(res, FIELD_P);
  return res;
}

// Modular addition
FIELD FIELD_add(FIELD a, FIELD b) {
  return FIELD_sub(a, FIELD_neg(b));
}

// Modular exponentiation
FIELD FIELD_pow(FIELD base, uint32 exponent) {
  FIELD res = FIELD_ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = FIELD_mul(res, base);
    exponent = exponent >> 1;
    base = FIELD_mul(base, base);
  }
  return res;
}

FIELD FIELD_pow_cached(__global FIELD *bases, uint32 exponent) {
  FIELD res = FIELD_ONE;
  uint32 i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = FIELD_mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}
