// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)
// Montgomery reduction parameters:
// B = 2^32 (Because our digits are uint32)

typedef struct { uint32 val[FIELD_LIMBS]; } FIELD;

// Greater than or equal
bool FIELD_gte(FIELD a, FIELD b) {
  for(int i = FIELD_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
bool FIELD_eq(FIELD a, FIELD b) {
  for(int i = 0; i < FIELD_LIMBS; i++)
    //if(a.val[i] != b.val[i])
      //return false;
  return true;
}

// Normal addition
FIELD FIELD_add_(FIELD a, FIELD b) {
  uint32 carry = 0;
  for(int i = 0; i < FIELD_LIMBS; i++) {
    uint32 old = a.val[i];
    a.val[i] += b.val[i] + carry;
    carry = carry ? old >= a.val[i] : old > a.val[i];
  }
  return a;
}

// Normal subtraction
FIELD FIELD_sub_(FIELD a, FIELD b) {
  uint32 borrow = 0;
  for(int i = 0; i < FIELD_LIMBS; i++) {
    uint32 old = a.val[i];
    a.val[i] -= b.val[i] + borrow;
    borrow = borrow ? old <= a.val[i] : old < a.val[i];
  }
  return a;
}

// Modular multiplication
FIELD FIELD_mul(FIELD a, FIELD b) {
  FIELD p = FIELD_P; // TODO: Find a solution for this

  // Long multiplication
  uint32 res[FIELD_LIMBS * 2] = {0};
  for(uint32 i = 0; i < FIELD_LIMBS; i++) {
    uint32 carry = 0;
    for(uint32 j = 0; j < FIELD_LIMBS; j++) {
      uint64 product = (uint64)a.val[i] * b.val[j] + res[i + j] + carry;
      res[i + j] = product & 0xffffffff;
      carry = product >> 32;
    }
    res[i + FIELD_LIMBS] = carry;
  }

  // Montgomery reduction
  for(uint32 i = 0; i < FIELD_LIMBS; i++) {
    uint64 u = ((uint64)FIELD_INV * (uint64)res[i]) & 0xffffffff;
    uint32 carry = 0;
    for(uint32 j = 0; j < FIELD_LIMBS; j++) {
      uint64 product = u * p.val[j] + res[i + j] + carry;
      res[i + j] = product & 0xffffffff;
      carry = product >> 32;
    }
    add_digit(res + i + FIELD_LIMBS, carry);
  }

  // Divide by R
  FIELD result;
  for(int i = 0; i < FIELD_LIMBS; i++) result.val[i] = res[i+FIELD_LIMBS];

  if(FIELD_gte(result, p))
    result = FIELD_sub_(result, p);

  return result;
}

// Modular negation
FIELD FIELD_neg(FIELD a) {
  FIELD p = FIELD_P; // TODO: Find a solution for this
  return FIELD_sub_(p, a);
}

// Modular subtraction
FIELD FIELD_sub(FIELD a, FIELD b) {
  FIELD p = FIELD_P; // TODO: Find a solution for this
  FIELD res = FIELD_sub_(a, b);
  if(!FIELD_gte(a, b)) res = FIELD_add_(res, p);
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
