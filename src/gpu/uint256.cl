typedef uint uint32;
typedef ulong uint64;

typedef struct {
  uint32 val[8];
} uint256;

#define ONE ((uint256){0xfffffffe,0x00000001,0x00034802,0x5884b7fa,0xecbc4ff5,0x998c4fef,0xacc5056f,0x1824b159})
#define P0INV ((uint32)4294967295)

void add_digit(uint32 *res, uint32 index, uint32 num) {
  while(true) {
    uint32 old = res[index];
    res[index] += num;
    if(res[index] < old) {
      num = 1;
      index++;
    } else break;
  }
}

void sub_digit(uint32 *res, uint32 index, uint32 num) {
  while(true) {
    uint32 old = res[index];
    res[index] -= num;
    if(res[index] > old) {
      num = 1;
      index++;
    } else break;
  }
}

bool gte(uint256 a, uint256 b) {
  for(int i = 7; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

uint256 add(uint256 a, uint256 b) {
  for(int i = 0; i < 8; i++)
    add_digit(a.val, i, b.val[i]);
  return a;
}

uint256 sub(uint256 a, uint256 b) {
  for(int i = 0; i < 8; i++) {
    sub_digit(a.val, i, b.val[i]);
  }
  return a;
}

uint256 mul_reduce(uint256 a, uint256 b) {
  uint256 P = {0x00000001,0xffffffff,0xfffe5bfe,0x53bda402,
               0x09a1d805,0x3339d808,0x299d7d48,0x73eda753};
  uint32 res[16] = {0};
  for(int i = 0; i < 8; i++) {
    for(int j = 0; j < 8; j++) {
      uint64 total = (uint64)a.val[i] * (uint64)b.val[j];
      uint32 lo = total & 0xffffffff;
      uint32 hi = total >> 32;
      add_digit(res, i + j, lo);
      add_digit(res, i + j + 1, hi);
    }
  }
  for (int i = 0; i < 8; i++)
  {
    uint64 u = ((uint64)P0INV * (uint64)res[i]) & 0xffffffff;
    for(int j = 0; j < 8; j++) {
      uint64 total = u * (uint64)P.val[j];
      uint32 lo = total & 0xffffffff;
      uint32 hi = total >> 32;
      add_digit(res, i + j, lo);
      add_digit(res, i + j + 1, hi);
    }
  }
  uint256 result;
  for(int i = 0; i < 8; i++) result.val[i] = res[i+8];
  if(gte(result, P))
    result = sub(result, P);
  return result;
}

uint256 mulmod(uint256 a, uint256 b) {
  return mul_reduce(a, b);
}

uint256 negmod(uint256 a) {
  uint256 P = {0x00000001,0xffffffff,0xfffe5bfe,0x53bda402,
               0x09a1d805,0x3339d808,0x299d7d48,0x73eda753};
  return sub(P, a);
}

uint256 submod(uint256 a, uint256 b) {
  uint256 P = {0x00000001,0xffffffff,0xfffe5bfe,0x53bda402,
               0x09a1d805,0x3339d808,0x299d7d48,0x73eda753};
  uint256 res = sub(a, b);

  if(!gte(a, b)) res = add(res, P);

  return res;
}

uint256 addmod(uint256 a, uint256 b) {
  return submod(a, negmod(b));
}

uint256 powmod(uint256 b, uint64 p) {
  uint256 res = ONE;
  while(p > 0) {
    if (p & 1)
      res = mulmod(res, b);
    p = p >> 1;
    b = mulmod(b, b);
  }
  return res;
}
