typedef uint uint32;
typedef ulong uint64;

// Adds `num` to `i`th digit of `res` and propagates carry in case of overflow
void add_digit(uint32 *res, uint32 num) {
  uint32 old = *res;
  *res += num;
  if(*res < old) {
    res++;
    while(++(*(res++)) == 0);
  }
}
