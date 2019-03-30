typedef struct {
  unsigned long a;
  unsigned long b;
  unsigned long c;
  unsigned long d;
} uint256;

uint256 add(uint256 a, uint256 b)
{
    uint256 c;
    c.a = a.a + b.a;
    c.b = a.b + b.b + (c.a < a.a);
    c.c = a.c + b.c + (c.b < a.b);
    c.d = a.d + b.d + (c.c < a.c);
    return c;
}

__kernel void fft(__global int* buffer) {
  buffer[get_global_id(0)] *= 5;
}
