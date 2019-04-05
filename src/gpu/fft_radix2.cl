// NYU Implementation
// research link https://pdfs.semanticscholar.org/a605/0a1ea2ce396e410d6e39d6c1980b5659c1ed.pdf

#define TWOPI 6.28318530718

// fft stage called logn times from host
__kernel void fft_radix2(__ global float2* src, __ global float2* dst, const int p, const int t) {
  const int gid = get_global_id(0);
  const int k = gid & (p-1);
  src += gid;
  dst += (gid << 1) - k;

  const float2 in1 = src[0];
  const float2 in2 = src[t];

  const float theta = -TWOPI * k / p;
  float cs;
  float sn = sincos(theta, &cs);
  const float2 temp = (float2) (in2.x * cs - in2.y * sn, in2.y * cs + in2.x * sn);

  dst[0] = in1 + temp;
  dst[p] = in1 - temp;
}