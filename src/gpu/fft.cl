__kernel void fft(__global int* buffer) {
  buffer[get_global_id(0)] *= 5;
}
