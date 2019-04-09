#include <opencl_memory>
#include <opencl_work_item>

kernel void example_kernel(cl::global_ptr<int[]> input)
{
  cl::local<int[256]> array;

  uint gid = cl::get_global_id(0);
  array[gid] = input[gid];
  // ...
}