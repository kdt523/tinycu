// CUDA twin of tests/kernels/int_hash.tc
extern "C" __global__ void int_hash(const int* x, int* h, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // TinyCU ints wrap; signed overflow is UB in CUDA C++, so the twin uses unsigned.
    unsigned v = (unsigned)x[i];
    v = v * 1103515245u + 12345u;
    v = v * 1664525u + 1013904223u - (unsigned)i;
    h[i] = (int)v;
  }
}
