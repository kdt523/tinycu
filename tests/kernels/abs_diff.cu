// CUDA twin of tests/kernels/abs_diff.tc
extern "C" __global__ void abs_diff(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = fabsf(a[i] - b[i]);
  }
}
