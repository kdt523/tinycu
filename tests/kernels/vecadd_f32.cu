// CUDA twin of tests/kernels/vecadd_f32.tc
extern "C" __global__ void vecadd_f32(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
