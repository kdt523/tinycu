// CUDA twin of tests/kernels/stencil_1d.tc
extern "C" __global__ void stencil_1d(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (i == 0 || i == n - 1) {
      y[i] = x[i];
    } else {
      y[i] = 0.25f * x[i - 1] + 0.5f * x[i] + 0.25f * x[i + 1];
    }
  }
}
