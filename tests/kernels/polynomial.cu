// CUDA twin of tests/kernels/polynomial.tc
extern "C" __global__ void polynomial(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = ((3.0f * v - 2.0f) * v + 0.5f) * v - 7.25f;
  }
}
