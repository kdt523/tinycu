// CUDA twin of tests/kernels/square.tc
extern "C" __global__ void square(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] * x[i];
  }
}
