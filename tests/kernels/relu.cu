// CUDA twin of tests/kernels/relu.tc
extern "C" __global__ void relu(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    if (v < 0.0f) {
      v = 0.0f;
    }
    y[i] = v;
  }
}
