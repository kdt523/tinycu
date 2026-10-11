// CUDA twin of tests/kernels/leaky_relu.tc
extern "C" __global__ void leaky_relu(const float* x, float* y, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    if (v < 0.0f) {
      y[i] = alpha * v;
    } else {
      y[i] = v;
    }
  }
}
