// CUDA twin of tests/kernels/dot_partial.tc
extern "C" __global__ void dot_partial(const float* a, const float* b, float* partial, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int chunks = (n + 15) / 16;
  if (t < chunks) {
    float acc = 0.0f;
    for (int j = t * 16; j < t * 16 + 16 && j < n; j++) {
      acc += a[j] * b[j];
    }
    partial[t] = acc;
  }
}
