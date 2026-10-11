// CUDA twin of tests/kernels/matmul_naive.tc
extern "C" __global__ void matmul_naive(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * n) {
    int row = idx / n;
    int col = idx % n;
    float acc = 0.0f;
    for (int k = 0; k < n; k++) {
      acc += a[row * n + k] * b[k * n + col];
    }
    c[idx] = acc;
  }
}
