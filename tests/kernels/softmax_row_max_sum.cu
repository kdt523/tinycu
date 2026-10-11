// CUDA twin of tests/kernels/softmax_row_max_sum.tc
extern "C" __global__ void softmax_row_max_sum(const float* x, float* rmax, float* rsum, int rows,
                                               int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows) {
    float m = x[r * cols];
    for (int j = 1; j < cols; j++) {
      m = fmaxf(m, x[r * cols + j]);
    }
    float s = 0.0f;
    for (int j = 0; j < cols; j++) {
      s += x[r * cols + j] - m;
    }
    rmax[r] = m;
    rsum[r] = s;
  }
}
