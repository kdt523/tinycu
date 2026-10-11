// CUDA twin of tests/kernels/grid_stride_loop.tc
extern "C" __global__ void grid_stride_loop(const float* x, float* y, float s, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    y[i] = x[i] * s + 1.0f;
  }
}
