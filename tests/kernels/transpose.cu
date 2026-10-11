// CUDA twin of tests/kernels/transpose.tc
extern "C" __global__ void transpose(const float* in, float* out, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows * cols) {
    int r = idx / cols;
    int c = idx % cols;
    out[c * rows + r] = in[idx];
  }
}
