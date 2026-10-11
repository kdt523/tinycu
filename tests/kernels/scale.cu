// CUDA twin of tests/kernels/scale.tc
extern "C" __global__ void scale(float* x, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = x[i] * s;
  }
}
