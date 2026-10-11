// CUDA twin of tests/kernels/clamp.tc
extern "C" __global__ void clamp(float* x, float lo, float hi, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = fminf(fmaxf(x[i], lo), hi);
  }
}
