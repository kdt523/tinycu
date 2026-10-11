// CUDA twin of tests/kernels/loop_sum.tc
extern "C" __global__ void loop_sum(const int* x, int* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int s = 0;
    for (int j = 0; j < x[i]; j++) {
      s += j * (j % 5);
    }
    o[i] = s;
  }
}
