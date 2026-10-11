// CUDA twin of tests/kernels/reduce_shared.tc
extern "C" __global__ void reduce_shared(const float* in, float* out, int n) {
  __shared__ float buf[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  float v = 0.0f;
  if (i < n) {
    v = in[i];
  }
  buf[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s) {
      buf[tid] += buf[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = buf[0];
  }
}
