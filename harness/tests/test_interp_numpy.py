"""The interpreter (our oracle) checked against independent NumPy reference implementations.

All float comparisons are bit-exact: every kernel here performs IEEE f32 operations whose
results NumPy reproduces exactly (no FMA contraction on either side).
"""
import numpy as np
import pytest

from tcharness.tinycu import interp

RNG = np.random.default_rng(1234)
BLOCK = 64


def blocks(n, block=BLOCK):
    return (n + block - 1) // block


def rand_f32(n, lo=-100.0, hi=100.0):
    return RNG.uniform(lo, hi, n).astype(np.float32)


def rand_i32(n, lo=-1000, hi=1000):
    return RNG.integers(lo, hi, n, dtype=np.int32)


def assert_bits_equal(got, want):
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.dtype == want.dtype and got.shape == want.shape
    if got.dtype == np.float32:
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
    else:
        np.testing.assert_array_equal(got, want)


GUARD = "int i = blockIdx.x * blockDim.x + threadIdx.x;\n  if (i < n) {\n"

SIZES = [1, 63, 64, 1000]  # includes non-multiples of the block size


@pytest.mark.parametrize("n", SIZES)
def test_vecadd_i32(n):
    a, b, c = rand_i32(n), rand_i32(n), np.zeros(n, np.int32)
    src = "kernel vecadd(int* a, int* b, int* c, int n) {\n  " + GUARD + "    c[i] = a[i] + b[i];\n  }\n}\n"
    r = interp(src, [a, b, c, n], blocks(n), BLOCK)
    assert r.ok, r.error
    assert_bits_equal(r.outputs[2], a + b)


@pytest.mark.parametrize("n", SIZES)
def test_saxpy(n):
    alpha, x, y = np.float32(2.5), rand_f32(n), rand_f32(n)
    src = "kernel saxpy(float a, float* x, float* y, int n) {\n  " + GUARD + "    y[i] = a * x[i] + y[i];\n  }\n}\n"
    r = interp(src, [alpha, x, y, n], blocks(n), BLOCK)
    assert r.ok, r.error
    assert_bits_equal(r.outputs[2], alpha * x + y)


def test_relu_and_leaky():
    n = 500
    x = rand_f32(n)
    src = """kernel relu(float* x, float* y, float* z, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = max(v, 0.0);
    if (v < 0.0) { z[i] = 0.01 * v; } else { z[i] = v; }
  }
}"""
    r = interp(src, [x, np.zeros(n, np.float32), np.zeros(n, np.float32), n], blocks(n), BLOCK)
    assert r.ok, r.error
    assert_bits_equal(r.outputs[1], np.maximum(x, np.float32(0)))
    assert_bits_equal(r.outputs[2], np.where(x < 0, np.float32(0.01) * x, x))


def test_clamp():
    n = 300
    x = rand_f32(n)
    src = """kernel clamp(float* x, float lo, float hi, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  x[i] = min(max(x[i], lo), hi);
}"""
    r = interp(src, [x, np.float32(-10), np.float32(25.5), n], blocks(n), BLOCK)
    assert r.ok, r.error
    assert_bits_equal(r.outputs[0], np.minimum(np.maximum(x, np.float32(-10)), np.float32(25.5)))


def test_int_hash_wraparound():
    n = 256
    x = rand_i32(n, -2**31, 2**31 - 1)
    src = """kernel hash(int* x, int* h, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int v = x[i];
    v = v * 1103515245 + 12345;
    v = v * 1664525 + 1013904223 - i;
    h[i] = v;
  }
}"""
    r = interp(src, [x, np.zeros(n, np.int32), n], blocks(n), BLOCK)
    assert r.ok, r.error
    v = x.astype(np.int64)
    v = (v * 1103515245 + 12345).astype(np.int64) & 0xFFFFFFFF
    v = (v * 1664525 + 1013904223 - np.arange(n)) & 0xFFFFFFFF
    assert_bits_equal(r.outputs[1], v.astype(np.uint32).view(np.int32))


def test_polynomial_horner():
    n = 200
    x = rand_f32(n, -2, 2)
    src = """kernel poly(float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = ((3.0 * v - 2.0) * v + 0.5) * v - 7.25;
  }
}"""
    r = interp(src, [x, np.zeros(n, np.float32), n], blocks(n), BLOCK)
    assert r.ok, r.error
    f = np.float32
    assert_bits_equal(r.outputs[1], ((f(3) * x - f(2)) * x + f(0.5)) * x - f(7.25))


def test_loop_sum_and_break_continue():
    n = 100
    x = rand_i32(n, 0, 50)
    src = """kernel k(int* x, int* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int s = 0;
    for (int j = 0; j < x[i]; j++) {
      if (j % 3 == 0) { continue; }
      if (j > 40) { break; }
      s += j * j;
    }
    o[i] = s;
  }
}"""
    r = interp(src, [x, np.zeros(n, np.int32), n], blocks(n), BLOCK)
    assert r.ok, r.error
    want = [sum(j * j for j in range(v) if j % 3 != 0 and j <= 40) for v in x]
    assert_bits_equal(r.outputs[1], np.array(want, np.int32))


def test_while_collatz():
    n = 128
    x = np.arange(1, n + 1, dtype=np.int32)
    src = """kernel collatz(int* x, int* steps, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int v = x[i];
    int c = 0;
    while (v != 1 && c < 1000) {
      if (v % 2 == 0) { v = v / 2; } else { v = 3 * v + 1; }
      c++;
    }
    steps[i] = c;
  }
}"""
    r = interp(src, [x, np.zeros(n, np.int32), n], blocks(n), BLOCK)
    assert r.ok, r.error

    def collatz(v):
        c = 0
        while v != 1 and c < 1000:
            v = v // 2 if v % 2 == 0 else 3 * v + 1
            c += 1
        return c

    assert_bits_equal(r.outputs[1], np.array([collatz(int(v)) for v in x], np.int32))


def test_branchy_nested_if():
    n = 400
    x = rand_i32(n, -50, 50)
    src = """kernel branchy(int* x, int* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int v = x[i];
    if (v < 0) {
      if (v < -25) { o[i] = -2; } else { o[i] = -1; }
    } else if (v == 0) {
      o[i] = 0;
    } else {
      if (v % 2 == 0 && v > 10 || v == 7) { o[i] = 2; } else { o[i] = 1; }
    }
  }
}"""
    r = interp(src, [x, np.zeros(n, np.int32), n], blocks(n), BLOCK)
    assert r.ok, r.error
    want = np.select(
        [x < -25, x < 0, x == 0, ((x % 2 == 0) & (x > 10)) | (x == 7)], [-2, -1, 0, 2], 1
    ).astype(np.int32)
    assert_bits_equal(r.outputs[1], want)


def test_grid_stride_loop():
    n = 1000
    x = rand_f32(n)
    src = """kernel scale(float* x, float s, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    x[i] = x[i] * s;
  }
}"""
    r = interp(src, [x, np.float32(-1.5), n], 3, 32)  # 96 threads cover 1000 elements
    assert r.ok, r.error
    assert_bits_equal(r.outputs[0], x * np.float32(-1.5))


def test_abs_sqrt_casts():
    n = 256
    x = rand_f32(n, -1000, 1000)
    src = """kernel k(float* x, float* s, int* t, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    s[i] = sqrt(abs(x[i]));
    t[i] = (int)x[i] + abs((int)x[i] - 3) % 7;
  }
}"""
    r = interp(src, [x, np.zeros(n, np.float32), np.zeros(n, np.int32), n], blocks(n), BLOCK)
    assert r.ok, r.error
    assert_bits_equal(r.outputs[1], np.sqrt(np.abs(x)))
    ti = np.trunc(x).astype(np.int32)
    assert_bits_equal(r.outputs[2], (ti + np.fmod(np.abs(ti - 3), 7)).astype(np.int32))


def test_shared_reduction():
    n, block = 1000, 128
    x = rand_f32(n, -1, 1)
    src = """kernel reduce(float* in, float* out, int n) {
  shared float buf[128];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  float v = 0.0;
  if (i < n) { v = in[i]; }
  buf[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s) { buf[tid] += buf[tid + s]; }
    __syncthreads();
  }
  if (tid == 0) { out[blockIdx.x] = buf[0]; }
}"""
    nb = blocks(n, block)
    r = interp(src, [x, np.zeros(nb, np.float32), n], nb, block)
    assert r.ok, r.error
    # Reproduce the exact tree order in NumPy for a bit-exact comparison.
    padded = np.zeros(nb * block, np.float32)
    padded[:n] = x
    want = []
    for b in range(nb):
        buf = padded[b * block:(b + 1) * block].copy()
        s = block // 2
        while s > 0:
            buf[:s] = buf[:s] + buf[s:2 * s]
            s //= 2
        want.append(buf[0])
    assert_bits_equal(r.outputs[1], np.array(want, np.float32))


def test_matmul_naive():
    m, k, nn = 7, 5, 9
    a, b = rand_f32(m * k, -2, 2), rand_f32(k * nn, -2, 2)
    src = """kernel matmul(float* a, float* b, float* c, int m, int k, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m * n) {
    int row = idx / n;
    int col = idx % n;
    float acc = 0.0;
    for (int j = 0; j < k; j++) {
      acc += a[row * k + j] * b[j * n + col];
    }
    c[idx] = acc;
  }
}"""
    r = interp(src, [a, b, np.zeros(m * nn, np.float32), m, k, nn], blocks(m * nn), BLOCK)
    assert r.ok, r.error
    A, B = a.reshape(m, k), b.reshape(k, nn)
    want = np.zeros((m, nn), np.float32)
    for j in range(k):  # same accumulation order as the kernel
        want = want + A[:, j:j + 1] * B[j:j + 1, :]
    assert_bits_equal(r.outputs[2], want.reshape(-1))


def test_runtime_error_reported():
    src = "kernel k(int* o, int n) { o[threadIdx.x] = 1; }"
    r = interp(src, [np.zeros(4, np.int32), 0], 1, 8)
    assert not r.ok
    assert "out-of-bounds store to 'o' at index 4" in r.error
