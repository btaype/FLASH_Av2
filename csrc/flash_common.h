#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " debe ser un tensor CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " debe ser contiguo")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

constexpr int MAX_HEAD_DIM = 128;
constexpr int BLOCK_N = 64;
constexpr int MAX_BLOCK_N = 128;
constexpr int THREADS = 128;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int WARP_SIZE = 32;
constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK;

#ifdef __CUDACC__
template <typename scalar_t>
__device__ inline float to_float(scalar_t x) {
    return static_cast<float>(x);
}

template <>
__device__ inline float to_float<c10::Half>(c10::Half x) {
    return __half2float(static_cast<__half>(x));
}

template <typename scalar_t>
__device__ inline scalar_t from_float(float x) {
    return static_cast<scalar_t>(x);
}

template <>
__device__ inline c10::Half from_float<c10::Half>(float x) {
    return c10::Half(x);
}

template <typename scalar_t>
__device__ inline void atomic_add_value(scalar_t* address, float value) {
    atomicAdd(address, from_float<scalar_t>(value));
}

template <>
__device__ inline void atomic_add_value<c10::Half>(c10::Half* address, float value) {
    atomicAdd(reinterpret_cast<__half*>(address), __float2half(value));
}
#endif

__host__ inline void check_qkv(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    TORCH_CHECK(q.sizes() == k.sizes(), "q y k deben tener la misma forma");
    TORCH_CHECK(q.sizes() == v.sizes(), "q y v deben tener la misma forma");
    TORCH_CHECK(q.dim() == 4, "q, k y v deben tener forma [B, H, N, D]");
    const int64_t head_dim = q.size(3);
    TORCH_CHECK(
        head_dim == 64 || head_dim == 128,
        "esta version solo soporta head_dim = 64 o 128");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q y k deben tener el mismo dtype");
    TORCH_CHECK(q.scalar_type() == v.scalar_type(), "q y v deben tener el mismo dtype");
}

__host__ __device__ inline int64_t offset4(int b, int h, int n, int d, int H, int N, int D) {
    return (((int64_t)b * H + h) * N + n) * D + d;
}
