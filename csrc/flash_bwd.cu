#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <vector>

#include "flash_common.h"

template <typename scalar_t>
__global__ void flash_bwd_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ out,
    const float* __restrict__ m,
    const float* __restrict__ l,
    scalar_t* __restrict__ dq,
    scalar_t* __restrict__ dk,
    scalar_t* __restrict__ dv,
    int B,
    int H,
    int N,
    int D,
    bool causal,
    float scale) {
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int tid = threadIdx.x;

    __shared__ float scores[BLOCK_N];
    __shared__ float probs[BLOCK_N];
    __shared__ float dscores[BLOCK_N];
    __shared__ float dq_acc[MAX_HEAD_DIM];
    __shared__ float partial[MAX_HEAD_DIM];
    __shared__ float delta;

    const int row_base = offset4(batch, head, row, 0, H, N, D);
    const int stats_index = (batch * H + head) * N + row;
    const float row_m = m[stats_index];
    const float row_l = l[stats_index];

    if (tid < D) {
        dq_acc[tid] = 0.0f;
        partial[tid] = to_float(dout[row_base + tid]) * to_float(out[row_base + tid]);
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            sum += partial[d];
        }
        delta = sum;
    }
    __syncthreads();

    for (int col_start = 0; col_start < N; col_start += BLOCK_N) {
        const int col = col_start + tid;

        if (tid < BLOCK_N) {
            const bool valid = col < N && (!causal || col <= row);
            float score = -INFINITY;
            if (valid) {
                const int k_base = offset4(batch, head, col, 0, H, N, D);
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    dot += to_float(q[row_base + d]) * to_float(k[k_base + d]);
                }
                score = dot * scale;
            }
            scores[tid] = score;
            probs[tid] = expf(score - row_m) / row_l;
        }
        __syncthreads();

        if (tid < BLOCK_N) {
            const int col_j = col_start + tid;
            float dp = 0.0f;
            if (col_j < N && (!causal || col_j <= row)) {
                const int v_base = offset4(batch, head, col_j, 0, H, N, D);
                for (int d = 0; d < D; ++d) {
                    dp += to_float(dout[row_base + d]) * to_float(v[v_base + d]);
                }
            }
            dscores[tid] = probs[tid] * (dp - delta) * scale;
        }
        __syncthreads();

        if (tid < D) {
            float sum = 0.0f;
            const int tile_count = min(BLOCK_N, N - col_start);
            for (int j = 0; j < tile_count; ++j) {
                const int col_j = col_start + j;
                const int k_index = offset4(batch, head, col_j, tid, H, N, D);
                sum += dscores[j] * to_float(k[k_index]);
            }
            dq_acc[tid] += sum;
        }

        const int tile_count = min(BLOCK_N, N - col_start);
        for (int index = tid; index < tile_count * D; index += blockDim.x) {
            const int j = index / D;
            const int d = index % D;
            const int col_j = col_start + j;
            if (col_j < N && (!causal || col_j <= row)) {
                const int kv_index = offset4(batch, head, col_j, d, H, N, D);
                const float q_val = to_float(q[row_base + d]);
                const float do_val = to_float(dout[row_base + d]);
                atomic_add_value(&dk[kv_index], dscores[j] * q_val);
                atomic_add_value(&dv[kv_index], probs[j] * do_val);
            }
        }
        __syncthreads();
    }

    if (tid < D) {
        dq[row_base + tid] = from_float<scalar_t>(dq_acc[tid]);
    }
}

std::vector<torch::Tensor> flash_bwd(
    torch::Tensor dout,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor m,
    torch::Tensor l,
    bool causal,
    double scale) {
    auto dq = torch::empty_like(q);
    auto dk = torch::zeros_like(k);
    auto dv = torch::zeros_like(v);

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);

    dim3 grid(N, H, B);
    dim3 block(THREADS);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "flash_bwd_kernel", [&] {
        flash_bwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            dout.data_ptr<scalar_t>(),
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            m.data_ptr<float>(),
            l.data_ptr<float>(),
            dq.data_ptr<scalar_t>(),
            dk.data_ptr<scalar_t>(),
            dv.data_ptr<scalar_t>(),
            B,
            H,
            N,
            D,
            causal,
            static_cast<float>(scale));
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dq, dk, dv};
}
