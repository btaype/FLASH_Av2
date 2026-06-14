#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdlib>
#include <vector>

#include "flash_common.h"
#include "softmax_online.h"

struct FwdTileConfig {
    int block_m;
    int block_n;
};

__host__ inline FwdTileConfig choose_fwd_tile(int N, int D) {
    const char* forced_block_n = std::getenv("F_ATTENCION_V2_BLOCK_N");
    if (forced_block_n != nullptr) {
        const int block_n = std::atoi(forced_block_n);
        if (block_n == 128) {
            return {ROWS_PER_BLOCK, 128};
        }
        if (block_n == 64) {
            return {ROWS_PER_BLOCK, 64};
        }
    }
    if (D <= 64 && N >= 1024) {
        return {ROWS_PER_BLOCK, 128};
    }
    return {ROWS_PER_BLOCK, 64};
}

template <typename scalar_t, int BLOCK_M, int BLOCK_N_TILE>
__global__ void flash_fwd_kernel_tiled(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    float* __restrict__ m_out,
    float* __restrict__ l_out,
    int B,
    int H,
    int N,
    int D,
    bool causal,
    float scale) {
    const int row = blockIdx.x * BLOCK_M + threadIdx.x / WARP_SIZE;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const bool active_warp = warp_id < BLOCK_M;
    const bool row_valid = active_warp && row < N;

    __shared__ float scores[BLOCK_M][BLOCK_N_TILE];
    __shared__ float probs[BLOCK_M][BLOCK_N_TILE];
    __shared__ float acc[BLOCK_M][MAX_HEAD_DIM];
    __shared__ float m_shared[BLOCK_M];
    __shared__ float l_shared[BLOCK_M];
    __shared__ float tile_max[BLOCK_M];
    __shared__ float tile_sum[BLOCK_M];

    if (active_warp) {
        for (int d = lane; d < D; d += WARP_SIZE) {
            acc[warp_id][d] = 0.0f;
        }
        if (lane == 0) {
            m_shared[warp_id] = -INFINITY;
            l_shared[warp_id] = 0.0f;
        }
    }
    __syncthreads();

    const int q_base = offset4(batch, head, row, 0, H, N, D);

    for (int col_start = 0; col_start < N; col_start += BLOCK_N_TILE) {
        const int tile_count = min(BLOCK_N_TILE, N - col_start);

        if (active_warp) {
            for (int j = 0; j < tile_count; ++j) {
                const int col = col_start + j;
                const bool valid = row_valid && col < N && (!causal || col <= row);
                float partial = 0.0f;
                if (valid) {
                    const int k_base = offset4(batch, head, col, 0, H, N, D);
                    for (int d = lane; d < D; d += WARP_SIZE) {
                        partial += to_float(q[q_base + d]) * to_float(k[k_base + d]);
                    }
                }

                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    partial += __shfl_down_sync(0xffffffff, partial, offset);
                }
                if (lane == 0) {
                    scores[warp_id][j] = valid ? partial * scale : -INFINITY;
                }
            }
            for (int j = tile_count + lane; j < BLOCK_N_TILE; j += WARP_SIZE) {
                scores[warp_id][j] = -INFINITY;
            }
        }
        __syncthreads();

        if (active_warp && lane == 0) {
            float local_max = -INFINITY;
            if (row_valid) {
                for (int j = 0; j < tile_count; ++j) {
                    local_max = fmaxf(local_max, scores[warp_id][j]);
                }
            }
            tile_max[warp_id] = local_max;
        }
        __syncthreads();

        if (active_warp) {
            const float old_m = m_shared[warp_id];
            const float new_m = fmaxf(old_m, tile_max[warp_id]);
            const float alpha = online_alpha(old_m, new_m);

            if (lane == 0) {
                float local_sum = 0.0f;
                if (row_valid) {
                    for (int j = 0; j < tile_count; ++j) {
                        const float p = expf(scores[warp_id][j] - new_m);
                        probs[warp_id][j] = p;
                        local_sum += p;
                    }
                }
                for (int j = tile_count; j < BLOCK_N_TILE; ++j) {
                    probs[warp_id][j] = 0.0f;
                }
                tile_sum[warp_id] = local_sum;
            }
            __syncwarp();

            for (int d = lane; d < D; d += WARP_SIZE) {
                float value_sum = 0.0f;
                if (row_valid) {
                    for (int j = 0; j < tile_count; ++j) {
                        const int col_j = col_start + j;
                        const int v_index = offset4(batch, head, col_j, d, H, N, D);
                        value_sum += probs[warp_id][j] * to_float(v[v_index]);
                    }
                }
                acc[warp_id][d] = alpha * acc[warp_id][d] + value_sum;
            }

            if (lane == 0) {
                l_shared[warp_id] = alpha * l_shared[warp_id] + tile_sum[warp_id];
                m_shared[warp_id] = new_m;
            }
        }
        __syncthreads();
    }

    if (row_valid) {
        for (int d = lane; d < D; d += WARP_SIZE) {
            out[q_base + d] = from_float<scalar_t>(acc[warp_id][d] / l_shared[warp_id]);
        }
    }
    if (row_valid && lane == 0) {
        const int stats_index = (batch * H + head) * N + row;
        m_out[stats_index] = m_shared[warp_id];
        l_out[stats_index] = l_shared[warp_id];
    }
}

std::vector<torch::Tensor> flash_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale) {
    auto out = torch::empty_like(q);
    auto stats_options = q.options().dtype(torch::kFloat32);
    auto m = torch::empty({q.size(0), q.size(1), q.size(2)}, stats_options);
    auto l = torch::empty({q.size(0), q.size(1), q.size(2)}, stats_options);

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);
    const FwdTileConfig tile = choose_fwd_tile(N, D);

    dim3 grid((N + tile.block_m - 1) / tile.block_m, H, B);
    dim3 block(THREADS);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "flash_fwd_kernel", [&] {
        if (tile.block_n == 128) {
            flash_fwd_kernel_tiled<scalar_t, ROWS_PER_BLOCK, 128><<<grid, block, 0, stream>>>(
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                B,
                H,
                N,
                D,
                causal,
                static_cast<float>(scale));
        } else {
            flash_fwd_kernel_tiled<scalar_t, ROWS_PER_BLOCK, 64><<<grid, block, 0, stream>>>(
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                B,
                H,
                N,
                D,
                causal,
                static_cast<float>(scale));
        }
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, m, l};
}
