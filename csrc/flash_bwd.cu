#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdlib>
#include <vector>

#include "flash_common.h"

struct BwdTileConfig {
    int block_m;
    int block_n;
};

__host__ inline BwdTileConfig choose_bwd_tile(int N, int D) {
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
__global__ void flash_bwd_dq_kernel_tiled(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ out,
    const float* __restrict__ m,
    const float* __restrict__ l,
    scalar_t* __restrict__ dq,
    float* __restrict__ delta_out,
    int B,
    int H,
    int N,
    int D,
    bool causal,
    float scale) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int row = blockIdx.x * BLOCK_M + warp_id;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const bool active_warp = warp_id < BLOCK_M;
    const bool row_valid = active_warp && row < N;

    __shared__ float scores[BLOCK_M][BLOCK_N_TILE];
    __shared__ float probs[BLOCK_M][BLOCK_N_TILE];
    __shared__ float dscores[BLOCK_M][BLOCK_N_TILE];
    __shared__ float dq_acc[BLOCK_M][MAX_HEAD_DIM];
    __shared__ float delta_shared[BLOCK_M];

    const int row_base = offset4(batch, head, row, 0, H, N, D);
    const int stats_index = (batch * H + head) * N + row;
    const float row_m = row_valid ? m[stats_index] : -INFINITY;
    const float row_l = row_valid ? l[stats_index] : 1.0f;

    if (active_warp) {
        float delta_partial = 0.0f;
        for (int d = lane; d < D; d += WARP_SIZE) {
            dq_acc[warp_id][d] = 0.0f;
            if (row_valid) {
                delta_partial += to_float(dout[row_base + d]) * to_float(out[row_base + d]);
            }
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            delta_partial += __shfl_down_sync(0xffffffff, delta_partial, offset);
        }
        if (lane == 0) {
            delta_shared[warp_id] = delta_partial;
            if (row_valid) {
                delta_out[stats_index] = delta_partial;
            }
        }
    }
    __syncwarp();

    for (int col_start = 0; col_start < N; col_start += BLOCK_N_TILE) {
        const int tile_count = min(BLOCK_N_TILE, N - col_start);

        if (active_warp) {
            for (int j = 0; j < tile_count; ++j) {
                const int col = col_start + j;
                const bool valid = row_valid && (!causal || col <= row);
                float dot = 0.0f;
                if (valid) {
                    const int k_base = offset4(batch, head, col, 0, H, N, D);
                    for (int d = lane; d < D; d += WARP_SIZE) {
                        dot += to_float(q[row_base + d]) * to_float(k[k_base + d]);
                    }
                }
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    dot += __shfl_down_sync(0xffffffff, dot, offset);
                }
                if (lane == 0) {
                    const float score = valid ? dot * scale : -INFINITY;
                    scores[warp_id][j] = score;
                    probs[warp_id][j] = valid ? expf(score - row_m) / row_l : 0.0f;
                }
            }
            for (int j = tile_count + lane; j < BLOCK_N_TILE; j += WARP_SIZE) {
                scores[warp_id][j] = -INFINITY;
                probs[warp_id][j] = 0.0f;
                dscores[warp_id][j] = 0.0f;
            }
        }
        __syncwarp();

        if (active_warp) {
            for (int j = 0; j < tile_count; ++j) {
                const int col_j = col_start + j;
                const bool valid = row_valid && (!causal || col_j <= row);
                float dp = 0.0f;
                if (valid) {
                    const int v_base = offset4(batch, head, col_j, 0, H, N, D);
                    for (int d = lane; d < D; d += WARP_SIZE) {
                        dp += to_float(dout[row_base + d]) * to_float(v[v_base + d]);
                    }
                }
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    dp += __shfl_down_sync(0xffffffff, dp, offset);
                }
                if (lane == 0) {
                    dscores[warp_id][j] = probs[warp_id][j] * (dp - delta_shared[warp_id]) * scale;
                }
            }
        }
        __syncwarp();

        if (row_valid) {
            for (int d = lane; d < D; d += WARP_SIZE) {
                float sum = 0.0f;
                for (int j = 0; j < tile_count; ++j) {
                    const int col_j = col_start + j;
                    const int k_index = offset4(batch, head, col_j, d, H, N, D);
                    sum += dscores[warp_id][j] * to_float(k[k_index]);
                }
                dq_acc[warp_id][d] += sum;
            }
        }

        __syncwarp();
    }

    if (row_valid) {
        for (int d = lane; d < D; d += WARP_SIZE) {
            dq[row_base + d] = from_float<scalar_t>(dq_acc[warp_id][d]);
        }
    }
}

template <typename scalar_t>
__global__ void flash_bwd_kv_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ m,
    const float* __restrict__ l,
    const float* __restrict__ delta,
    scalar_t* __restrict__ dk,
    scalar_t* __restrict__ dv,
    int B,
    int H,
    int N,
    int D,
    bool causal,
    float scale) {
    const int col = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    __shared__ float warp_score[WARPS_PER_BLOCK];
    __shared__ float warp_dp[WARPS_PER_BLOCK];
    __shared__ float ds_shared;
    __shared__ float p_shared;

    float dk_acc = 0.0f;
    float dv_acc = 0.0f;
    const int start_row = causal ? col : 0;

    for (int row = start_row; row < N; ++row) {
        const int row_base = offset4(batch, head, row, 0, H, N, D);
        const int k_base = offset4(batch, head, col, 0, H, N, D);
        const int stats_index = (batch * H + head) * N + row;

        float score_value = 0.0f;
        float dp_value = 0.0f;
        if (tid < D) {
            score_value = to_float(q[row_base + tid]) * to_float(k[k_base + tid]);
            dp_value = to_float(dout[row_base + tid]) * to_float(v[k_base + tid]);
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            score_value += __shfl_down_sync(0xffffffff, score_value, offset);
            dp_value += __shfl_down_sync(0xffffffff, dp_value, offset);
        }

        if (lane == 0) {
            warp_score[warp_id] = score_value;
            warp_dp[warp_id] = dp_value;
        }
        __syncthreads();

        if (tid == 0) {
            float score_sum = 0.0f;
            float dp_sum = 0.0f;
            for (int warp = 0; warp < WARPS_PER_BLOCK; ++warp) {
                score_sum += warp_score[warp];
                dp_sum += warp_dp[warp];
            }
            const float score = score_sum * scale;
            const float p = expf(score - m[stats_index]) / l[stats_index];
            p_shared = p;
            ds_shared = p * (dp_sum - delta[stats_index]) * scale;
        }
        __syncthreads();

        if (tid < D) {
            dk_acc += ds_shared * to_float(q[row_base + tid]);
            dv_acc += p_shared * to_float(dout[row_base + tid]);
        }
        __syncthreads();
    }

    if (tid < D) {
        const int kv_index = offset4(batch, head, col, tid, H, N, D);
        dk[kv_index] = from_float<scalar_t>(dk_acc);
        dv[kv_index] = from_float<scalar_t>(dv_acc);
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
    auto dk = torch::empty_like(k);
    auto dv = torch::empty_like(v);
    auto stats_options = q.options().dtype(torch::kFloat32);
    auto delta = torch::empty({q.size(0), q.size(1), q.size(2)}, stats_options);

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);
    const BwdTileConfig tile = choose_bwd_tile(N, D);

    dim3 grid((N + tile.block_m - 1) / tile.block_m, H, B);
    dim3 kv_grid(N, H, B);
    dim3 block(THREADS);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "flash_bwd_kernel", [&] {
        if (tile.block_n == 128) {
            flash_bwd_dq_kernel_tiled<scalar_t, ROWS_PER_BLOCK, 128><<<grid, block, 0, stream>>>(
                dout.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                dq.data_ptr<scalar_t>(),
                delta.data_ptr<float>(),
                B,
                H,
                N,
                D,
                causal,
                static_cast<float>(scale));
        } else {
            flash_bwd_dq_kernel_tiled<scalar_t, ROWS_PER_BLOCK, 64><<<grid, block, 0, stream>>>(
                dout.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                dq.data_ptr<scalar_t>(),
                delta.data_ptr<float>(),
                B,
                H,
                N,
                D,
                causal,
                static_cast<float>(scale));
        }

        flash_bwd_kv_kernel<scalar_t><<<kv_grid, block, 0, stream>>>(
            dout.data_ptr<scalar_t>(),
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            m.data_ptr<float>(),
            l.data_ptr<float>(),
            delta.data_ptr<float>(),
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
