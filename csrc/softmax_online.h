#pragma once

__device__ inline float online_alpha(float old_m, float new_m) {
    return isfinite(old_m) ? expf(old_m - new_m) : 0.0f;
}

__device__ inline float masked_score(bool valid, float score) {
    return valid ? score : -INFINITY;
}
