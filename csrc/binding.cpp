#include <torch/extension.h>

#include <vector>

#include "flash_common.h"

std::vector<torch::Tensor> flash_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale);

std::vector<torch::Tensor> flash_bwd(
    torch::Tensor dout,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor m,
    torch::Tensor l,
    bool causal,
    double scale);

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale) {
    check_qkv(q, k, v);
    return flash_fwd(q, k, v, causal, scale);
}

std::vector<torch::Tensor> backward(
    torch::Tensor dout,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor m,
    torch::Tensor l,
    bool causal,
    double scale) {
    CHECK_INPUT(dout);
    CHECK_INPUT(out);
    CHECK_INPUT(m);
    CHECK_INPUT(l);
    check_qkv(q, k, v);
    TORCH_CHECK(dout.sizes() == q.sizes(), "dout debe tener la misma forma que q");
    TORCH_CHECK(out.sizes() == q.sizes(), "out debe tener la misma forma que q");
    TORCH_CHECK(m.dim() == 3 && l.dim() == 3, "m y l deben tener forma [B, H, N]");
    return flash_bwd(dout, q, k, v, out, m, l, causal, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &forward, "FlashAttention-2 forward");
    module.def("backward", &backward, "FlashAttention-2 backward");
}
