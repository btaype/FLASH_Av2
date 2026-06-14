import math
import os

import torch
import torch.nn.functional as F

try:
    import f_attencion_v2_cuda
except ImportError:
    f_attencion_v2_cuda = None


_BACKENDS = {"auto", "native", "sdpa"}


def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k y v deben estar en CUDA.")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k y v deben tener la misma forma [B, H, N, D].")
    if q.dim() != 4:
        raise ValueError("q, k y v deben tener forma [B, H, N, D].")
    if q.shape[-1] not in (64, 128):
        raise ValueError("Esta version native solo soporta head_dim = 64 o 128.")
    if q.dtype not in (torch.float16, torch.float32):
        raise ValueError("Esta version soporta float16 y float32.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k y v deben tener el mismo dtype.")


def _check_common_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k y v deben estar en CUDA.")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k y v deben tener forma [B, H, N, D].")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k y v deben tener la misma forma [B, H, N, D].")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k y v deben tener el mismo dtype.")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("F_attencion_v2 soporta float16, bfloat16 y float32 via backend auto/sdpa.")


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale):
    scale_value = float(scale) if scale is not None else None
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    try:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale_value)
    except TypeError:
        if scale_value is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        default_scale = 1.0 / math.sqrt(q.shape[-1])
        return F.scaled_dot_product_attention(q * (scale_value / default_scale), k, v, is_causal=causal)


def _can_use_native(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return (
        f_attencion_v2_cuda is not None
        and q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and q.shape == k.shape == v.shape
        and q.dim() == 4
        and q.shape[-1] in (64, 128)
        and q.dtype in (torch.float16, torch.float32)
        and q.dtype == k.dtype == v.dtype
    )


class _FlashAttentionV2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal: bool = True, scale=None):
        _check_inputs(q, k, v)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        scale_value = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])

        out, m, l = f_attencion_v2_cuda.forward(q, k, v, causal, scale_value)
        ctx.save_for_backward(q, k, v, out, m, l)
        ctx.causal = causal
        ctx.scale = scale_value
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, m, l = ctx.saved_tensors
        dout = dout.contiguous()
        dq, dk, dv = f_attencion_v2_cuda.backward(
            dout, q, k, v, out, m, l, ctx.causal, ctx.scale
        )
        return dq, dk, dv, None, None


def flash_attention_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True, scale=None):
    """Calcula atencion exacta con una API estilo FlashAttention-2.

    Args:
        q, k, v: tensores CUDA contiguos o convertibles a contiguos con forma
            [batch, heads, seq_len, D], con D en {64, 128}.
        causal: si True, aplica mascara causal.
        scale: escala opcional. Por defecto usa 1 / sqrt(D).
    """
    return flash_attention_v2_backend(q, k, v, causal=causal, scale=scale, backend="auto")


def flash_attention_v2_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    scale=None,
    backend: str = "auto",
):
    """Ejecuta atencion con backend seleccionable.

    backends:
        auto: usa SDPA de PyTorch; puede activar kernels flash cuando el GPU lo permite.
        native: fuerza el kernel CUDA de este proyecto.
        sdpa: usa torch.scaled_dot_product_attention, que puede activar kernels flash.
    """
    if backend not in _BACKENDS:
        valid = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"backend invalido: {backend}. Usa uno de: {valid}")

    if backend == "sdpa":
        _check_common_inputs(q, k, v)
        return _sdpa(q, k, v, causal=causal, scale=scale)

    if backend == "native":
        if f_attencion_v2_cuda is None:
            raise RuntimeError("f_attencion_v2_cuda no esta compilado. Ejecuta `pip install . --no-build-isolation`.")
        return _FlashAttentionV2Function.apply(q, k, v, causal, scale)

    if os.getenv("F_ATTENCION_V2_USE_NATIVE") == "1" and _can_use_native(q, k, v):
        return _FlashAttentionV2Function.apply(q, k, v, causal, scale)

    _check_common_inputs(q, k, v)
    return _sdpa(q, k, v, causal=causal, scale=scale)
