import argparse
import math
import time

import torch

from f_attencion_v2 import flash_attention_v2_backend


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    synchronize()
    return start.elapsed_time(end) / iters


def attention_flops(batch: int, heads: int, seqlen: int, head_dim: int, causal: bool) -> float:
    # Approximate forward attention FLOPs: QK^T and P@V.
    scale = 0.5 if causal else 1.0
    return 4.0 * batch * heads * seqlen * seqlen * head_dim * scale


def eager_attention(q, k, v, causal: bool, scale=None):
    scale_value = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale_value
    if causal:
        seqlen = q.shape[-2]
        mask = torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def benchmark_fwd_bwd(fn, q, k, v, warmup: int, iters: int) -> float:
    def run():
        for tensor in (q, k, v):
            tensor.grad = None
        out = fn(q, k, v)
        out.backward(torch.ones_like(out))

    return time_ms(run, warmup=warmup, iters=iters)


def benchmark_case(batch: int, heads: int, seqlen: int, head_dim: int, dtype, causal: bool, warmup: int, iters: int):
    q = torch.randn(batch, heads, seqlen, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    def ours():
        return flash_attention_v2_backend(q, k, v, causal=causal, backend="native")

    def eager():
        return eager_attention(q, k, v, causal=causal)

    ours_out = ours()
    ref_out = eager()
    max_error = (ours_out - ref_out).abs().max().item()

    ours_ms = time_ms(ours, warmup=warmup, iters=iters)
    eager_ms = time_ms(eager, warmup=warmup, iters=iters)
    ours_fwd_bwd_ms = benchmark_fwd_bwd(
        lambda q_, k_, v_: flash_attention_v2_backend(q_, k_, v_, causal=causal, backend="native"),
        q,
        k,
        v,
        warmup=warmup,
        iters=iters,
    )
    eager_fwd_bwd_ms = benchmark_fwd_bwd(
        lambda q_, k_, v_: eager_attention(q_, k_, v_, causal=causal),
        q,
        k,
        v,
        warmup=warmup,
        iters=iters,
    )

    flops = attention_flops(batch, heads, seqlen, head_dim, causal)
    ours_tflops = flops / (ours_ms / 1000.0) / 1e12
    eager_tflops = flops / (eager_ms / 1000.0) / 1e12
    speedup_fwd = eager_ms / ours_ms if ours_ms > 0 else float("nan")
    speedup_fwd_bwd = eager_fwd_bwd_ms / ours_fwd_bwd_ms if ours_fwd_bwd_ms > 0 else float("nan")

    print(
        f"causal={causal} batch={batch} heads={heads} seqlen={seqlen} head_dim={head_dim} "
        f"| ours {ours_ms:.4f} ms {ours_tflops:.4f} TFLOPs/s "
        f"| eager {eager_ms:.4f} ms {eager_tflops:.4f} TFLOPs/s "
        f"| speedup_fwd {speedup_fwd:.2f}x "
        f"| speedup_fwd_bwd {speedup_fwd_bwd:.2f}x "
        f"| max_error {max_error:.4e}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark para F_attencion_v2.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--no-causal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no esta disponible.")

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    causal = not args.no_causal

    print("Benchmark F_attencion_v2")
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"dtype: {dtype}")
    print("Nota: este benchmark fuerza backend native de f_attencion_v2_cuda.")

    for head_dim in args.head_dims:
        if head_dim not in (64, 128):
            raise ValueError("F_attencion_v2 solo soporta head_dim 64 y 128.")
        for seqlen in args.seq_lens:
            benchmark_case(
                batch=args.batch,
                heads=args.heads,
                seqlen=seqlen,
                head_dim=head_dim,
                dtype=dtype,
                causal=causal,
                warmup=args.warmup,
                iters=args.iters,
            )


if __name__ == "__main__":
    main()
