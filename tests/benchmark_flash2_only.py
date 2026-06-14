import argparse
import math

import torch

from flash_attn import flash_attn_qkvpacked_func
from flash_attn.utils.benchmark import benchmark_fwd_bwd


def attention_flops(batch, seqlen, headdim, nheads, causal, mode):
    fwd = 4 * batch * seqlen * seqlen * nheads * headdim // (2 if causal else 1)
    if mode == "fwd":
        return fwd
    if mode == "bwd":
        return 2.5 * fwd
    return 3.5 * fwd


def efficiency_tflops(flops, seconds):
    if math.isnan(seconds) or seconds <= 0:
        return 0.0
    return flops / seconds / 1e12


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark solo FlashAttention-2 oficial.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--headdims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--model-dim", type=int, default=2048)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no esta disponible.")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"dtype: {dtype}")
    print(f"causal: {args.causal}")

    for headdim in args.headdims:
        if args.model_dim % headdim != 0:
            print(f"skip headdim={headdim}: model_dim={args.model_dim} no es divisible")
            continue
        nheads = args.model_dim // headdim
        batch_sizes = args.batch_sizes or [max(1, 16384 // seqlen) for seqlen in args.seq_lens]

        for batch_size, seqlen in zip(batch_sizes, args.seq_lens):
            torch.cuda.empty_cache()
            try:
                qkv = torch.randn(
                    batch_size,
                    seqlen,
                    3,
                    nheads,
                    headdim,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                time_f, time_b = benchmark_fwd_bwd(
                    flash_attn_qkvpacked_func,
                    qkv,
                    0.0,
                    causal=args.causal,
                    repeats=args.repeats,
                    verbose=False,
                )
                fwd_s = time_f[1].mean
                bwd_s = time_b[1].mean
                total_s = fwd_s + bwd_s
                fwd_tflops = efficiency_tflops(
                    attention_flops(batch_size, seqlen, headdim, nheads, args.causal, "fwd"),
                    fwd_s,
                )
                bwd_tflops = efficiency_tflops(
                    attention_flops(batch_size, seqlen, headdim, nheads, args.causal, "bwd"),
                    bwd_s,
                )
                total_tflops = efficiency_tflops(
                    attention_flops(batch_size, seqlen, headdim, nheads, args.causal, "fwd_bwd"),
                    total_s,
                )
                print(
                    f"headdim={headdim} batch={batch_size} seqlen={seqlen} nheads={nheads} | "
                    f"fwd {fwd_tflops:.2f} TFLOPs/s | "
                    f"bwd {bwd_tflops:.2f} TFLOPs/s | "
                    f"fwd+bwd {total_tflops:.2f} TFLOPs/s"
                )
            except RuntimeError as exc:
                print(f"FAILED headdim={headdim} batch={batch_size} seqlen={seqlen}: {exc}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
