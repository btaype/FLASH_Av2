import argparse
import csv
import math
from pathlib import Path

import torch

from f_attencion_v2 import flash_attention_v2_backend


def flops(batch, seqlen, headdim, nheads, causal, mode):
    base = 4.0 * batch * seqlen * seqlen * nheads * headdim
    if causal:
        base *= 0.5
    if mode == "fwd":
        return base
    if mode == "bwd":
        return 2.5 * base
    if mode == "fwd_bwd":
        return 3.5 * base
    raise ValueError(f"modo invalido: {mode}")


def tflops(total_flops, ms):
    if ms <= 0 or math.isnan(ms):
        return 0.0
    return total_flops / (ms / 1000.0) / 1e12


def sync():
    torch.cuda.synchronize()


def measure_forward(q, k, v, causal, warmup, iters, backend):
    def run():
        return flash_attention_v2_backend(q, k, v, causal=causal, backend=backend)

    for _ in range(warmup):
        run()
    sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    sync()
    return start.elapsed_time(end) / iters


def measure_fwd_bwd(q, k, v, causal, warmup, iters, backend):
    def run():
        for tensor in (q, k, v):
            if tensor.grad is not None:
                tensor.grad = None
        out = flash_attention_v2_backend(q, k, v, causal=causal, backend=backend)
        grad = torch.ones_like(out)
        out.backward(grad)

    for _ in range(warmup):
        run()
    sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    sync()
    return start.elapsed_time(end) / iters


def make_plot(rows, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no esta instalado; se omiten graficas.")
        return

    for mode in ("fwd", "fwd_bwd"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        fig.suptitle(f"FlashAttention-2 local GPU {mode}")

        panels = [
            (False, 64, axes[0][0]),
            (False, 128, axes[0][1]),
            (True, 64, axes[1][0]),
            (True, 128, axes[1][1]),
        ]
        for causal, headdim, ax in panels:
            data = [
                r for r in rows
                if r["causal"] == causal and r["headdim"] == headdim and r["status"] == "ok"
            ]
            data.sort(key=lambda r: r["seqlen"])
            xs = [r["seqlen"] for r in data]
            ys = [r[f"{mode}_tflops"] for r in data]
            ax.bar([str(x) for x in xs], ys)
            ax.set_title(f"causal={causal}, headdim={headdim}")
            ax.set_xlabel("Sequence length")
            ax.set_ylabel("TFLOPs/s")
            for idx, value in enumerate(ys):
                ax.text(idx, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        path = out_dir / f"flash2_local_{mode}.png"
        fig.savefig(path, dpi=160)
        print(f"grafica: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark local de F_attencion_v2 native para tu GPU.")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--headdims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--causal-values", choices=["true", "false", "both"], default="both")
    parser.add_argument("--model-dim", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--backend", choices=["native", "native_fast", "official", "sdpa", "auto"], default="native")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out-dir", default="local_benchmark_results")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no esta disponible.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "flash2_local_results.csv"

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    causal_values = [False, True] if args.causal_values == "both" else [args.causal_values == "true"]

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"dtype: {dtype}")
    print(f"batch: {args.batch}")
    print(f"seq_lens: {args.seq_lens}")
    print(f"headdims: {args.headdims}")
    print(f"backend: {args.backend}")
    print(f"salida CSV: {csv_path}")

    rows = []
    for causal in causal_values:
        for headdim in args.headdims:
            if args.model_dim % headdim != 0:
                print(f"skip headdim={headdim}: model_dim {args.model_dim} no divisible")
                continue
            nheads = args.model_dim // headdim

            for seqlen in args.seq_lens:
                torch.cuda.empty_cache()
                row = {
                    "gpu": torch.cuda.get_device_name(),
                    "dtype": args.dtype,
                    "batch": args.batch,
                    "seqlen": seqlen,
                    "headdim": headdim,
                    "nheads": nheads,
                    "causal": causal,
                    "fwd_ms": "",
                    "fwd_bwd_ms": "",
                    "fwd_tflops": 0.0,
                    "fwd_bwd_tflops": 0.0,
                    "status": "ok",
                    "error": "",
                }
                try:
                    q_fwd = torch.randn(
                        args.batch, nheads, seqlen, headdim,
                        device="cuda", dtype=dtype
                    )
                    k_fwd = torch.randn_like(q_fwd)
                    v_fwd = torch.randn_like(q_fwd)
                    fwd_ms = measure_forward(q_fwd, k_fwd, v_fwd, causal, args.warmup, args.iters, args.backend)

                    q_bwd = q_fwd.detach().clone().requires_grad_(True)
                    k_bwd = k_fwd.detach().clone().requires_grad_(True)
                    v_bwd = v_fwd.detach().clone().requires_grad_(True)
                    fwd_bwd_ms = measure_fwd_bwd(q_bwd, k_bwd, v_bwd, causal, args.warmup, args.iters, args.backend)

                    row["fwd_ms"] = round(fwd_ms, 6)
                    row["fwd_bwd_ms"] = round(fwd_bwd_ms, 6)
                    row["fwd_tflops"] = round(tflops(flops(args.batch, seqlen, headdim, nheads, causal, "fwd"), fwd_ms), 6)
                    row["fwd_bwd_tflops"] = round(tflops(flops(args.batch, seqlen, headdim, nheads, causal, "fwd_bwd"), fwd_bwd_ms), 6)
                    print(
                        f"causal={causal} headdim={headdim} seqlen={seqlen} "
                        f"| fwd {row['fwd_tflops']:.2f} TFLOPs/s "
                        f"| fwd+bwd {row['fwd_bwd_tflops']:.2f} TFLOPs/s"
                    )
                except RuntimeError as exc:
                    row["status"] = "failed"
                    row["error"] = str(exc).split("\n")[0]
                    print(f"FAILED causal={causal} headdim={headdim} seqlen={seqlen}: {row['error']}")
                    torch.cuda.empty_cache()
                rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV guardado: {csv_path}")

    if not args.no_plot:
        make_plot(rows, out_dir)


if __name__ == "__main__":
    main()
