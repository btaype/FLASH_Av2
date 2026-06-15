import argparse
import csv
import gc
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from f_attencion_v2 import flash_attention_v2_backend


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(dtype=x.dtype) * self.weight


class SyntheticTransformerBlock(nn.Module):
    """Bloque tipo Llama/Qwen para probar scheduling entre Attention y MLP."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        head_dim: int,
        mlp_ratio: float,
        backend: str,
        causal: bool,
    ):
        super().__init__()
        if hidden_dim != heads * head_dim:
            raise ValueError("hidden_dim debe ser igual a heads * head_dim.")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = head_dim
        self.backend = backend
        self.causal = causal

        mlp_dim = int(hidden_dim * mlp_ratio)
        self.norm1 = RMSNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm2 = RMSNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, hidden_dim, bias=False)

    def attention_stage(self, x: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = x.shape
        residual = x
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm)
        qkv = qkv.view(batch, seqlen, 3, self.heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2).contiguous()
        k = qkv[:, :, 1].transpose(1, 2).contiguous()
        v = qkv[:, :, 2].transpose(1, 2).contiguous()
        attn = flash_attention_v2_backend(
            q,
            k,
            v,
            causal=self.causal,
            backend=self.backend,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seqlen, self.hidden_dim)
        return residual + self.out_proj(attn)

    def mlp_stage(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm2(x)
        hidden = F.silu(self.gate_proj(x_norm)) * self.up_proj(x_norm)
        return residual + self.down_proj(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_stage(self.attention_stage(x))


def split_microbatches(x: torch.Tensor, microbatch: int) -> list[torch.Tensor]:
    if x.shape[0] % microbatch != 0:
        raise ValueError("batch debe ser divisible por microbatch.")
    return list(x.split(microbatch, dim=0))


def run_full_batch(block: SyntheticTransformerBlock, x: torch.Tensor, microbatch: int) -> torch.Tensor:
    del microbatch
    return block(x)


def run_micro_serial(block: SyntheticTransformerBlock, x: torch.Tensor, microbatch: int) -> torch.Tensor:
    outputs = []
    for chunk in split_microbatches(x, microbatch):
        outputs.append(block(chunk))
    return torch.cat(outputs, dim=0)


_INTERLEAVED_STATE: dict[
    tuple[int, int],
    tuple[torch.cuda.Stream, torch.cuda.Stream, list[torch.cuda.Event], list[torch.cuda.Event]],
] = {}


def get_interleaved_state(
    num_chunks: int,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream, list[torch.cuda.Event], list[torch.cuda.Event]]:
    device = torch.cuda.current_device()
    key = (device, num_chunks)
    state = _INTERLEAVED_STATE.get(key)
    if state is None:
        state = (
            torch.cuda.Stream(device=device),
            torch.cuda.Stream(device=device),
            [torch.cuda.Event() for _ in range(num_chunks)],
            [torch.cuda.Event() for _ in range(num_chunks)],
        )
        _INTERLEAVED_STATE[key] = state
    return state


def run_interleaved(block: SyntheticTransformerBlock, x: torch.Tensor, microbatch: int) -> torch.Tensor:
    chunks = split_microbatches(x, microbatch)
    current_stream = torch.cuda.current_stream()
    attn_stream, mlp_stream, attn_done, mlp_done = get_interleaved_state(len(chunks))
    attn_outputs: list[torch.Tensor | None] = [None] * len(chunks)
    outputs: list[torch.Tensor | None] = [None] * len(chunks)

    for idx, chunk in enumerate(chunks):
        with torch.cuda.stream(attn_stream):
            chunk.record_stream(attn_stream)
            attn_outputs[idx] = block.attention_stage(chunk)
            attn_outputs[idx].record_stream(attn_stream)
            attn_done[idx].record(attn_stream)

        if idx > 0:
            prev = idx - 1
            with torch.cuda.stream(mlp_stream):
                mlp_stream.wait_event(attn_done[prev])
                assert attn_outputs[prev] is not None
                attn_outputs[prev].record_stream(mlp_stream)
                outputs[prev] = block.mlp_stage(attn_outputs[prev])
                outputs[prev].record_stream(mlp_stream)
                mlp_done[prev].record(mlp_stream)

    last = len(chunks) - 1
    with torch.cuda.stream(mlp_stream):
        mlp_stream.wait_event(attn_done[last])
        assert attn_outputs[last] is not None
        attn_outputs[last].record_stream(mlp_stream)
        outputs[last] = block.mlp_stage(attn_outputs[last])
        outputs[last].record_stream(mlp_stream)
        mlp_done[last].record(mlp_stream)

    for event in mlp_done:
        current_stream.wait_event(event)
    return torch.cat([out for out in outputs if out is not None], dim=0)


def run_interleaved_prealloc(block: SyntheticTransformerBlock, x: torch.Tensor, microbatch: int) -> torch.Tensor:
    if torch.is_grad_enabled():
        return run_interleaved(block, x, microbatch)

    chunks = split_microbatches(x, microbatch)
    current_stream = torch.cuda.current_stream()
    attn_stream, mlp_stream, attn_done, mlp_done = get_interleaved_state(len(chunks))
    attn_outputs: list[torch.Tensor | None] = [None] * len(chunks)
    output = torch.empty_like(x)

    for idx, chunk in enumerate(chunks):
        with torch.cuda.stream(attn_stream):
            chunk.record_stream(attn_stream)
            attn_outputs[idx] = block.attention_stage(chunk)
            attn_done[idx].record(attn_stream)

        if idx > 0:
            prev = idx - 1
            start = prev * microbatch
            end = start + microbatch
            with torch.cuda.stream(mlp_stream):
                mlp_stream.wait_event(attn_done[prev])
                assert attn_outputs[prev] is not None
                mlp_out = block.mlp_stage(attn_outputs[prev])
                output[start:end].copy_(mlp_out)
                mlp_done[prev].record(mlp_stream)

    last = len(chunks) - 1
    start = last * microbatch
    end = start + microbatch
    with torch.cuda.stream(mlp_stream):
        mlp_stream.wait_event(attn_done[last])
        assert attn_outputs[last] is not None
        mlp_out = block.mlp_stage(attn_outputs[last])
        output[start:end].copy_(mlp_out)
        mlp_done[last].record(mlp_stream)

    for event in mlp_done:
        current_stream.wait_event(event)
    return output


RUNNERS = {
    "full": run_full_batch,
    "micro_serial": run_micro_serial,
    "interleaved": run_interleaved,
    "interleaved_prealloc": run_interleaved_prealloc,
}


def train_full_batch(block: SyntheticTransformerBlock, x: torch.Tensor, microbatch: int) -> None:
    del microbatch
    out = block(x)
    out.float().square().mean().backward()


def train_micro_accum(block: SyntheticTransformerBlock, x: torch.Tensor, microbatch: int) -> None:
    chunks = split_microbatches(x, microbatch)
    loss_scale = 1.0 / len(chunks)
    for chunk in chunks:
        out = block(chunk)
        loss = out.float().square().mean() * loss_scale
        loss.backward()


ACCUM_TRAINERS = {
    "full": train_full_batch,
    "micro_serial": train_micro_accum,
}


def block_fwd_flops(batch: int, seqlen: int, hidden_dim: int, mlp_ratio: float, causal: bool) -> float:
    qkv = 6.0 * batch * seqlen * hidden_dim * hidden_dim
    out_proj = 2.0 * batch * seqlen * hidden_dim * hidden_dim
    mlp_dim = int(hidden_dim * mlp_ratio)
    mlp = 6.0 * batch * seqlen * hidden_dim * mlp_dim
    attn = 4.0 * batch * seqlen * seqlen * hidden_dim
    if causal:
        attn *= 0.5
    return qkv + out_proj + mlp + attn


def tflops(total_flops: float, ms: float) -> float:
    if ms <= 0 or math.isnan(ms):
        return 0.0
    return total_flops / (ms / 1000.0) / 1e12


def make_input(args, dtype: torch.dtype) -> torch.Tensor:
    x = torch.randn(
        args.batch,
        args.seqlen,
        args.hidden_dim,
        device="cuda",
        dtype=dtype,
    )
    if args.requires_input_grad:
        x.requires_grad_(True)
    return x


def reset_grads(block: nn.Module, x: torch.Tensor) -> None:
    block.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None


def measure_runner(block, x, mode, runner, microbatch, measure, warmup, iters) -> tuple[float, float, float]:
    def run_once():
        if measure == "fwd":
            with torch.no_grad():
                return runner(block, x, microbatch)
        if measure == "fwd_bwd_accum":
            reset_grads(block, x)
            ACCUM_TRAINERS[mode](block, x, microbatch)
            return None
        reset_grads(block, x)
        out = runner(block, x, microbatch)
        loss = out.float().square().mean()
        loss.backward()
        return out

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_once()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    peak_alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
    return ms, peak_alloc_mb, peak_reserved_mb


@torch.no_grad()
def verify_outputs(block, x, microbatch) -> dict[str, float]:
    torch.cuda.synchronize()
    full = run_full_batch(block, x, microbatch)
    torch.cuda.synchronize()
    errors = {}
    for mode in ("micro_serial", "interleaved", "interleaved_prealloc"):
        out = RUNNERS[mode](block, x, microbatch)
        torch.cuda.synchronize()
        diff = (out - full).abs()
        errors[f"{mode}_max_error"] = float(diff.max().item())
        errors[f"{mode}_mean_error"] = float(diff.mean().item())
    return errors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark de scheduling intercalado Attention/MLP por microbatches."
    )
    parser.add_argument("--backend", choices=["native", "native_fast", "official", "sdpa", "auto"], default="native_fast")
    parser.add_argument(
        "--mode",
        choices=["full", "micro_serial", "interleaved", "interleaved_prealloc", "all"],
        default="all",
    )
    parser.add_argument("--measure", choices=["fwd", "fwd_bwd", "fwd_bwd_accum"], default="fwd")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--requires-input-grad", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default="interleaved_benchmark_results")
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no esta disponible.")
    if args.batch % args.microbatch != 0:
        raise ValueError("batch debe ser divisible por microbatch.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    block = SyntheticTransformerBlock(
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        head_dim=args.head_dim,
        mlp_ratio=args.mlp_ratio,
        backend=args.backend,
        causal=args.causal,
    ).to(device="cuda", dtype=dtype)
    block.train(args.measure == "fwd_bwd")
    x = make_input(args, dtype)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "interleaved_block_results.csv"

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"backend: {args.backend}")
    print(f"measure: {args.measure}")
    print(f"batch={args.batch} microbatch={args.microbatch} seqlen={args.seqlen}")
    print(f"hidden_dim={args.hidden_dim} heads={args.heads} head_dim={args.head_dim}")

    errors = {} if args.no_verify else verify_outputs(block, x.detach(), args.microbatch)
    if errors:
        print(
            "verify "
            f"micro_serial max={errors['micro_serial_max_error']:.4e} "
            f"interleaved max={errors['interleaved_max_error']:.4e} "
            f"interleaved_prealloc max={errors['interleaved_prealloc_max_error']:.4e}"
        )

    modes = list(RUNNERS) if args.mode == "all" else [args.mode]
    rows = []
    full_ms = None
    micro_serial_ms = None
    total_flops = block_fwd_flops(args.batch, args.seqlen, args.hidden_dim, args.mlp_ratio, args.causal)
    if args.measure in ("fwd_bwd", "fwd_bwd_accum"):
        total_flops *= 3.0

    for mode in modes:
        torch.cuda.empty_cache()
        runner = RUNNERS[mode]
        if args.measure == "fwd_bwd_accum" and mode not in ACCUM_TRAINERS:
            row = {
                "mode": mode,
                "backend": args.backend,
                "measure": args.measure,
                "batch": args.batch,
                "microbatch": args.microbatch,
                "seqlen": args.seqlen,
                "hidden_dim": args.hidden_dim,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "causal": args.causal,
                "ms": "",
                "tokens_s": "",
                "tflops": "",
                "speedup_vs_full": "",
                "speedup_vs_micro_serial": "",
                "peak_memory_mb": "",
                "reserved_memory_mb": "",
                "status": "unsupported",
                "error": "fwd_bwd_accum solo esta implementado para full y micro_serial",
                **{key: round(value, 8) for key, value in errors.items()},
            }
            rows.append(row)
            print(f"{mode:20s} unsupported | {row['error']}")
            continue
        try:
            ms, peak_alloc_mb, peak_reserved_mb = measure_runner(
                block, x, mode, runner, args.microbatch, args.measure, args.warmup, args.iters
            )
        except torch.OutOfMemoryError as exc:
            error = str(exc).splitlines()[0]
            reset_grads(block, x)
            gc.collect()
            torch.cuda.empty_cache()
            row = {
                "mode": mode,
                "backend": args.backend,
                "measure": args.measure,
                "batch": args.batch,
                "microbatch": args.microbatch,
                "seqlen": args.seqlen,
                "hidden_dim": args.hidden_dim,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "causal": args.causal,
                "ms": "",
                "tokens_s": "",
                "tflops": "",
                "speedup_vs_full": "",
                "speedup_vs_micro_serial": "",
                "peak_memory_mb": "",
                "reserved_memory_mb": "",
                "status": "oom",
                "error": error,
                **{key: round(value, 8) for key, value in errors.items()},
            }
            rows.append(row)
            print(f"{mode:20s} OOM | {row['error']}")
            continue
        if mode == "full":
            full_ms = ms
        if mode == "micro_serial":
            micro_serial_ms = ms
        speedup_vs_full = (full_ms / ms) if full_ms else 1.0
        speedup_vs_micro_serial = (micro_serial_ms / ms) if micro_serial_ms else 1.0
        row = {
            "mode": mode,
            "backend": args.backend,
            "measure": args.measure,
            "batch": args.batch,
            "microbatch": args.microbatch,
            "seqlen": args.seqlen,
            "hidden_dim": args.hidden_dim,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "causal": args.causal,
            "ms": round(ms, 6),
            "tokens_s": round(args.batch * args.seqlen / (ms / 1000.0), 3),
            "tflops": round(tflops(total_flops, ms), 6),
            "speedup_vs_full": round(speedup_vs_full, 6),
            "speedup_vs_micro_serial": round(speedup_vs_micro_serial, 6),
            "peak_memory_mb": round(peak_alloc_mb, 3),
            "reserved_memory_mb": round(peak_reserved_mb, 3),
            "status": "ok",
            "error": "",
            **{key: round(value, 8) for key, value in errors.items()},
        }
        rows.append(row)
        print(
            f"{mode:20s} {ms:.4f} ms | {row['tokens_s']:.1f} tok/s "
            f"| {row['tflops']:.2f} TFLOPs/s "
            f"| speedup_vs_full {speedup_vs_full:.3f}x "
            f"| speedup_vs_micro_serial {speedup_vs_micro_serial:.3f}x "
            f"| peak_mem {peak_alloc_mb:.1f} MiB"
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV guardado: {csv_path}")


if __name__ == "__main__":
    main()
