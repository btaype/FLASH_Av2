import argparse
import csv
import gc
import math
from pathlib import Path
from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint

from benchmark_interleaved_block import (
    SyntheticTransformerBlock,
    block_fwd_flops,
    make_input,
    reset_grads,
    run_full_batch,
    run_interleaved,
    run_interleaved_prealloc,
    run_micro_serial,
    split_microbatches,
)


ForwardFn = Callable[[SyntheticTransformerBlock, torch.Tensor, int], torch.Tensor]
TrainFn = Callable[[SyntheticTransformerBlock, torch.Tensor, int, int], None]


_PRIORITY_STATE: dict[
    tuple[int, int, str],
    tuple[torch.cuda.Stream, torch.cuda.Stream, list[torch.cuda.Event], list[torch.cuda.Event]],
] = {}


def get_priority_state(
    num_chunks: int,
    priority_mode: str,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream, list[torch.cuda.Event], list[torch.cuda.Event]]:
    device = torch.cuda.current_device()
    key = (device, num_chunks, priority_mode)
    state = _PRIORITY_STATE.get(key)
    if state is not None:
        return state

    least_priority, greatest_priority = torch.cuda.Stream.priority_range()
    attn_priority = 0
    mlp_priority = 0
    if priority_mode == "attn_high":
        attn_priority = greatest_priority
        mlp_priority = least_priority
    elif priority_mode == "mlp_high":
        attn_priority = least_priority
        mlp_priority = greatest_priority

    state = (
        torch.cuda.Stream(device=device, priority=attn_priority),
        torch.cuda.Stream(device=device, priority=mlp_priority),
        [torch.cuda.Event() for _ in range(num_chunks)],
        [torch.cuda.Event() for _ in range(num_chunks)],
    )
    _PRIORITY_STATE[key] = state
    return state


def run_interleaved_priority(
    block: SyntheticTransformerBlock,
    x: torch.Tensor,
    microbatch: int,
    priority_mode: str,
    prealloc: bool,
) -> torch.Tensor:
    chunks = split_microbatches(x, microbatch)
    current_stream = torch.cuda.current_stream()
    attn_stream, mlp_stream, attn_done, mlp_done = get_priority_state(len(chunks), priority_mode)
    attn_outputs: list[torch.Tensor | None] = [None] * len(chunks)
    outputs: list[torch.Tensor | None] = [None] * len(chunks)
    output = torch.empty_like(x) if prealloc and not torch.is_grad_enabled() else None

    def finish_chunk(idx: int) -> None:
        start = idx * microbatch
        end = start + microbatch
        with torch.cuda.stream(mlp_stream):
            mlp_stream.wait_event(attn_done[idx])
            assert attn_outputs[idx] is not None
            attn_outputs[idx].record_stream(mlp_stream)
            mlp_out = block.mlp_stage(attn_outputs[idx])
            if output is None:
                outputs[idx] = mlp_out
                outputs[idx].record_stream(mlp_stream)
            else:
                output[start:end].copy_(mlp_out)
            mlp_done[idx].record(mlp_stream)

    for idx, chunk in enumerate(chunks):
        with torch.cuda.stream(attn_stream):
            chunk.record_stream(attn_stream)
            attn_outputs[idx] = block.attention_stage(chunk)
            attn_outputs[idx].record_stream(attn_stream)
            attn_done[idx].record(attn_stream)
        if idx > 0:
            finish_chunk(idx - 1)

    finish_chunk(len(chunks) - 1)
    for event in mlp_done:
        current_stream.wait_event(event)

    if output is not None:
        return output
    return torch.cat([out for out in outputs if out is not None], dim=0)


def run_interleaved_attn_high(block: SyntheticTransformerBlock, x: torch.Tensor, microbatch: int) -> torch.Tensor:
    return run_interleaved_priority(block, x, microbatch, "attn_high", prealloc=False)


def run_interleaved_mlp_high(block: SyntheticTransformerBlock, x: torch.Tensor, microbatch: int) -> torch.Tensor:
    return run_interleaved_priority(block, x, microbatch, "mlp_high", prealloc=False)


def run_interleaved_attn_high_prealloc(
    block: SyntheticTransformerBlock,
    x: torch.Tensor,
    microbatch: int,
) -> torch.Tensor:
    return run_interleaved_priority(block, x, microbatch, "attn_high", prealloc=True)


def run_interleaved_mlp_high_prealloc(
    block: SyntheticTransformerBlock,
    x: torch.Tensor,
    microbatch: int,
) -> torch.Tensor:
    return run_interleaved_priority(block, x, microbatch, "mlp_high", prealloc=True)


FWD_RUNNERS: dict[str, ForwardFn] = {
    "full": run_full_batch,
    "micro_serial": run_micro_serial,
    "interleaved": run_interleaved,
    "interleaved_prealloc": run_interleaved_prealloc,
    "interleaved_attn_high": run_interleaved_attn_high,
    "interleaved_mlp_high": run_interleaved_mlp_high,
    "interleaved_attn_high_prealloc": run_interleaved_attn_high_prealloc,
    "interleaved_mlp_high_prealloc": run_interleaved_mlp_high_prealloc,
}


def cat_window(chunks: list[torch.Tensor]) -> torch.Tensor:
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=0)


def scaled_loss(out: torch.Tensor, window_chunks: int, num_chunks: int) -> torch.Tensor:
    return out.float().square().mean() * (window_chunks / num_chunks)


def train_full(block: SyntheticTransformerBlock, x: torch.Tensor, microbatch: int, accum_window: int) -> None:
    del microbatch
    del accum_window
    out = block(x)
    out.float().square().mean().backward()


def train_micro_window(
    block: SyntheticTransformerBlock,
    x: torch.Tensor,
    microbatch: int,
    accum_window: int,
) -> None:
    chunks = split_microbatches(x, microbatch)
    num_chunks = len(chunks)
    for start in range(0, num_chunks, accum_window):
        window_chunks = chunks[start : start + accum_window]
        out = block(cat_window(window_chunks))
        scaled_loss(out, len(window_chunks), num_chunks).backward()


def checkpoint_block_forward(block: SyntheticTransformerBlock, x: torch.Tensor) -> torch.Tensor:
    return checkpoint(lambda value: block(value), x, use_reentrant=False)


def checkpoint_stage_forward(block: SyntheticTransformerBlock, x: torch.Tensor) -> torch.Tensor:
    attn = checkpoint(lambda value: block.attention_stage(value), x, use_reentrant=False)
    return checkpoint(lambda value: block.mlp_stage(value), attn, use_reentrant=False)


def train_micro_checkpoint_block(
    block: SyntheticTransformerBlock,
    x: torch.Tensor,
    microbatch: int,
    accum_window: int,
) -> None:
    chunks = split_microbatches(x, microbatch)
    num_chunks = len(chunks)
    for start in range(0, num_chunks, accum_window):
        window_chunks = chunks[start : start + accum_window]
        out = checkpoint_block_forward(block, cat_window(window_chunks))
        scaled_loss(out, len(window_chunks), num_chunks).backward()


def train_micro_checkpoint_stages(
    block: SyntheticTransformerBlock,
    x: torch.Tensor,
    microbatch: int,
    accum_window: int,
) -> None:
    chunks = split_microbatches(x, microbatch)
    num_chunks = len(chunks)
    for start in range(0, num_chunks, accum_window):
        window_chunks = chunks[start : start + accum_window]
        out = checkpoint_stage_forward(block, cat_window(window_chunks))
        scaled_loss(out, len(window_chunks), num_chunks).backward()


def train_interleaved_window(
    runner: ForwardFn,
    block: SyntheticTransformerBlock,
    x: torch.Tensor,
    microbatch: int,
    accum_window: int,
) -> None:
    chunks = split_microbatches(x, microbatch)
    num_chunks = len(chunks)
    for start in range(0, num_chunks, accum_window):
        window_chunks = chunks[start : start + accum_window]
        window_x = cat_window(window_chunks)
        out = runner(block, window_x, microbatch)
        scaled_loss(out, len(window_chunks), num_chunks).backward()


def train_interleaved(block: SyntheticTransformerBlock, x: torch.Tensor, microbatch: int, accum_window: int) -> None:
    train_interleaved_window(run_interleaved, block, x, microbatch, accum_window)


def train_interleaved_attn_high(
    block: SyntheticTransformerBlock,
    x: torch.Tensor,
    microbatch: int,
    accum_window: int,
) -> None:
    train_interleaved_window(run_interleaved_attn_high, block, x, microbatch, accum_window)


def train_interleaved_mlp_high(
    block: SyntheticTransformerBlock,
    x: torch.Tensor,
    microbatch: int,
    accum_window: int,
) -> None:
    train_interleaved_window(run_interleaved_mlp_high, block, x, microbatch, accum_window)


TRAIN_RUNNERS: dict[str, TrainFn] = {
    "full": train_full,
    "micro_window": train_micro_window,
    "micro_checkpoint_block": train_micro_checkpoint_block,
    "micro_checkpoint_stages": train_micro_checkpoint_stages,
    "interleaved_window": train_interleaved,
    "interleaved_attn_high_window": train_interleaved_attn_high,
    "interleaved_mlp_high_window": train_interleaved_mlp_high,
}


def parse_csv_list(value: str) -> list[str]:
    if value == "all":
        return ["all"]
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def measure_once(fn: Callable[[], object], warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    peak_alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
    return ms, peak_alloc_mb, peak_reserved_mb


def tflops(total_flops: float, ms: float) -> float:
    if ms <= 0 or math.isnan(ms):
        return 0.0
    return total_flops / (ms / 1000.0) / 1e12


def add_result(
    rows: list[dict[str, object]],
    *,
    status: str,
    measure: str,
    mode: str,
    args,
    total_flops: float,
    ms: float | None = None,
    peak_alloc_mb: float | None = None,
    peak_reserved_mb: float | None = None,
    error: str = "",
) -> None:
    row = {
        "status": status,
        "measure": measure,
        "mode": mode,
        "backend": args.backend,
        "batch": args.batch,
        "microbatch": args.microbatch,
        "seqlen": args.seqlen,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "accum_window": args.accum_window if measure == "fwd_bwd_accum" else "",
        "ms": round(ms, 6) if ms is not None else "",
        "tokens_s": round(args.batch * args.seqlen / (ms / 1000.0), 3) if ms else "",
        "model_tflops": round(tflops(total_flops, ms), 6) if ms else "",
        "peak_memory_mb": round(peak_alloc_mb, 3) if peak_alloc_mb is not None else "",
        "reserved_memory_mb": round(peak_reserved_mb, 3) if peak_reserved_mb is not None else "",
        "error": error,
    }
    rows.append(row)


def run_with_oom(label: str, fn: Callable[[], tuple[float, float, float]]) -> tuple[str, tuple[float, float, float] | None, str]:
    try:
        return "ok", fn(), ""
    except torch.OutOfMemoryError as exc:
        gc.collect()
        torch.cuda.empty_cache()
        return "oom", None, str(exc).splitlines()[0]
    except RuntimeError as exc:
        gc.collect()
        torch.cuda.empty_cache()
        return "error", None, f"{label}: {str(exc).splitlines()[0]}"


def print_leaderboard(rows: list[dict[str, object]]) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        print("\nSin resultados ok.")
        return

    print("\n=== leaderboard ===")
    for measure in sorted({str(row["measure"]) for row in ok_rows}):
        measure_rows = [row for row in ok_rows if row["measure"] == measure]
        measure_rows.sort(key=lambda row: float(row["ms"]))
        print(f"\n{measure}")
        for row in measure_rows[:8]:
            print(
                f"{str(row['mode']):30s} {float(row['ms']):10.4f} ms | "
                f"{float(row['model_tflops']):7.2f} TFLOPs/s | "
                f"peak {float(row['peak_memory_mb']):8.1f} MiB"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark experimental para maximizar scheduling interleaved y training windows."
    )
    parser.add_argument("--backend", choices=["native", "native_fast", "official", "sdpa", "auto"], default="native_fast")
    parser.add_argument("--measure", choices=["fwd", "fwd_bwd_accum", "both"], default="both")
    parser.add_argument("--fwd-modes", default="all", help="all o lista separada por comas.")
    parser.add_argument("--train-modes", default="all", help="all o lista separada por comas.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=16384)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--requires-input-grad", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--accum-window", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="Prueba torch.compile sobre el bloque.")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--out-dir", default="interleaved_benchmark_results/max")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no esta disponible.")
    if args.batch % args.microbatch != 0:
        raise ValueError("batch debe ser divisible por microbatch.")
    if args.accum_window < 1:
        raise ValueError("accum_window debe ser >= 1.")

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
    if args.compile:
        block = torch.compile(block, mode=args.compile_mode)

    x = make_input(args, dtype)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "interleaved_max_results.csv"

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"backend: {args.backend}")
    print(f"batch={args.batch} microbatch={args.microbatch} seqlen={args.seqlen}")
    print(f"hidden_dim={args.hidden_dim} heads={args.heads} head_dim={args.head_dim}")
    print(f"accum_window={args.accum_window}")

    fwd_flops = block_fwd_flops(args.batch, args.seqlen, args.hidden_dim, args.mlp_ratio, args.causal)
    train_flops = fwd_flops * 3.0
    rows: list[dict[str, object]] = []

    fwd_modes = list(FWD_RUNNERS) if args.fwd_modes == "all" else parse_csv_list(args.fwd_modes)
    train_modes = list(TRAIN_RUNNERS) if args.train_modes == "all" else parse_csv_list(args.train_modes)

    if args.measure in ("fwd", "both"):
        block.train(False)
        for mode in fwd_modes:
            runner = FWD_RUNNERS[mode]

            def fwd_call(runner=runner):
                return measure_once(lambda: runner(block, x, args.microbatch), args.warmup, args.iters)

            status, result, error = run_with_oom(mode, fwd_call)
            if result is None:
                add_result(rows, status=status, measure="fwd", mode=mode, args=args, total_flops=fwd_flops, error=error)
                print(f"{mode:30s} {status} | {error}")
                continue
            ms, peak_alloc_mb, peak_reserved_mb = result
            add_result(
                rows,
                status="ok",
                measure="fwd",
                mode=mode,
                args=args,
                total_flops=fwd_flops,
                ms=ms,
                peak_alloc_mb=peak_alloc_mb,
                peak_reserved_mb=peak_reserved_mb,
            )
            print(f"{mode:30s} {ms:.4f} ms | {tflops(fwd_flops, ms):.2f} TFLOPs/s | peak {peak_alloc_mb:.1f} MiB")

    if args.measure in ("fwd_bwd_accum", "both"):
        block.train(True)
        for mode in train_modes:
            trainer = TRAIN_RUNNERS[mode]

            def train_call(trainer=trainer):
                def run_train():
                    reset_grads(block, x)
                    trainer(block, x, args.microbatch, args.accum_window)
                    return None

                return measure_once(run_train, args.warmup, args.iters)

            status, result, error = run_with_oom(mode, train_call)
            if result is None:
                add_result(
                    rows,
                    status=status,
                    measure="fwd_bwd_accum",
                    mode=mode,
                    args=args,
                    total_flops=train_flops,
                    error=error,
                )
                print(f"{mode:30s} {status} | {error}")
                continue
            ms, peak_alloc_mb, peak_reserved_mb = result
            add_result(
                rows,
                status="ok",
                measure="fwd_bwd_accum",
                mode=mode,
                args=args,
                total_flops=train_flops,
                ms=ms,
                peak_alloc_mb=peak_alloc_mb,
                peak_reserved_mb=peak_reserved_mb,
            )
            print(f"{mode:30s} {ms:.4f} ms | {tflops(train_flops, ms):.2f} TFLOPs/s | peak {peak_alloc_mb:.1f} MiB")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV guardado: {csv_path}")
    print_leaderboard(rows)


if __name__ == "__main__":
    main()
