import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = [
    {"batch": 4, "microbatch": 1, "seqlen": 1024},
    {"batch": 4, "microbatch": 1, "seqlen": 2048},
    {"batch": 4, "microbatch": 2, "seqlen": 2048},
    {"batch": 6, "microbatch": 2, "seqlen": 4096},
    {"batch": 8, "microbatch": 2, "seqlen": 4096},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Corre una matriz de benchmarks interleaved y consolida CSV.")
    parser.add_argument("--backend", choices=["native", "native_fast", "official", "sdpa", "auto"], default="native_fast")
    parser.add_argument("--measure", choices=["fwd", "fwd_bwd", "fwd_bwd_accum"], default="fwd")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--accum-window", type=int, default=1)
    parser.add_argument("--out-dir", default="interleaved_benchmark_results/sweep")
    parser.add_argument("--csv-name", default="interleaved_sweep_results.csv")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    benchmark = script_dir / "benchmark_interleaved_block.py"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    rows = []
    for cfg in DEFAULT_CONFIGS:
        case_name = f"b{cfg['batch']}_mb{cfg['microbatch']}_s{cfg['seqlen']}"
        case_dir = out_dir / case_name
        cmd = [
            sys.executable,
            str(benchmark),
            "--backend",
            args.backend,
            "--mode",
            "all",
            "--measure",
            args.measure,
            "--batch",
            str(cfg["batch"]),
            "--microbatch",
            str(cfg["microbatch"]),
            "--seqlen",
            str(cfg["seqlen"]),
            "--hidden-dim",
            str(args.hidden_dim),
            "--heads",
            str(args.heads),
            "--head-dim",
            str(args.head_dim),
            "--dtype",
            args.dtype,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--accum-window",
            str(args.accum_window),
            "--out-dir",
            str(case_dir),
        ]
        if not args.verify:
            cmd.append("--no-verify")

        print(f"\n=== {case_name} ===")
        subprocess.run(cmd, check=True, env=env)

        case_csv = case_dir / "interleaved_block_results.csv"
        with case_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["case"] = case_name
                rows.append(row)

    csv_path = out_dir / args.csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["case", *[key for key in rows[0] if key != "case"]]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV consolidado: {csv_path}")


if __name__ == "__main__":
    main()
