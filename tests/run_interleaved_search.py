import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_CASES = [
    {"batch": 4, "microbatch": 1, "seqlen": 1024},
    {"batch": 4, "microbatch": 1, "seqlen": 2048},
    {"batch": 4, "microbatch": 2, "seqlen": 2048},
    {"batch": 6, "microbatch": 2, "seqlen": 4096},
    {"batch": 8, "microbatch": 2, "seqlen": 4096},
    {"batch": 8, "microbatch": 2, "seqlen": 8192},
    {"batch": 16, "microbatch": 2, "seqlen": 16384},
]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_args():
    parser = argparse.ArgumentParser(description="Busca casos donde interleaved gane de forma real.")
    parser.add_argument("--backend", choices=["native", "native_fast", "official", "sdpa", "auto"], default="native_fast")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--accum-windows", default="1,2,4")
    parser.add_argument("--out-dir", default="interleaved_benchmark_results/search")
    parser.add_argument("--case", action="append", help="Caso especifico, por ejemplo b16_mb2_s16384. Puede repetirse.")
    parser.add_argument("--max-seqlen", type=int, help="Limita la busqueda a seqlen <= este valor.")
    parser.add_argument("--skip-fwd", action="store_true")
    parser.add_argument("--skip-bwd", action="store_true")
    return parser.parse_args()


def run_case(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, env=env)


def load_rows(path: Path, extra: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({**extra, **row})
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def numeric(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def print_summary(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["measure"], row["case"], row.get("accum_window", ""))
        grouped.setdefault(key, []).append(row)

    print("\n=== resumen ===")
    for (measure, case, accum_window), group in grouped.items():
        ok_rows = [row for row in group if row.get("status") == "ok" and numeric(row, "ms") is not None]
        if not ok_rows:
            print(f"{measure} {case} window={accum_window or '-'} | sin filas ok")
            continue
        best = min(ok_rows, key=lambda row: numeric(row, "ms") or float("inf"))
        micro = next((row for row in ok_rows if row["mode"] == "micro_serial"), None)
        inter = next((row for row in ok_rows if row["mode"] == "interleaved"), None)
        prealloc = next((row for row in ok_rows if row["mode"] == "interleaved_prealloc"), None)
        details = [f"best={best['mode']} {best['ms']} ms"]
        if micro and inter:
            micro_ms = numeric(micro, "ms")
            inter_ms = numeric(inter, "ms")
            if micro_ms and inter_ms:
                details.append(f"interleaved_vs_micro={micro_ms / inter_ms:.3f}x")
        if micro and prealloc:
            micro_ms = numeric(micro, "ms")
            prealloc_ms = numeric(prealloc, "ms")
            if micro_ms and prealloc_ms:
                details.append(f"prealloc_vs_micro={micro_ms / prealloc_ms:.3f}x")
        print(f"{measure} {case} window={accum_window or '-'} | " + " | ".join(details))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    benchmark = script_dir / "benchmark_interleaved_block.py"
    env = os.environ.copy()
    all_rows = []

    selected_cases = []
    requested_cases = set(args.case or [])
    for case in DEFAULT_CASES:
        case_name = f"b{case['batch']}_mb{case['microbatch']}_s{case['seqlen']}"
        if requested_cases and case_name not in requested_cases:
            continue
        if args.max_seqlen is not None and case["seqlen"] > args.max_seqlen:
            continue
        selected_cases.append((case_name, case))

    if not selected_cases:
        raise ValueError("No hay casos para ejecutar con los filtros dados.")

    for case_name, case in selected_cases:
        base_cmd = [
            sys.executable,
            str(benchmark),
            "--backend",
            args.backend,
            "--mode",
            "all",
            "--batch",
            str(case["batch"]),
            "--microbatch",
            str(case["microbatch"]),
            "--seqlen",
            str(case["seqlen"]),
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
            "--no-verify",
        ]

        if not args.skip_fwd:
            case_dir = out_dir / "fwd" / case_name
            print(f"\n=== fwd {case_name} ===")
            run_case([*base_cmd, "--measure", "fwd", "--out-dir", str(case_dir)], env)
            all_rows.extend(load_rows(case_dir / "interleaved_block_results.csv", {"case": case_name}))

        if not args.skip_bwd:
            for window in parse_int_list(args.accum_windows):
                case_dir = out_dir / f"fwd_bwd_accum_w{window}" / case_name
                print(f"\n=== fwd_bwd_accum {case_name} window={window} ===")
                run_case(
                    [
                        *base_cmd,
                        "--measure",
                        "fwd_bwd_accum",
                        "--accum-window",
                        str(window),
                        "--out-dir",
                        str(case_dir),
                    ],
                    env,
                )
                all_rows.extend(load_rows(case_dir / "interleaved_block_results.csv", {"case": case_name}))

    csv_path = out_dir / "interleaved_search_results.csv"
    write_rows(csv_path, all_rows)
    print(f"\nCSV consolidado: {csv_path}")
    print_summary(all_rows)


if __name__ == "__main__":
    main()
