#!/usr/bin/env python3
"""
Automated sensor-count sweep using the MFEM dataset and 2d_temporal pipeline.

Runs 2d_temporal_mfem.py for a series of observation counts (default powers of
two from 2 through 256), saving each run's logs/plots under
MFEM_results/sensor_sweep/M_xxx and compiling a summary CSV.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "MFEM_results"
DEFAULT_DATASET = RESULTS_ROOT / "datasets" / "mfem_ex9_dataset_res96_grad.npz"
DEFAULT_SWEEP_SUBDIR = "sensor_sweep"


def run_command(cmd: list[str], *, env: dict[str, str]) -> None:
    """Run subprocess while streaming output."""
    print(f"[sensor_sweep] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def parse_metrics(log_path: Path) -> dict:
    """Extract best validation loss and first demo PSNRs from the log."""
    metrics = {"best_val": None, "demo_psnr_global": None, "demo_psnr_solution": None}
    lines = log_path.read_text().splitlines()
    best_re = re.compile(r"best\s+([0-9.eE+-]+)")
    psnr_re = re.compile(
        r"PSNR_global\s+([0-9.eE+-]+)\s+dB\s+\|\s+.*solution:\s+([0-9.eE+-]+)\s+dB"
    )
    for line in lines:
        match = best_re.search(line)
        if match:
            metrics["best_val"] = float(match.group(1))
    for line in lines:
        if "Demo @" in line:
            ps = psnr_re.search(line)
            if ps:
                metrics["demo_psnr_global"] = float(ps.group(1))
                metrics["demo_psnr_solution"] = float(ps.group(2))
                break
    return metrics


def move_results(run_root: Path) -> Path:
    """Move log/temporal_results into run_root and return log path."""
    run_root.mkdir(parents=True, exist_ok=True)
    log_src = RESULTS_ROOT / "log"
    temp_src = RESULTS_ROOT / "temporal_results"
    if log_src.exists():
        shutil.move(str(log_src), run_root / "log")
    if temp_src.exists():
        shutil.move(str(temp_src), run_root / "temporal_results")
    return run_root / "log" / "2d_temporal.log"


def sweep_counts(
    counts: Iterable[int],
    dataset_path: Path,
    epochs: int,
    batch_size: int,
    timesteps: int,
    sweep_root: Path,
    use_pysensors: bool,
    skip_existing: bool = True,
) -> List[dict]:
    """Iterate over sensor counts, running the temporal script each time."""
    summaries: List[dict] = []
    count_list = list(counts)
    total = len(count_list)
    for idx, M in enumerate(count_list, start=1):
        run_dir = sweep_root / f"M_{M:03d}"
        if skip_existing and run_dir.exists():
            print(f"[sensor_sweep] Skipping M={M}: results already exist in {run_dir}")
            log_path = run_dir / "log" / "2d_temporal.log"
            if log_path.exists():
                metrics = parse_metrics(log_path)
            else:
                metrics = {"best_val": None, "demo_psnr_global": None, "demo_psnr_solution": None}
            metrics.update({"M": M})
            summaries.append(metrics)
            continue

        print(f"[sensor_sweep] === Run {idx}/{total} with M={M} sensors ===")
        # Ensure clean dirs
        shutil.rmtree(RESULTS_ROOT / "log", ignore_errors=True)
        shutil.rmtree(RESULTS_ROOT / "temporal_results", ignore_errors=True)

        env = os.environ.copy()
        env_vars = {
            "GAPPY_TEMPORAL_DATASET": str(dataset_path),
            "GAPPY_TEMPORAL_USE_PYSENSORS": "1" if use_pysensors else "0",
            "GAPPY_TEMPORAL_M": str(M),
            "GAPPY_TEMPORAL_EPOCHS": str(epochs),
            "GAPPY_TEMPORAL_BATCH": str(batch_size),
            "GAPPY_TEMPORAL_STEPS": str(timesteps),
            "GAPPY_TEMPORAL_DEMOS": "1",
        }
        env.update(env_vars)
        cmd = [
            "python",
            "2d_temporal_mfem.py",
            "--skip-ex9",
            "--skip-convert",
            "--dataset-path",
            str(dataset_path),
        ]
        run_command(cmd, env=env)
        log_path = move_results(run_dir)
        if log_path.exists():
            metrics = parse_metrics(log_path)
        else:
            metrics = {"best_val": None, "demo_psnr_global": None, "demo_psnr_solution": None}
        metrics.update({"M": M})
        summaries.append(metrics)
    return summaries


def write_summary_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("M,best_val,demo_psnr_global,demo_psnr_solution\n")
        for row in rows:
            fh.write(
                f"{row['M']},"
                f"{row.get('best_val','')},"
                f"{row.get('demo_psnr_global','')},"
                f"{row.get('demo_psnr_solution','')}\n"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sensor-count sweep runner.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the MFEM dataset (.npz) to use.",
    )
    parser.add_argument(
        "--min-power",
        type=int,
        default=1,
        help="Minimum power of two for sensor count (default: 1 -> 2 sensors).",
    )
    parser.add_argument(
        "--max-power",
        type=int,
        default=8,
        help="Maximum power of two (default: 8 -> 256 sensors).",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per run.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size per run.")
    parser.add_argument("--timesteps", type=int, default=12, help="Demo timesteps.")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="If set, rerun experiments even if run folders already exist.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_SWEEP_SUBDIR,
        help="Subdirectory under MFEM_results/ to store sweep outputs.",
    )
    parser.add_argument(
        "--use-pysensors",
        action="store_true",
        help="Enable pysensors-based sensor placement during the sweep.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_path}' not found.")
    counts = [2 ** p for p in range(args.min_power, args.max_power + 1)]
    sweep_root = (RESULTS_ROOT / args.output_root)
    sweep_root.mkdir(parents=True, exist_ok=True)
    summaries = sweep_counts(
        counts=counts,
        dataset_path=dataset_path,
        epochs=args.epochs,
        batch_size=args.batch,
        timesteps=args.timesteps,
        sweep_root=sweep_root,
        use_pysensors=args.use_pysensors,
        skip_existing=not args.no_skip_existing,
    )
    write_summary_csv(summaries, sweep_root / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
