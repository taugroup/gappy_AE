#!/usr/bin/env python3
"""
End-to-end utility that runs MFEM example 9, converts its ParaView output
to a regular grid dataset, and then launches 2d_temporal.py for gappy
signal reconstruction using the generated data.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "MFEM_results"
DEFAULT_PARAVIEW_DIR = RESULTS_ROOT / "ParaView" / "Example9"
DEFAULT_DATASET_PATH = RESULTS_ROOT / "datasets" / "mfem_ex9_dataset.npz"


def run_command(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run a subprocess while streaming output."""
    display = " ".join(cmd)
    print(f"[2d_temporal_mfem] Running: {display}")
    workdir = cwd or RESULTS_ROOT
    workdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=str(workdir), env=env, check=True)


def generate_mfem_paraview(args: argparse.Namespace) -> None:
    ex9_script = REPO_ROOT / "PyMFEM" / "examples" / "ex9.py"
    if not ex9_script.exists():
        raise FileNotFoundError(f"Could not locate MFEM example script at {ex9_script}")
    cmd = [
        sys.executable,
        str(ex9_script),
        "--mesh",
        args.mesh,
        "--refine",
        str(args.refine),
        "--problem",
        str(args.problem),
        "--order",
        str(args.order),
        "--ode-solver",
        str(args.ode_solver),
        "--t-final",
        str(args.t_final),
        "--time-step",
        str(args.time_step),
        "--visualization-steps",
        str(args.vis_steps),
        "--device",
        args.device,
    ]
    if args.partial_assembly:
        cmd.append("--partial-assembly")
    if args.visualization:
        cmd.append("--visualization")
    cmd.append("--paraview-datafiles")
    run_command(cmd)


def convert_paraview_dataset(args: argparse.Namespace, dataset_path: Path) -> None:
    converter = REPO_ROOT / "MFEM_Dataset" / "make_ex9_dataset.py"
    if not converter.exists():
        raise FileNotFoundError(f"Could not locate converter script at {converter}")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(converter),
        "--paraview-dir",
        str(Path(args.paraview_dir).resolve()),
        "--resolution",
        str(args.resolution),
        "--output",
        str(dataset_path.resolve()),
        "--field",
        args.field_name,
        "--stride",
        str(args.stride),
    ]
    if args.include_gradients:
        cmd.append("--include-gradients")
    run_command(cmd)


def launch_temporal_training(args: argparse.Namespace, dataset_path: Path) -> None:
    temporal_script = REPO_ROOT / "2d_temporal.py"
    if not temporal_script.exists():
        raise FileNotFoundError(f"Could not locate temporal script at {temporal_script}")
    env = os.environ.copy()
    env["GAPPY_TEMPORAL_DATASET"] = str(dataset_path.resolve())
    for item in args.gappy_env:
        if "=" not in item:
            raise ValueError(f"Invalid --gappy-env entry '{item}'. Expected format KEY=VALUE.")
        key, value = item.split("=", 1)
        env[key] = value
    cmd = [sys.executable, str(temporal_script)]
    run_command(cmd, env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MFEM Example 9 dataset and feed it to 2d_temporal.py."
    )
    parser.add_argument(
        "--mesh",
        default="periodic-square.mesh",
        help="Mesh filename relative to PyMFEM/data (default: periodic-square.mesh).",
    )
    parser.add_argument("--refine", type=int, default=1, help="Uniform refinement levels for MFEM.")
    parser.add_argument("--problem", type=int, default=0, help="MFEM problem setup index.")
    parser.add_argument("--order", type=int, default=3, help="Finite element order for MFEM.")
    parser.add_argument("--ode-solver", type=int, default=4, help="MFEM ODE solver choice.")
    parser.add_argument("--t-final", type=float, default=0.5, help="Final simulation time for MFEM.")
    parser.add_argument("--time-step", type=float, default=0.05, help="Time step used by MFEM.")
    parser.add_argument("--vis-steps", type=int, default=5, help="Visualization cadence for MFEM.")
    parser.add_argument("--device", default="cpu", help="Device string passed to MFEM.")
    parser.add_argument("--partial-assembly", action="store_true", help="Enable MFEM partial assembly.")
    parser.add_argument("--visualization", action="store_true", help="Enable MFEM GLVis streaming.")
    parser.add_argument(
        "--paraview-dir",
        default=str(DEFAULT_PARAVIEW_DIR),
        help="Directory where MFEM writes ParaView files.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Grid resolution for dataset interpolation.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET_PATH),
        help="Output path for the generated .npz dataset.",
    )
    parser.add_argument("--field-name", default="solution", help="Field name extracted from ParaView files.")
    parser.add_argument("--include-gradients", action="store_true", help="Store spatial gradients in the dataset.")
    parser.add_argument("--stride", type=int, default=1, help="Take every n-th snapshot during conversion.")
    parser.add_argument(
        "--skip-ex9",
        action="store_true",
        help="Skip running MFEM Example 9 (assume ParaView outputs already exist).",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip conversion step (assume dataset already exists).",
    )
    parser.add_argument(
        "--only-generate",
        action="store_true",
        help="Stop after dataset creation without launching 2d_temporal.py.",
    )
    parser.add_argument(
        "--gappy-env",
        action="append",
        default=[],
        help="Extra KEY=VALUE environment overrides for 2d_temporal.py (repeatable).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    dataset_path = Path(args.dataset_path)

    if not args.skip_ex9:
        generate_mfem_paraview(args)
    else:
        print("[2d_temporal_mfem] Skipping MFEM Example 9 execution.")

    if not args.skip_convert:
        convert_paraview_dataset(args, dataset_path)
    else:
        print(f"[2d_temporal_mfem] Skipping conversion; expecting dataset at {dataset_path}.")

    if args.only_generate:
        print("[2d_temporal_mfem] Dataset generation complete; skipping temporal training.")
        return 0

    launch_temporal_training(args, dataset_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
