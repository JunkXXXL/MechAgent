#!/usr/bin/env python3
"""Run generated pendulum simulations and compare their trajectories.

Every simulation is copied to an isolated working directory before execution, so
files in ``results`` (especially the reference CSV) are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "numpy is required. Run this script with the same Python environment "
        "that is used for the simulations."
    ) from exc


DEFAULT_RESULTS_ROOT = Path("results/2026-09-29")
DEFAULT_REFERENCE = DEFAULT_RESULTS_ROOT / "Эталонный запрос/2/double_pendulum_trajectories.csv"
TRAJECTORY_COLUMNS = ("x1", "y1", "x2", "y2")


@dataclass
class Trajectory:
    time: np.ndarray
    coordinates: np.ndarray


@dataclass
class Result:
    simulation: str
    script: str
    status: str = "pending"
    exit_code: int | None = None
    duration_seconds: float | None = None
    trajectory_csv: str = ""
    samples_compared: int | None = None
    reference_coverage_percent: float | None = None
    rmse_mass1: float | None = None
    rmse_mass2: float | None = None
    rmse_coordinates: float | None = None
    nrmse_percent: float | None = None
    mae_coordinates: float | None = None
    max_error_mass1: float | None = None
    max_error_mass2: float | None = None
    max_error: float | None = None
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run every Python simulation below a results directory and compare "
            "the generated mass trajectories with a reference CSV."
        )
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("trajectory_comparison_2026-09-29"),
        help="Directory for isolated runs, logs, and reports.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of simulations run in parallel (default: up to 4).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout for one simulation in seconds (default: 120).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N simulations; useful for a quick test.",
    )
    return parser.parse_args()


def normalized_name(name: str) -> str:
    """Map headers such as '# time [s]' or 'x_1' to compact identifiers."""
    return "".join(character for character in name.lower() if character.isalnum())


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        sample = stream.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def read_trajectory(path: Path) -> Trajectory:
    delimiter = detect_delimiter(path)
    data = np.genfromtxt(
        path,
        delimiter=delimiter,
        names=True,
        dtype=np.float64,
        encoding="utf-8-sig",
        autostrip=True,
        invalid_raise=False,
    )
    if data.dtype.names is None:
        raise ValueError("CSV does not contain a named header")

    data = np.atleast_1d(data)
    names = list(data.dtype.names)
    time_name = next(
        (name for name in names if name.strip() == "t"),
        None,
    )
    columns = {normalized_name(name): name for name in data.dtype.names}
    coordinate_names = [columns.get(name) for name in TRAJECTORY_COLUMNS]
    if time_name is None or any(name is None for name in coordinate_names):
        available = ", ".join(data.dtype.names)
        raise ValueError(
            "required columns t/time, x1, y1, x2, y2 were not found; "
            f"available: {available}"
        )

    time_values = np.asarray(data[time_name], dtype=np.float64)
    coordinates = np.column_stack(
        [np.asarray(data[name], dtype=np.float64) for name in coordinate_names]
    )
    finite = np.isfinite(time_values) & np.all(np.isfinite(coordinates), axis=1)
    time_values = time_values[finite]
    coordinates = coordinates[finite]
    if len(time_values) < 2:
        raise ValueError("trajectory contains fewer than two finite samples")

    order = np.argsort(time_values, kind="stable")
    time_values = time_values[order]
    coordinates = coordinates[order]
    time_values, unique_indices = np.unique(time_values, return_index=True)
    coordinates = coordinates[unique_indices]
    if len(time_values) < 2:
        raise ValueError("trajectory contains fewer than two distinct time samples")
    return Trajectory(time=time_values, coordinates=coordinates)


def compare_trajectory(candidate: Trajectory, reference: Trajectory) -> dict[str, float | int]:
    in_range = (reference.time >= candidate.time[0]) & (reference.time <= candidate.time[-1])
    comparison_time = reference.time[in_range]
    if len(comparison_time) < 2:
        raise ValueError("candidate and reference time ranges do not overlap")

    interpolated = np.column_stack(
        [
            np.interp(comparison_time, candidate.time, candidate.coordinates[:, index])
            for index in range(4)
        ]
    )
    differences = interpolated - reference.coordinates[in_range]
    mass1_distance = np.linalg.norm(differences[:, 0:2], axis=1)
    mass2_distance = np.linalg.norm(differences[:, 2:4], axis=1)
    reference_scale = math.sqrt(float(np.mean(reference.coordinates[in_range] ** 2)))
    rmse_coordinates = math.sqrt(float(np.mean(differences**2)))

    return {
        "samples_compared": len(comparison_time),
        "reference_coverage_percent": 100.0 * len(comparison_time) / len(reference.time),
        "rmse_mass1": math.sqrt(float(np.mean(mass1_distance**2))),
        "rmse_mass2": math.sqrt(float(np.mean(mass2_distance**2))),
        "rmse_coordinates": rmse_coordinates,
        "nrmse_percent": (
            100.0 * rmse_coordinates / reference_scale if reference_scale > 0.0 else math.nan
        ),
        "mae_coordinates": float(np.mean(np.abs(differences))),
        "max_error_mass1": float(np.max(mass1_distance)),
        "max_error_mass2": float(np.max(mass2_distance)),
        "max_error": float(max(np.max(mass1_distance), np.max(mass2_distance))),
    }


def safe_run_directory(output_dir: Path, relative_script: Path) -> Path:
    # There is one generated script per source directory, but including the stem
    # keeps this correct if that layout changes later.
    return output_dir / "runs" / relative_script.parent / relative_script.stem


def find_trajectory_csv(run_dir: Path) -> tuple[Path, Trajectory]:
    compatible: list[tuple[int, int, Path, Trajectory]] = []
    errors: list[str] = []
    for csv_path in sorted(run_dir.rglob("*.csv")):
        try:
            trajectory = read_trajectory(csv_path)
            name_score = 1 if "traject" in csv_path.name.lower() else 0
            compatible.append((name_score, len(trajectory.time), csv_path, trajectory))
        except Exception as exc:  # An auxiliary CSV may legitimately lack positions.
            errors.append(f"{csv_path.name}: {exc}")
    if not compatible:
        detail = "; ".join(errors[:3]) if errors else "no CSV files were produced"
        raise FileNotFoundError(detail)
    _, _, csv_path, trajectory = max(compatible, key=lambda item: (item[0], item[1]))
    return csv_path, trajectory


def run_one(
    source: Path,
    relative_script: Path,
    output_dir: Path,
    reference: Trajectory,
    timeout: float,
) -> Result:
    simulation = relative_script.parent.as_posix()
    result = Result(simulation=simulation, script=str(source))
    run_dir = safe_run_directory(output_dir, relative_script)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    copied_script = run_dir / source.name
    shutil.copy2(source, copied_script)

    environment = os.environ.copy()
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    started = time.perf_counter()
    try:
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr_file:
            completed = subprocess.run(
                [sys.executable, copied_script.name],
                cwd=run_dir,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=timeout,
                check=False,
            )
        result.exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        result.status = "timeout"
        result.error = f"simulation exceeded the {timeout:g} s timeout"
        return result
    except Exception as exc:
        result.status = "run_error"
        result.error = str(exc)
        return result
    finally:
        result.duration_seconds = time.perf_counter() - started

    try:
        csv_path, candidate = find_trajectory_csv(run_dir)
        result.trajectory_csv = str(csv_path)
        for key, value in compare_trajectory(candidate, reference).items():
            setattr(result, key, value)
        if result.exit_code == 0:
            result.status = "ok"
        else:
            result.status = "run_failed_with_trajectory"
            result.error = f"process returned exit code {result.exit_code}; see stderr.log"
    except Exception as exc:
        result.status = "run_failed" if result.exit_code else "no_comparable_trajectory"
        prefix = f"process returned exit code {result.exit_code}; " if result.exit_code else ""
        result.error = prefix + str(exc)
    return result


def is_below(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover_scripts(results_root: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    scripts = []
    for source in results_root.rglob("*.py"):
        resolved = source.resolve()
        if is_below(resolved, output_dir.resolve()):
            continue
        scripts.append((resolved, resolved.relative_to(results_root.resolve())))
    return sorted(scripts, key=lambda item: item[1].as_posix().casefold())


def optional_number(value: object) -> object:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def write_reports(results: Iterable[Result], output_dir: Path, reference_path: Path) -> None:
    ordered = sorted(
        results,
        key=lambda row: (
            row.rmse_coordinates is None,
            -(row.reference_coverage_percent or 0.0),
            row.rmse_coordinates if row.rmse_coordinates is not None else math.inf,
            row.simulation.casefold(),
        ),
    )
    report_csv = output_dir / "comparison_results.csv"
    fieldnames = list(Result.__dataclass_fields__)
    with report_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered:
            writer.writerow({key: optional_number(value) for key, value in asdict(row).items()})

    status_counts: dict[str, int] = {}
    for row in ordered:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
    comparable = [row for row in ordered if row.rmse_coordinates is not None]
    summary = {
        "reference": str(reference_path),
        "total_simulations": len(ordered),
        "comparable_trajectories": len(comparable),
        "status_counts": status_counts,
        "best_full_coverage_simulations": [
            {
                "simulation": row.simulation,
                "rmse_coordinates": row.rmse_coordinates,
                "nrmse_percent": row.nrmse_percent,
            }
            for row in sorted(
                (item for item in comparable if (item.reference_coverage_percent or 0.0) >= 99.999),
                key=lambda item: item.rmse_coordinates or math.inf,
            )[:10]
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    if importlib.util.find_spec("matplotlib") is None:
        raise SystemExit(
            "matplotlib is not installed for this Python interpreter; the generated "
            "simulations import it. Activate the project environment first."
        )
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    results_root = args.results_root.resolve()
    reference_path = args.reference.resolve()
    output_dir = args.output_dir.resolve()
    if not results_root.is_dir():
        raise SystemExit(f"Results directory does not exist: {results_root}")
    if not reference_path.is_file():
        raise SystemExit(f"Reference CSV does not exist: {reference_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read before launching anything. Runs are isolated as an additional safeguard.
    reference = read_trajectory(reference_path)
    scripts = discover_scripts(results_root, output_dir)
    if args.limit is not None:
        scripts = scripts[: max(0, args.limit)]
    if not scripts:
        raise SystemExit("No simulation scripts were found")

    print(f"Found {len(scripts)} simulations; reference has {len(reference.time)} samples.")
    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one,
                source,
                relative,
                output_dir,
                reference,
                args.timeout,
            ): relative
            for source, relative in scripts
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            relative = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # Keep the batch alive if the runner itself hits an edge case.
                result = Result(
                    simulation=relative.parent.as_posix(),
                    script=str(results_root / relative),
                    status="runner_error",
                    error=str(exc),
                )
            results.append(result)
            metric = (
                f", RMSE={result.rmse_coordinates:.6g}"
                if result.rmse_coordinates is not None
                else ""
            )
            print(f"[{completed_count}/{len(scripts)}] {result.status}: {result.simulation}{metric}")

    write_reports(results, output_dir, reference_path)
    comparable = sum(result.rmse_coordinates is not None for result in results)
    print(f"Comparable trajectories: {comparable}/{len(results)}")
    print(f"CSV report: {output_dir / 'comparison_results.csv'}")
    print(f"JSON summary: {output_dir / 'summary.json'}")
    return 0 if comparable else 1


if __name__ == "__main__":
    raise SystemExit(main())
