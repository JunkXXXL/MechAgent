#!/usr/bin/env python3
"""Build an experiment-name by run-number heatmap of rmse_mass2."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "numpy and matplotlib are required. Run the script in the same Python "
        "environment as the simulations."
    ) from exc


DEFAULT_INPUT = Path("trajectory_comparison_2026-09-29/comparison_results.csv")
DEFAULT_OUTPUT = Path("trajectory_comparison_2026-09-29/rmse_mass2_heatmap.png")


@dataclass
class Experiment:
    name: str
    number: int
    max_error_mass2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a heatmap with experiment names as rows, run numbers as "
            "columns, and rmse_mass2 as cell values."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Comparison CSV file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output image path.")
    parser.add_argument(
        "--row-order",
        choices=("name", "input"),
        default="name",
        help="Order experiment names alphabetically or by first CSV appearance.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output resolution.")
    return parser.parse_args()


def read_experiments(path: Path) -> list[Experiment]:
    if not path.is_file():
        raise FileNotFoundError(f"Comparison file does not exist: {path}")

    experiments: list[Experiment] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"simulation", "rmse_mass2", "max_error_mass2"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            simulation = (row.get("simulation") or "").strip()
            name, separator, raw_number = simulation.rpartition("/")
            if not separator or not name:
                raise ValueError(
                    f"Row {row_number}: simulation must have the form "
                    f"'experiment name/number', got {simulation!r}"
                )
            try:
                number = int(raw_number)
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_number}: invalid experiment number {raw_number!r}"
                ) from exc
            raw_value = (row.get("max_error_mass2") or "").strip()
            try:
                value = float(raw_value) if raw_value else math.nan
            except ValueError:
                value = math.nan
            if not math.isfinite(value):
                value = math.nan
            experiments.append(Experiment(name=name, number=number, max_error_mass2=value))

    if not experiments:
        raise ValueError("The comparison CSV contains no experiments")
    if not any(math.isfinite(item.max_error_mass2) for item in experiments):
        raise ValueError("The rmse_mass2 column contains no finite numeric values")
    return experiments


def build_matrix(
    experiments: list[Experiment], row_order: str
) -> tuple[list[str], list[int], np.ndarray]:
    names = list(dict.fromkeys(item.name for item in experiments))
    if row_order == "name":
        names.sort(key=str.casefold)
    numbers = sorted({item.number for item in experiments})

    name_indices = {name: index for index, name in enumerate(names)}
    number_indices = {number: index for index, number in enumerate(numbers)}
    matrix = np.full((len(names), len(numbers)), np.nan, dtype=float)
    occupied: set[tuple[int, int]] = set()
    for item in experiments:
        position = (name_indices[item.name], number_indices[item.number])
        if position in occupied:
            raise ValueError(
                f"Duplicate result for experiment {item.name!r}, number {item.number}"
            )
        occupied.add(position)
        matrix[position] = item.max_error_mass2 # max_error_mass2
    return names, numbers, matrix


def format_value(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.4g}"


def create_heatmap(
    names: list[str], numbers: list[int], matrix: np.ndarray, output: Path, dpi: int
) -> None:
    masked = np.ma.masked_invalid(matrix)

    valid_values = matrix[np.isfinite(matrix)]
    color_min = float(valid_values.min())
    color_max = float(valid_values.max())
    # Matplotlib needs a non-zero normalization interval for equal values.
    if color_min == color_max:
        color_max = color_min + max(abs(color_min) * 1e-9, 1e-12)

    longest_name = max(len(name) for name in names)
    figure_width = min(28.0, max(16.0, 10.0 + longest_name * 0.095))
    figure_height = max(8.0, 0.62 * len(names) + 2.0)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))

    color_map = plt.colormaps["YlOrRd"].copy()
    color_map.set_bad(color="#d9d9d9")
    image = axis.imshow(
        masked,
        cmap=color_map,
        vmin=color_min,
        vmax=color_max,
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_xticks(np.arange(len(numbers)), labels=[str(number) for number in numbers])
    axis.set_yticks(np.arange(len(names)), labels=names)
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")
    axis.set_xlabel("Номер эксперимента", labelpad=12)
    axis.set_ylabel("Название эксперимента")
    axis.tick_params(axis="x", labelsize=11, pad=8)
    axis.tick_params(axis="y", labelsize=8, length=0)
    axis.set_title(
        "Отклонение траектории второго груза от эталона",
        fontsize=14,
        pad=36,
    )

    color_range = color_max - color_min
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if math.isfinite(value):
                relative_color = (value - color_min) / color_range
                text_color = "white" if relative_color > 0.65 else "black"
            else:
                text_color = "#555555"
            axis.text(
                column_index,
                row_index,
                format_value(value),
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_color,
            )

    axis.set_xticks(np.arange(-0.5, len(numbers), 1.0), minor=True)
    axis.set_yticks(np.arange(-0.5, len(names), 1.0), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.7)
    axis.tick_params(which="minor", bottom=False, left=False)

    color_bar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    color_bar.set_label("RMSE второго груза", rotation=270, labelpad=18)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")

    experiments = read_experiments(args.input)
    names, numbers, matrix = build_matrix(experiments, args.row_order)
    create_heatmap(names, numbers, matrix, args.output, args.dpi)
    valid_count = sum(math.isfinite(item.max_error_mass2) for item in experiments)
    print(f"Experiments: {len(experiments)}; numeric max_error_mass2 values: {valid_count}")
    print(f"Heatmap saved to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
