#!/usr/bin/env python3
"""Create an editable heatmap table with experiment groups as columns."""

from __future__ import annotations

import argparse
import csv
import math
import textwrap
from pathlib import Path


DEFAULT_SOURCE = Path("trajectory_comparison_2026-09-29/comparison_results.csv")
DEFAULT_TABLE = Path("trajectory_comparison_2026-09-29/group_mean_rmse_table.csv")
DEFAULT_OUTPUT = Path("trajectory_comparison_2026-09-29/group_mean_rmse_heatmap.png")
COUNT_ROW_LABEL = "Количество max_error_mass2 < 1"
OBSOLETE_ROW_LABELS = {"Среднее rmse_mass2", "Среднее max_error_mass2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count max_error_mass2 < 1 for each experiment group and draw a heatmap "
            "whose columns are group names. Extra rows added to the editable table "
            "are preserved."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Do not recalculate the built-in count row from comparison_results.csv.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help=(
            "Recalculate from comparison_results.csv and render the image without "
            "changing the editable CSV table."
        ),
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Create or update the editable CSV table without rendering the image.",
    )
    return parser.parse_args()


def split_simulation(value: str, row_number: int) -> tuple[str, int]:
    group_name, separator, raw_number = value.strip().rpartition("/")
    if not separator or not group_name:
        raise ValueError(
            f"Row {row_number}: simulation must have the form 'group name/number', "
            f"got {value!r}"
        )
    try:
        return group_name, int(raw_number)
    except ValueError as exc:
        raise ValueError(
            f"Row {row_number}: invalid experiment number {raw_number!r}"
        ) from exc


def calculate_group_counts(source: Path) -> dict[str, int]:
    if not source.is_file():
        raise FileNotFoundError(f"Comparison report does not exist: {source}")

    counts: dict[str, int] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"simulation", "max_error_mass2"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            group_name, _ = split_simulation(row.get("simulation") or "", row_number)
            counts.setdefault(group_name, 0)
            raw_value = (row.get("max_error_mass2") or "").strip()
            if not raw_value:
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            if math.isfinite(value) and value < 1:
                counts[group_name] += 1

    return dict(sorted(counts.items(), key=lambda item: item[0].casefold()))


def update_editable_table(table: Path, counts: dict[str, int]) -> None:
    label_column = "metric"
    existing_columns: list[str] = []
    rows: list[dict[str, str]] = []

    if table.is_file():
        with table.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"Editable table has no header: {table}")
            label_column = reader.fieldnames[0]
            existing_columns = reader.fieldnames[1:]
            rows = [dict(row) for row in reader]

    rows = [
        row for row in rows
        if (row.get(label_column) or "").strip() not in OBSOLETE_ROW_LABELS
    ]
    experiment_columns = existing_columns + [
        name for name in counts if name not in existing_columns
    ]
    count_row = next(
        (row for row in rows if (row.get(label_column) or "").strip() == COUNT_ROW_LABEL),
        None,
    )
    if count_row is None:
        count_row = {label_column: COUNT_ROW_LABEL}
        rows.insert(0, count_row)
    for group_name, value in counts.items():
        count_row[group_name] = str(value)

    fieldnames = [label_column, *experiment_columns]
    table.parent.mkdir(parents=True, exist_ok=True)
    temporary = table.with_suffix(table.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    temporary.replace(table)


def read_editable_table(table: Path) -> tuple[list[str], list[str], list[list[float]]]:
    if not table.is_file():
        raise FileNotFoundError(f"Editable heatmap table does not exist: {table}")

    with table.open("r", encoding="utf-8-sig") as stream: # utf-8-sig CP1251
        reader = csv.DictReader(stream, delimiter=",")
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            raise ValueError("The editable table needs a label column and experiment columns")
        label_column = reader.fieldnames[0]
        experiment_names = reader.fieldnames[1:]
        row_labels: list[str] = []
        matrix: list[list[float]] = []
        for row_number, row in enumerate(reader, start=2):
            label = (row.get(label_column) or "").strip()
            if not label:
                continue
            values: list[float] = []
            for experiment_name in experiment_names:
                raw_value = (row.get(experiment_name) or "").strip()
                if not raw_value:
                    values.append(math.nan)
                    continue
                try:
                    value = float(str(raw_value).replace(',','.'))
                except ValueError as exc:
                    raise ValueError(
                        f"Row {row_number}, column {experiment_name!r}: "
                        f"expected a number, got {raw_value!r}"
                    ) from exc
                values.append(value if math.isfinite(value) else math.nan)
            row_labels.append(label)
            matrix.append(values)

    if not row_labels:
        raise ValueError("The editable table contains no data rows")
    return experiment_names, row_labels, matrix


def build_in_memory_table(
    table: Path, counts: dict[str, int]
) -> tuple[list[str], list[str], list[list[float]]]:
    if table.is_file():
        experiment_names, row_labels, matrix = read_editable_table(table)
    else:
        experiment_names, row_labels, matrix = [], [], []

    for name in counts:
        if name not in experiment_names:
            experiment_names.append(name)
            for row in matrix:
                row.append(math.nan)

    kept_rows = [
        (label, values)
        for label, values in zip(row_labels, matrix)
        if label not in OBSOLETE_ROW_LABELS and label != COUNT_ROW_LABEL
    ]
    count_values = [float(counts[name]) if name in counts else math.nan for name in experiment_names]
    return (
        experiment_names,
        [COUNT_ROW_LABEL, *(label for label, _ in kept_rows)],
        [count_values, *(values for _, values in kept_rows)],
    )


def format_cell(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.4g}"


def create_heatmap(
    experiment_names: list[str],
    row_labels: list[str],
    matrix_values: list[list[float]],
    output: Path,
    dpi: int,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Pillow is required to render the heatmap. "
            "The editable CSV can still be generated with --data-only."
        ) from exc

    valid_values = [
        value for row in matrix_values for value in row if math.isfinite(value)
    ]
    if not valid_values:
        raise ValueError("The editable table contains no finite values")
    color_min = min(valid_values)
    color_max = max(valid_values)
    if color_min == color_max:
        color_max = color_min + max(abs(color_min) * 1e-9, 1e-12)
    scale = dpi / 180
    cell_width = round(190 * scale)
    cell_height = round(58 * scale)
    label_width = round(305 * scale)
    left_margin = round(24 * scale)
    top_margin = round(35 * scale)
    header_height = round(225 * scale)
    bottom_margin = round(105 * scale)
    width = left_margin + label_width + cell_width * len(experiment_names) + left_margin
    height = top_margin + header_height + cell_height * len(row_labels) + bottom_margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    font = ImageFont.truetype(str(font_path), round(16 * scale)) if font_path.is_file() else ImageFont.load_default()
    bold_font = ImageFont.truetype(str(bold_path), round(17 * scale)) if bold_path.is_file() else font
    title_font = ImageFont.truetype(str(bold_path), round(26 * scale)) if bold_path.is_file() else font
    title = "Запуски с max_error_mass2 < 1 по группам"
    draw.text((left_margin, top_margin), title, fill="#202020", font=title_font)

    grid_x = left_margin + label_width
    grid_y = top_margin + header_height
    for column_index, name in enumerate(experiment_names):
        wrapped = textwrap.fill(name, width=18)
        draw.multiline_text(
            (grid_x + column_index * cell_width + round(8 * scale), grid_y - round(180 * scale)),
            wrapped,
            fill="#202020",
            font=font,
            spacing=round(3 * scale),
        )

    def cell_color(value: float) -> tuple[int, int, int]:
        if not math.isfinite(value):
            return (217, 217, 217)
        fraction = (value - color_min) / (color_max - color_min)
        stops = [(255, 255, 204), (255, 237, 160), (254, 178, 76), (240, 59, 32), (189, 0, 38)]
        scaled = fraction * (len(stops) - 1)
        index = min(int(scaled), len(stops) - 2)
        blend = scaled - index
        return tuple(round(a + (b - a) * blend) for a, b in zip(stops[index], stops[index + 1]))

    for row_index, label in enumerate(row_labels):
        y0 = grid_y + row_index * cell_height
        draw.text((left_margin, y0 + round(18 * scale)), label, fill="#202020", font=bold_font)
        for column_index, value in enumerate(matrix_values[row_index]):
            x0 = grid_x + column_index * cell_width
            fill = cell_color(value)
            draw.rectangle(
                (x0, y0, x0 + cell_width - 1, y0 + cell_height - 1),
                fill=fill,
                outline="#ffffff",
                width=max(1, round(scale)),
            )
            text_value = format_cell(value)
            box = draw.textbbox((0, 0), text_value, font=bold_font)
            text_width = box[2] - box[0]
            text_height = box[3] - box[1]
            text_color = "#ffffff" if math.isfinite(value) and (value - color_min) / (color_max - color_min) > 0.68 else "#202020"
            draw.text(
                (x0 + (cell_width - text_width) / 2, y0 + (cell_height - text_height) / 2 - box[1]),
                text_value,
                fill=text_color,
                font=bold_font,
            )

    legend_y = grid_y + cell_height * len(row_labels) + round(35 * scale)
    legend_width = round(360 * scale)
    legend_height = round(14 * scale)
    for offset in range(legend_width):
        value = color_min + offset / max(legend_width - 1, 1) * (color_max - color_min)
        draw.line((grid_x + offset, legend_y, grid_x + offset, legend_y + legend_height), fill=cell_color(value))
    draw.text((grid_x, legend_y + legend_height + round(6 * scale)), format_cell(color_min), fill="#202020", font=font)
    draw.text((grid_x + legend_width + round(10 * scale), legend_y - round(2 * scale)), f"{format_cell(color_max)} подходящих запусков", fill="#202020", font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(dpi, dpi))


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    if args.no_write and (args.no_refresh or args.data_only):
        raise SystemExit("--no-write cannot be combined with --no-refresh or --data-only")
    if args.no_write:
        experiment_names, row_labels, matrix = build_in_memory_table(
            args.table, calculate_group_counts(args.source)
        )
    else:
        if not args.no_refresh:
            update_editable_table(args.table, calculate_group_counts(args.source))
            print(f"Editable table updated: {args.table.resolve()}")
        if args.data_only:
            return 0
        experiment_names, row_labels, matrix = read_editable_table(args.table)
    create_heatmap(experiment_names, row_labels, matrix, args.output, args.dpi)
    print(f"Heatmap saved to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
