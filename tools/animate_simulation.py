"""Render a saved MacPherson trajectory: python animate_simulation.py specification.json timeseries.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = (
    "t", "x", "z_s", "v_s", "z_u", "v_u", "H", "H_dot", "q", "delta",
    "c_s", "c_d", "spring_vertical_force", "damper_vertical_force",
    "suspension_vertical_force", "tire_vertical_force",
)


def load_inputs(specification_path, timeseries_path):
    """Validate the fixed SI CSV contract and the existing specification schema.

    State displacements are relative to equilibrium, positive upwards. R_t must
    be a numeric radius in metres. Other specification expressions are never run.
    Returns (channels, radius, static_tire_compression).
    """
    with Path(specification_path).open(encoding="utf-8-sig") as stream:
        try:
            specification = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Некорректный specification.json: {exc}") from exc
    if not isinstance(specification, dict) or not isinstance(specification.get("parameters"), list):
        raise ValueError("specification.json: требуется объект с массивом parameters")
    parameters = {}
    for record in specification["parameters"]:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ValueError("specification.json: каждый параметр должен иметь строковое name")
        name = record["name"]
        if name in parameters:
            raise ValueError(f"specification.json: повтор параметра {name}")
        parameters[name] = record
    if "R_t" not in parameters:
        raise ValueError("specification.json: отсутствует обязательный параметр R_t (радиус шины, m)")
    record = parameters["R_t"]
    if record.get("unit") != "m":
        raise ValueError("specification.json: R_t должен иметь единицу m")
    try:
        if isinstance(record.get("value"), bool):
            raise ValueError
        radius = float(record["value"])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("specification.json: R_t.value должен быть числом, а не выражением") from exc
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("specification.json: R_t должен быть конечным положительным числом")

    rows = []
    with Path(timeseries_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        duplicates = sorted({name for name in header if header.count(name) > 1})
        if duplicates:
            raise ValueError("CSV: повторяющиеся колонки: " + ", ".join(duplicates))
        missing = [name for name in REQUIRED_COLUMNS if name not in header]
        if missing:
            raise ValueError("CSV: отсутствуют обязательные колонки: " + ", ".join(missing))
        indices = [header.index(name) for name in REQUIRED_COLUMNS]
        for line, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"CSV, строка {line}: число значений не соответствует заголовку")
            values = []
            for name, index in zip(REQUIRED_COLUMNS, indices):
                try:
                    value = float(row[index])
                except ValueError as exc:
                    raise ValueError(f"CSV, строка {line}, {name}: требуется число") from exc
                if not math.isfinite(value):
                    raise ValueError(f"CSV, строка {line}, {name}: требуется конечное число")
                values.append(value)
            rows.append(values)
    if len(rows) < 2:
        raise ValueError("CSV: требуется не менее двух состояний")
    array = np.asarray(rows)
    data = dict(zip(REQUIRED_COLUMNS, array.T))
    if np.any(np.diff(data["t"]) <= 0):
        raise ValueError("CSV: время t должно строго возрастать")
    if not np.allclose(data["q"], data["z_u"] - data["z_s"], rtol=1e-6, atol=1e-8):
        raise ValueError("CSV: нарушено соглашение q = z_u - z_s (метры)")
    baseline = data["delta"] - data["H"] + data["z_u"]
    delta0 = float(baseline[0])
    if not np.allclose(baseline, delta0, rtol=1e-6, atol=1e-8):
        raise ValueError("CSV: delta - H + z_u должна быть постоянной статической деформацией")
    if not 0 <= delta0 < radius:
        raise ValueError("CSV/JSON: требуется 0 <= статическая деформация шины < R_t")
    return data, radius, delta0


def _build_animation(data, radius, delta0):
    """Adapt the reference model's schematic to saved channels, without dynamics."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Ellipse, Rectangle

    fps = 30
    t = data["t"]
    count = max(2, math.ceil(float(t[-1] - t[0]) * fps) + 1)
    frame_t = np.linspace(t[0], t[-1], count)
    frames = {key: np.interp(frame_t, t, values) for key, values in data.items()}
    wheel0 = radius - delta0
    # Display dimensions, as in the reference schematic, are not hardpoint geometry.
    body0 = wheel0 + 0.82
    fig = Figure(figsize=(12, 6))
    FigureCanvasAgg(fig)
    scene, history = fig.subplots(1, 2)
    fig.suptitle("MacPherson: движение подвески", fontsize=16)
    scene.set(title="Боковая схема: 2 степени свободы, масштаб 1:1",
              xlabel="Расстояние от колеса, м", ylabel="Высота, м",
              xlim=(-1.2, 1.2),
              ylim=(min(float(data["H"].min()), wheel0 + float(data["z_u"].min()) - radius) - .15,
                    max(body0 + float(data["z_s"].max()) + .3, float(data["H"].max()) + .3)))
    scene.set_aspect("equal", adjustable="box")
    scene.grid(alpha=.15)
    envelope_line, = scene.plot([], [], color="#d28a00", lw=2, label="Огибающая H (CSV)")
    envelope, = scene.plot([], [], "o", color="#d28a00")
    tire = Ellipse((0, wheel0), 2 * radius, 2 * wheel0,
                   facecolor="#d9e0e7", edgecolor="#263445", lw=3)
    body = Rectangle((-.62, body0), 1.24, .22, color="#3276ac")
    scene.add_patch(tire)
    scene.add_patch(body)
    hub, = scene.plot([], [], "o", color="#263445", ms=8)
    spring, = scene.plot([], [], color="#29854d", lw=2.5, label="Пружина")
    damper, = scene.plot([], [], color="#b34f55", lw=3, label="Амортизатор")
    axle, = scene.plot([], [], color="#263445", lw=3)
    status = scene.text(.02, .98, "", transform=scene.transAxes, va="top", fontsize=9,
                        bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    scene.legend(loc="lower left", fontsize=8)
    for key, color, label in (("z_s", "#3276ac", "Кузов z_s"),
                              ("z_u", "#263445", "Колесо z_u"),
                              ("H", "#d28a00", "Огибающая H")):
        history.plot(t, 1000 * data[key], color=color, label=label)
    history.set(xlabel="Время, с", ylabel="Перемещение, мм",
                title="Вертикальные перемещения", xlim=(t[0], t[-1]))
    history.grid(alpha=.25)
    history.legend(loc="upper right", fontsize=9)
    cursor = history.axvline(t[0], color="#b34f55", lw=1.5)
    body_dot, = history.plot([], [], "o", color="#3276ac")
    wheel_dot, = history.plot([], [], "o", color="#263445")
    fig.text(.5, .025, "Пружина и амортизатор схематические. Показана сохранённая огибающая H, не профиль h(x).",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))

    def update(index):
        time, x = frame_t[index], frames["x"][index]
        zs, zu, compression = (frames[key][index] for key in ("z_s", "z_u", "delta"))
        wz, bz = wheel0 + zu, body0 + zs
        tire.center = (0, wz)
        tire.height = 2 * max(radius - max(compression, 0), .02 * radius)
        body.set_y(bz)
        hub.set_data([0], [wz])
        axle.set_data([-.36, .36], [wz, wz])
        bottom, top = wz + .07, bz
        coil_y = np.linspace(bottom + .08, top - .08, 17)
        coil_x = -.25 + .065 * np.where(np.arange(17) % 2, 1, -1)
        spring.set_data(np.r_[-.25, -.25, coil_x, -.25, -.25],
                        np.r_[bottom, bottom + .08, coil_y, top - .08, top])
        mid = bottom + .55 * (top - bottom)
        damper.set_data([.25, .25, np.nan, .19, .19, .31, .31, np.nan, .25, .25],
                        [bottom, mid, np.nan, bottom + .1, mid + .07, mid + .07,
                         bottom + .1, np.nan, mid, top])
        # Only the recorded spatial interval is drawn; no extrapolated road.
        envelope_line.set_data(data["x"] - x, data["H"])
        envelope.set_data([0], [frames["H"][index]])
        cursor.set_xdata([time, time])
        body_dot.set_data([time], [1000 * zs])
        wheel_dot.set_data([time], [1000 * zu])
        status.set_text(f"t = {time:.2f} с    x = {x:.2f} м\n"
                        f"Кузов: {1000 * zs:+.1f} мм    Колесо: {1000 * zu:+.1f} мм\n"
                        f"Ход подвески: {1000 * frames['q'][index]:+.1f} мм\n"
                        f"Сжатие шины: {1000 * max(compression, 0):.1f} мм"
                        + ("\nГеометрический зазор шины" if compression <= 0 else ""))
        return (envelope_line, envelope, tire, body, hub, spring, damper, axle,
                status, cursor, body_dot, wheel_dot)

    animation = FuncAnimation(fig, update, frames=count, interval=1000 / fps,
                              blit=False, repeat=True)
    update(0)
    return animation


def animate_simulation(specification_path, timeseries_path) -> Path:
    """Read exactly two files and save macpherson_animation.html beside the CSV.

    Playback is 30 fps, real time. Existing output is replaced. Returns its path.
    No generated Python modules are imported and no physical equations are solved.
    """
    data, radius, delta0 = load_inputs(specification_path, timeseries_path)
    animation = _build_animation(data, radius, delta0)
    from matplotlib import rc_context

    try:
        # Embed every frame, even when the animation exceeds Matplotlib's 20 MB default.
        with rc_context({"animation.embed_limit": float("inf")}):
            html = animation.to_jshtml(default_mode="loop")
        output = Path(timeseries_path).resolve().with_name("macpherson_animation.html")
        output.write_text(
            '<!doctype html><html lang="ru"><meta charset="utf-8">'
            '<title>MacPherson — 2D анимация</title>'
            '<body style="font-family:sans-serif;text-align:center">'
            '<p>Нажмите ▶ для воспроизведения. Ползунок выбирает момент времени.</p>'
            + html + '</body></html>', encoding="utf-8")
        return output
    finally:
        animation._fig.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path, help="Путь к specification.json")
    parser.add_argument("timeseries", type=Path, help="Путь к timeseries.csv")
    args = parser.parse_args()
    try:
        print(f"Animation saved: {animate_simulation(args.specification, args.timeseries)}")
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
