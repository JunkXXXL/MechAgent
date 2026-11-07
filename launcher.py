"""Generate one suspension model and compare its trajectories with the reference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime
from uuid import uuid4

import numpy as np

from reference_model.standalone_macpherson_tire import MacPhersonTireModel, PlateauRoad
from run_trace import RunTrace, jsonable


PROJECT_DIR = Path(__file__).resolve().parent


def evaluation_config() -> dict:
    """Fixed SI benchmark; state order is [z_s, v_s, z_u, v_u]."""
    return {
        "t": np.linspace(0.0, 3.0, 3001).tolist(),
        "initial_state": [0.0, 0.0, 0.0, 0.0],
        "sprung_mass": 365.0,
        "unsprung_mass": 45.0,
        "speed": 2.5,
        "road": {"height_m": 0.04, "start_x": 1.0, "length": 2.0},
    }


def evaluation_request(prompt: str, config: dict) -> str:
    # Only execution inputs and the output contract are given to the agent,
    # never reference trajectories or reference implementation source.
    return prompt + "\n\n" + """Обязательный интерфейс для последующего сравнения модели:
В итоговом автономном model.py реализуй функцию simulate_evaluation(config).
Она должна запускать именно построенную тобой модель и возвращать словарь
с одномерными массивами 't', 'z_s', 'z_u'. Все массивы соответствуют точно
config['t']; единицы СИ. z_s и z_u — отклонения вверх от статического равновесия.
Используй config['initial_state'] в порядке [z_s, v_s, z_u, v_u], массы
config['sprung_mass'], config['unsprung_mass'], скорость config['speed'].
Дорога: height_m при start_x <= x <= start_x + length, иначе 0;
параметры берутся из config['road'], x = speed*t.
Эти значения задают условия оценочного запуска, даже если в запросе другие
начальные условия. Остальные физические параметры и уравнения бери из запроса.
Интегрирование — классический RK4. При импорте файла симуляция не запускается.
Не импортируй модули проекта и не читай внешние эталонные данные.
Сохрани также обычный самостоятельный запуск программы под main guard.
Фиксированная конфигурация оценочного запуска (JSON):
""" + json.dumps(config, ensure_ascii=False)


# Run generated definitions in a child process; never import them into launcher.
EVALUATION_RUNNER = """
import json, runpy, sys
import numpy as np
namespace = runpy.run_path(sys.argv[1], run_name='__evaluation__')
with open(sys.argv[2], encoding='utf-8') as stream:
    config = json.load(stream)
result = namespace['simulate_evaluation'](config)
t = np.asarray(config['t'], dtype=float)
arrays = [np.asarray(result[key], dtype=float) for key in ('t', 'z_s', 'z_u')]
if any(a.shape != t.shape or not np.all(np.isfinite(a)) for a in arrays):
    raise ValueError('Evaluation outputs must be finite vectors matching config t')
if not np.allclose(arrays[0], t, rtol=0, atol=1e-12):
    raise ValueError('Evaluation time grid differs from config t')
if not np.allclose([arrays[1][0], arrays[2][0]],
                   [config['initial_state'][0], config['initial_state'][2]],
                   rtol=0, atol=1e-12):
    raise ValueError('Evaluation initial displacements differ from config')
np.savetxt(sys.argv[3], np.column_stack(arrays), delimiter=',',
           header='t,z_s,z_u', comments='')
"""


def execute_model(model_path: Path, folder: Path, trace: RunTrace) -> dict:
    environment = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    command = [sys.executable, "-c", EVALUATION_RUNNER, str(model_path),
               str(folder / "evaluation_config.json"), str(folder / "generated.csv")]
    trace.add("evaluation_start", command=command)
    try:
        process = subprocess.run(command, cwd=folder, env=environment,
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=300, check=False)
    except subprocess.TimeoutExpired as exc:
        log = f"Evaluation timed out after 300 seconds.\n{exc.stdout or ''}\n{exc.stderr or ''}"
        (folder / "execution.log").write_text(log, encoding="utf-8")
        trace.add("evaluation_error", error=log)
        raise RuntimeError(log) from exc
    log = f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
    (folder / "execution.log").write_text(log, encoding="utf-8")
    trace.add("evaluation_execution", returncode=process.returncode,
              stdout=process.stdout, stderr=process.stderr)
    if process.returncode:
        raise RuntimeError(f"Generated model evaluation failed.\n{log}")
    data = np.genfromtxt(folder / "generated.csv", delimiter=",", names=True)
    return {key: data[key] for key in ("t", "z_s", "z_u")}


def compare_trajectories(generated: dict, reference: dict) -> dict[str, float]:
    """Discrete Euclidean L2 on a shared grid, without dt or sample normalization."""
    t = np.asarray(reference["t"], dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Reference time must be finite and strictly increasing")
    generated_t = np.asarray(generated["t"], dtype=float)
    if generated_t.shape != t.shape or not np.allclose(generated_t, t, rtol=0, atol=1e-12):
        raise ValueError("Trajectories must use the same time grid")
    metrics = {}
    for key in ("z_s", "z_u"):
        actual = np.asarray(generated[key], dtype=float)
        expected = np.asarray(reference[key], dtype=float)
        if any(a.shape != t.shape or not np.all(np.isfinite(a)) for a in (actual, expected)):
            raise ValueError(f"Invalid trajectory: {key}")
        metrics[key] = float(np.linalg.norm(actual - expected))
    return metrics


def launch(prompt: str, folder: Path, *, enable_insight: bool = True) -> dict:
    """Persist the complete run, including a trace if any stage fails."""
    # Lazy import allows --help and numerical tests without initializing LLM clients.
    from main import extract_generated_code, run_experiment

    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    trace = RunTrace()
    config = evaluation_config()
    result = None
    metrics = None
    error = None
    try:
        (folder / "request.txt").write_text(prompt, encoding="utf-8")
        (folder / "evaluation_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        request = evaluation_request(prompt, config)
        (folder / "agent_request.txt").write_text(request, encoding="utf-8")
        trace.add("benchmark_config", config=config, max_iterations=3)
        print("Запуск агента (максимум 3 итерации)...", flush=True)
        result = run_experiment(request, trace, artifact_dir=folder,
                                enable_insight=enable_insight)
        if not result or result.get("error") == "yes":
            raise RuntimeError("Agent did not return a validated model")
        (folder / "agent_execution.log").write_text(str(result.get("sim_logs", "")),
                                                     encoding="utf-8")
        model_path = folder / "model.py"
        model_path.write_text(extract_generated_code(result), encoding="utf-8")
        print("Запуск полученной и эталонной моделей...", flush=True)
        generated = execute_model(model_path, folder, trace)
        model = MacPhersonTireModel(sprung_mass=config["sprung_mass"],
                                   unsprung_mass=config["unsprung_mass"], speed=config["speed"])
        reference = model.simulate(np.asarray(config["t"]), PlateauRoad(**config["road"]),
                                   initial_state=tuple(config["initial_state"]))
        np.savetxt(folder / "reference.csv",
                   np.column_stack([reference[k] for k in ("t", "z_s", "z_u")]),
                   delimiter=",", header="t,z_s,z_u", comments="")
        metrics = compare_trajectories(generated, reference)
        (folder / "metrics.json").write_text(json.dumps({
            "definition": "sqrt(sum((z_generated - z_reference)**2))",
            "samples": len(config["t"]), "l2": metrics,
        }, indent=2), encoding="utf-8")
        trace.add("trajectory_comparison", l2=metrics)
        print("L2 = sqrt(sum((z_model - z_reference)^2)); 3001 отсчёт, СИ")
        print(f"z_s: {metrics['z_s']:.10g}")
        print(f"z_u: {metrics['z_u']:.10g}")
        return metrics
    except Exception:
        error = traceback.format_exc()
        trace.add("launcher_error", error=error)
        (folder / "error.log").write_text(error, encoding="utf-8")
        raise
    finally:
        (folder / "trace.json").write_text(json.dumps(jsonable({
            "prompt": prompt, "status": "error" if error else "success",
            "events": trace.events, "final_state": result, "metrics": metrics,
            "error": error,
        }), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Результаты: {folder}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("prompt", nargs="?", help="Один запрос текстом")
    source.add_argument("--prompt-file", type=Path, help="UTF-8 текстовый файл с одним запросом")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results" / "launcher",
                        help="Корневая папка; каждый запуск создаёт уникальный подкаталог")
    parser.add_argument("--skip-insight", action="store_true")
    args = parser.parse_args()
    try:
        prompt = args.prompt_file.read_text(encoding="utf-8-sig") if args.prompt_file else args.prompt
        if not prompt.strip():
            parser.error("Запрос должен быть непустым")
        name = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid4().hex[:8]
        launch(prompt.strip(), args.output_dir.expanduser() / name,
               enable_insight=not args.skip_insight)
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
