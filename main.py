"""Запуск серии экспериментов с агентом из YAML-файла."""

from __future__ import annotations

import argparse
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from langchain.messages import HumanMessage
from langchain_core.messages import BaseMessage

from pipeline import agent


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запускает по одному эксперименту для каждого промпта из YAML."
    )
    parser.add_argument(
        "prompts_file", type=Path, help="Путь к YAML-файлу с промптами."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Корневая папка результатов (по умолчанию: ./results).",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> list[tuple[str, str]]:
    """Загрузить именованные промпты из YAML mapping или списка объектов."""
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Файл с промптами не найден: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if isinstance(data, dict):
        prompts = list(data.items())
    elif isinstance(data, list):
        prompts = []
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict) or "prompt" not in item:
                raise ValueError(
                    "Элемент списка YAML должен содержать поля 'name' и 'prompt'."
                )
            prompts.append((item.get("name", f"experiment_{index}"), item["prompt"]))
    else:
        raise ValueError("Корень YAML должен быть словарём или списком промптов.")

    normalized: list[tuple[str, str]] = []
    for name, prompt in prompts:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Промпт '{name}' должен быть непустой строкой.")
        normalized.append((str(name), prompt.strip()))

    if not normalized:
        raise ValueError("В YAML-файле нет промптов.")
    return normalized


def safe_filename(name: str) -> str:
    """Преобразовать имя промпта в безопасное имя Markdown-файла."""
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" ._")
    return filename or "experiment"


def unique_report_path(folder: Path, prompt_name: str) -> Path:
    base_name = safe_filename(prompt_name)
    candidate = folder / f"{base_name}.md"
    suffix = 2
    while candidate.exists():
        candidate = folder / f"{base_name}_{suffix}.md"
        suffix += 1
    return candidate


def render_value(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json(indent=2)
    return str(value)


def message_content(message: Any) -> str:
    if isinstance(message, BaseMessage):
        return render_value(message.content)
    return render_value(message)


def build_report(
    prompt_name: str,
    prompt: str,
    started_at: datetime,
    finished_at: datetime,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> str:
    lines = [
        f"# Эксперимент: {prompt_name}",
        "",
        f"- Начало: `{started_at.isoformat(timespec='seconds')}`",
        f"- Завершение: `{finished_at.isoformat(timespec='seconds')}`",
        f"- Длительность: `{(finished_at - started_at).total_seconds():.2f} с`",
        f"- Статус: `{'ошибка' if error else 'успешно'}`",
        "",
        "## Исходный промпт",
        "",
        prompt,
        "",
    ]

    if error:
        lines.extend(["## Ошибка", "", "```text", error.rstrip(), "```", ""])
        return "\n".join(lines)

    assert result is not None
    generation = result.get("generation")
    if generation is not None:
        lines.extend(
            [
                "## Сгенерированное решение",
                "",
                "```json",
                render_value(generation),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Логи исполнения",
            "",
            "```text",
            render_value(result.get("sim_logs")),
            "```",
            "",
        ]
    )

    messages = result.get("messages", [])
    if messages:
        lines.extend(["## Результат работы агента", ""])
        for message in messages:
            role = getattr(message, "type", type(message).__name__)
            lines.extend([f"### {role}", "", message_content(message), ""])

    return "\n".join(lines)


def run_experiment(prompt: str) -> dict[str, Any]:
    return agent.invoke(
        {
            "messages": [HumanMessage(prompt)],
            "error": "no",
            "generation": "",
            "agent_code_history": [],
            "agent_error_history": [],
        }
    )


def main() -> int:
    args = parse_args()
    try:
        prompts = load_prompts(args.prompts_file)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Не удалось загрузить промпты: {exc}")
        return 2

    experiment_date = datetime.now().astimezone().strftime("%Y-%m-%d")
    result_folder = args.output_dir.expanduser().resolve() / experiment_date
    result_folder.mkdir(parents=True, exist_ok=True)

    print(f"Найдено промптов: {len(prompts)}")
    print(f"Папка результатов: {result_folder}")

    failed = 0
    for index, (name, prompt) in enumerate(prompts, start=1):
        print(f"\n[{index}/{len(prompts)}] Запуск эксперимента: {name}")
        started_at = datetime.now().astimezone()
        result: dict[str, Any] | None = None
        error: str | None = None

        try:
            result = run_experiment(prompt)
            print("Эксперимент завершён успешно.")
            print("Логи исполнения:")
            print(render_value(result.get("sim_logs")) or "<логи отсутствуют>")
            messages = result.get("messages", [])
            print("Результат работы агента:")
            print(
                message_content(messages[-1])
                if messages
                else "<результат отсутствует>"
            )
        except Exception:
            failed += 1
            error = traceback.format_exc()
            print("Эксперимент завершился с ошибкой.")
            print(error)

        finished_at = datetime.now().astimezone()
        report_path = unique_report_path(result_folder, name)
        report = build_report(name, prompt, started_at, finished_at, result, error)
        report_path.write_text(report, encoding="utf-8")
        print(f"Отчёт сохранён: {report_path}")

    print(
        f"\nЭксперименты завершены: {len(prompts) - failed} успешно, "
        f"{failed} с ошибкой."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
