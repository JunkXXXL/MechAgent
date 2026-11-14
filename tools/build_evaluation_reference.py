"""Build the shared reference dataset used to evaluate all prompt groups."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SCRIPT = (
    ROOT / "results" / "2026-09-05" / "Reference_prompt" / "2" / "Reference_prompt.py"
)
OUTPUT = ROOT / "results" / "2026-09-05" / "evaluation_reference.json"


def load_reference_module():
    spec = importlib.util.spec_from_file_location("reference_prompt", REFERENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reference script: {REFERENCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    reference = load_reference_module()
    parameters = {
        "l1": 5.0,
        "l2": 1.0,
        "m1": 1.0,
        "m2": 3.0,
        "theta1_deg": 10.0,
        "theta2_deg": 15.0,
        "omega1": 0.0,
        "omega2": 0.0,
        "g": 9.81,
    }
    time_grid = {"T": 20.0, "dt": 0.005, "n_steps": 4000, "n_points": 4001}
    result = reference.run_simulation(
        T=time_grid["T"],
        dt=time_grid["dt"],
        m1=parameters["m1"],
        m2=parameters["m2"],
        l1=parameters["l1"],
        l2=parameters["l2"],
        g=parameters["g"],
        theta1_0=reference.np.deg2rad(parameters["theta1_deg"]),
        theta2_0=reference.np.deg2rad(parameters["theta2_deg"]),
        omega1_0=parameters["omega1"],
        omega2_0=parameters["omega2"],
    )
    payload = {
        "description": (
            "Shared benchmark for all experiment groups. Dynamics/integrator and time "
            "grid come from Reference_prompt/2/Reference_prompt.py; explicitly supplied "
            "task parameters override that function's differing m1 and l2 defaults."
        ),
        "parameter_interpretation": {
            "theta": "angular deviation from the downward vertical, radians internally",
            "omega": "initial angular velocity in rad/s",
        },
        "parameters": parameters,
        "time_grid": time_grid,
        "rmse_definition": "sqrt(mean((x-x_ref)^2 + (y-y_ref)^2))",
        "trajectory": {
            name: result[name].tolist() for name in ("t", "x1", "y1", "x2", "y2")
        },
        "source": str(REFERENCE_SCRIPT.relative_to(ROOT)).replace("\\\\", "/"),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
