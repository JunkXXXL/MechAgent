from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
import os
from pathlib import Path
from uuid import uuid4
import subprocess
import sys
import tempfile
import yaml
from dotenv import load_dotenv
from langgraph.config import get_stream_writer

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent

with (PROJECT_DIR / "agents.yaml").open(encoding="utf-8") as stream:
    prompts_agents = yaml.safe_load(stream)
with (PROJECT_DIR / "tasks.yaml").open(encoding="utf-8") as stream:
    prompts_tasks = yaml.safe_load(stream)


class code(BaseModel):
    """Python source returned by an implementation agent."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


class PhysicalParameter(BaseModel):
    name: str = Field(description="Stable Python identifier")
    symbol: str
    meaning: str
    value: str = Field(description="Value or explicit time-dependent expression")
    unit: str = Field(description="SI unit; use 1 for dimensionless quantities")
    original_value_and_unit: str = Field(description="Original user value, or not provided")
    source: Literal["user", "inferred"]
    category: Literal["parameter", "constant", "initial_condition", "simulation_control"]
    significant: bool = Field(description="Whether this must be exposed as a runtime input")


class SimulationSpecification(BaseModel):
    system_description: str
    parameters: list[PhysicalParameter]
    state_variables_and_units: str
    assumptions: list[str]
    numerical_method: str
    output_requirements: str


agent_input_classifier = ChatOpenAI(model="gpt-5.2-2025-12-11").with_structured_output(SimulationSpecification)
input_classifier_prompt = SystemMessage(prompts_agents["input_classifier"]["goal"] +
                                        prompts_agents["input_classifier"]["backstory"])

agent_code_builder = ChatOpenAI(model="gpt-5.2-2025-12-11").with_structured_output(code)
agent_code_prompt = SystemMessage(prompts_agents["code_builder"]["goal"] +
                                  prompts_agents["code_builder"]["backstory"])

agent_subsystem_designer = ChatOpenAI(model="gpt-5.2-2025-12-11")
subsystem_prompt = SystemMessage(prompts_agents["subsystem_designer"]["goal"] +
                                prompts_agents["subsystem_designer"]["backstory"])
agent_model_assembler = ChatOpenAI(model="gpt-5.2-2025-12-11").with_structured_output(code)
assembly_prompt = SystemMessage(prompts_agents["model_assembler"]["goal"] +
                               prompts_agents["model_assembler"]["backstory"])

agent_error_diagnoser = ChatOpenAI(model="gpt-5.2-2025-12-11")
agent_error_prompt = SystemMessage(prompts_agents["Error_diagnosis"]["goal"] +
                                   prompts_agents["Error_diagnosis"]["backstory"])

agent_insighter = ChatOpenAI(model="gpt-5-nano")
insighter_prompt = SystemMessage(prompts_agents["MechanicalInsight"]["goal"] +
                                 prompts_agents["MechanicalInsight"]["backstory"])


def decide_to_finish(state: dict)-> Literal["regenerate", "make insight", "finish"]:
    print("decide_to_finish")
    if state["error"] == "yes":
        if state.get("iterations", 0) >= state.get("max_iterations", 3):
            raise RuntimeError("Model validation failed after the maximum number of attempts.\n" +
                               state.get("sim_logs", ""))
        return "regenerate"
    else:
        return "make insight" if state.get("enable_insight", True) else "finish"

CODE_EXECUTION_TIMEOUT_SECONDS = 300


def _run_python_code(source: str, label: str) -> subprocess.CompletedProcess[str]:
    """Run generated code in a separate interpreter and return its process result."""
    try:
        with tempfile.TemporaryDirectory(prefix="mechagent-check-") as directory:
            script_path = Path(directory) / "generated_simulation.py"
            script_path.write_text(source, encoding="utf-8")

            environment = os.environ.copy()
            # A generated matplotlib program must not wait for a GUI during validation.
            environment.setdefault("MPLBACKEND", "Agg")
            return subprocess.run(
                [sys.executable, str(script_path)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=CODE_EXECUTION_TIMEOUT_SECONDS,
                env=environment,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        raise RuntimeError(
            f"{label} exceeded {CODE_EXECUTION_TIMEOUT_SECONDS} seconds.\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        ) from exc


def _build_failure_result(code_solution: code, source: str, error: str):
    diagnosis_request = HumanMessage(
        prompts_tasks["diagnosis_error"]["description"].format(error, source)
    )
    error_message = agent_error_diagnoser.invoke([agent_error_prompt, diagnosis_request])
    return {
        "generation": code_solution,
        "messages": [error_message],
        "agent_error_history": [diagnosis_request, error_message],
        "error": "yes",
        "sim_logs": error,
    }


def code_execution_import_check(state: dict):
    writer = get_stream_writer()
    code_solution = state["generation"]
    imports = code_solution.imports
    source = f"{imports}\n{code_solution.code}"

    try:
        import_result = _run_python_code(imports, "Import check")
        writer({"stage": "import_check", "returncode": import_result.returncode,
                "stdout": import_result.stdout, "stderr": import_result.stderr})
        if import_result.returncode != 0:
            error = (
                f"Import check failed with exit code {import_result.returncode}.\n"
                f"stdout:\n{import_result.stdout}\nstderr:\n{import_result.stderr}"
            )
            print("---CODE IMPORT CHECK: FAILED---")
            return _build_failure_result(code_solution, source, error)

        execution_result = _run_python_code(source, "Execution check")
        writer({"stage": "execution_check", "returncode": execution_result.returncode,
                "stdout": execution_result.stdout, "stderr": execution_result.stderr})
        if execution_result.returncode != 0:
            error = (
                f"Execution failed with exit code {execution_result.returncode}.\n"
                f"stdout:\n{execution_result.stdout}\nstderr:\n{execution_result.stderr}"
            )
            print("---CODE EXECUTION CHECK: FAILED---")
            return _build_failure_result(code_solution, source, error)
    except (OSError, RuntimeError) as exc:
        writer({"stage": "code_check_error", "error": str(exc)})
        print("---CODE EXECUTION CHECK: FAILED---")
        return _build_failure_result(code_solution, source, str(exc))

    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "error": "no",
        "sim_logs": execution_result.stdout + execution_result.stderr,
    }

def _source(solution: code) -> str:
    return f"{solution.imports}\n\n{solution.code}\n"


def _feedback(state: dict) -> str:
    if state.get("error") != "yes":
        return ""
    return ("Previous complete program:\n" + _source(state["generation"]) +
            "\nValidation diagnosis:\n" + str(state["messages"][-1].content))


def _implementation_context(state: dict) -> dict:
    return {
        "request": state["user_request"],
        "specification": state["input_classifier_response"],
        "architecture": state["subsystem_description"],
        "feedback": _feedback(state),
    }


def code_generation(state: dict):
    """Third agent: implement the physical subsystem function contracts."""
    request = HumanMessage(prompts_tasks["code_building"]["description"].format(
        **_implementation_context(state)
    ))
    conversation = [agent_code_prompt, request]
    generated = agent_code_builder.invoke(conversation)
    (Path(state["artifact_dir"]) / "subsystems.py").write_text(
        _source(generated), encoding="utf-8"
    )
    return {
        "subsystem_generation": generated,
        "iterations": state.get("iterations", 0) + 1,
        "agent_code_history": conversation + [AIMessage(content=generated.model_dump_json())],
    }


def model_assembly(state: dict):
    """Fourth agent: connect functions; retain their source verbatim in the result."""
    subsystems = state["subsystem_generation"]
    request = HumanMessage(prompts_tasks["model_assembly"]["description"].format(
        **_implementation_context(state), subsystems=_source(subsystems)
    ))
    conversation = [assembly_prompt, request]
    assembly = agent_model_assembler.invoke(conversation)
    combined = code(
        prefix=assembly.prefix,
        imports=subsystems.imports + "\n" + assembly.imports,
        code=subsystems.code + "\n\n" + assembly.code,
    )
    (Path(state["artifact_dir"]) / "model.py").write_text(_source(combined), encoding="utf-8")
    return {
        "generation": combined,
        "agent_assembly_history": conversation + [AIMessage(content=assembly.model_dump_json())],
    }


def input_classifier(state: dict):
    """First agent: preserve the user request and specify quantities in SI units."""
    request = state.get("user_request")
    if not request:
        request = "\n\n".join(
            str(message.content) for message in state.get("messages", [])
            if isinstance(message, HumanMessage)
        )
    if not request or not request.strip():
        raise ValueError("A nonempty user request is required")
    specification = agent_input_classifier.invoke([
        input_classifier_prompt, HumanMessage(request)
    ])
    description = specification.model_dump_json(indent=2)
    artifact_dir = Path(state.get("artifact_dir") or
                        PROJECT_DIR / "output" / uuid4().hex).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "specification.json").write_text(description, encoding="utf-8")
    return {
        "user_request": request,
        "specification": specification,
        "input_classifier_response": description,
        "artifact_dir": str(artifact_dir),
        "iterations": 0,
        "error": "no",
        "messages": [AIMessage(content=description)],
    }


def subsystem_design(state: dict):
    """Second agent: persist the architecture before any code is generated."""
    request = HumanMessage(prompts_tasks["subsystem_design"]["description"].format(
        request=state["user_request"], specification=state["input_classifier_response"]
    ))
    message = agent_subsystem_designer.invoke([subsystem_prompt, request])
    if not isinstance(message.content, str) or not message.content.strip():
        raise ValueError("Subsystem designer must return a nonempty Markdown document")
    markdown = message.content.strip()
    if markdown.startswith("```markdown\n") or markdown.startswith("```md\n"):
        markdown = markdown.split("\n", 1)[1].removesuffix("```").rstrip()
    path = Path(state["artifact_dir"]) / "physical_subsystems.md"
    path.write_text(markdown + "\n", encoding="utf-8")
    return {
        "subsystem_description": markdown,
        "subsystem_description_path": str(path),
        "messages": [AIMessage(content=markdown)],
    }


def create_insight(state: dict):
    print("CREATE Insight")
    generation = state["generation"]
    sim_logs = state["sim_logs"]

    insight_request = HumanMessage(
        prompts_tasks["create_insight"]["description"].format(
            state["input_classifier_response"],
            str(generation),
            sim_logs,
        )
    )
    insight = agent_insighter.invoke([insighter_prompt, insight_request])
    (Path(state["artifact_dir"]) / "simulation_report.md").write_text(
        str(insight.content), encoding="utf-8"
    )
    return {"messages": [insight]}
