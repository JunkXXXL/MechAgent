from langchain import messages
from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, AnyMessage, HumanMessage, AIMessage
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import yaml
from dotenv import load_dotenv

load_dotenv()

with open("agents.yaml", 'r', encoding="utf-8") as stream:
    prompts_agents = yaml.safe_load(stream)
with open("tasks.yaml", 'r', encoding="utf-8") as stream:
    prompts_tasks = yaml.safe_load(stream)
class code(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


agent_input_classifier = ChatOpenAI(model="gpt-5-nano", temperature=0)
input_classifier_prompt = SystemMessage(prompts_agents["input_classifier"]["goal"] +
                                        prompts_agents["input_classifier"]["backstory"])

agent_code_builder = ChatOpenAI(model="gpt-5-nano", temperature=0).with_structured_output(code)
agent_code_prompt = SystemMessage(prompts_agents["code_builder"]["goal"] +
                                  prompts_agents["code_builder"]["backstory"])

agent_error_diagnoser = ChatOpenAI(model="gpt-5-nano", temperature=0)
agent_error_prompt = SystemMessage(prompts_agents["Error_diagnosis"]["goal"] +
                                   prompts_agents["Error_diagnosis"]["backstory"])

agent_insighter = ChatOpenAI(model="gpt-5-nano", temperature=0.3)
insighter_prompt = SystemMessage(prompts_agents["MechanicalInsight"]["goal"] +
                                 prompts_agents["MechanicalInsight"]["backstory"])


def decide_to_finish(state: dict)-> Literal["regenerate", "make insight"]:
    print("decide_to_finish")
    if state["error"] == "yes":
        return "regenerate"
    else:
        return "make insight"

CODE_EXECUTION_TIMEOUT_SECONDS = 30


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
    }


def code_execution_import_check(state: dict):
    code_solution = state["generation"]
    imports = code_solution.imports
    source = f"{imports}\n{code_solution.code}"

    try:
        import_result = _run_python_code(imports, "Import check")
        if import_result.returncode != 0:
            error = (
                f"Import check failed with exit code {import_result.returncode}.\n"
                f"stdout:\n{import_result.stdout}\nstderr:\n{import_result.stderr}"
            )
            print("---CODE IMPORT CHECK: FAILED---")
            return _build_failure_result(code_solution, source, error)

        execution_result = _run_python_code(source, "Execution check")
        if execution_result.returncode != 0:
            error = (
                f"Execution failed with exit code {execution_result.returncode}.\n"
                f"stdout:\n{execution_result.stdout}\nstderr:\n{execution_result.stderr}"
            )
            print("---CODE EXECUTION CHECK: FAILED---")
            return _build_failure_result(code_solution, source, error)
    except (OSError, RuntimeError) as exc:
        print("---CODE EXECUTION CHECK: FAILED---")
        return _build_failure_result(code_solution, source, str(exc))

    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "error": "no",
        "sim_logs": execution_result.stdout + execution_result.stderr,
    }

def code_generation(state: dict):
    print("CODE GENERATION")
    error = state["error"]

    if error != "yes":
        code_request = HumanMessage(
            prompts_tasks["code_building"]["description"].format(
                state["input_classifier_response"]
            )
        )
        conversation = [agent_code_prompt, code_request]
        generated_code = agent_code_builder.invoke(conversation)
        generated_message = AIMessage(content=generated_code.model_dump_json())
        return {
            "agent_code_history": conversation + [generated_message],
            "generation": generated_code,
        }
    else:
        correction_request = HumanMessage(state["messages"][-1].content)
        conversation = state["agent_code_history"] + [correction_request]
        generated_code = agent_code_builder.invoke(conversation)
        generated_message = AIMessage(content=generated_code.model_dump_json())
        # Reducer уже хранит предыдущую историю, поэтому возвращаем только новые сообщения.
        return {
            "agent_code_history": [correction_request, generated_message],
            "generation": generated_code,
        }

def input_classifier(state: dict):
    print("INPUT CLASSIFIER")
    messages = [input_classifier_prompt] + state["messages"]
    message = agent_input_classifier.invoke(messages)
    return {"messages": [message],
            "input_classifier_response": message.content
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
    return {"messages": [insight]}
