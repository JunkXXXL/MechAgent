from langchain import messages
from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, AnyMessage, HumanMessage, AIMessage
from contextlib import redirect_stdout, redirect_stderr
import io
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

def code_excecution_import_check(state: dict):
    code_solution = state["generation"]

    code = state["generation"].imports + "\n" + state["generation"].code
    print("CODEEEE \n", code)

    #check import
    stdout = io.StringIO()
    stderr = io.StringIO()
    # try:
    #     exec(imports)
    # except Exception as e:
    #     print("---CODE IMPORT CHECK: FAILED---")
    #     error_message = [HumanMessage(f"Your solution failed the import test: {e}")]
    #     messages += error_message
    #     return {
    #         "generation": code_solution,
    #         "messages": messages,
    #         "error": "yes",
    #     }
    #check code
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            #exec(imports + "\n" + code)
            exec(code)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")

        if isinstance(e, ModuleNotFoundError):
            missing_package = e.name or "неизвестный пакет"
            print(f"Отсутствует необходимый пакет: {missing_package}")

        diagnosis_request = HumanMessage(
            prompts_tasks["diagnosis_error"]["description"].format(e, code)
        )
        diagnosis_messages = [agent_error_prompt, diagnosis_request]
        error_message = agent_error_diagnoser.invoke(diagnosis_messages)
        return {
            "generation": code_solution,
            "messages": [error_message],
            "agent_error_history": [diagnosis_request, error_message],
            "error": "yes",
        }

    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "error": "no",
        "sim_logs": str(stdout.getvalue())
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
    print("GENERATION AAA " + str(generation))
    print("LOGS AAA " + sim_logs)

    insight_request = HumanMessage(
        prompts_tasks["create_insight"]["description"].format(
            state["input_classifier_response"],
            str(generation),
            sim_logs,
        )
    )
    insight = agent_insighter.invoke([insighter_prompt, insight_request])
    return {"messages": [insight]}
