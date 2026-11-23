from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated, List
from langchain.messages import SystemMessage, AnyMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import operator
from code_nodes import *
from dotenv import load_dotenv

load_dotenv()


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """
    error: str = Field(description="Binary flag for control flow to indicate whether test error was tripped")
    messages: Annotated[List[AnyMessage], operator.add] = Field(description="With user question, error messages, reasoning")
    generation: code = Field(description="Code solution")
    input_classifier_response: str = Field(description="Extention of human request")

    agent_code_history: Annotated[List[AnyMessage], operator.add] = Field(description="agent_code_builder history")
    agent_error_history: Annotated[List[AnyMessage], operator.add] = Field(description="agent_error history")
    sim_logs: str = Field(description="code logs")


agent_builder = StateGraph(GraphState)

agent_builder.add_node("input_classifier", input_classifier)
agent_builder.add_node("code_generation", code_generation)
agent_builder.add_node("code_execution_import_check", code_execution_import_check)
agent_builder.add_node("create_insight", create_insight)
#agent_builder.add_node("decide_to_finish", decide_to_finish)



agent_builder.add_edge(START, "input_classifier")
agent_builder.add_edge("input_classifier", "code_generation")
agent_builder.add_edge("code_generation", "code_execution_import_check")
agent_builder.add_conditional_edges("code_execution_import_check", decide_to_finish,
                       {"make insight":"create_insight",
                        "regenerate":"code_generation"})
agent_builder.add_edge("create_insight", END)

agent = agent_builder.compile()
