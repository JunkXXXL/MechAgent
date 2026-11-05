"""Sequential agents for specifying, decomposing and implementing a physical model."""

import operator

from langchain.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from code_nodes import (
    code, input_classifier, subsystem_design, code_generation, model_assembly,
    code_execution_import_check, create_insight, decide_to_finish,
    SimulationSpecification,
)


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_request: str
    specification: SimulationSpecification
    input_classifier_response: str
    subsystem_description: str
    subsystem_description_path: str
    artifact_dir: str
    subsystem_generation: code
    generation: code
    agent_code_history: Annotated[list[AnyMessage], operator.add]
    agent_assembly_history: Annotated[list[AnyMessage], operator.add]
    agent_error_history: Annotated[list[AnyMessage], operator.add]
    error: str
    sim_logs: str
    iterations: int
    max_iterations: int
    enable_insight: bool


def build_pipeline(*, enable_insight: bool = True):
    """Build a graph; insight can also be disabled per invocation in state."""
    builder = StateGraph(GraphState)
    builder.add_node("input_classifier", input_classifier)
    builder.add_node("subsystem_design", subsystem_design)
    builder.add_node("code_generation", code_generation)
    builder.add_node("model_assembly", model_assembly)
    builder.add_node("code_execution_import_check", code_execution_import_check)
    if enable_insight:
        builder.add_node("create_insight", create_insight)
        builder.add_edge("create_insight", END)

    builder.add_edge(START, "input_classifier")
    builder.add_edge("input_classifier", "subsystem_design")
    builder.add_edge("subsystem_design", "code_generation")
    builder.add_edge("code_generation", "model_assembly")
    builder.add_edge("model_assembly", "code_execution_import_check")
    builder.add_conditional_edges(
        "code_execution_import_check", decide_to_finish,
        {"regenerate": "code_generation",
         "make insight": "create_insight" if enable_insight else END,
         "finish": END},
    )
    return builder.compile()


agent = build_pipeline()


if __name__ == "__main__":

    agent = build_pipeline()
    png_data = agent.get_graph().draw_mermaid_png()
    with open("graph.png", 'wb') as fl:
        fl.write(png_data)

    print("end")

