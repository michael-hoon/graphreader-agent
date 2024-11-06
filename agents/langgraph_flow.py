from typing import Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent_data_models import (
    OverallState,
    InputState,
    OutputState,
)

from agent_nodes import (
    rational_plan_node,
    initial_node_selection,
    atomic_fact_check,
    chunk_check,
    answer_reasoning,
    neighbor_select
)

load_dotenv()

####################################################
# LangGraph State Conditions (for conditional edges)
####################################################

def atomic_fact_condition(state: OverallState,) -> Literal["neighbor_select", "chunk_check"]:
    if state.get("chosen_action") == "stop_and_read_neighbor":
        return "neighbor_select"
    elif state.get("chosen_action") == "read_chunk":
        return "chunk_check"

def chunk_condition(state: OverallState,) -> Literal["answer_reasoning", "chunk_check", "neighbor_select"]:
    if state.get("chosen_action") == "termination":
        return "answer_reasoning"
    elif state.get("chosen_action") in ["read_subsequent_chunk", "read_previous_chunk", "search_more"]:
        return "chunk_check"
    elif state.get("chosen_action") == "search_neighbor":
        return "neighbor_select"

def neighbor_condition(state: OverallState,) -> Literal["answer_reasoning", "atomic_fact_check"]:
    if state.get("chosen_action") == "termination":
        return "answer_reasoning"
    elif state.get("chosen_action") == "read_neighbor_node":
        return "atomic_fact_check"
    
####################################################
# LangGraph Control Flow
####################################################

builder = StateGraph(OverallState, input=InputState, output=OutputState)
builder.add_node(rational_plan_node)
builder.add_node(initial_node_selection)
builder.add_node(atomic_fact_check)
builder.add_node(chunk_check)
builder.add_node(answer_reasoning)
builder.add_node(neighbor_select)

builder.add_edge(START, "rational_plan_node")
builder.add_edge("rational_plan_node", "initial_node_selection")
builder.add_edge("initial_node_selection", "atomic_fact_check")
builder.add_conditional_edges(
    "atomic_fact_check",
    atomic_fact_condition,
)
builder.add_conditional_edges(
    "chunk_check",
    chunk_condition,
)
builder.add_conditional_edges(
    "neighbor_select",
    neighbor_condition,
)
builder.add_edge("answer_reasoning", END)

# pass checkpointer for persistence
graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "1"}} # need some way to set the thread_id across different streamlit sessions

graph.invoke({"question":"How is deep learning used in nuclear safety research?"}, config=config)

# graph.invoke({"question":"What did I just ask you?"}, config=config)

# def invoke_graph(st_messages, callables):
#     if not isinstance(callables, list):
#         raise TypeError("callables must be a list")
#     return graph.invoke({"messages":st_messages}, config={"callables":callables})