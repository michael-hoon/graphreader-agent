import os
from typing import Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection

from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

from .state import (
    OverallState,
)

from .agent_nodes import (
    semantic_router,
    clarification,
    general_query,
    rational_plan_node,
    initial_node_selection,
    atomic_fact_check,
    chunk_check,
    answer_reasoning,
    neighbor_select,
    summarize_conversation,
)

from streamlit_app.postgres_db import get_connection

load_dotenv()

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

####################################################
# LangGraph State Conditions (for conditional edges)
####################################################

def atomic_fact_condition(
        state: OverallState,
    ) -> Literal["neighbor_select", "chunk_check"]:
    if state.get("chosen_action") == "stop_and_read_neighbor":
        return "neighbor_select"
    elif state.get("chosen_action") == "read_chunk":
        return "chunk_check"

def chunk_condition(
        state: OverallState,
    ) -> Literal["answer_reasoning", "chunk_check", "neighbor_select"]:
    if state.get("chosen_action") == "termination":
        return "answer_reasoning"
    elif state.get("chosen_action") in ["read_subsequent_chunk", "read_previous_chunk", "search_more"]:
        return "chunk_check"
    elif state.get("chosen_action") == "search_neighbor":
        return "neighbor_select"

def neighbor_condition(
        state: OverallState,
    ) -> Literal["answer_reasoning", "atomic_fact_check"]:
    if state.get("chosen_action") == "termination":
        return "answer_reasoning"
    elif state.get("chosen_action") == "read_neighbor_node":
        return "atomic_fact_check"
    
def route_query_condition(
    state: OverallState,
    ) -> Literal["rational_plan_node", "clarification", "general_query"]:
    """Determine the next step based on the query classification.

    Args:
        state (OverallState): The current state of the agent, including the router's classification.

    Returns:
        Literal["rational_plan_node", "clarification", "general_query"]: The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    if state.get("route") == "research":
        return "rational_plan_node"
    elif state.get("route") == "clarification":
        return "clarification"
    elif state.get("route") == "general_query":
        return "general_query"
    else:
        raise ValueError(f"Unknown router type {state.get('route')}")
    
def summary_condition(
        state: OverallState,
    ) -> Literal["summarize_conversation", END]:
    messages = state["messages"]
    # summarise only if conversation exceeds 3 turns (6 messages)
    if len(messages) > 6:
        return "summarize_conversation"
    return END
    
####################################################
# LangGraph Control Flow
####################################################

with get_connection() as conn:
    checkpointer = PostgresSaver(conn)

    # NOTE: need to call .setup() the first time you're using the checkpointer
    # checkpointer.setup()

    agent = StateGraph(OverallState)

    agent.add_node(semantic_router)
    agent.add_node(clarification)
    agent.add_node(general_query)
    agent.add_node(rational_plan_node)
    agent.add_node(initial_node_selection)
    agent.add_node(atomic_fact_check)
    agent.add_node(chunk_check)
    agent.add_node(answer_reasoning)
    agent.add_node(neighbor_select)
    agent.add_node(summarize_conversation)

    agent.add_edge(START, "semantic_router")

    agent.add_conditional_edges(
        "semantic_router",
        route_query_condition,
    )
    agent.add_conditional_edges(
        "atomic_fact_check",
        atomic_fact_condition,
    )
    agent.add_conditional_edges(
        "chunk_check",
        chunk_condition,
    )
    agent.add_conditional_edges(
        "neighbor_select",
        neighbor_condition,
    )
    agent.add_conditional_edges(
        "answer_reasoning",
        summary_condition,
    )
    agent.add_conditional_edges(
        "clarification",
        summary_condition,
    )
    agent.add_conditional_edges(
        "general_query",
        summary_condition,
    )
    agent.add_edge("rational_plan_node", "initial_node_selection")
    agent.add_edge("initial_node_selection", "atomic_fact_check")
    agent.add_edge("summarize_conversation", END)

    graph = agent.compile(
        checkpointer=checkpointer,
    )

def chatbot_response(
        input_text: str,
        config: dict = None,
    ) -> str:
    """
    Generate a response from the chatbot based on the user's input text.
    
    Args:
        input_text (str): The user's input text to generate a response for.
        config (dict, optional): Configuration settings for the chatbot. Defaults to None.
        
    Returns:
        str: The chatbot's response to the user's input text.
    """

    # ensure that callables is a list as you can have multiple callbacks
    if not isinstance(config.get("callbacks"), list):
        raise TypeError("callables must be a list")
    output_text = graph.invoke({'messages': [HumanMessage(content=input_text)]}, config=config)
    return output_text['messages'][-1].content