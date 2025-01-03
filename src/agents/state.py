from pydantic import BaseModel, Field
from typing import List, Annotated, TypedDict, Literal
from operator import add

from langgraph.graph import MessagesState

# class InputState(MessagesState):
#     question: str
#     summary: str # the summary here acts as the conversation summary thus far

# class OutputState(MessagesState):
#     answer: str
#     analysis: str
#     previous_actions: List[str]

#NOTE: tracking with only one state handler for now, can't figure out how to manage message updates with multiple state classes, but secondary for now

class OverallState(MessagesState):
    question: str
    route: Literal["clarification", "research", "general_query"]
    rational_plan: str
    notebook: str
    previous_actions: Annotated[List[str], add]
    check_atomic_facts_queue: List[str]
    check_chunks_queue: List[str]
    neighbor_check_queue: List[str]
    chosen_action: str
    answer: str
    analysis: str
    summary: str

class Router(BaseModel):
    route: Literal["clarification", "research", "general_query"] = Field(
        description="""Chosen route by the model"""
    )

class Node(BaseModel):
    key_element: str = Field(
        description="""Key element or name of a relevant node"""
    )

    score: int = Field(
        description="""Relevance to the potential answer by assigning a score between 0 and 100. A score of 100 implies a high likelihood of relevance to the answer, whereas a score of 0 suggests minimal relevance."""
    )

class InitialNodes(BaseModel):
    initial_nodes: List[Node] = Field(
        description="List of relevant nodes to the question and plan"
    )

class AtomicFactOutput(BaseModel):
    updated_notebook: str = Field(
        description="""First, combine your current notebook with new insights and findings about the question from current atomic facts, creating a more complete version of the notebook that contains more valid information."""
    )

    rational_next_action: str = Field(
        description="""Based on the given question, the rational plan, previous actions, and notebook content, analyze how to choose the next action."""
    )

    chosen_action: str = Field(
        description="""
        1. read_chunk(List[ID]): Choose this action if you believe that a text chunk linked to an atomic fact may hold the necessary information to answer the question. This will allow you to access more complete and detailed information.
        2. stop_and_read_neighbor(): Choose this action if you ascertain that all text chunks lack valuable information."""
    )

class ChunkOutput(BaseModel):
    updated_notebook: str = Field(
        description="""First, combine your previous notes with new insights and findings about the question from current text chunks, creating a more complete version of the notebook that contains more valid information."""
    )

    rational_next_move: str = Field(
        description="""Based on the given question, rational plan, previous actions, and notebook content, analyze how to choose the next action."""
    )

    chosen_action: str = Field(
        description="""
        1. search_more(): Choose this action if you think that the essential information necessary to answer the question is still lacking.
        2. read_previous_chunk(): Choose this action if you feel that the previous text chunk contains valuable information for answering the question.
        3. read_subsequent_chunk(): Choose this action if you feel that the subsequent text chunk contains valuable information for answering the question.
        4. termination(): Choose this action if you believe that the information you have currently obtained is enough to answer the question. This will allow you to summarize the gathered information and provide a final answer.
        """
    )
    
class NeighborOutput(BaseModel):
    rational_next_move: str = Field(
        description="""Based on the given question, rational plan, previous actions, and notebook content, analyze how to choose the next action."""
    )

    chosen_action: str = Field(
        description="""You have the following Action Options:
        1. read_neighbor_node(key element of node): Choose this action if you believe that any of the neighboring nodes may contain information relevant to the question. Note that you should focus on one neighbor node at a time.
        2. termination(): Choose this action if you believe that none of the neighboring nodes possess information that could answer the question."""
    )
    
class AnswerReasonOutput(BaseModel):
    analyze: str = Field(
        description="""You should first analyze each notebook content before providing a final answer. During the analysis, consider complementary information from other notes and employ a majority voting strategy to resolve any inconsistencies."""
    )

    final_answer: str = Field(
        description="""When generating the final answer, ensure that you take into account all available information."""
    )