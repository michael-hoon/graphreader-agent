from .state import (
    OverallState,
    Router,
    InitialNodes,
    AtomicFactOutput,
    ChunkOutput,
    NeighborOutput,
    AnswerReasonOutput,
)
from .kg_queries import (
    get_atomic_facts,
    get_neighbors_by_key_element,
    get_subsequent_chunk_id,
    get_previous_chunk_id,
    get_chunk,
)

from .prompts import (
    RATIONAL_PLAN_SYSTEM,
    INITIAL_NODE_SYSTEM,
    ATOMIC_FACT_CHECK_SYSTEM,
    CHUNK_READ_SYSTEM,
    NEIGHBOR_SELECT_SYSTEM,
    ANSWER_REASONING_SYSTEM,
    SEMANTIC_ROUTER_SYSTEM,
    GENERAL_SYSTEM_PROMPT,
    MORE_INFO_SYSTEM_PROMPT,
) #TODO i think these can be set by using a configurable file in langgraph, makes things easier

from .utils import parse_function

from typing import Dict, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
neo4j_graph = Neo4jGraph(refresh_schema=False)

########################################
# Semantic Router Node
########################################

def semantic_router(
        state: OverallState,
        semantic_router_system: str=SEMANTIC_ROUTER_SYSTEM
    ) -> OverallState:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type).
    """
    messages = [SystemMessage(content=semantic_router_system)] + state["messages"]
    chosen_route = model.with_structured_output(Router).invoke(messages)
    print("-" * 20)
    print("Step: Routing Query")
    print(f"Chosen route: {chosen_route.route}")
    return {
        "route": chosen_route.route,
        "question": state["messages"][-1].content #NOTE kinda hardcoded here since the last message is always from the Human, might have issues next time if we want to implement breakpoints in the graph, check again
        }

########################################
# General Query Node
########################################

def general_query(
        state: OverallState,
        general_system_prompt: str=GENERAL_SYSTEM_PROMPT
    ) -> OverallState:
    question = state.get("question")
    messages = [SystemMessage(content=general_system_prompt)] + [HumanMessage(content=question)] + state["messages"]

    general_reply = model.invoke(messages)
    print("-" * 20)
    print("Step: General Query")
    print(f"General query response: {general_reply.content}")
    return {
        "messages": [general_reply],
        "answer": general_reply.content,
    }

########################################
# Query Clarification Node
########################################

def clarification(
        state: OverallState,
        more_info_system_prompt: str=MORE_INFO_SYSTEM_PROMPT
    ) -> OverallState:
    question = state.get("question")
    messages = [SystemMessage(content=more_info_system_prompt)] + [HumanMessage(content=question)] + state["messages"]

    clarification_reply = model.invoke(messages)
    print("-" * 20)
    print("Step: Clarification")
    print(f"Clarification response: {clarification_reply.content}")
    return {
        "messages": [clarification_reply],
        "answer": clarification_reply.content,
    }

########################################
# Rational Plan Node
########################################

def rational_plan_node(
        state: OverallState, 
        rational_plan_system: str=RATIONAL_PLAN_SYSTEM
        ) -> OverallState:
    summary = state.get("summary", "")
    if summary:
        conversation_summary = f"Summary of conversation earlier (for context): {summary}"
        rational_plan_system += conversation_summary
    messages = [SystemMessage(content=rational_plan_system)] + state["messages"] # doing this does not affect the "messages" state in the object, so the system prompts do not get updated in the state and we do not summarise it
    rational_plan = model.invoke(messages) # returns AIMessage
    print("-" * 20)
    print(f"Step: rational_plan")
    print(f"Rational plan: {rational_plan.content}")
    return {
        "rational_plan": rational_plan.content, # as rational_plan is now an AIMessage object since we're not passing StrOutputParser() anymore
        "previous_actions": ["rational_plan"], # here we just append it into the previous_actions list, since again, we defined it with an `add` reducer function
        "messages": [rational_plan] # this will be a AIMessage List, which will then be added into the messages state (add_messages reducer function) since we are inheriting from the MessagesState class for OverallState
    }

########################################
# Initial Node Selection
########################################

# create and use a vector index to retrieve a list of input nodes for the prompt.

neo4j_vector = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    index_name="keyelements",
    node_label="KeyElement",
    text_node_properties=["id"],
    embedding_node_property="embedding",
    retrieval_query="RETURN node.id AS text, score, {} AS metadata"
)

def get_potential_nodes(question: str) -> List[str]:
    data = neo4j_vector.similarity_search(question, k=50)
    return [el.page_content for el in data]

def initial_node_selection(
        state: OverallState,
        initial_node_system: str=INITIAL_NODE_SYSTEM,
    ) -> OverallState:
    potential_nodes = get_potential_nodes(state.get("question"))
    question = state.get("question")
    rational_plan = state.get("rational_plan")
    initial_node_human = f"Question: {question}\nPlan: {rational_plan}\nNodes: {potential_nodes}"
    messages = [SystemMessage(content=initial_node_system)] + [HumanMessage(content=initial_node_human)] + state["messages"] # check again later to see whether to add the nodes as a humanmessage or just add in the system prompt and do a .format() would be better
    initial_nodes = model.with_structured_output(InitialNodes).invoke(messages) # returns AIMessage

    print("-" * 20)
    print("Step: Initial Node Selection")
    print(f"Initial nodes: {initial_nodes.initial_nodes}")

    # paper uses 5 initial nodes
    check_atomic_facts_queue = [
        el.key_element
        for el in sorted(
            initial_nodes.initial_nodes,
            key=lambda node: node.score,
            reverse=True,
        )
    ][:5]

    print(f"Selected nodes: {check_atomic_facts_queue}")
    return {
        "check_atomic_facts_queue": check_atomic_facts_queue,
        "previous_actions": ["initial_node_selection"],
    }

########################################
# Atomic Fact Check Node
########################################

def atomic_fact_check(
        state: OverallState,
        atomic_fact_check_system: str=ATOMIC_FACT_CHECK_SYSTEM
    ) -> OverallState:
    print("-" * 20)
    print(f"Step: atomic_fact_check")
    print(
        f"Reading atomic facts about: {state.get('check_atomic_facts_queue')}"
    )
    question = state.get("question")
    rational_plan = state.get("rational_plan")
    notebook = state.get("notebook")
    previous_actions = state.get("previous_actions")
    atomic_facts = get_atomic_facts(state.get("check_atomic_facts_queue")) # check if setting a configuration file would be better for this instead of calling all the variables

    human_message = f"Question: {question}\nPlan: {rational_plan}\nPrevious actions: {previous_actions}\nNotebook: {notebook}\nAtomic facts: {atomic_facts}"
    messages = [SystemMessage(content=atomic_fact_check_system)] + [HumanMessage(content=human_message)] + state["messages"]

    atomic_facts_results = model.with_structured_output(AtomicFactOutput).invoke(messages)

    notebook = atomic_facts_results.updated_notebook
    print(
        f"Rational for next action after atomic check: {atomic_facts_results.rational_next_action}"
    )
    chosen_action = parse_function(atomic_facts_results.chosen_action)
    print(f"Chosen action: {chosen_action}")
    response = {
        "notebook": notebook,
        "chosen_action": chosen_action.get("function_name"),
        "check_atomic_facts_queue": [],
        "previous_actions": [
            f"atomic_fact_check({state.get('check_atomic_facts_queue')})"
        ],
    }
    if chosen_action.get("function_name") == "stop_and_read_neighbor":
        neighbors = get_neighbors_by_key_element(
            state.get("check_atomic_facts_queue")
        )
        response["neighbor_check_queue"] = neighbors
    elif chosen_action.get("function_name") == "read_chunk":
        response["check_chunks_queue"] = chosen_action.get("arguments")[0]
    return response

########################################
# Chunk Read Node
########################################

def chunk_check(
        state: OverallState,
        chunk_read_system: str=CHUNK_READ_SYSTEM,
    ) -> OverallState:
    check_chunks_queue = state.get("check_chunks_queue")
    chunk_id = check_chunks_queue.pop()
    print("-" * 20)
    print(f"Step: read chunk({chunk_id})")

    chunks_text = get_chunk(chunk_id)

    question = state.get("question")
    rational_plan = state.get("rational_plan")
    notebook = state.get("notebook")
    previous_actions = state.get("previous_actions")
    chunk = chunks_text
    human_message = f"Question: {question}\nPlan: {rational_plan}\nPrevious actions: {previous_actions}\nNotebook: {notebook}\nChunk: {chunk}"
    messages = [SystemMessage(content=chunk_read_system)] + [HumanMessage(content=human_message)] + state["messages"]

    read_chunk_results = model.with_structured_output(ChunkOutput).invoke(messages)

    notebook = read_chunk_results.updated_notebook
    print(
        f"Rational for next action after reading chunks: {read_chunk_results.rational_next_move}"
    )
    chosen_action = parse_function(read_chunk_results.chosen_action)
    print(f"Chosen action: {chosen_action}")
    response = {
        "notebook": notebook,
        "chosen_action": chosen_action.get("function_name"),
        "previous_actions": [f"read_chunks({chunk_id})"],
    }
    if chosen_action.get("function_name") == "read_subsequent_chunk":
        subsequent_id = get_subsequent_chunk_id(chunk_id)
        check_chunks_queue.append(subsequent_id)
    elif chosen_action.get("function_name") == "read_previous_chunk":
        previous_id = get_previous_chunk_id(chunk_id)
        check_chunks_queue.append(previous_id)
    elif chosen_action.get("function_name") == "search_more":
        # Go over to next chunk
        # Else explore neighbors
        if not check_chunks_queue:
            response["chosen_action"] = "search_neighbor"
            # Get neighbors/use vector similarity
            print(f"Neighbor rational: {read_chunk_results.rational_next_move}")
            neighbors = get_potential_nodes(
                read_chunk_results.rational_next_move
            )
            response["neighbor_check_queue"] = neighbors

    response["check_chunks_queue"] = check_chunks_queue
    return response

########################################
# Neighbour Selection Node
########################################

def neighbor_select(
        state: OverallState,
        neighbor_select_system: str=NEIGHBOR_SELECT_SYSTEM,
    ) -> OverallState:
    print("-" * 20)
    print(f"Step: neighbor select")
    print(f"Possible candidates: {state.get('neighbor_check_queue')}")

    question = state.get("question")
    rational_plan = state.get("rational_plan")
    notebook = state.get("notebook")
    nodes = state.get("neighbor_check_queue")
    previous_actions = state.get("previous_actions")

    human_message = f"Question: {question}\nPlan: {rational_plan}\nPrevious actions: {previous_actions}\nNotebook: {notebook}\nNeighbor nodes: {nodes}"
    messages = [SystemMessage(content=neighbor_select_system)] + [HumanMessage(content=human_message)] + state["messages"]

    neighbor_select_results = model.with_structured_output(NeighborOutput).invoke(messages)

    print(
        f"Rational for next action after selecting neighbor: {neighbor_select_results.rational_next_move}"
    )
    chosen_action = parse_function(neighbor_select_results.chosen_action)
    print(f"Chosen action: {chosen_action}")
    # Empty neighbor select queue
    response = {
        "chosen_action": chosen_action.get("function_name"),
        "neighbor_check_queue": [],
        "previous_actions": [
            f"neighbor_select({chosen_action.get('arguments', [''])[0] if chosen_action.get('arguments', ['']) else ''})"
        ],
    }
    if chosen_action.get("function_name") == "read_neighbor_node":
        response["check_atomic_facts_queue"] = [
            chosen_action.get("arguments")[0]
        ]
    return response

########################################
# Answer Reasoning Node
########################################

def answer_reasoning(
        state: OverallState,
        answer_reasoning_system: str=ANSWER_REASONING_SYSTEM
    ) -> OverallState:
    print("-" * 20)
    print("Step: Answer Reasoning")

    question = state.get("question")
    notebook = state.get("notebook")

    human_message = f"Question: {question}\nNotebook: {notebook}"

    messages = [SystemMessage(content=answer_reasoning_system)] + [HumanMessage(content=human_message)] + state["messages"]

    final_answer = model.with_structured_output(AnswerReasonOutput).invoke(messages)
    return {
        "answer": final_answer.final_answer,
        "analysis": final_answer.analyze,
        "previous_actions": ["answer_reasoning"],
        "messages": [final_answer.final_answer]
    }

########################################
# Conversation Summary Node
########################################

def summarize_conversation(
        state: OverallState
    ) -> OverallState:
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)] # Add prompt to our history
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]] # Delete all but the 2 most recent messages
    return {
        "summary": response.content, 
        "messages": delete_messages
        }