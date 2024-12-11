from typing import Any, Dict, Literal, Optional, Type, TypeVar
from uuid import UUID
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig, ensure_config

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
)

T = TypeVar("T", bound="BaseConfiguration")

class BaseConfiguration(BaseModel):
    """
    Configuration base class for indexing operations.

    This class defines the parameters needed for configuring the indexing processes, including embedding model selection.
    """

    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Name of the embedding model to use. Must be a valid embedding model name.",
        __template_metadata__={"kind": "embeddings"},
    )

    thread_id: Optional[UUID] = Field(
        default=None,
        description="Unique identifier for the current user thread. Set dynamically at runtime."
    )

    callbacks: Optional[list[Any]] = Field(
        default=None,
        description="List of callbacks to be executed during processing."
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """
        Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable", {})
        callbacks = config.get("callbacks", [])

        # Extract fields from the class that match the config keys
        field_keys = cls.__fields__.keys()
        extracted_config = {k: v for k, v in configurable.items() if k in field_keys}
        
        # Return an instance, incorporating extracted config and callbacks
        return cls(**extracted_config, callbacks=callbacks)
    #     # return cls(**{k: v for k, v in configurable.items() if k in cls.__fields__})

class AgentConfiguration(BaseConfiguration):
    """
    The configuration for the agents, including model type and prompts.
    """

    # Models
    kg_processing_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "The language model used for routing processing the knowledge graph."
        ),
        __template_metadata__={"kind": "llm"},
    )

    router_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "The language model used for routing of user queries."
        ),
        __template_metadata__={"kind": "llm"},
    )

    research_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "The language model used in research subgraph."
        ),
        __template_metadata__={"kind": "llm"},
    )

    reasoning_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "The language model used in research subgraph."
        ),
        __template_metadata__={"kind": "llm"},
    )

    # Prompts
    router_system_prompt: str = Field(
        default=SEMANTIC_ROUTER_SYSTEM,
        description="The system prompt used for classifying user questions to route them to the correct node.",
    )

    more_info_system_prompt: str = Field(
        default=MORE_INFO_SYSTEM_PROMPT,
        description="The system prompt used for asking for more information from the user.",
    )

    general_system_prompt: str = Field(
        default=GENERAL_SYSTEM_PROMPT,
        description="The system prompt used for responding to general questions.",
    )

    rational_plan_system_prompt: str = Field(
        default=RATIONAL_PLAN_SYSTEM,
        description="The system prompt used for generating a research plan based on the user's question.",
    )

    initial_node_system_prompt: str = Field(
        default=INITIAL_NODE_SYSTEM,
        description="The system prompt used to search for initial nodes in the knowledge graph.",
    )

    atomic_fact_check_system_prompt: str = Field(
        default=ATOMIC_FACT_CHECK_SYSTEM,
        description="The system prompt used to read and analyse relevance of atomic facts from initial nodes.",
    )

    chunk_read_system_prompt: str = Field(
        default=CHUNK_READ_SYSTEM,
        description="The system prompt used to read and analyse associated text chunks from relevant atomic facts.",
    )

    neighbor_select_system_prompt: str = Field(
        default=NEIGHBOR_SELECT_SYSTEM,
        description="The system prompt used to search surrounding neighbor nodes for more relevant information.",
    )

    answer_reasoning_system_prompt: str = Field(
        default=ANSWER_REASONING_SYSTEM,
        description="The system prompt used for generating responses.",
    )