import re
import ast
from typing import Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

def parse_function(input_str: str) -> Dict[str, Any]:
    """
    Regex to capture the function name and arguments from the agent output.

    Args:
        input_str (str): The agent output for selection function.
    
    Returns:
        Dict[str, Any]: The matched function name and arguments for execution.
    """
    pattern = r'(\w+)(?:\((.*)\))?'
    match = re.match(pattern, input_str)
    if match:
        function_name = match.group(1)  # extract function name
        raw_arguments = match.group(2)  # extract arguments as string        
        # if there are arguments, attempt to parse
        arguments = []
        if raw_arguments:
            try:
                # use ast.literal_eval to safely evaluate and convert the arguments
                parsed_args = ast.literal_eval(f'({raw_arguments})')  # wrap in tuple
                # ensure it's always treated as a tuple even with a single argument
                arguments = list(parsed_args) if isinstance(parsed_args, tuple) else [parsed_args]
            except (ValueError, SyntaxError):
                # if fail to parse, return raw argument string
                arguments = [raw_arguments.strip()]

        return {
            'function_name': function_name,
            'arguments': arguments
        }
    else:
        return None

def match_llm(model: str) -> BaseChatModel:
    """
    Connect to the configured chat model.
    """
    match model:
        case "gpt-4o-mini":
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        case "llama-3.1":
            return ChatOllama(model="qwen2.5:32b")
        case _:
            raise ValueError(f"Unsupported chat model: {model}")

def match_embedding_model(embedding: str) -> Embeddings:
    """
    Connect to the configured embedding model.
    """
    match model:
        case "text-embedding-3-small":
            return OpenAIEmbeddings(model="text-embedding-3-small")
        case "nomic-embed-text":
            return OllamaEmbeddings(model="nomic-embed-text")
        case _:
            raise ValueError(f"Unsupported embedding model: {embedding}")