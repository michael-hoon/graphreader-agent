import re
import ast
from typing import Dict, Any

def parse_function(input_str: str) -> Dict[str, Any]:
    # regex to capture the function name and arguments from the model output
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