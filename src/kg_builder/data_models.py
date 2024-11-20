from pydantic import BaseModel, Field
from typing import List

class AtomicFact(BaseModel):
    key_elements: List[str] = Field(description="""Critical concepts central to the research context, including specific terms (e.g., hypotheses, methods, 
        models, datasets, metrics, findings). These are pivotal entities related to the atomic fact and can be nouns, actions, or descriptors directly tied to the research study.""")

    atomic_fact: str = Field(description="""The smallest, indivisible factual statements, presented as concise sentences. These include 
        hypotheses, findings, methodologies, results, and causal relationships essential to understanding the paper's core contributions.""")

class Extraction(BaseModel):
    atomic_facts: List[AtomicFact] = Field(description="List of atomic facts representing key research insights and findings from the paper.")