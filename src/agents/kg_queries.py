from typing import List, Dict
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph

load_dotenv()

neo4j_graph = Neo4jGraph(refresh_schema=False)

def get_atomic_facts(key_elements: List[str]) -> List[Dict[str, str]]:
    data = neo4j_graph.query("""
    MATCH (k:KeyElement)<-[:HAS_KEY_ELEMENT]-(fact)<-[:HAS_ATOMIC_FACT]-(chunk)
    WHERE k.id IN $key_elements
    RETURN distinct chunk.id AS chunk_id, fact.text AS text
    """, params={"key_elements": key_elements})
    return data

def get_neighbors_by_key_element(key_elements):
    print(f"Key elements: {key_elements}")
    data = neo4j_graph.query("""
    MATCH (k:KeyElement)<-[:HAS_KEY_ELEMENT]-()-[:HAS_KEY_ELEMENT]->(neighbor)
    WHERE k.id IN $key_elements AND NOT neighbor.id IN $key_elements
    WITH neighbor, count(*) AS count
    ORDER BY count DESC LIMIT 50
    RETURN collect(neighbor.id) AS possible_candidates
    """, params={"key_elements":key_elements})
    return data

def get_subsequent_chunk_id(chunk_id):
    data = neo4j_graph.query("""
    MATCH (c:Chunk)-[:NEXT]->(next)
    WHERE c.id = $id
    RETURN next.id AS next
    """, params={"id": chunk_id})
    return data

def get_previous_chunk_id(chunk_id):
    data = neo4j_graph.query("""
    MATCH (c:Chunk)<-[:NEXT]-(previous)
    WHERE c.id = $id
    RETURN previous.id AS previous
    """, params={"id": chunk_id})
    return data

def get_chunk(chunk_id: str) -> List[Dict[str, str]]:
    data = neo4j_graph.query("""
    MATCH (c:Chunk)
    WHERE c.id = $chunk_id
    RETURN c.id AS chunk_id, c.text AS text
    """, params={"chunk_id": chunk_id})
    return data