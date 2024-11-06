from langchain_community.graphs import Neo4jGraph

from dotenv import load_dotenv
load_dotenv()

class Neo4jUploader:
    def __init__(self):
        self.graph = Neo4jGraph(refresh_schema=False)
        self.set_constraints()

    def set_constraints(self):
        # enforces unique IDs on all 4 node types
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:AtomicFact) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:KeyElement) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
        ]
        for query in constraints:
            self.graph.query(query)

    def upload_data(self, data, document_name):
        import_query = """
            MERGE (d:Document {id:$document_name})
            WITH d
            UNWIND $data AS row
            MERGE (c:Chunk {id: row.chunk_id})
            SET c.text = row.chunk_text,
                c.index = row.index,
                c.document_name = $document_name
            MERGE (d)-[:HAS_CHUNK]->(c)
            WITH c, row
            UNWIND row.atomic_facts AS af
            MERGE (a:AtomicFact {id: af.id})
            SET a.text = af.atomic_fact
            MERGE (c)-[:HAS_ATOMIC_FACT]->(a)
            WITH c, a, af
            UNWIND af.key_elements AS ke
            MERGE (k:KeyElement {id: ke})
            MERGE (a)-[:HAS_KEY_ELEMENT]->(k)
        """
        self.graph.query(import_query, params={"data": data, "document_name": document_name})

    def create_next_relationships(self, document_name):
        # link chunk nodes in sequence by order in the document structure
        next_relationship_query = """
            MATCH (c:Chunk) WHERE c.document_name = $document_name
            WITH c ORDER BY c.index WITH collect(c) AS nodes
            UNWIND range(0, size(nodes) -2) AS index
            WITH nodes[index] AS start, nodes[index + 1] AS end
            MERGE (start)-[:NEXT]->(end)
        """
        self.graph.query(next_relationship_query, params={"document_name": document_name})
