from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()

class Neo4jResetter:
    def __init__(self):
        self.graph = Neo4jGraph(refresh_schema=False)

    def reset_graph(self) -> None:
        """
        Delete all nodes and relationships in the graph. Node labels and relationship types are preserved.
        """
        reset_query = """
            MATCH (n)
            DETACH DELETE n
        """
        try:
            self.graph.query(reset_query)
            print("Knowledge graph reset successfully.")
        except Exception as e:
            print(f"An error occurred while resetting the graph: {e}")
