from neo4j import GraphDatabase

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_graph_data(self, graph_projection_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._get_graph_nodes_and_relationships, graph_projection_name)
            return result

    @staticmethod
    def _get_graph_nodes_and_relationships(tx, graph_projection_name):
        # Substitua a consulta abaixo conforme necessário para se adequar ao seu modelo de dados e à projeção do grafo desejada
        query = (
            "MATCH (n) "
            "WHERE n.projection = $projection "
            "RETURN n.id AS id, n.row AS row, n.column AS column, n.label AS label"
        )
        result = tx.run(query, projection=graph_projection_name)
        return [record for record in result]

# Uso da classe
uri = "neo4j://localhost:7687"
user = "neo4j"
password = "password"
connector = Neo4jConnector(uri, user, password)
graph_data = connector.get_graph_data("roadmaptec_imunologia_abc")
connector.close()

print(graph_data)