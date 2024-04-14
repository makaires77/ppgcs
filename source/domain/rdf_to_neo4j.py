from rdflib import Graph
from neo4j import GraphDatabase

class RDFtoNeo4j:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()

    def load_rdf_ontology(self, url):
        g = Graph()
        g.parse(url)
        return g

    def map_rdf_to_neo4j(self, graph):
        nodes = []
        relationships = []
        
        for subj, pred, obj in graph:
            nodes.append(subj)
            nodes.append(obj)
            relationships.append((subj, pred, obj))
        
        # Remove duplicates
        nodes = list(set(nodes))
        return nodes, relationships

    def optimize_neo4j(self):
        with self.driver.session() as session:
            # Create indexes for faster query performance
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Resource) ON (n.uri)")
            session.run("CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.type)")
    
    def insert_data_into_neo4j(self, nodes, relationships):
        with self.driver.session() as session:
            # Using write transactions for batch processing
            session.write_transaction(self._insert_nodes, nodes)
            session.write_transaction(self._insert_relationships, relationships)

    @staticmethod
    def _insert_nodes(tx, nodes):
        for node in nodes:
            tx.run("MERGE (:Resource {uri: $uri})", uri=str(node))

    @staticmethod
    def _insert_relationships(tx, relationships):
        for subj, pred, obj in relationships:
            query = (
                "MATCH (a:Resource {uri: $subj}), (b:Resource {uri: $obj}) "
                "MERGE (a)-[:RELATIONSHIP {type: $pred}]->(b)"
            )
            tx.run(query, subj=str(subj), obj=str(obj), pred=str(pred))

if __name__ == "__main__":
    # Exemplo de uso
    rdf_to_neo = RDFtoNeo4j("bolt://localhost:7687", "neo4j", "password")
    rdf_to_neo.optimize_neo4j()
    
    rdf_graph = rdf_to_neo.load_rdf_ontology("path_to_your_ontology.owl")
    nodes, relationships = rdf_to_neo.map_rdf_to_neo4j(rdf_graph)
    rdf_to_neo.insert_data_into_neo4j(nodes, relationships)
    
    rdf_to_neo.close()