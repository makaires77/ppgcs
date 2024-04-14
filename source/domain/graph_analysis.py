from neo4j import GraphDatabase

class GraphAnalysis:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.write_transaction(lambda tx: tx.run(query, parameters))
            return [record for record in result]

    # Métodos de cálculo de centralidade a partir de parâmetros dinâmicos
    def degree_centrality(self, graph_name, node_label, relationship_type):
        query = f"""
        CALL gds.degree.stream('{graph_name}', {{
            nodeProjection: '{node_label}',
            relationshipProjection: '{relationship_type}'
        }})
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name AS node, score
        ORDER BY score DESC
        """
        return self.execute_query(query)

    def betweenness_centrality(self, graph_name, node_label, relationship_type):
        query = f"""
        CALL gds.betweenness.stream('{graph_name}', {{
            nodeProjection: '{node_label}',
            relationshipProjection: '{relationship_type}'
        }})
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name AS node, score
        ORDER BY score DESC
        """
        return self.execute_query(query)

    def closeness_centrality(self, graph_name, node_label, relationship_type):
        query = f"""
        CALL gds.alpha.closeness.stream('{graph_name}', {{
            nodeProjection: '{node_label}',
            relationshipProjection: '{relationship_type}'
        }})
        YIELD nodeId, centrality
        RETURN gds.util.asNode(nodeId).name AS node, centrality
        ORDER BY centrality DESC
        """
        return self.execute_query(query)

    # Função genérica para rodar identificação de comunidades
    def run_community_algorithm(self, algorithm_name, graph_name, node_labels, relationship_types, config=None):
        if config is None:
            config = {}
        query = f"""
        CALL gds.{algorithm_name}.stream('{graph_name}', {{
            nodeProjection: '{node_labels}',
            relationshipProjection: '{relationship_types}',
            configuration: $config
        }})
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).name AS node, communityId
        """
        parameters = {"config": config}
        return self.execute_query(query, parameters)

    # Identificar comunidades para usar parâmetros
    def label_propagation(self, graph_name, node_labels, relationship_types, config=None):
        return self.run_community_algorithm("labelPropagation", graph_name, node_labels, relationship_types, config)

    def wcc(self, graph_name, node_labels, relationship_types, config=None):
        return self.run_community_algorithm("wcc", graph_name, node_labels, relationship_types, config)

    # Ajustar método para predição de links e menor caminho para incluir parâmetros
    def predict_links(self, graph_name, node_label, relationship_type):
        query = f"""
        CALL gds.linkPrediction.adamicAdar.stream('{graph_name}', {{
            nodeProjection: '{node_label}',
            relationshipProjection: '{relationship_type}'
        }})
        YIELD node1, node2, score
        RETURN gds.util.asNode(node1).name AS node1, gds.util.asNode(node2).name AS node2, score
        ORDER BY score DESC, node1, node2
        """
        return self.execute_query(query)

    def shortest_path(self, graph_name, startNode, endNode, node_label, relationship_type):
        query = f"""
        MATCH (start:{node_label} {{name: $startNode}}), (end:{node_label} {{name: $endNode}})
        CALL gds.alpha.shortestPath.stream('{graph_name}', {{
            nodeProjection: '{node_label}',
            relationshipProjection: {{
                {relationship_type}: {{
                    type: '{relationship_type}',
                    properties: 'weight',
                    orientation: 'UNDIRECTED'
                }}
            }},
            startNode: id(start),
            endNode: id(end),
            relationshipWeightProperty: 'weight'
        }})
        YIELD nodeId, cost
        RETURN gds.util.asNode(nodeId).name AS node, cost
        """
        parameters = {'startNode': startNode, 'endNode': endNode}
        return self.execute_query(query, parameters)
    
    # Método genérico para executar algoritmos de similaridade
    def run_similarity_algorithm(self, algorithm_name, graph_name, node_labels, relationship_types, config=None):
        if config is None:
            config = {}
        query = f"""
        CALL gds.alpha.similarity.{algorithm_name}.stream('{graph_name}', {{
            nodeProjection: '{node_labels}',
            relationshipProjection: '{relationship_types}',
            configuration: $config
        }})
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS node1, gds.util.asNode(node2).name AS node2, similarity
        ORDER BY similarity DESC, node1, node2
        """
        parameters = {"config": config}
        return self.execute_query(query, parameters)

    # Métodos específicos para cada tipo de similaridade
    def jaccard_similarity(self, graph_name, node_labels, relationship_types, config=None):
        return self.run_similarity_algorithm("jaccard", graph_name, node_labels, relationship_types, config)

    def cosine_similarity(self, graph_name, node_labels, relationship_types, config=None):
        return self.run_similarity_algorithm("cosine", graph_name, node_labels, relationship_types, config)

    def pearson_similarity(self, graph_name, node_labels, relationship_types, config=None):
        return self.run_similarity_algorithm("pearson", graph_name, node_labels, relationship_types, config)
