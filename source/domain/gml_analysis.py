from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt

class GraphAnalysis:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.G = nx.Graph()

    def close(self):
        self.driver.close()

    def fetch_graph(self):
        """
        Extrair dados do Neo4j e construir um grafo usando NetworkX.
        """
        with self.driver.session() as session:
            result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m")
            for record in result:
                self.G.add_edge(record['n']['id'], record['m']['id'])

    def apply_newman_clustering(self):
        """
        Aplicar o algoritmo de clustering de Newman.
        """
        communities = nx.algorithms.community.greedy_modularity_communities(self.G)
        return communities

    def visualize_communities(self, communities):
        """
        Visualizar as comunidades identificadas.
        """
        color_map = []
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'lime']
        node_community = {}
        for idx, community in enumerate(communities):
            for node in community:
                node_community[node] = colors[idx % len(colors)]

        color_map = [node_community[node] for node in self.G.nodes()]
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, node_color=color_map, with_labels=True, node_size=500)
        plt.title('Communities Identified by Newman Clustering')
        plt.show()

    def run_analysis(self):
        """
        Executar o fluxo completo de análise.
        """
        self.fetch_graph()
        communities = self.apply_newman_clustering()
        self.visualize_communities(communities)

# Usage
uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

graph_analysis = GraphAnalysis(uri, user, password)
graph_analysis.run_analysis()
graph_analysis.close()