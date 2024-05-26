from typing import Union, List, Dict, Tuple, Any
from pymatgen.core import Composition
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Classes Genéricas para manipular Grafo (Node, Edge, Graph)
class Node:
    def __init__(self, id: str, data: Any = None):
        self.id = id
        self.data = data

class Edge:
    def __init__(self, source: Node, target: Node, weight: float = None):
        self.source = source
        self.target = target
        self.weight = weight

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def get_node_by_id(self, node_id):
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> List[Node]:
        """Retorna os vizinhos de um nó."""
        neighbors = []
        for edge in self.edges:
            if edge.source.id == node_id:
                neighbors.append(edge.target)
            elif edge.target.id == node_id:
                neighbors.append(edge.source)
        return neighbors

    def get_edge_weight(self, source_node_id: str, target_node_id: str) -> float:
        """Retorna o peso da aresta entre dois nós."""
        for edge in self.edges:
            if (edge.source.id == source_node_id and edge.target.id == target_node_id) or \
               (edge.source.id == target_node_id and edge.target.id == source_node_id):
                return edge.weight
        return None  # Aresta não encontrada

    def remove_node(self, node_id: str):
        """Remove um nó e suas arestas do grafo."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.edges = [e for e in self.edges if e.source.id != node_id and e.target.id != node_id]

    def remove_edge(self, source_node_id: str, target_node_id: str):
        """Remove uma aresta do grafo."""
        self.edges = [e for e in self.edges if not (
            (e.source.id == source_node_id and e.target.id == target_node_id) or
            (e.source.id == target_node_id and e.target.id == source_node_id)
        )]

    def get_subgraph(self, node_ids: List[str]) -> Graph:
        """Retorna um subgrafo contendo apenas os nós especificados."""
        subgraph = Graph()
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if node:
                subgraph.add_node(node)
        for edge in self.edges:
            if edge.source.id in node_ids and edge.target.id in node_ids:
                subgraph.add_edge(edge)
        return subgraph

    def merge_nodes(self, node_id1: str, node_id2: str, new_node_id: str):
        """Combina dois nós em um único nó."""
        node1 = self.get_node_by_id(node_id1)
        node2 = self.get_node_by_id(node_id2)
        if node1 is None or node2 is None:
            raise ValueError("One or both nodes not found in the graph.")

        new_node = Node(new_node_id, data={**node1.data, **node2.data})  # Combina os dados dos nós
        self.add_node(new_node)

        # Atualiza as arestas para apontar para o novo nó
        for edge in self.edges:
            if edge.source.id == node_id1 or edge.source.id == node_id2:
                edge.source = new_node
            if edge.target.id == node_id1 or edge.target.id == node_id2:
                edge.target = new_node

        self.remove_node(node_id1)
        self.remove_node(node_id2)


class TextPropertyGraphAdapter:
    def __init__(self, texts: List[str], embeddings: KeyedVectors, similarity_threshold: float = 0.8):
        self.texts = texts
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold

    def create_graph(self) -> Graph:
        graph = Graph()
        for i, text in enumerate(self.texts):
            node = self._create_node(i, text)
            graph.add_node(node)

            for j, other_text in enumerate(self.texts):
                if i != j:
                    similarity = self._calculate_similarity(text, other_text)
                    if similarity > self.similarity_threshold:
                        edge = Edge(node, graph.nodes[j], weight=similarity)
                        graph.add_edge(edge)
        return graph

    def _create_node(self, id: int, text: str) -> Node:
        embedding = self.embeddings[text]
        return Node(id, {"text": text, "embedding": embedding})

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        return cosine_similarity([self.embeddings[text1]], [self.embeddings[text2]])[0][0]  # Cosine similarity


class RecommenderSystem:
    def __init__(self, graph: Graph, embedding: KeyedVectors, distance_threshold: float, forbidden_nodes: List[str] = None):
        self.graph = graph
        self.embedding = embedding
        self.distance_threshold = distance_threshold
        self.forbidden_nodes = forbidden_nodes or []

    def get_recommendations_for_node(self, node_id: str, top_n: int = None) -> List[Tuple[Node, float]]:
        node = self.graph.get_node_by_id(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in the graph.")

        recommendations = []
        for other_node in self.graph.nodes.values():
            if other_node.id != node_id and other_node.id not in self.forbidden_nodes:
                distance = 1 - self._calculate_similarity(node.data['embedding'], other_node.data['embedding'])  # Convertendo similaridade em distância
                if distance < self.distance_threshold:
                    recommendations.append((other_node, distance))

        recommendations.sort(key=lambda x: x[1])
        return recommendations[:top_n] if top_n else recommendations

    def _calculate_similarity(self, embedding1, embedding2):
        return cosine_similarity([embedding1], [embedding2])[0][0]

    def find_similar_documents(self, query_text: str, top_n: int = 5) -> List[Tuple[Node, float]]:
        query_embedding = self.embedding[query_text]
        
        similarities = []
        for node in self.graph.nodes.values():
            similarity = self._calculate_similarity(query_embedding, node.data['embedding'])
            similarities.append((node, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)  # Ordenar por similaridade decrescente
        return similarities[:top_n]

    def detect_communities(self) -> List[set]:
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(self.graph.nodes.keys())
        nx_graph.add_weighted_edges_from([(e.source.id, e.target.id, e.weight) for e in self.graph.edges])
        communities_generator = community.girvan_newman(nx_graph)
        top_level_communities = next(communities_generator)
        return sorted(map(sorted, top_level_communities))

    def predict_links(self, metric="adamic_adar_index") -> List[Tuple[Node, Node, float]]:
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(self.graph.nodes.keys())
        nx_graph.add_weighted_edges_from([(e.source.id, e.target.id, e.weight) for e in self.graph.edges])

        if metric == "adamic_adar_index":
            predictions = nx.adamic_adar_index(nx_graph)
        elif metric == "jaccard_coefficient":
            predictions = nx.jaccard_coefficient(nx_graph)
        # Adicione outras métricas de link prediction aqui
        else:
            raise ValueError(f"Invalid metric: {metric}")

        return [(self.graph.get_node_by_id(u), self.graph.get_node_by_id(v), p) for u, v, p in predictions]

    def find_shortest_path(self, source_node_id: str, target_node_id: str) -> List[Node]:
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(self.graph.nodes.keys())
        nx_graph.add_weighted_edges_from([(e.source.id, e.target.id, 1/e.weight) for e in self.graph.edges])  # Inverte os pesos para encontrar o caminho mais curto
        path = nx.shortest_path(nx_graph, source=source_node_id, target=target_node_id, weight='weight')
        return [self.graph.get_node_by_id(node_id) for node_id in path]

    def calculate_max_flow(self, source_node_id: str, target_node_id: str) -> float:
        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from(self.graph.nodes.keys())
        nx_graph.add_weighted_edges_from([(e.source.id, e.target.id, e.weight) for e in self.graph.edges])
        return nx.maximum_flow_value(nx_graph, source_node_id, target_node_id, capacity='weight')
