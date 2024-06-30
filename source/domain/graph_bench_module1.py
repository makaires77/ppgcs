import networkx as nx
import numpy as np
import torch
import time
import psutil
import shutil
from numba import cuda
import cupy as cp
import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constantes
MAX_ITERATIONS = 100
CONVERGENCE_THRESHOLD = 0.001

class OverlappingCommunityDetector:
    def __init__(self, H, must_link=None, cannot_link=None, num_communities=None, other_params=None):
        self.H = H
        self.must_link = must_link if must_link else []
        self.cannot_link = cannot_link if cannot_link else []
        self.num_communities = num_communities
        self.other_params = other_params if other_params else {}

        # Detectar número de comunidades automaticamente se não for fornecido
        if num_communities is None:
            self.num_communities = self._estimate_num_communities()

    def _estimate_num_communities(self):
        """Estima o número de comunidades com base no tamanho do hipergrafo."""
        try:
            num_nodes = self.H.number_of_nodes()
            avg_degree = np.mean([d for _, d in self.H.degree()])

            # Exemplo de estimativa simples (pode ser ajustado ou substituído)
            estimated_num_communities = max(1, int(np.sqrt(num_nodes) / avg_degree))

            logging.info(f"Número estimado de comunidades: {estimated_num_communities}")
            return estimated_num_communities

        except Exception as e:
            logging.error(f"Erro na estimativa do número de comunidades: {e}")
            return 10  # Ou outro valor que faça sentido para o seu problema

    def _benchmark_cpu(self):
        """Realiza um benchmark de CPU com o hipergrafo."""
        try:
            start_time = time.time()

            # Cálculo de métricas do hipergrafo
            num_nodes = self.H.number_of_nodes()
            num_edges = self.H.number_of_edges()
            avg_degree = np.mean([d for _, d in self.H.degree()])
            density = nx.density(self.H)

            # Simulação de propagação de rótulos (SLPA simplificado)
            labels = {node: i for i, node in enumerate(self.H.nodes())}
            for _ in range(10):
                new_labels = {}
                for node in self.H.nodes():
                    neighbor_labels = [labels[neighbor] for neighbor in self.H.neighbors(node)]
                    most_common_label = max(set(neighbor_labels), key=neighbor_labels.count)
                    new_labels[node] = most_common_label
                labels = new_labels

            # Cálculo de modularidade (simplificado)
            communities = set(labels.values())
            modularity = 0
            for community in communities:
                nodes_in_community = [node for node, label in labels.items() if label == community]
                subgraph = self.H.subgraph(nodes_in_community)
                internal_edges = subgraph.number_of_edges()
                total_degree = sum(d for _, d in subgraph.degree())
                modularity += internal_edges / num_edges - (total_degree / (2 * num_edges))**2

            end_time = time.time()
            return end_time - start_time
        except Exception as e:
            logging.error(f"Erro no benchmark da CPU: {e}")
            return float('inf')  # Retorna um tempo infinito em caso de erro

    def _benchmark_gpu(self):
        """Realiza um benchmark de GPU com o hipergrafo."""
        try:
            start_time = time.time()

            # Transferência de dados para a GPU
            H_gpu = nx.to_scipy_sparse_matrix(self.H)
            H_gpu = cp.sparse.csr_matrix(H_gpu)

            # Cálculo de métricas do hipergrafo na GPU
            num_nodes = H_gpu.shape[0]
            num_edges = H_gpu.nnz // 2  # Dividir por 2 para não contar arestas duplicadas
            degrees = cp.diff(H_gpu.indptr)
            avg_degree = cp.mean(degrees)
            density = 2 * num_edges / (num_nodes * (num_nodes - 1))

            # Simulação de propagação de rótulos na GPU (SLPA simplificado)
            labels = cp.arange(num_nodes)  # Inicialização
            for _ in range(10):
                new_labels = cp.zeros_like(labels)
                for i in range(num_nodes):
                    start, end = H_gpu.indptr[i], H_gpu.indptr[i + 1]
                    neighbor_labels = labels[H_gpu.indices[start:end]]
                    unique_labels, counts = cp.unique(neighbor_labels, return_counts=True)
                    most_common_label = unique_labels[cp.argmax(counts)]
                    new_labels[i] = most_common_label
                labels = new_labels

            # Cálculo de modularidade na GPU (simplificado)
            communities = cp.unique(labels)
            modularity = 0
            for community in communities:
                community_mask = labels == community
                community_degrees = degrees[community_mask]
                internal_edges = cp.sum(H_gpu[community_mask, :][:, community_mask]) // 2
                total_degree = cp.sum(community_degrees)
                modularity += internal_edges / num_edges - (total_degree / (2 * num_edges))**2

            end_time = time.time()
            return end_time - start_time

        except Exception as e:
            logging.error(f"Erro no benchmark da GPU: {e}")
            return float('inf')  # Retorna um tempo infinito em caso de erro

    def _has_gpu(self):
        """Verifica se há GPUs disponíveis."""
        try:
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except Exception as e:
            logging.error(f"Erro na detecção de GPU: {e}")
            return False

    def _detect_communities_gpu(self, H_gpu):
        """Implementação CUDA da detecção de comunidades."""
        try:
            # Inicialização das comunidades
            num_nodes = H_gpu.shape[0]
            communities = cp.random.randint(0, self.num_communities, size=num_nodes)
            
            # Pré-processamento das restrições de pares
            must_link_gpu = cp.array(self.must_link)
            cannot_link_gpu = cp.array(self.cannot_link)

            # Iterações do algoritmo
            prev_modularity = -1  # Inicializa a modularidade anterior com um valor inválido
            for _ in range(MAX_ITERATIONS):
                for node_idx in range(num_nodes):
                    # Calcular ganho de modularidade para cada comunidade candidata
                    gains = cp.zeros(self.num_communities)
                    for community_idx in range(self.num_communities):
                        gains[community_idx] = self._calculate_modularity_gain_gpu(
                            H_gpu, node_idx, community_idx, must_link_gpu, cannot_link_gpu
                        )

                    # Encontrar a comunidade com o maior ganho
                    best_community = cp.argmax(gains)

                    # Mover o nó para a melhor comunidade se o ganho for positivo e respeitar as restrições
                    if gains[best_community] > 0 and self._is_move_valid(node_idx, best_community, must_link_gpu, cannot_link_gpu):
                        communities[node_idx] = best_community

                # Verificar convergência
                current_modularity = self._calculate_modularity_gpu(H_gpu, communities)  # Calcula a modularidade atual
                if abs(current_modularity - prev_modularity) < CONVERGENCE_THRESHOLD:  
                    break  # Para se a mudança na modularidade for pequena
                prev_modularity = current_modularity

            return communities

        except Exception as e:
            logging.error(f"Erro na detecção de comunidades na GPU: {e}")