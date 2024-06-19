import networkx as nx
import numpy as np
import torch
import time
import psutil
import shutil
from numba import cuda
import cupy as cp
import logging
import platform
import importlib.util

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constantes (ajuste conforme necessário)
MAX_ITERATIONS = 100
CONVERGENCE_THRESHOLD = 0.001

class OverlappingCommunityDetector:
    def __init__(self, H, must_link, cannot_link, num_communities=None, other_params=None):
        self.H = H
        self.must_link = must_link
        self.cannot_link = cannot_link
        self.num_communities = num_communities
        self.other_params = other_params

        # Detectar número de comunidades automaticamente se não for fornecido
        if num_communities is None:
            self.num_communities = self._estimate_num_communities()

    def _benchmark_cpu(self, H):

        """Realiza um benchmark de CPU com o hipergrafo.
            Explicação:

            Cálculo de Métricas: Calculamos algumas métricas básicas do hipergrafo, como número de nós, número de arestas, grau médio e densidade. Essas operações envolvem iterações sobre os nós e arestas do hipergrafo, sendo representativas da carga computacional em CPU para manipulação de dados.

            Simulação de Propagação de Rótulos: Simulamos uma versão simplificada do algoritmo SLPA (Speaker-Listener Label Propagation Algorithm), que é um algoritmo iterativo para detecção de comunidades. A cada iteração, cada nó atualiza seu rótulo com base nos rótulos de seus vizinhos. Essa simulação representa a carga computacional em CPU para algoritmos iterativos em grafos.

            Cálculo de Modularidade: Calculamos uma versão simplificada da modularidade, que é uma métrica para avaliar a qualidade de uma divisão de um grafo em comunidades. Essa operação envolve iterações sobre as comunidades e seus nós, representando a carga computacional em CPU para cálculos de métricas em grafos.

        As operações escolhidas são apenas exemplos e podem ser adaptadas de acordo com as características específicas do seu algoritmo.

        O número de iterações na simulação de propagação de rótulos e a complexidade do cálculo de modularidade podem ser ajustados para aumentar ou diminuir a carga computacional do benchmark.

        O tempo de execução da função benchmark_cpu será usado como referência para comparar com o tempo de execução na GPU e decidir onde executar os cálculos.
        """
        try:

            # Operações representativas de carga computacional em CPU
            start_time = time.time()

            # 1. Cálculo de métricas do hipergrafo
            num_nodes = H.number_of_nodes()
            num_edges = H.number_of_edges()
            avg_degree = np.mean([d for _, d in H.degree()])
            density = nx.density(H)

            # 2. Simulação de propagação de rótulos (SLPA simplificado)
            labels = {node: i for i, node in enumerate(H.nodes())}  # Inicialização
            for _ in range(10):  # Número de iterações (arbitrário)
                new_labels = {}
                for node in H.nodes():
                    neighbor_labels = [labels[neighbor] for neighbor in H.neighbors(node)]
                    most_common_label = max(set(neighbor_labels), key=neighbor_labels.count)
                    new_labels[node] = most_common_label
                labels = new_labels

            # 3. Cálculo de modularidade (simplificado)
            communities = set(labels.values())
            modularity = 0
            for community in communities:
                nodes_in_community = [node for node, label in labels.items() if label == community]
                subgraph = H.subgraph(nodes_in_community)
                internal_edges = subgraph.number_of_edges()
                total_degree = sum(d for _, d in subgraph.degree())
                modularity += internal_edges / num_edges - (total_degree / (2 * num_edges))**2

            end_time = time.time()
            return end_time - start_time
        except Exception as e:
            logging.error(f"Erro no benchmark da CPU: {e}")
            return float('inf')  # Retorna um tempo infinito em caso de erro

    def _benchmark_gpu(self, H):
        """Realiza um benchmark de GPU com o hipergrafo H.
        
        Explicação:

        Transferência de Dados: O hipergrafo H é convertido para uma matriz esparsa SciPy e, em seguida, para uma matriz esparsa CuPy (H_gpu) para ser processado na GPU.
        
        Cálculo de Métricas na GPU: As métricas do hipergrafo são calculadas usando operações de álgebra linear na matriz esparsa H_gpu na GPU.
        Simulação de Propagação de Rótulos na GPU: A simulação do SLPA é adaptada para usar operações de array do CuPy na GPU.
        
        Cálculo de Modularidade na GPU: O cálculo da modularidade também é adaptado para usar operações de array do CuPy na GPU.
        Observações:

        Esta implementação assume que você tem o CuPy instalado, que é uma biblioteca que permite usar NumPy na GPU.
        
        As operações na GPU são geralmente mais rápidas que na CPU, mas a transferência de dados entre CPU e GPU pode ser um gargalo.
        
        O tempo de execução da função benchmark_gpu será usado para comparar com o tempo de execução na CPU e decidir onde executar os cálculos.
        """
        try:
            start_time = time.time()

            # 1. Transferência de dados para a GPU
            H_gpu = nx.to_scipy_sparse_matrix(H)
            H_gpu = cp.sparse.csr_matrix(H_gpu)

            # 2. Cálculo de métricas do hipergrafo na GPU
            num_nodes = H_gpu.shape[0]
            num_edges = H_gpu.nnz // 2  # Dividir por 2 para não contar arestas duplicadas
            degrees = cp.diff(H_gpu.indptr)
            avg_degree = cp.mean(degrees)
            density = 2 * num_edges / (num_nodes * (num_nodes - 1))

            # 3. Simulação de propagação de rótulos na GPU (SLPA simplificado)
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

            # 4. Cálculo de modularidade na GPU (simplificado)
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
            return None  # Retorna None em caso de erro na GPU

    def _calculate_modularity_gain_gpu(self, H_gpu, node_idx, community_idx, must_link_gpu, cannot_link_gpu):
        """Calcula o ganho de modularidade para um nó se mover para uma comunidade na GPU."""
        try:
            # Parâmetros (substitua pelos valores reais)
            resolution = self.other_params.get('resolution', 1.0)
            must_link_weight = self.other_params.get('must_link_weight', 1.0)
            cannot_link_weight = self.other_params.get('cannot_link_weight', 1.0)

            # Variáveis auxiliares
            num_nodes = H_gpu.shape[0]
            num_edges = H_gpu.nnz // 2  # Dividir por 2 para não contar arestas duplicadas
            degrees = cp.diff(H_gpu.indptr)
            node_degree = degrees[node_idx]
            community_mask = communities == community_idx
            community_degrees = degrees[community_mask]
            total_degree = cp.sum(community_degrees)

            # Cálculo da modularidade atual da comunidade
            internal_edges = cp.sum(H_gpu[community_mask, :][:, community_mask]) // 2
            current_modularity = internal_edges / num_edges - (total_degree / (2 * num_edges))**2

            # Cálculo da modularidade se o nó for movido para a comunidade
            new_internal_edges = internal_edges + cp.sum(H_gpu[node_idx, community_mask])
            new_total_degree = total_degree + node_degree
            new_modularity = new_internal_edges / num_edges - (new_total_degree / (2 * num_edges))**2

            # Cálculo do ganho de modularidade
            gain = new_modularity - current_modularity

            # Aplicar resolução
            gain *= resolution

            # Penalização por violar restrições de pares
            for i in range(must_link_gpu.shape[0]):
                if (must_link_gpu[i, 0] == node_idx and communities[must_link_gpu[i, 1]] != community_idx) or \
                (must_link_gpu[i, 1] == node_idx and communities[must_link_gpu[i, 0]] != community_idx):
                    gain -= must_link_weight  # Penaliza se o nó não estiver na mesma comunidade que um nó "must-link"

            for i in range(cannot_link_gpu.shape[0]):
                if cannot_link_gpu[i, 0] == node_idx and communities[cannot_link_gpu[i, 1]] == community_idx:
                    gain -= cannot_link_weight  # Penaliza se o nó estiver na mesma comunidade que um nó "cannot-link"

            return gain

        except Exception as e:
            logging.error(f"Erro no cálculo do ganho de modularidade na GPU: {e}")
            return 0  # Retorna ganho zero em caso de erro


    def _is_move_valid(self, node_idx, community_idx, must_link_gpu, cannot_link_gpu):
        """Verifica se mover um nó para uma comunidade viola alguma restrição de pares na GPU."""
        try:
            for i in range(must_link_gpu.shape[0]):
                if (must_link_gpu[i, 0] == node_idx and communities[must_link_gpu[i, 1]] != community_idx) or \
                (must_link_gpu[i, 1] == node_idx and communities[must_link_gpu[i, 0]] != community_idx):
                    return False  # Violação de must-link

            for i in range(cannot_link_gpu.shape[0]):
                if cannot_link_gpu[i, 0] == node_idx and communities[cannot_link_gpu[i, 1]] == community_idx:
                    return False  # Violação de cannot-link

            return True  # Movimento válido

        except Exception as e:
            logging.error(f"Erro na verificação de restrições na GPU: {e}")
            return False  # Considera o movimento inválido em caso de erro


    def _calculate_modularity_gpu(self, H_gpu, communities):
        """Calcula a modularidade da partição atual das comunidades na GPU."""
        try:
            modularity = 0.0
            num_edges = H_gpu.nnz // 2

            for community_idx in range(self.num_communities):
                community_mask = communities == community_idx
                community_degrees = cp.diff(H_gpu.indptr)[community_mask]
                internal_edges = cp.sum(H_gpu[community_mask, :][:, community_mask]) // 2
                total_degree = cp.sum(community_degrees)
                modularity += internal_edges / num_edges - (total_degree / (2 * num_edges))**2

            return modularity

        except Exception as e:
            logging.error(f"Erro no cálculo da modularidade na GPU: {e}")
            return -1  # Retorna um valor inválido em caso de erro

    def _detect_communities_cpu(self):
        """Implementação da detecção de comunidades na CPU."""
        # Inicialização das comunidades (mesma lógica da GPU, mas usando NumPy)
        num_nodes = self.H.number_of_nodes()
        communities = np.random.randint(0, self.num_communities, size=num_nodes)

        # Pré-processamento das restrições de pares
        must_link_np = np.array(self.must_link)
        cannot_link_np = np.array(self.cannot_link)

        # Iterações do algoritmo
        prev_modularity = -1  # Inicializa a modularidade anterior com um valor inválido
        for _ in range(MAX_ITERATIONS):
            for node_idx in range(num_nodes):
                # Calcular ganho de modularidade para cada comunidade candidata
                gains = np.zeros(self.num_communities)
                for community_idx in range(self.num_communities):
                    gains[community_idx] = self._calculate_modularity_gain_cpu(
                        node_idx, community_idx, must_link_np, cannot_link_np
                    )

                # Encontrar a comunidade com o maior ganho
                best_community = np.argmax(gains)

                # Mover o nó para a melhor comunidade se o ganho for positivo e respeitar as restrições
                if gains[best_community] > 0 and self._is_move_valid(node_idx, best_community, must_link_np, cannot_link_np):
                    communities[node_idx] = best_community

            # Verificar convergência
            current_modularity = self._calculate_modularity_cpu(communities)  # Calcula a modularidade atual
            if abs(current_modularity - prev_modularity) < CONVERGENCE_THRESHOLD:  
                break  # Para se a mudança na modularidade for pequena
            prev_modularity = current_modularity

        return communities

    def _calculate_modularity_gain_cpu(self, node_idx, community_idx, must_link_np, cannot_link_np):
        """Calcula o ganho de modularidade para um nó se mover para uma comunidade na CPU."""
        try:
            # Parâmetros
            resolution = self.other_params.get('resolution', 1.0)
            must_link_weight = self.other_params.get('must_link_weight', 1.0)
            cannot_link_weight = self.other_params.get('cannot_link_weight', 1.0)

            # Variáveis auxiliares
            num_nodes = self.H.number_of_nodes()
            num_edges = self.H.number_of_edges()
            degrees = np.array([d for _, d in self.H.degree()])
            node_degree = degrees[node_idx]
            community_mask = communities == community_idx
            community_degrees = degrees[community_mask]
            total_degree = np.sum(community_degrees)

            # Cálculo da modularidade atual da comunidade
            H_matrix = nx.to_numpy_array(self.H)
            internal_edges = np.sum(H_matrix[community_mask, :][:, community_mask]) // 2
            current_modularity = internal_edges / num_edges - (total_degree / (2 * num_edges))**2

            # Cálculo da modularidade se o nó for movido para a comunidade
            new_internal_edges = internal_edges + np.sum(H_matrix[node_idx, community_mask])
            new_total_degree = total_degree + node_degree
            new_modularity = new_internal_edges / num_edges - (new_total_degree / (2 * num_edges))**2

            # Cálculo do ganho de modularidade
            gain = new_modularity - current_modularity

            # Aplicar resolução
            gain *= resolution

            # Penalização por violar restrições de pares
            for i in range(must_link_np.shape[0]):
                if (must_link_np[i, 0] == node_idx and communities[must_link_np[i, 1]] != community_idx) or \
                (must_link_np[i, 1] == node_idx and communities[must_link_np[i, 0]] != community_idx):
                    gain -= must_link_weight  # Penaliza se o nó não estiver na mesma comunidade que um nó "must-link"

            for i in range(cannot_link_np.shape[0]):
                if cannot_link_np[i, 0] == node_idx and communities[cannot_link_np[i, 1]] == community_idx:
                    gain -= cannot_link_weight  # Penaliza se o nó estiver na mesma comunidade que um nó "cannot-link"

            return gain

        except Exception as e:
            logging.error(f"Erro no cálculo do ganho de modularidade na CPU: {e}")
            return 0  # Retorna ganho zero em caso de erro

    def _is_move_valid(self, node_idx, community_idx, must_link_np, cannot_link_np):
        """Verifica se mover um nó para uma comunidade viola alguma restrição de pares na CPU."""
        try:
            for i in range(must_link_np.shape[0]):
                if (must_link_np[i, 0] == node_idx and communities[must_link_np[i, 1]] != community_idx) or \
                (must_link_np[i, 1] == node_idx and communities[must_link_np[i, 0]] != community_idx):
                    return False  # Violação de must-link

            for i in range(cannot_link_np.shape[0]):
                if cannot_link_np[i, 0] == node_idx and communities[cannot_link_np[i, 1]] == community_idx:
                    return False  # Violação de cannot-link

            return True  # Movimento válido

        except Exception as e:
            logging.error(f"Erro na verificação de restrições na CPU: {e}")
            return False  # Considera o movimento inválido em caso de erro


    def _calculate_modularity_cpu(self, communities):
        """Calcula a modularidade da partição atual das comunidades na CPU."""
        try:
            modularity = 0.0
            num_edges = self.H.number_of_edges()
            H_matrix = nx.to_numpy_array(self.H)  # Convertendo para matriz NumPy

            for community_idx in range(self.num_communities):
                community_mask = communities == community_idx
                degrees = np.array([d for _, d in self.H.degree()])
                community_degrees = degrees[community_mask]
                internal_edges = np.sum(H_matrix[community_mask, :][:, community_mask]) // 2
                total_degree = np.sum(community_degrees)
                modularity += internal_edges / num_edges - (total_degree / (2 * num_edges))**2

            return modularity
        except Exception as e:
            logging.error(f"Erro no cálculo da modularidade na CPU: {e}")
            return -1  # Retorna um valor inválido em caso de erro
        
    def detect_communities_benchmarking(self):
        """Detecta comunidades com benchmarking e escolha automática de CPU/GPU."""
        try:
            # Informações sobre o ambiente de execução
            logging.info("Informações do sistema:")
            logging.info(f"  - Sistema operacional: {platform.system()} {platform.release()}")
            logging.info(f"  - Processador: {platform.processor()}")
            logging.info(f"  - Memória RAM total: {psutil.virtual_memory().total / (1024**3):.2f} GB")
            logging.info(f"  - Número de GPUs disponíveis: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

            # Benchmarking
            cpu_time = self._benchmark_cpu(self.H)  # Passando self.H como argumento
            if self._has_gpu():
                gpu_time = self._benchmark_gpu(self.H)  # Passando self.H como argumento
            else:
                gpu_time = float('inf')

            logging.info("\nResultados do Benchmarking:")
            logging.info(f"  - Tempo de execução na CPU: {cpu_time:.2f} segundos")
            logging.info(f"  - Tempo de execução na GPU: {gpu_time:.2f} segundos")

            # Decisão de execução
            if self._has_gpu() and gpu_time < cpu_time and psutil.virtual_memory().available > self.H.size():
                logging.info("Executando na GPU")
                H_gpu = cuda.to_device(self.H)
                communities = self._detect_communities_gpu(H_gpu)
                communities = cuda.from_device(communities)
            else:
                logging.info("Executando na CPU")
                communities = self._detect_communities_cpu()

            return communities

        except Exception as e:
            logging.error(f"Erro na detecção de comunidades: {e}")
            return None
        
    def _check_libraries(self):
        """Verifica se as bibliotecas necessárias estão instaladas."""
        required_libraries = {
            "networkx": nx,
            "numpy": np,
            "torch": torch,
            "psutil": psutil,
            "shutil": shutil,
            "numba": cuda,  # Verifica se a CUDA está disponível no Numba
            "cupy": cp
        }

        missing_libraries = []
        for lib_name, module in required_libraries.items():
            if module is None or not importlib.util.find_spec(lib_name):
                missing_libraries.append(lib_name)

        if missing_libraries:
            logging.error(f"Bibliotecas ausentes: {', '.join(missing_libraries)}")
            return False
        else:
            logging.info("Todas as bibliotecas necessárias estão instaladas.")
            return True