import networkx as nx
import numpy as np
import torch
import time
import psutil
import shutil
from numba import cuda
import cupy as cp

# Funções auxiliares para benchmarks e detecção de hardware
def benchmark_cpu(H):
    """Realiza um benchmark de CPU com o hipergrafo H.
        Explicação:

        Cálculo de Métricas: Calculamos algumas métricas básicas do hipergrafo, como número de nós, número de arestas, grau médio e densidade. Essas operações envolvem iterações sobre os nós e arestas do hipergrafo, sendo representativas da carga computacional em CPU para manipulação de dados.

        Simulação de Propagação de Rótulos: Simulamos uma versão simplificada do algoritmo SLPA (Speaker-Listener Label Propagation Algorithm), que é um algoritmo iterativo para detecção de comunidades. A cada iteração, cada nó atualiza seu rótulo com base nos rótulos de seus vizinhos. Essa simulação representa a carga computacional em CPU para algoritmos iterativos em grafos.

        Cálculo de Modularidade: Calculamos uma versão simplificada da modularidade, que é uma métrica para avaliar a qualidade de uma divisão de um grafo em comunidades. Essa operação envolve iterações sobre as comunidades e seus nós, representando a carga computacional em CPU para cálculos de métricas em grafos.

    As operações escolhidas são apenas exemplos e podem ser adaptadas de acordo com as características específicas do seu algoritmo.

    O número de iterações na simulação de propagação de rótulos e a complexidade do cálculo de modularidade podem ser ajustados para aumentar ou diminuir a carga computacional do benchmark.

    O tempo de execução da função benchmark_cpu será usado como referência para comparar com o tempo de execução na GPU e decidir onde executar os cálculos.
    """
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

def benchmark_gpu(H):
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

def has_gpu():
    """Verifica se há GPUs disponíveis."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


# Função principal do algoritmo
def overlapping_community_detection(H, must_link, cannot_link, num_communities=None, other_params=None):
    """Detecta comunidades sobrepostas em um hipergrafo com restrições de pares."""

    # Pré-processamento
    # ... (carregar dados, construir hipergrafo, normalização, etc.)

    # Detecção de aceleradores e restrições de hardware
    disk_space = shutil.disk_usage('/').free
    ram_available = psutil.virtual_memory().available
    gpu_available = has_gpu()

    # Benchmarking
    cpu_time = benchmark_cpu(H)
    if gpu_available:
        gpu_time = benchmark_gpu(H)
    else:
        gpu_time = float('inf')  # Se não houver GPU, defina um tempo infinito

    # Decisão de execução
    if gpu_available and gpu_time < cpu_time and ram_available > H.size():
        # Executar na GPU
        H_gpu = cuda.to_device(H)
        communities = detect_communities_gpu(H_gpu, must_link, cannot_link, num_communities, other_params)
        communities = cuda.from_device(communities)
    else:
        # Executar na CPU
        communities = detect_communities_cpu(H, must_link, cannot_link, num_communities, other_params)

    return communities

@cuda.jit
def detect_communities_gpu(H_gpu, must_link, cannot_link, num_communities, other_params=None):
    """Implementação CUDA da detecção de comunidades em um hipergrafo."""

    # Inicialização das comunidades
    num_nodes = H_gpu.shape[0]
    communities = cp.random.randint(0, num_communities, size=num_nodes)
    
    # Pré-processamento das restrições de pares
    must_link_gpu = cp.array(must_link)
    cannot_link_gpu = cp.array(cannot_link)

    # Iterações do algoritmo
    prev_modularity = -1  # Inicializa a modularidade anterior com um valor inválido
    for _ in range(MAX_ITERATIONS):
        for node_idx in range(num_nodes):
            # Calcular ganho de modularidade para cada comunidade candidata
            gains = cp.zeros(num_communities)
            for community_idx in range(num_communities):
                gains[community_idx] = calculate_modularity_gain_gpu(
                    H_gpu, node_idx, community_idx, must_link_gpu, cannot_link_gpu, other_params
                )

            # Encontrar a comunidade com o maior ganho
            best_community = cp.argmax(gains)

            # Mover o nó para a melhor comunidade se o ganho for positivo e respeitar as restrições
            if gains[best_community] > 0 and is_move_valid(node_idx, best_community, must_link_gpu, cannot_link_gpu):
                communities[node_idx] = best_community

        # Verificar convergência
        current_modularity = calculate_modularity_gpu(H_gpu, communities)  # Calcula a modularidade atual
        if abs(current_modularity - prev_modularity) < CONVERGENCE_THRESHOLD:  
            break  # Para se a mudança na modularidade for pequena
        prev_modularity = current_modularity

# Função auxiliar para calcular o ganho de modularidade na GPU
@cuda.jit(device=True)
def calculate_modularity_gain_gpu(H_gpu, node_idx, community_idx, must_link_gpu, cannot_link_gpu, other_params):
    """Calcula o ganho de modularidade para um nó se mover para uma comunidade."""

    # Parâmetros (substitua pelos valores reais)
    resolution = other_params.get('resolution', 1.0)
    must_link_weight = other_params.get('must_link_weight', 1.0)
    cannot_link_weight = other_params.get('cannot_link_weight', 1.0)

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

# Função auxiliar para verificar a validade de um movimento
@cuda.jit(device=True)
def is_move_valid(node_idx, community_idx, must_link_gpu, cannot_link_gpu):
    """Verifica se mover um nó para uma comunidade viola alguma restrição de pares.
    """
    for i in range(must_link_gpu.shape[0]):
        if (must_link_gpu[i, 0] == node_idx and communities[must_link_gpu[i, 1]] != community_idx) or \
           (must_link_gpu[i, 1] == node_idx and communities[must_link_gpu[i, 0]] != community_idx):
            return False  # Violação de must-link

    for i in range(cannot_link_gpu.shape[0]):
        if cannot_link_gpu[i, 0] == node_idx and communities[cannot_link_gpu[i, 1]] == community_idx:
            return False  # Violação de cannot-link

    return True  # Movimento válido

# Função auxiliar para calcular a modularidade na GPU
@cuda.jit(device=True)
def calculate_modularity_gpu(H_gpu, communities):
    """Calcula a modularidade da partição atual das comunidades.
    """
    modularity = 0.0
    num_edges = H_gpu.nnz // 2

    for community_idx in range(num_communities):
        community_mask = communities == community_idx
        community_degrees = cp.diff(H_gpu.indptr)[community_mask]
        internal_edges = cp.sum(H_gpu[community_mask, :][:, community_mask]) // 2
        total_degree = cp.sum(community_degrees)
        modularity += internal_edges / num_edges - (total_degree / (2 * num_edges))**2

    return modularity

# Função auxiliar para calcular a modularidade na GPU
@cuda.jit
def detect_communities_gpu(H_gpu, must_link, cannot_link, num_communities, other_params):
    """Calcula a modularidade da partição atual das comunidades."""
    modularity = 0.0
    num_edges = H_gpu.nnz // 2

    for community_idx in range(num_communities):
        community_mask = communities == community_idx
        community_degrees = cp.diff(H_gpu.indptr)[community_mask]
        internal_edges = cp.sum(H_gpu[community_mask, :][:, community_mask]) // 2
        total_degree = cp.sum(community_degrees)
        modularity += internal_edges / num_edges - (total_degree / (2 * num_edges))**2

    return modularity

# Carregar o hipergrafo (substituir pelo seu arquivo)
H = nx.read_hypergraph("dados_hipergrafo.txt")

# Restrições de pares (exemplo)
must_link = [(1, 2), (3, 4)]
cannot_link = [(1, 3), (2, 4)]

# Executar o algoritmo
communities = overlapping_community_detection(H, must_link, cannot_link, num_communities=5)

# Imprimir as comunidades encontradas
print(communities)