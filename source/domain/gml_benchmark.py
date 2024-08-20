import dgl
import time
import torch
import numpy as np
import torch_geometric
import matplotlib.pyplot as plt
from git import Repo
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def create_pytorch_geometric_graph(X, edge_index):
    """
    Cria um grafo PyTorch Geometric a partir dos embeddings e índices de arestas.

    Args:
        X: Tensor com os embeddings dos nós.
        edge_index: Tensor com os índices das arestas.

    Returns:
        Objeto Data do PyTorch Geometric representando o grafo.
    """
    # Converter edge_index para o formato esperado pelo PyTorch Geometric
    edge_index = torch.LongTensor(edge_index)

    # Criar o objeto Data
    data = Data(x=X, edge_index=edge_index)

    return data

def create_dgl_graph(X, edge_index):
    """
    Cria um grafo DGL a partir dos embeddings e índices de arestas.

    Args:
        X: Tensor com os embeddings dos nós.
        edge_index: Tensor com os índices das arestas.

    Returns:
        Objeto DGLGraph representando o grafo.
    """
    # Converter edge_index para o formato esperado pelo DGL
    u, v = edge_index
    g = dgl.graph((u, v))

    # Adicionar os embeddings como features dos nós
    g.ndata['feat'] = X

    return g

def benchmark_operation(graph, library_name, operation_func):
    """
    Realiza o benchmarking de uma operação em um grafo específico.

    Args:
        graph: Objeto representando o grafo (PyTorch Geometric ou DGL).
        library_name: Nome da biblioteca ("PyTorch Geometric" ou "DGL").
        operation_func: Função que implementa a operação a ser realizada no grafo.

    Returns:
        Tempo de execução da operação em segundos.
    """
    start_time = time.time()
    operation_func(graph)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo {library_name}: {elapsed_time:.2f} segundos")
    return elapsed_time

def plot_benchmark_results(results):
    """
    Plota os resultados do benchmarking em um gráfico de barras.

    Args:
        results: Dicionário com os resultados do benchmarking, 
                 no formato {model_name: {"PyTorch Geometric": tempo_pyg, "DGL": tempo_dgl}}.
    """
    labels = list(results.keys())
    pyg_times = [results[label]["PyTorch Geometric"] for label in labels]
    dgl_times = [results[label]["DGL"] for label in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar([i - width/2 for i in x], pyg_times, width, label='PyTorch Geometric')
    rects2 = ax.bar([i + width/2 for i in x], dgl_times, width, label='DGL')

    ax.set_ylabel('Tempo (segundos)')
    ax.set_title('Benchmarking PyTorch Geometric vs. DGL')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()

## Funções para tarefas a comparar entre vários modelos pré-treinados
# generate_embeddings utiliza o modelo pré-treinado para gerar os embeddings dos dados de texto.
# calculate_node_similarity e calculate_edge_similarity calculam a similaridade entre nós e arestas, respectivamente.
# classify_nodes e classify_edges realizam a classificação de nós e arestas usando o algoritmo K-means.

def generate_embeddings(model, data):
    """
    Gera embeddings para os dados de texto usando o modelo pré-treinado.

    Args:
        model: Modelo pré-treinado SentenceTransformer.
        data: Lista de textos a serem incorporados.

    Returns:
        Tensor com os embeddings gerados.
    """
    # Caminho para o arquivo JSON com lista de currículos
    repo = Repo(search_parent_directories=True)
    root_folder = repo.working_tree_dir
    filename = os.path.join("_data", "in_csv", "docents_dict_list.json")

    # Utilizar função vectorize_competences da classe CompetenceExtraction
    competence_extractor = CompetenceExtraction(curricula_file=os.path.join(root_folder, filename))
    embeddings = competence_extractor.vectorize_competences(data)
    return torch.tensor(embeddings)  # Converter para tensor PyTorch


def calculate_node_similarity(embeddings):
    """
    Calcula a similaridade entre os nós com base nos embeddings.

    Args:
        embeddings: Tensor com os embeddings dos nós.

    Returns:
        Matriz de similaridade entre os nós.
    """
    similarity_matrix = cosine_similarity(embeddings.cpu().numpy()) 
    return similarity_matrix

def calculate_edge_similarity(graph):
    """
    Calcula a similaridade entre as arestas com base nos embeddings dos nós conectados.

    Args:
        graph: Objeto representando o grafo (PyTorch Geometric ou DGL).

    Returns:
        Matriz de similaridade entre as arestas.
    """
    if isinstance(graph, torch_geometric.data.Data):
        # PyTorch Geometric
        edge_embeddings = graph.x[graph.edge_index]
        edge_embeddings = edge_embeddings.reshape(-1, 2 * graph.x.shape[1])  # Concatenar embeddings dos nós conectados
    elif isinstance(graph, dgl.DGLGraph):
        # DGL
        edge_embeddings = torch.cat([graph.ndata['feat'][graph.edges()[0]], graph.ndata['feat'][graph.edges()[1]]], dim=1)

    similarity_matrix = cosine_similarity(edge_embeddings.cpu().numpy())
    return similarity_matrix

def classify_nodes(embeddings, num_clusters):
    """
    Classifica os nós em clusters usando K-means.

    Args:
        embeddings: Tensor com os embeddings dos nós.
        num_clusters: Número de clusters desejado.

    Returns:
        Vetor com os rótulos dos clusters para cada nó.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings.cpu().numpy())
    labels = kmeans.labels_
    return labels

def classify_edges(similarity_matrix, num_clusters):
    """
    Classifica as arestas em clusters usando K-means com base na matriz de similaridade.

    Args:
        similarity_matrix: Matriz de similaridade entre as arestas.
        num_clusters: Número de clusters desejado.

    Returns:
        Vetor com os rótulos dos clusters para cada aresta.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(similarity_matrix)
    labels = kmeans.labels_
    return labels