import os
import json
import GPUtil
import numpy as np

from gml_embeddings_analyser import EmbeddingsMulticriteriaAnalysis
from sentence_transformers import SentenceTransformer
from retry import retry
from git import Repo

def print_gpu_memory():
    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print(f"GPU {i}: Carga: {gpu.load*100}% | {gpu.name} | memória utilizada: {gpu.memoryUsed}/{gpu.memoryTotal} MB")

@retry(exceptions=Exception, tries=3, delay=1, backoff=2)
def gerar_resultados_clustering(embeddings_dict, analise):
    """Gera os resultados do clustering com tentativas de retry."""
    try:
        print(f'\nIniciando tentativas de geração de avaliação...')
        print_gpu_memory()
        resultados = analise.evaluate_clustering_cpu(embeddings_dict)
    except Exception as e:
        print(f"Erro na função gerar_resultados_clustering: {e}")
        raise  # Re-raise a exceção para que o retry funcione
    
    return resultados

def salvar_resultados(resultados):
    """
    Autor: Marcos Aires (Nov.2024)
    Salva os resultados em um arquivo JSON local.
    """
    try:
        # Informar caminho para arquivo usando raiz do repositório Git como referência
        repo = Repo(search_parent_directories=True)
        root_folder = repo.working_tree_dir
        folder_data_output = os.path.join(str(root_folder), '_embeddings')
        filename = 'resultados.json'
        pathfilename = os.path.join(folder_data_output, filename)

        # Criar a pasta _embeddings se ela não existir
        os.makedirs(folder_data_output, exist_ok=True)

        # Salvar o dicionário de resultados em arquivo JSON
        with open(pathfilename, 'w') as f:
            json.dump(resultados, f, indent=4)

        # Imprimir o nome do arquivo e o número de modelos
        print(f"Arquivo de resultados salvo: {filename}")
        print(f"Número de modelos avaliados: {len(resultados)}")

    except Exception as e:
        print(f"Erro ao salvar os resultados: {e}")

def salvar_resultados_cpu(resultados, filename="resultados_clustering.json"):
    """
    Salva os resultados da avaliação de clustering em um arquivo JSON.
    """
    try:
        # Converte os valores float32 para float64
        for model_name in resultados:
            for algorithm in resultados[model_name]:
                for result in resultados[model_name][algorithm]["resultados"]:
                    for metric in result:
                        # Correção: Passar a classe np.float32
                        if np.float32(result[metric]) is np.float32:   
                            result[metric] = float(result[metric])

        with open(filename, 'w') as f:
            json.dump(resultados, f, indent=4)
        print(f"Resultados salvos em: {filename}")
    except Exception as e:
        print(f"Erro ao salvar os resultados: {e}")

def load_resultados(filename="resultados.json"):
    """
    Autor: Marcos Aires (Nov.2024)
    Carrega os resultados de um arquivo JSON local.
    """
    # Informar caminho para arquivo JSON usando raiz do repositório Git como referência
    repo = Repo(search_parent_directories=True)
    root_folder = repo.working_tree_dir
    folder_data_output = os.path.join(str(root_folder), '_embeddings')
    pathfilename = os.path.join(folder_data_output, filename)
    
    # Carregar o dicionário de resultados de arquivo JSON
    with open(pathfilename, 'r') as f:
        resultados = json.load(f)

    if not isinstance(resultados, dict):
        raise TypeError("O arquivo JSON não contém um dicionário.")

    return resultados

## Análise multicritério da qualidade de clustering com diferentes modelos de embeedings

# Definir os nomes de modelo do SentenceTransformer a serem comparados
model_names = [
    'sentence-transformers/gtr-t5-large',
    'distiluse-base-multilingual-cased-v2',
    'paraphrase-multilingual-MiniLM-L12-v2',
    'all-mpnet-base-v2',
    'all-distilroberta-v1',
    'all-MiniLM-L6-v2',
]

# Criar uma instância da classe EmbeddingsMulticriteriaAnalysis
analise = EmbeddingsMulticriteriaAnalysis(
    model_names=model_names,
    models= [SentenceTransformer(model_name) for model_name in model_names]
)

## Carregar os embeddings previamente gerados (em caso de erro na avaliação a seguir)
embeddings_dict = analise.load_embeddings_dict("embeddings_funding.pt")

# Chamar a função com retry
resultados = gerar_resultados_clustering(embeddings_dict, analise)

# Salvar os resultados
salvar_resultados_cpu(resultados)