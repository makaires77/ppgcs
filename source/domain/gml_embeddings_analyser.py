import os
import gc
import ast
import time
import cudf
import cuml.metrics

import torch
import string
import jinja2
import logging
import warnings
import traceback
import unicodedata
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoModel
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# Avaliação de clustering somente rodam em CPU
# from sklearn.cluster import KMeans, DBSCAN, HDBSCAN

# Avaliação de clustering para rodar em GPU
import cuml
import cupy as cp
from cuml.cluster import KMeans, DBSCAN, HDBSCAN
from cuml.metrics.cluster import silhouette_score

from tqdm.auto import tqdm
from git import Repo
tqdm.pandas()

from gml_funding_preprocessor import ENPreprocessor, BRPreprocessor

# Configurar o logging (opcional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning or UserWarning)
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")


class EmbeddingsMulticriteriaAnalysis:

    def __init__(self, model_names, models, algorithms=[KMeans, DBSCAN, HDBSCAN], pesos=None, n_rodadas=1, n_splits=5):
        """
        Autor Marcos Aires (Nov.2024)
        Inicializa a classe com os embeddings, algoritmos de clustering, pesos para cada critério,
        número de rodadas e número de splits para validação cruzada.

        Args:
            data: O dataframe contendo os dados.
            model_names: Uma lista com os nomes dos modelos de embedding.
            models: Uma lista com os modelos de embedding.
            algorithms: Uma lista de classes de algoritmos de clustering a serem usados.
            pesos: Um dicionário com os pesos para cada critério.
                   As chaves devem ser 'silhouette', 'calinski_harabasz', 'davies_bouldin' e 'tempo'.
                   Se None, todos os critérios terão o mesmo peso.
            n_rodadas: Número de rodadas para calcular a média e o desvio padrão das métricas.
            n_splits: Número de splits para validação cruzada.
        """
        self.model_names = model_names
        self.models = models
        self.embeddings = {}  # Para armazenar os embeddings gerados por cada modelo
        self.algorithms = algorithms
        self.n_rodadas = n_rodadas
        self.n_splits = n_splits
        self.en_preprocessor = ENPreprocessor()  # Criar instância do pré-processador para inglês
        self.br_preprocessor = BRPreprocessor()  # Criar instância do pré-processador para português
        self.data = self.create_embedding_column()  # Carregar dataframe arquivo de fomento
        
        self.show_models_info() # Exibir dados de cada modelo

        # self.generate_embeddings()
        self.generate_embeddings_batch()
        # self.generate_embeddings_batch_no_grad()
        # self.generate_embeddings_optimzed()
        
        self.resultados = self.evaluate_clustering()

        # Define os pesos dos critérios
        if pesos is None:
            self.pesos = {
                'silhouette': 0.25,
                'calinski_harabasz': 0.25,
                'davies_bouldin': 0.25,
                'tempo': 0.25
            }
        else:
            self.pesos = pesos


    def show_models_info(self):
        """
        Exibe informações sobre os modelos pré-treinados, 
        como o número de features, tipo, tamanho e outras características.
        """
        from huggingface_hub import hf_hub_download
        import os

        print()
        print("-"*75)
        for model in self.models:
            try:
                # Obter informações do modelo
                print(f"{type(model.model_card_data.keys())}")
                print(f"{model.get('base_model')}")
                print(f"Comprimento Máximo: {model.get_max_seq_length()}")
                print(f"Número de features: {model.get_sentence_embedding_dimension()}")
                print("-"*75)
            except Exception as e:
                pass

    def create_embedding_column(self, use_cudf=True):
        """
        Creates the 'texto_para_embedding' column in the df_fomento dataframe by combining selected data and applying preprocessing.

        Args:
            use_cudf: Whether to use cuDF for DataFrame operations (default: True)

        Returns:
            The updated dataframe with the 'texto_para_embedding' column.
        """

        # Informar caminho para arquivo CSV usando raiz do repositório Git como referência
        repo = Repo(search_parent_directories=True)
        root_folder = repo.working_tree_dir
        folder_data_output = os.path.join(root_folder, '_data', 'out_json') # type: ignore
        filename = 'df_fomento_geral.csv'
        pathfilename = os.path.join(folder_data_output, filename)
        pdf = pd.read_csv(pathfilename, header=0)

        def convert_to_dict(text):
            try:
                return ast.literal_eval(text)
            except ValueError:
                return None

        def generate_embedding_text_helper(row, cols_geninfo, cols_details, cols_moreinf):
            # Extrair dados das colunas específicas de interesse
            gen_info_text = ' '.join([str(row[col]) for col in cols_geninfo])
            details_text = ' '.join([str(row['detalhes'][col]) for col in cols_details if col in row['detalhes']])
            more_info_text = ' '.join([str(row['detalhes'][col]) for col in cols_moreinf if col in row['detalhes']])

            # Combinar textos de interesse em string única
            combined_text = f"{gen_info_text} {details_text} {more_info_text}"
            return combined_text

        pdf['detalhes'] = pdf['detalhes'].apply(convert_to_dict)

        # Definir as colunas para a geração dos embeddings
        cols_geninfo = ['financiadora','titulo','palavras-chave']
        cols_details = ['elegibilidade','descricao','valorfinanciado','datalimite']
        cols_moreinf = ['formasolicitacao']

        # Aplciar função de suporte para cada linha do dataframe de editais
        pdf['texto_para_embedding'] = pdf.apply(
            lambda row: generate_embedding_text_helper(
                row,
                cols_geninfo,
                cols_details,
                cols_moreinf
            ),
            axis=1
        )

        # Converter o DataFrame do pandas para cuDF (se o parâmetro de entrada for True)
        df = cudf.from_pandas(pdf) if use_cudf else pdf

        return df

    def detect_predominant_language(self, sentences):
        """
        Detecta o idioma predominante em uma lista de sentenças.
        """
        en_preprocessor = ENPreprocessor()
        idiomas = [en_preprocessor.detect_language(sentence) for sentence in sentences]
        idioma_predominante = max(set(idiomas), key=idiomas.count)
        return idioma_predominante

    ## Tentativa de carregar também modelo e dados para GPU dando erro
    # def generate_embeddings(self):
    #     """
    #     Gera embeddings para os textos usando os modelos especificados, processando em lotes.
    #     """
    #     for model_name, model in zip(self.model_names, self.models):
    #         try:
    #             sentences = self.data['texto_para_embedding'].to_arrow().to_pylist()

    #             # Limpa a memória da GPU
    #             gc.collect()
    #             torch.cuda.empty_cache()

    #             if torch.cuda.is_available():
    #                 device = torch.device('cuda')
    #                 print(f"Gerando embeedings em GPU com modelo {model_name}.")
    #             else:
    #                 device = torch.device('cpu')
    #                 print(f"GPU não disponível. Gerando embeedings em CPU com modelo {model_name}.")

    #             inicio = time.time()

    #             # Instanciar os pré-processadores dentro da função
    #             en_preprocessor = ENPreprocessor()
    #             br_preprocessor = BRPreprocessor()

    #             # Detectar o idioma predominante (apenas uma vez)
    #             idioma_predominante = self.detect_predominant_language(sentences)

    #             batch_size = 128  # Defina o tamanho do lote
    #             embeddings_list = []
    #             for i in tqdm(range(0, len(sentences), batch_size), desc="Processando sentenças", unit=f"batch (batch_size {batch_size})"):
    #                 batch = sentences[i: i + batch_size]
    #                 processed_sentences = []
    #                 for sentence in batch:
    #                     if idioma_predominante == 'en':
    #                         processed_sentence = en_preprocessor.preprocess_text(sentence)
    #                     elif idioma_predominante == 'pt':
    #                         processed_sentence = br_preprocessor.preprocess_text(sentence)
    #                     else:
    #                         processed_sentence = sentence  # Ou aplicar um pré-processamento padrão
    #                     processed_sentences.append(processed_sentence)

    #                 # Gerar embeddings em lote
    #                 embeddings_batch = model.encode(processed_sentences, convert_to_tensor=True, device=device, batch_size=batch_size)
    #                 embeddings_list.append(embeddings_batch)

    #             fim = time.time()

    #             # Concatenar os embeddings de todos os lotes
    #             embeddings = torch.cat(embeddings_list, dim=0)
    #             self.embeddings[model_name] = embeddings.cpu().numpy()

    #             # Limpar o cache da GPU
    #             gc.collect()
    #             torch.cuda.empty_cache()

    #         except Exception as e:
    #             print(f"Erro ao gerar embeddings com modelo {model_name}: {e}")


    ## funcionando antes da otimização de uso da VRAM
    def generate_embeddings(self):
        """
        Gera embeddings para os textos usando os modelos especificados.
        """
        for model_name, model in zip(self.model_names, self.models):
            try:
                sentences = self.data['texto_para_embedding'].to_arrow().to_pylist()   # type: ignore

                # Limpa a memória da GPU
                gc.collect()
                torch.cuda.empty_cache()

                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    print(f"\nGerando embeedings em GPU com modelo {model_name}.")
                else:
                    device = torch.device('cpu')
                    print(f"GPU não disponível. Gerando embeedings em CPU com modelo {model_name}.")

                inicio = time.time()
                
                # Pré-processar cada texto de acordo com o idioma
                processed_sentences = []
                for sentence in tqdm(sentences, desc="Processando sentenças", unit="sentença"):
                    idioma = self.en_preprocessor.detect_language(sentence)  # Detectar o idioma
                    if idioma == 'en':
                        processed_sentence = self.en_preprocessor.preprocess_text(sentence)
                    elif idioma == 'pt':
                        processed_sentence = self.br_preprocessor.preprocess_text(sentence)
                    else:
                        # Tratar caso o idioma não seja detectado ou suportado
                        processed_sentence = sentence  # Ou aplicar um pré-processamento padrão
                    processed_sentences.append(processed_sentence)

                embeddings = model.encode(processed_sentences, convert_to_tensor=True, device=device)
                self.embeddings[model_name] = embeddings.cpu().numpy()

                # Calcula o tempo de execução em segundos
                fim = time.time()
                tempo_execucao_segundos = fim - inicio
                horas, resto = divmod(tempo_execucao_segundos, 3600)
                minutos, segundos = divmod(resto, 60)
                print(f"Tempo de execução: {int(horas):02d}:{int(minutos):02d}:{int(segundos):02d} para modelo {model_name}\n")

            except Exception as e:
                print(f"    Erro ao gerar embeddings com modelo {model_name}: {e}")

    def generate_embeddings_batch(self):
        """
        Gera embeddings para os textos usando os modelos especificados, processando em lotes.
        """
        for model_name, model in zip(self.model_names, self.models):
            try:
                sentences = self.data['texto_para_embedding'].to_arrow().to_pylist() # type: ignore

                # Limpa a memória da GPU antes de cada lote
                gc.collect()
                torch.cuda.empty_cache()

                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    model = model.to(device)  # Transferir o modelo para a GPU (nem todos modelos)
                    print(f"\nGerando embeedings em GPU com modelo {model_name}.")
                else:
                    device = torch.device('cpu')
                    print(f"GPU não disponível. Gerando embeedings em CPU com modelo {model_name}.")

                inicio = time.time()

                # Pré-processar cada texto de acordo com o idioma, em lotes
                batch_size = 512  # Defina o tamanho do lote
                processed_sentences = []
                for i in tqdm(range(0, len(sentences), batch_size), desc="Processando sentenças", unit=f"batch (batch_size {batch_size})"):
                    batch = sentences[i: i + batch_size]
                    for sentence in batch:
                        idioma = self.en_preprocessor.detect_language(sentence)  # Detectar o idioma
                        if idioma == 'en':
                            processed_sentence = self.en_preprocessor.preprocess_text(sentence)
                        elif idioma == 'pt':
                            processed_sentence = self.br_preprocessor.preprocess_text(sentence)
                        else:
                            # Tratar caso o idioma não seja detectado ou suportado
                            processed_sentence = sentence  # Ou aplicar um pré-processamento padrão
                        processed_sentences.append(processed_sentence)

                    # Gerar embeddings em lote
                    embeddings = model.encode(processed_sentences, convert_to_tensor=True, device=device, batch_size=batch_size)
                    processed_sentences = []  # Limpar a lista para o próximo lote

                    # Limpar a memória da GPU após cada lote
                    gc.collect()
                    torch.cuda.empty_cache()

                self.embeddings[model_name] = embeddings.cpu().numpy()

                # Calcula o tempo de execução em segundos
                fim = time.time()
                tempo_execucao_segundos = fim - inicio
                horas, resto = divmod(tempo_execucao_segundos, 3600)
                minutos, segundos = divmod(resto, 60)
                print(f"Tempo de execução: {int(horas):02d}:{int(minutos):02d}:{int(segundos):02d} para modelo {model_name}\n")

            except Exception as e:
                print(f"Erro ao gerar embeddings com modelo {model_name}: {e}")


    def generate_embeddings_batch_no_grad(self):
        """
        Gera embeddings para os textos usando os modelos especificados, 
        processando em lotes e otimizando o uso da VRAM.
        """
        for model_name, model in zip(self.model_names, self.models):
            try:
                sentences = self.data['texto_para_embedding'].to_arrow().to_pylist() # type: ignore

                # Limpa a memória da GPU
                gc.collect()
                torch.cuda.empty_cache()

                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    model = model.to(device)  # Transferir modelo para GPU (quando possível)
                    print(f"\nGerando embeddings em GPU com modelo {model_name}")
                else:
                    device = torch.device('cpu')
                    print(f"GPU não disponível. Gerando embeddings em CPU com modelo {model_name}")

                inicio = time.time()

                batch_size = 1024  # Aumentar o batch_size (ajuste conforme a VRAM disponível)
                num_batches = len(sentences) // batch_size + (len(sentences) % batch_size > 0)

                # Pré-alocar o tensor de embeddings na GPU
                embeddings = torch.empty((len(sentences), model.get_sentence_embedding_dimension()), 
                                        dtype=torch.float32, device=device)

                with torch.no_grad():  # Desabilitar o cálculo de gradientes
                    for i in tqdm(range(num_batches), desc="Processando sentenças", unit=f"batch (batch_size {batch_size})"):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, len(sentences))
                        batch = sentences[start_idx:end_idx]

                        processed_sentences = []
                        for sentence in batch:
                            idioma = self.en_preprocessor.detect_language(sentence)  # Detectar o idioma
                            if idioma == 'en':
                                processed_sentence = self.en_preprocessor.preprocess_text(sentence)
                            elif idioma == 'pt':
                                processed_sentence = self.br_preprocessor.preprocess_text(sentence)
                            else:
                                # Tratar caso o idioma não seja detectado ou suportado
                                processed_sentence = sentence  # Ou aplicar um pré-processamento padrão
                            processed_sentences.append(processed_sentence)

                        # Gerar embeddings em lote e armazenar no tensor pré-alocado
                        embeddings[start_idx:end_idx] = model.encode(
                            processed_sentences, convert_to_tensor=True, device=device, batch_size=batch_size
                        )

                # Manter os embeddings na GPU (se possível)
                # self.embeddings[model_name] = embeddings  
                
                # OU Transferir para CPU, apenas se necessário
                self.embeddings[model_name] = embeddings.cpu().numpy()  

                # Calcula o tempo de execução em segundos
                fim = time.time()
                tempo_execucao_segundos = fim - inicio
                horas, resto = divmod(tempo_execucao_segundos, 3600)
                minutos, segundos = divmod(resto, 60)
                print(f"Tempo de execução: {int(horas):02d}:{int(minutos):02d}:{int(segundos):02d} para modelo {model_name}\n")

            except Exception as e:
                print(f"Erro ao gerar embeddings com modelo {model_name}: {e}")


    ## Otimizado
    def generate_embeddings_optimzed(self):
        """
        Gera embeddings para os textos usando os modelos especificados, processando em lotes.
        """
        for model_name, model in zip(self.model_names, self.models):
            try:
                sentences = self.data['texto_para_embedding'].to_arrow().to_pylist() #type: ignore

                # Imprimir informações sobre o uso da memória
                print(f"Uso de memória da GPU antes da limpeza: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

                # Limpa a memória da GPU
                gc.collect()
                torch.cuda.empty_cache()

                print(f"Uso de memória da GPU após a limpeza: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    print(f"\nGerando embeddings em GPU com modelo {model_name}")
                else:
                    device = torch.device('cpu')
                    print(f"GPU não disponível. Gerando embeddings em CPU com modelo {model_name}")

                inicio = time.time()

                # Instanciar os pré-processadores dentro da função
                en_preprocessor = ENPreprocessor()
                br_preprocessor = BRPreprocessor()

                # Detectar o idioma predominante (apenas uma vez)
                idioma_predominante = self.detect_predominant_language(sentences)

                batch_size = 128  # Defina o tamanho do lote
                embeddings_list = []
                for i in tqdm(range(0, len(sentences), batch_size), desc="Processando sentenças", unit=f"batch (batch_size {batch_size})"):
                    batch = sentences[i: i + batch_size]
                    processed_sentences = []
                    for sentence in batch:
                        if idioma_predominante == 'en':
                            processed_sentence = en_preprocessor.preprocess_text(sentence)
                        elif idioma_predominante == 'pt':
                            processed_sentence = br_preprocessor.preprocess_text(sentence)
                        else:
                            processed_sentence = sentence  # Ou aplicar um pré-processamento padrão
                        processed_sentences.append(processed_sentence)

                    # Gerar embeddings em lote
                    print(f"Gerando embeddings para o lote {i // batch_size + 1} de {len(sentences) // batch_size + 1}")
                    embeddings_batch = model.encode(processed_sentences, convert_to_tensor=True, device=device, batch_size=batch_size)

                    # Imprimir informações sobre os embeddings gerados
                    print(f"Tamanho do lote de embeddings: {embeddings_batch.shape}")
                    print(f"Dispositivo dos embeddings: {embeddings_batch.device}")

                    embeddings_list.append(embeddings_batch)

                    # Imprimir informações sobre o uso da memória
                    print(f"Uso de memória da GPU após o lote {i // batch_size + 1}: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

                # Concatenar os embeddings de todos os lotes
                embeddings = torch.cat(embeddings_list, dim=0)
                self.embeddings[model_name] = embeddings.cpu().numpy()

                # Calcula o tempo de execução em segundos
                fim = time.time()
                tempo_execucao_segundos = fim - inicio
                horas, resto = divmod(tempo_execucao_segundos, 3600)
                minutos, segundos = divmod(resto, 60)
                print(f"Tempo de execução: {int(horas):02d}:{int(minutos):02d}:{int(segundos):02d} para modelo {model_name}\n")

                # Limpar o cache da GPU
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Memória em uso na GPU após limpeza final: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

            except Exception as e:
                print(f"Erro ao gerar embeddings com modelo {model_name}: {e}")


    def evaluate_clustering(self):
        """
        Avalia o desempenho dos embeddings em tarefas de clustering
        usando diferentes algoritmos do cuML, múltiplas rodadas e 
        validação cruzada, e mede o tempo de execução de cada algoritmo.
        """
        print("Iniciando avaliação de clustering com cuML...")

        resultados = {}
        for model_name, embeddings in self.embeddings.items():
            print(f"Avaliando modelo: {model_name}")
            resultados[model_name] = {}

            # Converter os embeddings para arrays do CuPy
            embeddings_cp = cp.array(embeddings)  

            for algorithm in self.algorithms:
                resultados[model_name][algorithm.__name__] = {"medias": [], "desvios": [], "tempo": []}
                resultados_algoritmo = []
                tempos_execucao = []

                for _ in range(self.n_rodadas):
                    skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
                    resultados_split = []

                    # Usar embeddings_cp (array do CuPy) no StratifiedKFold
                    for train_index, test_index in skf.split(embeddings_cp, np.zeros(len(embeddings_cp))):  
                        X_train, X_test = embeddings_cp[train_index], embeddings_cp[test_index]
                        
                        # Inicializar o modelo de clustering do cuML
                        if algorithm.__name__ == "KMeans":
                            clustering_model = cuml.KMeans(n_clusters=8)  # Ajustar o número de clusters
                        elif algorithm.__name__ == "DBSCAN":
                            clustering_model = cuml.DBSCAN(eps=0.5, min_samples=5)  # Ajustar os parâmetros
                        elif algorithm.__name__ == "HDBSCAN":
                            clustering_model = cuml.HDBSCAN(min_cluster_size=5)  # Ajustar os parâmetros

                        # Mede o tempo de execução do algoritmo
                        inicio = time.time()
                        cluster_labels = clustering_model.fit_predict(X_train)
                        fim = time.time()
                        tempo_execucao = fim - inicio
                        tempos_execucao.append(tempo_execucao)

                        # Avalia apenas nos dados de teste
                        try:
                            # Converter para arrays do CuPy para as métricas do cuML
                            X_test_cp = cp.array(X_test)
                            cluster_labels_cp = cp.array(cluster_labels)

                            # accuracy_score = accuracy_score(y_true, y_pred)
                            silhouette_avg = silhouette_score(X_test_cp, cluster_labels_cp)
                            calinski_harabasz = calinski_harabasz_score(X_test_cp.get(), cluster_labels_cp.get())
                            davies_bouldin = davies_bouldin_score(X_test_cp.get(), cluster_labels_cp.get())

                            resultados_split.append({
                                "silhouette": silhouette_avg,
                                "calinski_harabasz": calinski_harabasz,
                                "davies_bouldin": davies_bouldin
                            })
                        # except ValueError:
                        except Exception as e:
                            print(f"Erro ao calcular métricas: {e}")
                            print(f"{algorithm.__name__}, modelo {model_name}. Pulando esta iteração.")

                    resultados_algoritmo.append(resultados_split)

                # Calcula a média e o desvio padrão das métricas
                resultados_algoritmo = cp.array(resultados_algoritmo)  # Usar CuPy para calcular a média e o desvio padrão
                medias = cp.mean(resultados_algoritmo, axis=0)
                desvios = cp.std(resultados_algoritmo, axis=0)

                resultados[model_name][algorithm.__name__] = {
                    "medias": medias.tolist(),  # Converter de volta para lista para compatibilidade
                    "desvios": desvios.tolist(),  # Converter de volta para lista para compatibilidade
                    "tempo": np.mean(tempos_execucao)  # Adiciona o tempo médio de execução
                }

        return resultados

    def evaluate_clustering_cpu(self):
        """
        Avalia o desempenho dos embeddings em tarefas de clustering
        usando diferentes algoritmos, múltiplas rodadas e validação cruzada,
        e mede o tempo de execução de cada algoritmo.
        """
        print("Iniciando avaliação de clustering...")

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        resultados = {}
        for model_name, embeddings in self.embeddings.items():
            print(f"Avaliando modelo: {model_name}")
            resultados[model_name] = {}

            # Converter embeddings para array do NumPy ANTES do StratifiedKFold
            embeddings_np = torch.tensor(embeddings).cpu().numpy()  

            for algorithm in self.algorithms:
                resultados[model_name][algorithm.__name__] = {"medias": [], "desvios": [], "tempo": []}
                resultados_algoritmo = []
                tempos_execucao = []

                for _ in range(self.n_rodadas):
                    skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
                    resultados_split = []

                    # Usar embeddings_np (array do NumPy) no StratifiedKFold
                    for train_index, test_index in skf.split(embeddings_np, np.zeros(len(embeddings_np))):
                        # Converter para tensor e transferir para a GPU
                        X_train, X_test = torch.tensor(embeddings_np[train_index]).to(device), torch.tensor(embeddings_np[test_index]).to(device)  
                        clustering_model = algorithm()

                        # Transferir o modelo para a GPU (se possível)
                        if hasattr(clustering_model, "to"):
                            clustering_model = clustering_model.to(device)

                        # Medir o tempo de execução do algoritmo
                        inicio = time.time()

                        # Converter para NumPy para o scikit-learn
                        cluster_labels = clustering_model.fit_predict(X_train.cpu().numpy())  
                        fim = time.time()
                        tempo_execucao = fim - inicio
                        tempos_execucao.append(tempo_execucao)

                        # Converter os tensores para arrays do NumPy
                        X_test_np = X_test.cpu().numpy()
                        cluster_labels_np = cluster_labels

                        # Avaliar apenas nos dados de teste
                        try:
                            silhouette_avg = silhouette_score(X_test_np, cluster_labels_np[test_index])
                            calinski_harabasz = calinski_harabasz_score(X_test_np, cluster_labels_np[test_index])
                            davies_bouldin = davies_bouldin_score(X_test_np, cluster_labels_np[test_index])

                            resultados_split.append({
                                "silhouette": silhouette_avg,
                                "calinski_harabasz": calinski_harabasz,
                                "davies_bouldin": davies_bouldin
                            })
                        except ValueError:
                            print(f"Erro ao calcular métricas para {algorithm.__name__} com modelo {model_name}. Pulando esta iteração.")

                    resultados_algoritmo.append(resultados_split)

                # Calcula a média e o desvio padrão das métricas
                resultados_algoritmo = np.array(resultados_algoritmo)
                medias = np.mean(resultados_algoritmo, axis=0)
                desvios = np.std(resultados_algoritmo, axis=0)

                resultados[model_name][algorithm.__name__] = {
                    "medias": medias,
                    "desvios": desvios,
                    "tempo": np.mean(tempos_execucao)  # Adiciona o tempo médio de execução
                }

                # Liberar memória não utilizada
                torch.cuda.empty_cache()

        return resultados

    def calcular_pontuacao_multicriterio(self):
        """
        Calcula a pontuação multicritério para cada algoritmo, combinando as métricas com os pesos.
        """
        pontuacoes = {}
        for model_name, model_results in self.resultados.items():
            pontuacoes[model_name] = {}
            max_valor = 0
            for algoritmo, resultados in model_results.items():
                medias = resultados["medias"]
                pontuacao = 0
                for i, metrica in enumerate(['silhouette', 'calinski_harabasz', 'davies_bouldin']):
                    valor = medias[i]
                    # Normaliza as métricas para ficarem na mesma escala (0 a 1)
                    if metrica == "silhouette":
                        valor_normalizado = (valor + 1) / 2  # Silhouette varia de -1 a 1
                    elif metrica == "davies_bouldin":
                        valor_normalizado = 1 / (valor + 1e-6)  # Davies-Bouldin é menor quanto melhor
                    elif metrica == "calinski_harabasz":
                        max_valor = np.max([resultados["medias"][i] for model_results in self.resultados.values() for resultados in model_results.values()])
                        valor_normalizado = valor / max_valor
                    pontuacao += self.pesos[metrica] * valor_normalizado

                # Adiciona o tempo de execução à pontuação
                tempo_execucao = resultados["tempo"]
                tempo_normalizado = 1 / (tempo_execucao + 1e-6)  # Tempo é menor quanto melhor
                pontuacao += self.pesos["tempo"] * tempo_normalizado

                pontuacoes[model_name][algoritmo] = pontuacao
                print(f"Pontuações: {pontuacoes}")

        return pontuacoes

    def escolher_melhor_modelo(self):
            """
            Escolhe o modelo com a maior pontuação multicritério.
            """
            print("Resultados:", self.resultados)
            pontuacoes = self.calcular_pontuacao_multicriterio()
            print("Pontuações:", pontuacoes)  # Adicione este print
            melhor_modelo = max(pontuacoes, key=lambda model_name: max(pontuacoes[model_name].values()))
            return melhor_modelo

    def generate_report(self):
        """
        Generates a report with the benchmarking results.
        """
        try:
            # 1. Preparar os dados para o relatório
            report_data = []
            for model_name, model_results in self.resultados.items():
                for algorithm_name, results in model_results.items():
                    medias = results["medias"]
                    desvios = results["desvios"]
                    tempo = results["tempo"]
                    report_data.append({
                        "Modelo": model_name,
                        "Algoritmo": algorithm_name,
                        "Silhouette": f"{medias[0]:.3f} ± {desvios[0]:.3f}",
                        "Calinski-Harabasz": f"{medias[1]:.3f} ± {desvios[1]:.3f}",
                        "Davies-Bouldin": f"{medias[2]:.3f} ± {desvios[2]:.3f}",
                        "Tempo (s)": f"{tempo:.2f}"
                    })

            # 2. Criar o dataframe do relatório
            df_report = pd.DataFrame(report_data)

            # 3. Salvar o relatório em formato CSV
            # Obter o caminho do diretório do repositório Git
            repo = Repo(search_parent_directories=True)
            root_folder = repo.working_tree_dir
            folder_data_output = os.path.join(str(root_folder), '_reports')
            os.makedirs(folder_data_output, exist_ok=True)  # Criar a pasta se não existir
            report_filename = os.path.join(folder_data_output, 'benchmark_report.csv')
            df_report.to_csv(report_filename, index=False)

            # 4. Gerar os gráficos do relatório
            self.generate_report_charts(df_report)

        except Exception as e:
            print(f"Erro ao gerar o relatório: {e}")
            traceback.print_exc()

    def generate_report_charts(self, df_report):
        """
        Generates charts for the benchmarking report.

        Args:
            df_report: The dataframe containing the report data.
        """
        try:
            # 1. Configurar o estilo dos gráficos
            sns.set_theme(style="whitegrid")

            # 2. Criar os gráficos
            # Gráfico de barras para Silhouette Score
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Modelo", y="Silhouette", hue="Algoritmo", data=df_report)
            plt.title("Silhouette Score")
            plt.ylabel("Silhouette Score")
            plt.xticks(rotation=45)
            plt.show()

            # Gráfico de barras para Calinski-Harabasz Index
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Modelo", y="Calinski-Harabasz", hue="Algoritmo", data=df_report)
            plt.title("Calinski-Harabasz Index")
            plt.ylabel("Calinski-Harabasz Index")
            plt.xticks(rotation=45)
            plt.show()

            # Gráfico de barras para Davies-Bouldin Index
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Modelo", y="Davies-Bouldin", hue="Algoritmo", data=df_report)
            plt.title("Davies-Bouldin Index")
            plt.ylabel("Davies-Bouldin Index")
            plt.xticks(rotation=45)
            plt.show()

            # Gráfico de barras para Tempo de Execução
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Modelo", y="Tempo (s)", hue="Algoritmo", data=df_report)
            plt.title("Tempo de Execução (s)")
            plt.ylabel("Tempo (s)")
            plt.xticks(rotation=45)
            plt.show()

        except Exception as e:
            print(f"Erro ao gerar os gráficos do relatório: {e}")
            traceback.print_exc()

    ## antes da otimização
    # def evaluate_clustering(self):
    #     """
    #     Avalia o desempenho dos embeddings em tarefas de clustering
    #     usando diferentes algoritmos, múltiplas rodadas e validação cruzada,
    #     e mede o tempo de execução de cada algoritmo.
    #     """
    #     print("Iniciando avaliação de clustering...")
    #     resultados = {}
    #     for model_name, embeddings in self.embeddings.items():
    #         print(f"Avaliando modelo: {model_name}")
    #         resultados[model_name] = {}
    #         for algorithm in self.algorithms:
    #             resultados[model_name][algorithm.__name__] = {"medias": [], "desvios": [], "tempo": []}
    #             resultados_algoritmo = []
    #             tempos_execucao = []
    #             for _ in range(self.n_rodadas):
    #                 skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
    #                 resultados_split = []
    #                 for train_index, test_index in skf.split(embeddings, np.zeros(len(embeddings))):
    #                     X_train, X_test = embeddings[train_index], embeddings[test_index]
    #                     clustering_model = algorithm()

    #                     # Mede o tempo de execução do algoritmo
    #                     inicio = time.time()
    #                     cluster_labels = clustering_model.fit_predict(X_train)
    #                     fim = time.time()
    #                     tempo_execucao = fim - inicio
    #                     tempos_execucao.append(tempo_execucao)

    #                     # Avalia apenas nos dados de teste
    #                     try:
    #                         silhouette_avg = silhouette_score(X_test, cluster_labels[test_index])
    #                         calinski_harabasz = calinski_harabasz_score(X_test, cluster_labels[test_index])
    #                         davies_bouldin = davies_bouldin_score(X_test, cluster_labels[test_index])

    #                         resultados_split.append({
    #                             "silhouette": silhouette_avg,
    #                             "calinski_harabasz": calinski_harabasz,
    #                             "davies_bouldin": davies_bouldin
    #                         })
    #                     except ValueError:
    #                         print(f"Erro ao calcular métricas para {algorithm.__name__} com modelo {model_name}. Pulando esta iteração.")

    #                 resultados_algoritmo.append(resultados_split)

    #             # Calcula a média e o desvio padrão das métricas
    #             resultados_algoritmo = np.array(resultados_algoritmo)
    #             medias = np.mean(resultados_algoritmo, axis=0)
    #             desvios = np.std(resultados_algoritmo, axis=0)

    #             resultados[model_name][algorithm.__name__] = {
    #                 "medias": medias,
    #                 "desvios": desvios,
    #                 "tempo": np.mean(tempos_execucao)  # Adiciona o tempo médio de execução
    #             }
    #     return resultados