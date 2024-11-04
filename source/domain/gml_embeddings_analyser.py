import os
import gc
import ast
import time
import cudf
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

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm
from git import Repo
tqdm.pandas()

# from transformers.tokenization_utils_base import TruncationStrategy
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import pipeline, TranslationPipeline
# from sentence_transformers import SentenceTransformer
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from langdetect import detect
# import contextualSpellCheck

from gml_funding_preprocessor import ENPreprocessor, BRPreprocessor

# Configurar o logging (opcional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning or UserWarning)
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")


class EmbeddingsMulticriteriaAnalysis:

    def __init__(self, model_names, models, algorithms=[KMeans, DBSCAN, HDBSCAN], pesos=None, n_rodadas=10, n_splits=5):
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
        self.embeddings = {}  # Armazena os embeddings gerados por cada modelo
        self.algorithms = algorithms
        self.n_rodadas = n_rodadas
        self.n_splits = n_splits
        self.en_preprocessor = ENPreprocessor()  # Criar instância do pré-processador para inglês
        self.br_preprocessor = BRPreprocessor()  # Criar instância do pré-processador para português
        self.data = self.create_embedding_column()  # Carregar o dataframe com as linhas do arquivo de fomento
        self.generate_embeddings_batch()
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

    def generate_embeddings(self):
        """
        Gera embeddings para os textos usando os modelos especificados.
        """
        for model_name, model in zip(self.model_names, self.models):
            try:
                sentences = self.data['texto_para_embedding'].to_arrow().to_pylist()  

                # Limpa a memória da GPU
                gc.collect()
                torch.cuda.empty_cache()

                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    print(f"Gerando embeedings em GPU com modelo {model_name}.")
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
                fim = time.time()

                self.embeddings[model_name] = embeddings.cpu().numpy()
            except Exception as e:
                print(f"    Erro ao gerar embeddings com modelo {model_name}: {e}")

    def generate_embeddings_batch(self):
        """
        Gera embeddings para os textos usando os modelos especificados, processando em lotes.
        """
        for model_name, model in zip(self.model_names, self.models):
            try:
                sentences = self.data['texto_para_embedding'].to_arrow().to_pylist()

                # Limpa a memória da GPU
                gc.collect()
                torch.cuda.empty_cache()

                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    print(f"Gerando embeedings em GPU com modelo {model_name}.")
                else:
                    device = torch.device('cpu')
                    print(f"GPU não disponível. Gerando embeedings em CPU com modelo {model_name}.")

                inicio = time.time()

                # Pré-processar cada texto de acordo com o idioma, em lotes
                batch_size = 64  # Defina o tamanho do lote
                processed_sentences = []
                for i in tqdm(range(0, len(sentences), batch_size), desc="Processando sentenças", unit=f"batch_size {batch_size}"):
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

                fim = time.time()

                self.embeddings[model_name] = embeddings.cpu().numpy()
            except Exception as e:
                print(f"Erro ao gerar embeddings com modelo {model_name}: {e}")

    def evaluate_clustering(self):
        """
        Avalia o desempenho dos embeddings em tarefas de clustering
        usando diferentes algoritmos, múltiplas rodadas e validação cruzada,
        e mede o tempo de execução de cada algoritmo.
        """
        print("Iniciando avaliação de clustering...")
        resultados = {}
        for model_name, embeddings in self.embeddings.items():
            print(f"Avaliando modelo: {model_name}")
            resultados[model_name] = {}
            for algorithm in self.algorithms:
                resultados[model_name][algorithm.__name__] = {"medias": [], "desvios": [], "tempo": []}
                resultados_algoritmo = []
                tempos_execucao = []
                for _ in range(self.n_rodadas):
                    skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
                    resultados_split = []
                    for train_index, test_index in skf.split(embeddings, np.zeros(len(embeddings))):
                        X_train, X_test = embeddings[train_index], embeddings[test_index]
                        clustering_model = algorithm()

                        # Mede o tempo de execução do algoritmo
                        inicio = time.time()
                        cluster_labels = clustering_model.fit_predict(X_train)
                        fim = time.time()
                        tempo_execucao = fim - inicio
                        tempos_execucao.append(tempo_execucao)

                        # Avalia apenas nos dados de teste
                        try:
                            silhouette_avg = silhouette_score(X_test, cluster_labels[test_index])
                            calinski_harabasz = calinski_harabasz_score(X_test, cluster_labels[test_index])
                            davies_bouldin = davies_bouldin_score(X_test, cluster_labels[test_index])

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
            folder_data_output = os.path.join(root_folder, '_reports')
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