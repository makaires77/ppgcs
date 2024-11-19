
import os, ast, logging, time, cudf, jinja2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback
import torch
import json
import time
import gc

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.model_selection import StratifiedKFold
from git import Repo

from funding_analyser import BRPreprocessor, ENPreprocessor

class EmbeedingsMulticriteriaAnalysis:

    def __init__(self, data, model_names, models, algorithms=[KMeans, DBSCAN, HDBSCAN], pesos=None, n_rodadas=10, n_splits=5):
        """
        Autor: Marcos Aires (Nov.2024)
        Inicializa a classe com os embeddings, algoritmos de clustering, pesos para cada critério,
        número de rodadas e número de splits para validação cruzada.

        Args:
            embeddings: Os embeddings a serem avaliados.
            algorithms: Uma lista de classes de algoritmos de clustering a serem usados.
            pesos: Um dicionário com os pesos para cada critério.
                   As chaves devem ser 'silhouette', 'calinski_harabasz', 'davies_bouldin' e 'tempo'.
                   Se None, todos os critérios terão o mesmo peso.
            n_rodadas: Número de rodadas para calcular a média e o desvio padrão das métricas.
            n_splits: Número de splits para validação cruzada.
        """
        self.data = data
        self.model_names = model_names
        self.models = models
        self.embeddings = {}  # Armazena os embeddings gerados por cada modelo
        self.algorithms = algorithms
        self.n_rodadas = n_rodadas
        self.n_splits = n_splits
        self.generate_embeddings()  # Adicione esta linha
        self.resultados = self.evaluate_clustering()  # Chama o método e armazena o resultado

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
        Creates the 'texto_para_embedding' column in the df_fomento dataframe 
        by combining selected data and applying preprocessing.

        Args:
            use_cudf: Whether to use cuDF for DataFrame operations (default: True)

        Returns:
            The updated dataframe with the 'texto_para_embedding' column.
        """

        # Informar caminho para arquivo CSV usando raiz do repositório Git como referência
        repo = Repo(search_parent_directories=True)
        root_folder = repo.working_tree_dir
        folder_data_output = os.path.join(str(root_folder), '_data', 'out_json')
        filename = 'df_fomento_geral.csv'
        pathfilename = os.path.join(folder_data_output, filename)

        if use_cudf:
            try:
                import cudf
                df_fomento = cudf.read_csv(pathfilename, header=0)
            except ImportError:
                print("cuDF não está disponível. Usando Pandas.")
                df_fomento = pd.read_csv(pathfilename, header=0)
        else:
            df_fomento = pd.read_csv(pathfilename, header=0)

        def convert_to_dict(text):
            try:
                return ast.literal_eval(text)
            except ValueError:
                return None

        df_fomento['lista_de_projetos'] = df_fomento['lista_de_projetos'].apply(convert_to_dict)
        df_fomento = df_fomento.dropna(subset=['lista_de_projetos']).reset_index(drop=True)

        # -------------------------------------------------------------------------------------
        # Início da criação da coluna 'texto_para_embedding'
        # -------------------------------------------------------------------------------------

        # Combinar as colunas 'titulo', 'resumo' e 'palavras_chave' em uma única coluna 'texto_para_embedding'
        if use_cudf:
            df_fomento['texto_para_embedding'] = df_fomento['titulo'] + ' ' + \
                                                df_fomento['resumo'] + ' ' + \
                                                df_fomento['palavras_chave']
        else:
            df_fomento['texto_para_embedding'] = df_fomento['titulo'].astype(str) + ' ' + \
                                                df_fomento['resumo'].astype(str) + ' ' + \
                                                df_fomento['palavras_chave'].astype(str)

        # Pré-processamento do texto
        def preprocess_text(text, language='portuguese'):
            if language == 'portuguese':
                preprocessor = BRPreprocessor()
            elif language == 'english':
                preprocessor = ENPreprocessor()
            else:
                raise ValueError("Idioma não suportado.")
            return preprocessor.preprocess_text(text)

        df_fomento['texto_para_embedding'] = df_fomento['texto_para_embedding'].apply(preprocess_text)

        # -------------------------------------------------------------------------------------
        # Fim da criação da coluna 'texto_para_embedding'
        # -------------------------------------------------------------------------------------

        return df_fomento

    def generate_embeddings(self):
        """
        Gera embeddings para os textos usando os modelos especificados.
        """
        for model_name, model in zip(self.model_names, self.models):
            try:
                print(f"Gerando embeddings para o modelo {model_name}...")
                # sentences = self.data['texto_para_embedding'].to_arrow().to_pylist()
                sentences = self.data['texto_para_embedding'].tolist() #tentativa de simplificação

                # Limpa a memória da GPU
                gc.collect()
                torch.cuda.empty_cache()

                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    print(f"Usando GPU para gerar embeddings para o modelo {model_name}.")
                else:
                    device = torch.device('cpu')
                    print(f"GPU não disponível. Usando CPU para gerar embeddings para o modelo {model_name}.")

                inicio = time.time()
                embeddings = model.encode(sentences, convert_to_tensor=True, device=device)
                fim = time.time()

                self.embeddings[model_name] = embeddings.cpu().numpy()
            except Exception as e:
                print(f"Erro ao gerar embeddings para o modelo {model_name}: {e}")

    def evaluate_clustering(self):
        """
        Autor: Marcos Aires (Nov.2024)
        Avalia o desempenho dos embeddings em tarefas de clustering
        usando diferentes algoritmos, múltiplas rodadas e validação cruzada,
        e mede o tempo de execução de cada algoritmo.
        """
        resultados = {}
        for model_name, embeddings in self.embeddings.items():
            print(f"Iniciando avaliação de clustering para modelo: {model_name}")
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

    def adicionar_tempo_execucao(self, model_name, algorithm, tempo):
        """
        Autor: Marcos Aires (Nov.2024)
        Adiciona o tempo de execução para um algoritmo específico.

        Args:
            algoritmo: Nome do algoritmo de clustering.
            tempo: Tempo de execução em segundos.
        """
        self.resultados[model_name][algorithm]["tempo"] = tempo

    def calcular_pontuacao_multicriterio(self):
        """
        Autor: Marcos Aires (Nov.2024)
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
                    # else:  # Calinski-Harabasz
                    #     # Correção: Calcular o máximo entre todos os modelos
                    #     max_valor = np.max([r["medias"][i] for model_results in self.resultados.values() for r in model_results.values()])  
                    #     valor_normalizado = valor / max_valor
                    pontuacao += self.pesos[metrica] * valor_normalizado

                # Adiciona o tempo de execução à pontuação
                tempo_execucao = resultados["tempo"]
                tempo_normalizado = 1 / (tempo_execucao + 1e-6)  # Tempo é menor quanto melhor
                pontuacao += self.pesos["tempo"] * tempo_normalizado

                pontuacoes[model_name][algoritmo] = pontuacao
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


class EmbeddingEvaluator:
    def __init__(self, model_names, models, data):
        """
        Autor: Marcos Aires (Nov.2024)
        Initializes the class for benchmarking embedding models.

        Args:
            model_names: A list of names for the embedding models.
            models: A list of sentence embedding models (SentenceTransformer).
            data: A DataFrame containing the 'texto_para_embedding' column with preprocessed texts.
        """
        self.model_names = model_names
        self.models = models
        self.data = data

        # Set the JINJA_DEBUG environment variable to True
        os.environ['JINJA_DEBUG'] = 'True'

        template_dir = os.path.expanduser("~/ppgcs/source/template/")
        self.template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
        self.template_env = jinja2.Environment(loader=self.template_loader)
        self.template = self.template_env.get_template("benchmark_report.html")

    def generate_embeddings(self, model):
        """
        Autor: Marcos Aires (Nov.2024)
        Generates embeddings for the texts using the specified model.

        Args:
            model: The sentence embedding model to be used.

        Returns:
            The generated embeddings as a NumPy array.
        """
        try:
            sentences = self.data['texto_para_embedding'].to_arrow().to_pylist()
            
            # Tenta usar a GPU se disponível
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print("Usando GPU para gerar embeddings.")
            else:
                device = torch.device('cpu')
                print("GPU não disponível. Usando CPU para gerar embeddings.")

            embeddings = model.encode(sentences, convert_to_tensor=True, device=device)
        except Exception as e:
            print(f"Erro ao gerar embeedings: {e}")
            return None
        return embeddings.cpu().numpy()

    def evaluate_clustering(self, embeddings, algorithms=[KMeans, DBSCAN, HDBSCAN]):
        """
        Autor: Marcos Aires (Nov.2024)
        Evaluates the performance of embeddings in clustering tasks using different algorithms.

        Args:
            embeddings: The embeddings to be evaluated.
            algorithms: A list of clustering algorithm classes to be used.

        Returns:
            A dictionary with the benchmarking results for each algorithm.
        """
        results = {}
        for algorithm in algorithms:
            clustering_model = algorithm() 
            cluster_labels = clustering_model.fit_predict(embeddings)
            
            # Calculate the metrics
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
            davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
            
            # Store the results
            results[algorithm.__name__] = {
                "silhouette": silhouette_avg,
                "calinski_harabasz": calinski_harabasz,
                "davies_bouldin": davies_bouldin
            }
        return results

    def run_benchmark(self):
        """
        Autor: Marcos Aires (Nov.2024)
        Executes benchmarking of the embedding models.

        Returns:
            A dictionary containing the benchmarking results for each model.
        """
        benchmark_results = {}
        for model_name, model in zip(self.model_names, self.models):
            start_time = time.time()
            embeddings = self.generate_embeddings(model)
            end_time = time.time()
            elapsed_time = end_time - start_time

            clustering_results = self.evaluate_clustering(embeddings)

            # Ensure 'Tempo de execução' is numeric
            elapsed_time = float(elapsed_time)

            # Ensure values in 'Resultados de clustering' are numeric and handle potential errors
            for algorithm, score in clustering_results.items():
                try:
                    # Attempt to convert to float, handling potential ',' as decimal separator
                    clustering_results[algorithm] = float(str(score).replace(',', '.'))
                except ValueError:
                    logging.error(f"Erro ao converter score para float no algoritmo {algorithm} do modelo {model_name}: {score}")
                    # Handle the error gracefully - you can choose one of the following:
                    # 1. Set to a default value (e.g., 0 or -1)
                    # clustering_results[algorithm] = -1.0 
                    # 2. Remove the algorithm from the results
                    del clustering_results[algorithm]

            benchmark_results[model_name] = {
                'Tempo de execução': elapsed_time,
                'Resultados de clustering': clustering_results
            }

        return benchmark_results

    def generate_report(self):
        """
        Autor: Marcos Aires (Nov.2024)
        Generates an HTML report with the benchmarking results and the choice of the best model.
        """
        try:
            results = self.run_benchmark()
            print(f"\nResultados para gerar relatório: {results}")

            # Imprimir resultados do benchmarking
            for model_name, metrics in results.items():
                print(f"\nModelo: {model_name}")
                print(f"  Tempo de execução: {metrics['Tempo de execução']} segundos")
                print("  Resultados de clustering:")
                for algorithm, score in metrics['Resultados de clustering'].items():
                    print(f"    {algorithm}: {score:.3f}")
                print("\n")

            # Escolher o melhor modelo
            best_model = max(results, key=lambda x: sum(results[x]['Resultados de clustering'].values()) / results[x]['Tempo de execução'])

            print(f"Melhor modelo: {best_model}\n")

            # Visualizar os resultados de clustering (opcional)
            for model_name, metrics in results.items():
                plt.bar(metrics['Resultados de clustering'].keys(), metrics['Resultados de clustering'].values())
                plt.title(f'Resultados de Clustering para o Modelo {model_name}')
                plt.xlabel('Algoritmo de Clustering')
                plt.ylabel('Silhouette Score')
                plt.show()

            # Render the template with the provided data
            report_content = self.template.render(
                benchmark_results=results,
                best_model=best_model
            )

            # Save the report to an HTML file
            with open("benchmark_report.html", "w", encoding="utf-8") as f:
                f.write(report_content)

            print("Relatório de benchmarking gerado com sucesso: benchmark_report.html")

        except Exception as e:
            print(f'Erro ao gerar relatório HTML com Jinja2: {e}')
            traceback.print_exc()  # Imprime o traceback completo do erro

class DataPreprocessor:
    def __init__(self, 
                 curriculos_file, 
                 produtos_file, 
                 editais_file, 
                 use_gpu=True):
        """
        Initializes the DataPreprocessor.

        Args:
            curriculos_file: Path to the CSV file containing researcher CV data.
            produtos_file: Path to the JSON file containing strategic health products data.
            editais_file: Path to the CSV or JSON file containing funding opportunities data.
            use_gpu: Whether to use GPU-accelerated preprocessing for non-Portuguese text (default: True).
        """
        self.curriculos_file = curriculos_file
        self.produtos_file = produtos_file
        self.editais_file = editais_file
        self.use_gpu = use_gpu

        # Initialize preprocessors
        self.br_preprocessor = BRPreprocessor()
        self.en_preprocessor = ENPreprocessor()

    def load_data(self):
        """
        Loads data from the specified CSV and JSON files.

        Returns:
            A tuple containing three DataFrames: curriculos_df, produtos_df, editais_df.
        """
        try:
            curriculos_df = pd.read_csv(self.curriculos_file)
        except FileNotFoundError:
            logging.error(f"Arquivo de currículos não encontrado: {self.curriculos_file}")
            raise
        except Exception as e:
            logging.error(f"Erro ao carregar o arquivo de currículos: {e}")
            raise

        try:
            with open(self.produtos_file, 'r', encoding='utf-8') as f:
                produtos_data = json.load(f)
            produtos_df = pd.DataFrame(produtos_data)
        except FileNotFoundError:
            logging.error(f"Arquivo de produtos não encontrado: {self.produtos_file}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar o arquivo JSON de produtos: {e}")
            raise
        except Exception as e:
            logging.error(f"Erro ao carregar o arquivo de produtos: {e}")
            raise

        try:
            if self.editais_file.endswith('.csv'):
                editais_df = pd.read_csv(self.editais_file)
            elif self.editais_file.endswith('.json'):
                with open(self.editais_file, 'r', encoding='utf-8') as f:
                    editais_data = json.load(f)
                editais_df = pd.DataFrame(editais_data)
            else:
                raise ValueError("Formato de arquivo de editais não suportado. Use CSV ou JSON.")
        except FileNotFoundError:
            logging.error(f"Arquivo de editais não encontrado: {self.editais_file}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar o arquivo JSON de editais: {e}")
            raise
        except Exception as e:
            logging.error(f"Erro ao carregar o arquivo de editais: {e}")
            raise

        return curriculos_df, produtos_df, editais_df

    def preprocess_dataframe(self, df, column_name):
        """
        Preprocesses the specified column in the dataframe using the appropriate preprocessor.

        Args:
            df: The dataframe to be preprocessed.
            column_name: The name of the column containing the text to be preprocessed.

        Returns:
            The preprocessed text data as a list of lists of words.
        """
        try:
            # Apply preprocessing based on GPU availability
            if self.use_gpu:
                preprocessed_data = df[column_name].to_pandas().progress_apply(
                    lambda x: self.en_preprocessor.preprocess_text(x, self.en_preprocessor.tokenizer)
                )
            else:
                preprocessed_data = df[column_name].to_pandas().progress_apply(self.br_preprocessor.preprocess_text)

            return preprocessed_data.tolist()

        except Exception as e:
            logging.error(f"Erro durante o pré-processamento da coluna '{column_name}': {e}")
            raise

    def preprocess_data(self):
        """
        Loads and preprocesses the data from the CSV and JSON files.

        Returns:
            A tuple containing three lists of lists of words: 
            preprocessed_curriculos, preprocessed_produtos, preprocessed_editais.
        """

        # Load the data
        curriculos_df, produtos_df, editais_df = self.load_data()

        # Preprocess each dataframe
        preprocessed_curriculos = self.preprocess_dataframe(curriculos_df, 'texto_do_curriculo')  # Substitua 'texto_do_curriculo' pelo nome da coluna relevante no seu dataframe de currículos
        preprocessed_produtos = self.preprocess_dataframe(produtos_df, 'descricao_do_produto')    # Substitua 'descricao_do_produto' pelo nome da coluna relevante no seu dataframe de produtos
        preprocessed_editais = self.preprocess_dataframe(editais_df, 'texto_do_edital')        # Substitua 'texto_do_edital' pelo nome da coluna relevante no seu dataframe de editais

        return preprocessed_curriculos, preprocessed_produtos, preprocessed_editais