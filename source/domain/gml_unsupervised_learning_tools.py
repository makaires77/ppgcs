import os, ast, logging, time, cudf, jinja2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score

from funding_analyser import BRPreprocessor, ENPreprocessor

class EmbeddingEvaluator:
    def __init__(self, model_names, models, data):
        """
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
        Generates embeddings for the texts using the specified model.

        Args:
            model: The sentence embedding model to be used.

        Returns:
            The generated embeddings as a NumPy array.
        """
        try:
            sentences = self.data['texto_para_embedding'].to_arrow().to_pylist()
            embeddings = model.encode(sentences, convert_to_tensor=True, device=model.device)
        except Exception as e:
            print(f"Erro ao gerar embeedings: {e}")
            return None
        return embeddings.cpu().numpy()

    def evaluate_clustering(self, embeddings, algorithms=[KMeans, DBSCAN, HDBSCAN]):
        """
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
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            results[algorithm.__name__] = silhouette_avg
        return results

    def run_benchmark(self):
        """
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
        Generates an HTML report with the benchmarking results and the choice of the best model.
        """
        results = self.run_benchmark()
        print(f"\nResultados para gerar relatório: {results}")

        # Imprimir resultados do benchmarking
        for model_name, metrics in results.items():
            print(f"Modelo: {model_name}")
            print(f"  Tempo de execução: {metrics['Tempo de execução']:.2f} segundos")
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

        try:
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
            print('Erro ao gerar relatório HTML com Jinja2')
            print(e)
        

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

