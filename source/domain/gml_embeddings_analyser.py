import os
import gc
import ast
import time
import torch
import string
import jinja2
import logging
import warnings
import traceback
import unicodedata
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoModel
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

## Para avaliar clustering com algoritmos somente em CPU
# from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
# from sklearn.metrics import silhouette_score

## Para avaliar clustering com algoritmos que rodam em GPU
import cuml
import cudf
import cupy as cp
import numpy as np
import cuml.metrics
from cuml.cluster import KMeans, DBSCAN, HDBSCAN
from cuml.metrics.cluster import silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from tqdm.auto import tqdm
from git import Repo
tqdm.pandas()

from gml_funding_preprocessor import ENPreprocessor, BRPreprocessor

# Configurar o logging (opcional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning or UserWarning)
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")


class EmbeddingsMulticriteriaAnalysis:

    def __init__(self, model_names, models, algorithms=[KMeans, DBSCAN, HDBSCAN], pesos=None, n_splits=5):
        """
        Autor: Marcos Aires (Nov.2024)
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
            n_splits: Número de splits para validação cruzada.
        """
        self.model_names = model_names
        self.models = models
        self.embeddings = {}
        self.algorithms = algorithms
        self.n_splits = n_splits
        self.en_preprocessor = ENPreprocessor()  # Criar instância do pré-processador para inglês
        self.br_preprocessor = BRPreprocessor()  # Criar instância do pré-processador para português
        self.data = self.create_embedding_column()  # Carregar dataframe arquivo de fomento
        
        self.show_models_info() # Exibir dados de cada modelo

        ## Para gerar embeeding junto com instanciação descomentar uma das versões:
        # self.generate_embeddings_batch()

        # self.generate_embeddings()
        # self.generate_embeddings_batch_no_grad()
        # self.generate_embeddings_optimzed()        

        ## Para disparar a avaliação junto com instaciação descomentar:
        # self.resultados = self.evaluate_clustering()

        # Definir os pesos dos critérios
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
        Autor: Marcos Aires (Nov.2024)
        Exibe informações sobre os modelos pré-treinados, extraindo 
        dados do ModelCardData e tratando exceções.
        """

        print("\n" + "-"*115)
        for model in self.models:
            try:
                # Obter informações do card do modelo
                model_card_data = model.model_card_data
                print(f"    Modelo de Base: {model_card_data.base_model}")

                # try:
                #     # Listar os atributos disponíveis
                #     print("Atributos disponíveis no modelo:")
                #     for attr, value in vars(model_card_data).items():
                #         if value and value != 'deprecated':
                #             if 'and more' in value:
                #                 print(f"  - {attr}: {value.replace(', and more','')}")
                #             else:
                #                 print(f"  - {attr}: {value}")

                #         if attr == 'model':
                #             print(type(value))
                # except:
                #     pass

                try:
                    # Verificar se o atributo task_name existe
                    if hasattr(model_card_data, 'task_name'):
                        value = model_card_data.task_name
                        if value:
                            if 'and more' in value:
                                value = value.replace(', and more','')
                            print(f"Tarefas aplicáveis: {value}")
                    else:
                        pass
                except:
                    pass

                try:
                    # Verificar se o atributo task_name existe
                    if hasattr(model_card_data, 'output_dimensionality'):
                        print(f"Número de features: {model_card_data.output_dimensionality}")
                    else:
                        pass
                except:
                    pass

                try:
                    # Obter a quantidade de features
                    num_features = model.get_sentence_embedding_dimension()
                    if num_features:
                        print(f"Número de features: {num_features}")
                except:
                    pass

                try:
                    # Verificar se o atributo model_max_length existe
                    if hasattr(model_card_data, 'model_max_length'):
                        print(f"Comprimento Máximo: {model_card_data.model_max_length}")
                    else:
                        pass
                        # print(f"Comprimento Máximo: Informação não disponível")
                    print(f"Número de features: {model_card_data.output_dimensionality}")
                except:
                    pass

                try:
                    # Verificar se o atributo model_name existe
                    if hasattr(model_card_data, 'model'):
                        print(f"Detalhes da classe: {model_card_data.model}")
                    else:
                        pass
                except Exception as e:
                    print(e)

                try:
                    if hasattr(model_card_data, 'license'):
                        license = model_card_data.get('license', 'Licença não encontrada') 
                        if license is not None:
                            print(f"   Tipo de Licença: {license}")
                    if hasattr(model_card_data, 'train_datasets'):
                        datasets = f"{', '.join([d['name'] for d in model_card_data.train_datasets])}"
                        if datasets:
                            print(f"Datasets de Treino: {datasets}")
                except:
                    pass
                print("-"*115)

            except Exception as e:
                # print(f"Erro ao obter informações do modelo: {e}")
                print("-"*115)

    def create_embedding_column(self, use_cudf=True):
        """
        Autor: Marcos Aires (Nov.2024)
        Cria a coluna 'texto_para_embedding' no dataframe df_fomento combinando dados selecionados e aplicando pré-processamento.

        Argumentos:
        use_cudf: Se deve usar cuDF para operações DataFrame (padrão: True)

        Retorna:
        O dataframe atualizado com a coluna 'texto_para_embedding'.
        """

        # Informar caminho para arquivo CSV usando raiz do repositório Git como referência
        repo = Repo(search_parent_directories=True)
        root_folder = repo.working_tree_dir
        folder_data_output = os.path.join(str(root_folder), '_data', 'out_json')
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

        # Aplicar função de suporte para cada linha do dataframe de editais
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
        Autor: Marcos Aires (Nov.2024)
        Detecta o idioma predominante em uma lista de sentenças.
        """
        en_preprocessor = ENPreprocessor()
        idiomas = [en_preprocessor.detect_language(sentence) for sentence in sentences]
        idioma_predominante = max(set(idiomas), key=idiomas.count)
        return idioma_predominante

    ## Sem otimização de uso da VRAM da GPU
    def generate_embeddings(self):
        """
        Autor: Marcos Aires (Nov.2024)
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
        Autor: Marcos Aires (Nov.2024)
        Gera embeddings para os textos usando os modelos especificados, 
        processando em lotes e retornando um dicionário com os embeddings 
        para cada modelo, no formato adequado para uso com PyTorch.
        """

        embeddings_dict = {}  # Inicializa o dicionário de embeddings
        for model_name, model in zip(self.model_names, self.models):
            try:
                sentences = self.data['texto_para_embedding'].to_arrow().to_pylist() # type: ignore

                # Limpa a memória da GPU
                gc.collect()
                torch.cuda.empty_cache()

                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    model = model.to(device)
                    print(f"\nGerando embeddings em GPU com modelo {model_name}.")
                else:
                    device = torch.device('cpu')
                    print(f"GPU não disponível. Gerando embeddings em CPU com modelo {model_name}.")

                inicio = time.time()

                # Pré-processar cada texto de acordo com o idioma, em lotes
                batch_size = 512
                all_embeddings = []
                for i in tqdm(range(0, len(sentences), batch_size), desc="Processando sentenças", unit=f"batch (batch_size {batch_size})"):
                    batch = sentences[i: i + batch_size]
                    processed_sentences = []
                    for sentence in batch:
                        idioma = self.en_preprocessor.detect_language(sentence)
                        if idioma == 'en':
                            processed_sentence = self.en_preprocessor.preprocess_text(sentence)
                        elif idioma == 'pt':
                            processed_sentence = self.br_preprocessor.preprocess_text(sentence)
                        else:
                            processed_sentence = sentence
                        processed_sentences.append(processed_sentence)

                    # Gerar embeddings em lote e concatenar
                    embeddings = model.encode(processed_sentences, convert_to_tensor=True, device=device, batch_size=batch_size)
                    all_embeddings.append(embeddings)

                # Concatena todos os embeddings em um único tensor
                all_embeddings = torch.cat(all_embeddings, dim=0)

                # Armazenar os embeddings no dicionário self.embeddings
                self.embeddings[model_name] = all_embeddings 
                
                # Calcula o tempo de execução
                fim = time.time()
                tempo_execucao_segundos = fim - inicio
                horas, resto = divmod(tempo_execucao_segundos, 3600)
                minutos, segundos = divmod(resto, 60)
                print(f"Tempo de execução: {int(horas):02d}:{int(minutos):02d}:{int(segundos):02d} para modelo {model_name}\n")

            except Exception as e:
                print(f"Erro ao gerar embeddings com modelo {model_name}: {e}")

        return self.embeddings  # Retorna o dicionário de embeddings


    def generate_embeddings_batch_no_grad(self):
        """
        Autor: Marcos Aires (Nov.2024)
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


    ## Otimizado com processamento em lote na GPU e detecção de idioma predominante
    def generate_embeddings_optimzed(self):
        """
        Autor: Marcos Aires (Nov.2024)
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

                batch_size = 128  # Definir o tamanho do lote de acordo com tamanhos e capacidades
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

                # Limpar o cache da GPU
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Memória em uso na GPU após limpeza final: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

            except Exception as e:
                print(f"Erro ao gerar embeddings com modelo {model_name}: {e}")

    def save_embeddings(self, filename="embeddings.pt"):
        """
        Autor: Marcos Aires (Nov.2024)
        Salva os embeddings gerados em um arquivo PyTorch.
        """
        # Informar caminho para pasta raiz do repositório Git como referência
        repo = Repo(search_parent_directories=True)
        root_folder = repo.working_tree_dir
        folder_data_output = os.path.join(str(root_folder), '_embeddings')
        pathfilename = os.path.join(folder_data_output, filename)        
        torch.save(self.embeddings, pathfilename)
        
        return self.embeddings

    def load_embeddings(self, filename="embeddings.pt"):
        """
        Autor: Marcos Aires (Nov.2024)
        Carrega os embeddings de um arquivo PyTorch local.
        """
        # Informar caminho para arquivo CSV usando raiz do repositório Git como referência
        repo = Repo(search_parent_directories=True)
        root_folder = repo.working_tree_dir
        folder_data_output = os.path.join(str(root_folder), '_embeddings')
        pathfilename = os.path.join(folder_data_output, filename)
        self.embeddings = torch.load(pathfilename)


    def save_embeddings_dict(self, filename="embeddings_dict.pt"):
        """
        Autor: Marcos Aires (Nov.2024)
        Salva os embeddings gerados em um arquivo PyTorch, um para cada modelo.
        """
        try:
            # Informar caminho para arquivo usando raiz do repositório Git como referência
            repo = Repo(search_parent_directories=True)
            root_folder = repo.working_tree_dir
            folder_data_output = os.path.join(str(root_folder), '_embeddings')
            pathfilename = os.path.join(folder_data_output, filename)

            # Criar a pasta _embeddings se ela não existir
            os.makedirs(folder_data_output, exist_ok=True)

            # Salvar o dicionário de embeddings
            torch.save({model_name: torch.tensor(embeddings) for model_name, embeddings in self.embeddings.items()}, pathfilename)

            # Imprimir o nome do arquivo e o número de modelos
            print(f"Arquivo de embeddings salvo: {filename}")
            print(f"Número de modelos salvos: {len(self.embeddings)}")

        except Exception as e:
            print(f"Erro ao salvar os embeddings: {e}")


    def load_embeddings_dict(self, filename):
        """
        Carrega os embeddings de um arquivo PyTorch.
        """
        try:
            # Informar caminho para arquivo usando raiz do repositório Git como referência
            repo = Repo(search_parent_directories=True)
            root_folder = repo.working_tree_dir
            folder_data_output = os.path.join(str(root_folder), '_embeddings')
            pathfilename = os.path.join(folder_data_output, filename)

            # Verificar se o arquivo existe no local
            if not os.path.exists(pathfilename):
                # Se não existir, tentar carregar da pasta _embeddings
                folder_data_output = os.path.join(str(root_folder), '_embeddings')
                pathfilename = os.path.join(folder_data_output, filename)

                # Se ainda não existir, imprimir mensagem de erro
                if not os.path.exists(pathfilename):
                    print(f"Erro: Arquivo de embeddings não encontrado em {pathfilename}")
                    return None

            # Carregar o dicionário de embeddings
            embedding_dict = torch.load(pathfilename)

            # Imprimir o nome do arquivo e o número de modelos
            print(f"Arquivo de embeddings carregado: {filename}")
            print(f"Número de modelos carregados: {len(embedding_dict)}")

            return embedding_dict

        except Exception as e:
            print(f"Erro ao carregar os embeddings: {e}")
            return None

    ## Início da avaliação, refatorar em outra classe posterioremente
    def evaluate_clustering(self, embeddings_dict):
        """
        Autor: Marcos Aires (Nov.2024)
        Avalia o desempenho dos embeddings em tarefas de clustering
        usando diferentes algoritmos do cuML e 
        validação cruzada, e mede o tempo de execução de cada algoritmo.
        Usa métricas do cuML e scikit-learn conforme a disponibilidade.

        Args:
            embeddings_dict: Um dicionário com os embeddings para cada modelo.
                            As chaves são os nomes dos modelos e os valores 
                            são os embeddings correspondentes.
        """
        print("Iniciando avaliação de clustering com cuML...")

        resultados = {}
        for model_name, embeddings in embeddings_dict.items():  # Usar embeddings_dict como entrada
            print(f"Avaliando modelo: {model_name}")
            resultados[model_name] = {}

            # Converter os embeddings para arrays do CuPy
            embeddings_cp = cp.array(embeddings.cpu().numpy())  # Converter para NumPy antes de converter para CuPy

            for algorithm in self.algorithms:
                resultados[model_name][algorithm.__name__] = {"resultados": [], "tempo": []}
                tempos_execucao = []

                # Instanciação do StratifiedKFold
                skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
                splits = list(skf.split(embeddings_cp.get(), cp.zeros(len(embeddings_cp)).get()))

                resultados_split = []  # Inicializar resultados_split aqui

                for train_index, test_index in splits:
                    X_train, X_test = embeddings_cp[train_index], embeddings_cp[test_index]

                    try:
                        if len(X_test) > 1:
                            print(f"  Split {len(resultados_split) + 1}:")
                            print(f"    Tamanho de X_train: {X_train.shape}")
                            print(f"    Tamanho de X_test: {X_test.shape}")

                            if algorithm.__name__ == "KMeans":
                                clustering_model = cuml.KMeans(n_clusters=8)
                            elif algorithm.__name__ == "DBSCAN":
                                clustering_model = cuml.DBSCAN(eps=0.5, min_samples=5)
                            elif algorithm.__name__ == "HDBSCAN":
                                clustering_model = cuml.HDBSCAN(min_cluster_size=5)

                            inicio = time.time()
                            cluster_labels = clustering_model.fit_predict(X_test)
                            fim = time.time()
                            tempo_execucao = fim - inicio
                            tempos_execucao.append(tempo_execucao)

                            print(f"    Tamanho de cluster_labels: {cluster_labels.shape}")

                            X_test_cp = cp.array(X_test)
                            cluster_labels_cp = cp.array(cluster_labels)

                            print(f"    Tamanho de X_test_cp: {X_test_cp.shape}")
                            print(f"    Tamanho de cluster_labels_cp: {cluster_labels_cp.shape}")

                            silhouette_avg = silhouette_score(X_test_cp, cluster_labels_cp)
                            X_test_np = X_test_cp.get()
                            cluster_labels_np = cluster_labels_cp.get()

                            calinski_harabasz = calinski_harabasz_score(X_test_np, cluster_labels_np)
                            davies_bouldin = davies_bouldin_score(X_test_np, cluster_labels_np)

                            resultados_split.append({
                                "silhouette": silhouette_avg,
                                "calinski_harabasz": calinski_harabasz,
                                "davies_bouldin": davies_bouldin
                            })

                        else:
                            print("      Ignorando split com menos de 2 amostras.")

                    except Exception as e:
                        print(f"    Erro ao calcular métricas: {e}")
                        print(f"    {algorithm.__name__}, modelo {model_name}, split {len(resultados_split) + 1}. Pulando esta iteração.")

                resultados[model_name][algorithm.__name__]["resultados"] = resultados_split
                resultados[model_name][algorithm.__name__]["tempo"] = np.mean(tempos_execucao)

        return resultados


    def calcular_pontuacao_multicriterio(self, resultados):
        """
        Autor: Marcos Aires (Nov.2024)
        Calcula a pontuação multicritério para cada algoritmo, combinando as métricas com os pesos.
        """
        pontuacoes = {}
        for model_name, model_results in resultados.items():
            pontuacoes[model_name] = {}
            max_valor = 0
            for algoritmo, resultados in model_results.items():
                medias = resultados["medias"]
                pontuacao = 0
                for i, metrica in enumerate(['silhouette', 'calinski_harabasz', 'davies_bouldin']):
                    valor = medias[i]
                    # Normaliza as métricas para ficarem na mesma escala (0 a 1)
                    if metrica == "silhouette":
                        valor_normalizado = (valor + 1) / 2  # Silhouette varia de -1 a 1, quanto maior melhor qualidade de clustering, clusters mais coesos e separados
                    elif metrica == "davies_bouldin":
                        valor_normalizado = 1 / (valor + 1e-6)  # Davies-Bouldin é quanto menor melhor,  indica clusters mais distintos e compactos
                    elif metrica == "calinski_harabasz":
                        max_valor = np.max([resultados["medias"][i] for model_results in resultados.values() for resultados in model_results.values()])
                        valor_normalizado = valor / max_valor # Calinski_Harabasz é quanto maior melhor, sugere uma melhor separação
                    pontuacao += self.pesos[metrica] * valor_normalizado

                # Adiciona o tempo de execução à pontuação
                tempo_execucao = resultados["tempo"]
                tempo_normalizado = 1 / (tempo_execucao + 1e-6)  # Tempo é quanto menor melhor
                pontuacao += self.pesos["tempo"] * tempo_normalizado

                pontuacoes[model_name][algoritmo] = pontuacao
                print(f"Pontuações: {pontuacoes}")

        return pontuacoes


    def escolher_melhor_modelo(self, resultados):
            """
            Autor: Marcos Aires (Nov.2024)
            Escolhe o modelo com a maior pontuação multicritério.
            """
            print("Resultados:", resultados)
            pontuacoes = self.calcular_pontuacao_multicriterio(resultados)
            print("Pontuações:", pontuacoes)  # Adicione este print
            melhor_modelo = max(pontuacoes, key=lambda model_name: max(pontuacoes[model_name].values()))
            return melhor_modelo


    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def plot_clustering_results_bars(self, resultados):
        """
        Plota os resultados do clustering usando Plotly, exibindo gráficos de barras
        para comparar o desempenho de cada modelo em cada métrica para cada algoritmo.
        """
        metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        algorithms = ['KMeans', 'DBSCAN', 'HDBSCAN']

        # Ajustar o número de subplots de acordo com o número de métricas
        fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)

        # Loop pelas métricas
        for i, metric_name in enumerate(metrics):
            # Loop pelos algoritmos
            for j, algorithm_name in enumerate(algorithms):
                # Obter os valores da métrica para cada modelo
                model_values = {}
                for model_name, model_results in resultados.items():
                    # Obter os valores da métrica para cada split
                    metric_values = [result[metric_name] for result in model_results[algorithm_name]['resultados']]
                    model_values[model_name] = np.mean(metric_values)  # Calcular a média dos splits

                # Adicionar um gráfico de barras para a métrica e algoritmo, especificando a posição do subplot
                for k, model_name in enumerate(model_values.keys()):
                    fig.add_trace(go.Bar(
                        x=[algorithm_name],
                        y=[model_values[model_name]],
                        name=model_name,  # Usar o nome do modelo na legenda
                        showlegend=i == 0 and j == 0,  # Exibe a legenda apenas no primeiro subplot
                        legendgroup=model_name,  # Agrupa as barras por modelo
                        offsetgroup=k,  # Define o offset para agrupar as barras por modelo
                        marker_color=['blue', 'green'][k]  # Cores para os modelos
                    ), row=1, col=i+1)  # Especificar a linha e coluna do subplot

        # Configura o layout do gráfico
        fig.update_layout(
            title="Comparação do Desempenho dos Modelos",
            height=600,
            width=1200
        )

        # Ajustar os títulos dos eixos
        for i, metric_name in enumerate(metrics):
            fig.update_xaxes(title_text="Algoritmo", row=1, col=i+1)
            fig.update_yaxes(title_text="", row=1, col=i+1)

        # Adicionar anotações para indicar as melhores regiões
        fig.add_annotation(
            text="Maior Melhor",
            x=0.23, y=1,
            xref="paper", yref="paper",
            showarrow=True, arrowhead=4,
            ax=0, ay=30
        )
        fig.add_annotation(
            text="Maior Melhor",
            x=0.59, y=1,
            xref="paper", yref="paper",
            showarrow=True, arrowhead=4,
            ax=0, ay=30
        )
        fig.add_annotation(
            text="Menor Melhor",
            x=0.77, y=0.9,
            xref="paper", yref="paper",
            showarrow=True, arrowhead=4,
            ax=0, ay=-30
        )

        fig.show()

    ## Início da geração de relatórios
    def generate_report(self, resultados):
        """
        Autor: Marcos Aires (Nov.2024)
        Generates a report with the benchmarking results.
        """
        try:
            # 1. Preparar os dados para o relatório
            report_data = []
            for model_name, model_results in resultados.items():
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
            self.plot_clustering_results_bars(resultados)
            self.generate_report_charts(df_report)

        except Exception as e:
            print(f"Erro ao gerar o relatório: {e}")
            traceback.print_exc()


    def generate_report_charts(self, df_report):
        """
        Autor: Marcos Aires (Nov.2024)
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