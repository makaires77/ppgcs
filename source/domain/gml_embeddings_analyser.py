import os
import gc
import ast
import json
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
import plotly.graph_objects as go

from datetime import datetime
from genericpath import isfile
from transformers import AutoModel
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

## Para avaliar clustering com algoritmos que rodam em GPU
import cuml
import cudf
import cupy as cp
import numpy as np
import cuml.metrics
from sklearn.model_selection import KFold
from cuml.cluster import KMeans, DBSCAN, HDBSCAN
from cuml.metrics.cluster import silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

## Para avaliar clustering com algoritmos somente em CPU
# from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
# from sklearn.metrics import silhouette_score
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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
        self.model_stats = {}  # Dicionário para armazenar as estatísticas dos modelos        
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

    def get_model_size(self, model_name, base_path="~/.cache/huggingface/hub"):
        """
        Retorna o tamanho de um modelo em MB.
        """

        model_folders = {
            'sentence-transformers/gtr-t5-large': 'models--sentence-transformers--gtr-t5-large',
            'distiluse-base-multilingual-cased-v2': 'models--sentence-transformers--distiluse-base-multilingual-cased-v2',
            'all-distilroberta-v1': 'models--sentence-transformers--all-distilroberta-v1',
            'all-mpnet-base-v2': 'models--sentence-transformers--all-mpnet-base-v2',
            'paraphrase-multilingual-MiniLM-L12-v2':
            'models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2',
            'all-MiniLM-L6-v2': 'models--sentence-transformers--multi-qa-MiniLM-L6-cos-v1',
        }

        base_path = os.path.expanduser(base_path)

        # Constrói o caminho completo para o modelo
        model_name = "models--sentence-transformers--" + model_name
        if "/" in model_name:
            model_name = model_name.replace("sentence-transformers/", "")
        model_path = os.path.join(base_path, model_name)

        size_mb = 0

        if os.path.isdir(model_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(model_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
                        size_mb = total_size / (1024 * 1024)

        elif os.path.isfile(model_path):
            print(f"O caminho {model_path} é um arquivo e não uma pasta")
        else:
            print(f"O caminho {model_path} não foi encontrado")

        print(f"Pasta local para o modelo: {model_name}")
        print(f"Espaço memória necessária: {size_mb:.2f} MB")


    def get_models_sizes(self, base_path="~/.cache/huggingface/hub"):
        """
        Retorna o tamanho dos modelos na pasta local de cache do hugging face em MB.
        """        
        model_names = [
            'sentence-transformers/gtr-t5-large',
            'distiluse-base-multilingual-cased-v2',
            'paraphrase-multilingual-MiniLM-L12-v2',
            'all-mpnet-base-v2',
            'all-distilroberta-v1',
            'all-MiniLM-L6-v2',
        ]

        model_folders = {
            'sentence-transformers/gtr-t5-large': 'models--sentence-transformers--gtr-t5-large',
            'distiluse-base-multilingual-cased-v2': 'models--sentence-transformers--distiluse-base-multilingual-cased-v2',
            'all-distilroberta-v1': 'models--sentence-transformers--all-distilroberta-v1',
            'all-mpnet-base-v2': 'models--sentence-transformers--all-mpnet-base-v2',
            'paraphrase-multilingual-MiniLM-L12-v2':
            'models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2',
            'all-MiniLM-L6-v2': 'models--sentence-transformers--multi-qa-MiniLM-L6-cos-v1',
        }

        base_path = os.path.expanduser(base_path)

        for model_name in model_names:
            # Adiciona "models--" ao início do nome do modelo
            model_name = "models--sentence-transformers--" + model_name

            if "/" in model_name:
                model_name = model_name.replace("sentence-transformers/", "")

            # Constrói o caminho completo para o modelo
            model_path = os.path.join(base_path, model_name)
            size_mb = 0

            if os.path.isdir(model_path):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(model_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if not os.path.islink(fp):
                            total_size += os.path.getsize(fp)
                            size_mb = total_size / (1024 * 1024)

            elif os.path.isfile(model_path):
                print(f"O caminho {model_path} é um arquivo e não uma pasta")
            else:
                print(f"O caminho {model_path} não foi encontrado")

            print(f"Nome modelo pré-treinado: {model_name}")
            print(f"Tamanho ocupado em disco: {size_mb:.2f} MB\n")


    def get_all_models_with_size(self, base_path="~/.cache/huggingface/hub"):
        """
        Acessa as pastas de todos os modelos na pasta base e retorna o tamanho de cada um.
        """
        models_with_size = {}

        for model_name in os.listdir(os.path.expanduser(base_path)):
            model_name = "models--sentence-transformers--" + model_name
            if "/" in model_name:
                model_name = model_name.replace("sentence-transformers/", "")

            # Constrói o caminho completo para o modelo
            model_path = os.path.join(base_path, model_name)
            if os.path.isdir(model_path):
                size_mb = self.get_model_size(model_path)
                models_with_size[model_name] = size_mb
        
        # Imprime o resultado
        for model_name, size_mb in models_with_size.items():
            print(f"Modelo: {model_name}, Tamanho: {size_mb:.2f} MB")

        return models_with_size


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
                print(f"Nome modelo base original: {model_card_data.base_model}")

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
                            print(f"Treinado nas tarefas para: {value}")
                    else:
                        pass
                except:
                    pass

                try:
                    # Consultar o espaço em disco dos arquivos do modelo
                    model_size = self.get_model_size(model_card_data.base_model)
                    if model_size:
                        print(f"Espaço para armazenamento: {model_size}")
                    else:
                        pass
                except:
                    pass

                try:
                    # Verificar se o atributo dimensionalidade existe
                    if hasattr(model_card_data, 'output_dimensionality'):
                        print(f"Dimensionalidade vetorial: {model_card_data.output_dimensionality}")
                    else:
                        pass
                except:
                    pass

                try:
                    # Obter a quantidade de features
                    num_features = model.get_sentence_embedding_dimension()
                    if num_features:
                        print(f"Número de características: {num_features}")
                except:
                    pass

                try:
                    # Verificar se o atributo model_max_length existe
                    if hasattr(model_card_data, 'model_max_length'):
                        print(f"Maior tamanho de sentença: {model_card_data.model_max_length}")
                except:
                    pass

                try:
                    # Verificar se o atributo model_name existe
                    if hasattr(model_card_data, 'model'):
                        print(f"Detalhe módulos do modelo: {model_card_data.model}")
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
                            print(f"Datasets para treinamento: {datasets}")
                except:
                    pass
                print("-"*115)

            except Exception as e:
                # print(f"Erro ao obter informações do modelo: {e}")
                print("-"*115)


    def exportar_estatisticas_modelos(self, nome_arquivo="estatisticas_modelos.json"):
        """
        Exporta as estatísticas dos modelos para um arquivo JSON.
        """

        # Informar caminho para pasta raiz do repositório Git como referência
        repo = Repo(search_parent_directories=True)
        root_folder = repo.working_tree_dir
        folder_data_output = os.path.join(str(root_folder), '_embeddings')

        # Verificar se a pasta _embeddings existe, senão cria
        if not os.path.exists(folder_data_output):
            os.makedirs(folder_data_output)

        pathfilename = os.path.join(folder_data_output, nome_arquivo)        

        with open(pathfilename, 'w') as f:
            json.dump(self.model_stats, f, indent=4)


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
                
                # Calcula o tamanho em disco do modelo
                ## model_path = model.config._name_or_path  # Obtém o caminho do modelo

                size_mb = 0
                try:
                    tamanho_modelo_mb = self.get_model_size(model_name)
                except:
                    pass

                # Calcula o tempo de execução
                fim = time.time()
                tempo_execucao_segundos = fim - inicio
                horas, resto = divmod(tempo_execucao_segundos, 3600)
                minutos, segundos = divmod(resto, 60)
                tempo_execucao_formatado = f"{int(horas):02d}:{int(minutos):02d}:{int(segundos):02d}"
                print(f"Tamanho: {tamanho_modelo_mb:.2f} MB e tempo de execução: {tempo_execucao_formatado}")

                # Armazena as estatísticas do modelo
                self.model_stats[model_name] = {
                    'tempo_execucao': tempo_execucao_formatado,
                    'tamanho_mb': tamanho_modelo_mb
                }

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
        # Informar caminho para arquivo de embeddings usando raiz do repositório Git como referência
        repo = Repo(search_parent_directories=True)
        root_folder = repo.working_tree_dir
        folder_data_output = os.path.join(str(root_folder), '_embeddings')
        pathfilename = os.path.join(folder_data_output, filename)
        self.embeddings = torch.load(pathfilename)


    def save_embeddings_dict(self, filename="embeddings_dict.pt", manter_gradientes=True):
        """
        Autor: Marcos Aires (Nov.2024)
        Salva os embeddings gerados em um arquivo PyTorch, um para cada modelo.

        Obs.: a forma de salvar os embeddings define a manutenção do histórico de gradientes e o consumo de memória. O problema é que se for usado apenas torch.tensor(embeddings) se cria uma cópia do tensor embeddings, e essa cópia não compartilha o histórico de gradientes com o tensor original. Isso pode ser problemático se esses embeddings forem para treinar um modelo, pois as atualizações nos gradientes não serão propagadas para o tensor original. Para corrigir o erro e evitar potenciais problemas, deve-se usar os métodos clone().detach() ou clone().detach().requires_grad_(True) para criar uma cópia do tensor que compartilhe o histórico de gradientes, mantendo para futuros treinamentos.  
        
        Explicação:
            clone(): cria uma cópia do tensor, compartilhando os dados subjacentes, mas com um novo histórico de gradientes.

                detach(): remove o tensor do grafo computacional, tornando-o um tensor "folha" sem histórico de gradientes. Isso é útil para economizar memória quando o histórico de gradientes não é mais necessário. É preciso manter o histórico de gradientes para treinar o KGNN, ou outro modelo, usando os embeddings, ou fazer fine-tuning dos embeddings posteriormente.

                requires_grad_(True): define a flag requires_grad do tensor como True, o que significa que o histórico de gradientes será rastreado para esse tensor. E aconselhável remover o histórico de gradientes se for usar os embeddings apenas para inferência, ou se for armazenar os embeddings para uso posterior.

            Ao usar clone().detach() ou clone().detach().requires_grad_(True), garante-se que os embeddings salvos sejam cópias independentes do tensor original, evitando problemas com o histórico de gradientes e o consumo de memória.
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
            if manter_gradientes:
                torch.save(
                    {
                        model_name: embeddings.clone().detach().requires_grad_(True) 
                        for model_name, embeddings in self.embeddings.items()
                    }, 
                    pathfilename
                )
            else:
                torch.save(
                    {
                        model_name: embeddings.clone().detach() 
                        for model_name, embeddings in self.embeddings.items()
                    }, 
                    pathfilename
                )

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

    def load_statistics(self, filename="estatisticas_modelos.json"):
        """
        Autor: Marcos Aires (Nov.2024)
        Carrega JSON local contendo os tempos de geração de embeddings para cada modelo.
        """
        # Informar caminho para arquivo JSON usando raiz do repositório Git como referência
        repo = Repo(search_parent_directories=True)
        root_folder = repo.working_tree_dir
        folder_data_output = os.path.join(str(root_folder), '_embeddings')
        pathfilename = os.path.join(folder_data_output, filename)
        
        # Carregar o dicionário de resultados de arquivo JSON
        with open(pathfilename, 'r') as f:
            estatisticas = json.load(f)

        if not isinstance(estatisticas, dict):
            raise TypeError("O arquivo JSON não contém um dicionário.")

        return estatisticas

    def load_results(self, filename="resultados_clustering.json"):
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

    def evaluate_clustering_cpu(self, embeddings_dict):
        """
        Autor: Marcos Aires (Nov.2024)
        Avalia o desempenho dos embeddings em tarefas de clustering
        usando diferentes algoritmos do scikit-learn e 
        validação cruzada, e mede o tempo de execução de cada algoritmo.
        Usa métricas do scikit-learn.

        Args:
            embeddings_dict: Um dicionário com os embeddings para cada modelo.
                            As chaves são os nomes dos modelos e os valores 
                            são os embeddings correspondentes.
        """
        print(f"\nIniciando avaliação de clustering com scikit-learn (CPU)...")

        from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

        resultados = {}
        for model_name, embeddings in embeddings_dict.items():
            print(f"Avaliando modelo: {model_name}")
            resultados[model_name] = {}

            # Converter os embeddings para arrays do NumPy
            embeddings_np = embeddings.cpu().numpy()

            # Carregar tempos para gerar embeddings para cada modelo
            tempos_embedding = self.load_statistics()

            for algorithm in self.algorithms:
                resultados[model_name][algorithm.__name__] = {"resultados": [], "tempo": []}

                estatisticas = tempos_embedding.get(model_name)
                if estatisticas:
                    tempos_execucao = estatisticas.get('tempo_execucao')
                    tempos_execucao = self.tempo_para_horas(tempos_execucao)

                # Instanciação do StratifiedKFold
                skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
                splits = list(skf.split(embeddings_np, np.zeros(len(embeddings_np))))

                resultados_split = []

                for train_index, test_index in splits:
                    X_train, X_test = embeddings_np[train_index], embeddings_np[test_index]

                    try:
                        if len(X_test) > 1:
                            print(f"  Split {len(resultados_split) + 1}:")
                            print(f"    Tamanho de X_train: {X_train.shape}")
                            print(f"    Tamanho de X_test: {X_test.shape}")

                            if algorithm.__name__ == "KMeans":
                                clustering_model = KMeans(n_clusters=8, random_state=42)
                            elif algorithm.__name__ == "DBSCAN":
                                clustering_model = DBSCAN(eps=0.5, min_samples=5)
                            elif algorithm.__name__ == "HDBSCAN":
                                clustering_model = HDBSCAN(min_cluster_size=5)

                            cluster_labels = clustering_model.fit_predict(X_test)
                            print(f"    Tamanho de cluster_labels: {cluster_labels.shape}")

                            # Verifica se cluster_labels contém mais de um valor único
                            if len(np.unique(cluster_labels)) > 1:
                                silhouette_avg = silhouette_score(X_test, cluster_labels)
                                calinski_harabasz = calinski_harabasz_score(X_test, cluster_labels)
                                davies_bouldin = davies_bouldin_score(X_test, cluster_labels)

                                resultados_split.append({
                                    "silhouette": silhouette_avg,
                                    "calinski_harabasz": calinski_harabasz,
                                    "davies_bouldin": davies_bouldin
                                })
                            else:
                                logging.warning(f"      HDBSCAN, modelo {model_name}, split {len(resultados_split) + 1}. Apenas um cluster encontrado. Pulando esta iteração.")

                        else:
                            print("      Ignorando split com menos de 2 amostras.")

                    except ValueError as e:
                        logging.error(f"    Erro ao calcular métricas: ValueError: {e}")

                    except Exception as e:
                        print(f"    Erro inesperado ao calcular métricas: {e}")
                        print(f"    {algorithm.__name__}, modelo {model_name}, split {len(resultados_split) + 1}. Pulando esta iteração.")

                resultados[model_name][algorithm.__name__]["resultados"] = resultados_split
                resultados[model_name][algorithm.__name__]["tempo"] = tempos_execucao

        self.salvar_resultados_cpu(resultados, filename="resultados_clustering.json")

        return resultados

    def tempo_para_horas(self, tempo):
        """Converte uma string de tempo no formato HH:MM:SS para um float em fração de horas.

        Args:
            tempo: A string de tempo no formato HH:MM:SS.

        Returns:
            Um float representando o tempo em fração de horas.
        """
        horas, minutos, segundos = map(int, tempo.split(':'))
        return horas + minutos/60 + segundos/3600

    def identificar_tipos_de_dados(self, data):
        """
        Identifica os tipos de dados presentes em uma estrutura de dados.
        """
        tipos = set()
        
        def percorrer(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    tipos.add(type(v))
                    percorrer(v)
            elif isinstance(obj, list):
                for item in obj:
                    tipos.add(type(item))
                    percorrer(item)
            else:
                tipos.add(type(obj))

        percorrer(data)
        return tipos


    def salvar_resultados(self, resultados, filename = "resultados_clustering.json"):
        """
        Autor: Marcos Aires (Nov.2024)
        Salva os resultados em um arquivo JSON local.
        """
        import numpy as np

        def converter_para_json_serializavel(obj):
            """
            Converte um objeto em um formato serializável em JSON.
            """
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Converte arrays NumPy para listas
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)  # Converte float32 e float64 para float
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)  # Converte int32 e int64 para int
            return obj

        # Converte os valores não serializáveis para serializáveis
        resultados_serializaveis = json.loads(json.dumps(resultados, default=converter_para_json_serializavel))

        # # Chama a função para identificar os tipos de dados em 'resultados'
        # tipos_encontrados = self.identificar_tipos_de_dados(resultados)

        # # Imprime os tipos encontrados
        # print(f"Tipos de dados encontrados em 'resultados': {tipos_encontrados}")   

        try:
            # Informar caminho para arquivo usando raiz do repositório Git como referência
            repo = Repo(search_parent_directories=True)
            root_folder = repo.working_tree_dir
            folder_data_output = os.path.join(str(root_folder), '_embeddings')
            pathfilename = os.path.join(folder_data_output, filename)

            # Criar a pasta _embeddings se ela não existir
            os.makedirs(folder_data_output, exist_ok=True)

            # Salvar o dicionário de resultados em arquivo JSON
            with open(pathfilename, 'w') as f:
                json.dump(resultados_serializaveis, f, indent=4)

            # Imprimir o nome do arquivo e o número de modelos
            print(f"Arquivo de resultados salvo: {filename}")
            print(f"Número de modelos avaliados: {len(resultados)}")

        except Exception as e:
            print(f"Erro ao salvar os resultados: {e}")


    def salvar_resultados_cpu(self, resultados, filename="resultados_clustering.json"):
        """
        Salva os resultados da avaliação de clustering em um arquivo JSON.
        """
        try:
            def converter_para_json_serializavel(obj):
                """
                Converte um objeto em um formato serializável em JSON.
                """
                if isinstance(obj, np.ndarray):
                    return obj.tolist()  # Converte arrays NumPy para listas
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)  # Converte float32 e float64 para float
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)  # Converte int32 e int64 para int
                return obj

            # Converte os valores não serializáveis para serializáveis
            resultados_serializaveis = json.loads(json.dumps(resultados, default=converter_para_json_serializavel))

            try:
                # Informar caminho para arquivo usando raiz do repositório Git como referência
                repo = Repo(search_parent_directories=True)
                root_folder = repo.working_tree_dir
                folder_data_output = os.path.join(str(root_folder), '_embeddings')
                pathfilename = os.path.join(folder_data_output, filename)

                # Criar a pasta _embeddings se ela não existir
                os.makedirs(folder_data_output, exist_ok=True)

                # Salvar o dicionário de resultados em arquivo JSON
                with open(pathfilename, 'w') as f:
                    json.dump(resultados_serializaveis, f, indent=4)

                # Imprimir o nome do arquivo e o número de modelos
                print(f"Arquivo de resultados salvo: {filename}")
                print(f"Número de modelos avaliados: {len(resultados)}")

            except Exception as e:
                print(f"Erro ao salvar os resultados: {e}")

        except Exception as e:
            print(f"Erro ao salvar os resultados: {e}")

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
        print(f"\nIniciando avaliação de clustering com cuML...")

        resultados = {}
        for model_name, embeddings in embeddings_dict.items():  # Usar embeddings_dict como entrada
            print(f"Avaliando modelo: {model_name}")
            resultados[model_name] = {}

            # Converter os embeddings para arrays do CuPy
            embeddings_cp = cp.array(embeddings.cpu().numpy())  # Converter para NumPy antes de converter para CuPy

            # Carregar tempos para gerar embeeding para cada modelo
            tempos_embedding = self.load_statistics()

            for algorithm in self.algorithms:
                resultados[model_name][algorithm.__name__] = {"resultados": [], "tempo": []}
                estatisticas = tempos_embedding.get(model_name)
                if estatisticas:
                    tempos_execucao = estatisticas['tempo'].get('tempo_execucao')
                    tempos_execucao = self.tempo_para_horas(tempos_execucao)

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

                            
                            cluster_labels = clustering_model.fit_predict(X_test)
                            print(f"    Tamanho de cluster_labels: {cluster_labels.shape}")

                            # Verifica se cluster_labels contém mais de um valor único
                            if len(cp.unique(cluster_labels)) > 1:

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
                                logging.warning(f"      HDBSCAN, modelo {model_name}, split {len(resultados_split) + 1}. Apenas um cluster encontrado. Pulando esta iteração.")

                        else:
                            print("      Ignorando split com menos de 2 amostras.")

                    except ValueError as e:
                        logging.error(f"    Erro ao calcular métricas: ValueError: {e}")
 
                    except cp.cuda.memory.OutOfMemoryError:
                        logging.error(f"    Erro de memória ao calcular métricas: OutOfMemoryError")
            
                    except Exception as e:
                        print(f"    Erro inesperado ao calcular métricas: {e}")
                        print(f"    {algorithm.__name__}, modelo {model_name}, split {len(resultados_split) + 1}. Pulando esta iteração.")

                resultados[model_name][algorithm.__name__]["resultados"] = resultados_split
                resultados[model_name][algorithm.__name__]["tempo"] = tempos_execucao

        self.salvar_resultados(resultados)
        
        return resultados


    def calcular_pontuacao_multicriterio(self, resultados, pesos):  # Adiciona pesos como argumento
        """
        Autor: Marcos Aires (Nov.2024)
        Calcula a soma ponderada por pesos para cada algoritmo e métrica.
        """
        # Carregar tempos para gerar embeeding para cada modelo
        tempos_embedding = self.load_statistics()

        pontuacoes = {}

        for model_name, model_results in resultados.items():
            pontuacoes[model_name] = {}
            max_valor_calinski = 0  # Inicializa o valor máximo do Calinski-Harabasz
            for algoritmo, resultados_algoritmo in model_results.items():
                
                # Extrai as pontuações das métricas
                pontuacoes_metricas = {metrica: [] for metrica in ['silhouette', 'calinski_harabasz', 'davies_bouldin']}
                for resultado in resultados_algoritmo['resultados']:
                    for metrica in pontuacoes_metricas:
                        pontuacoes_metricas[metrica].append(resultado[metrica])
                
                # Calcula as médias das métricas
                medias = [np.mean(pontuacoes_metricas[metrica]) for metrica in pontuacoes_metricas]
                
                # Lê o tempo para gerar embeeding de cada modelo
                tempos = tempos_embedding.get(model_name)

                # Encontra o valor máximo do Calinski-Harabasz entre todos os modelos e algoritmos
                max_valor_calinski = max(max_valor_calinski, medias[1])

                pontuacao = 0
                for i, metrica in enumerate(['silhouette', 'calinski_harabasz', 'davies_bouldin']):
                    valor = medias[i]  # Usa as médias calculadas
                    # Normaliza as métricas para ficarem na mesma escala (0 a 1)
                    if metrica == "silhouette":
                        valor_normalizado = (valor + 1) / 2  # Silhouette varia de -1 a 1
                    elif metrica == "davies_bouldin":
                        # Normalização corrigida para Davies-Bouldin
                        valor_normalizado = 1 - (valor / (max(medias[2] for model_results in resultados.values() for resultados_algoritmo in model_results.values()) + 1e-6))
                    elif metrica == "calinski_harabasz":
                        valor_normalizado = valor / max_valor_calinski  #Calinski-Harabasz maior melhor
                    
                    pontuacao += pesos[metrica] * valor_normalizado

                if tempos:
                    # print(model_name)
                    # print(f"{type(tempos)}: {tempos}")
                    tempo_str = tempos.get('tempo_execucao')
                    
                    # Converte a string de tempo para um objeto datetime
                    tempo_obj = datetime.strptime(tempo_str, "%H:%M:%S")
                    
                    # Calcula o tempo total em segundos
                    total_segundos = (tempo_obj.hour*3600) + (tempo_obj.minute*60) + tempo_obj.second
                    
                    # Converte o total de segundos para float
                    total_segundos_float = float(total_segundos)

                    # # Adiciona o tempo de execução normalizado entre algoritmos à pontuação
                    # tempo_normalizado = 1 - (total_segundos_float / (max(resultados_algoritmo['tempo'] 
                    # for model_results in resultados.values() for resultados_algoritmo in model_results.values()) + 1e-6))

                    # print(tempo_normalizado)
                    
                    pontuacao += pesos["tempo"] * total_segundos_float

                pontuacoes[model_name][algoritmo] = np.round(pontuacao,4)

        return pontuacoes

    def escolher_melhor_modelo(self, resultados):
        """
        Escolhe o modelo com a maior pontuação multicritério.
        """
        # Define os pesos para cada métrica (ajuste os valores conforme necessário)
        pesos = {
            "silhouette": 0.3,
            "calinski_harabasz": 0.3,
            "davies_bouldin": 0.3,
            "tempo": 0.1
        }

        pontuacoes = self.calcular_pontuacao_multicriterio(resultados, pesos)

        # Calcula a pontuação média para cada modelo
        pontuacoes_modelos = {}
        for model_name, model_pontuacoes in pontuacoes.items():
            
            # Calcula a médias de pontuações
            media = np.mean(list(model_pontuacoes.values()))
            
            # Truncar a média após a quarta casa decimal
            media_truncada = "{:.4f}".format(media)
            
            pontuacoes_modelos[model_name] = media_truncada

        # Seleciona o modelo com a maior pontuação média
        melhor_modelo = max(pontuacoes_modelos, key=pontuacoes_modelos.get) # type: ignore
        print('Pontuações médias ponderadas por cada modelo e algoritmo')
        for i,j in pontuacoes_modelos.items():
            print(f"  {i}: {j}")

        return melhor_modelo


    def plot_clustering_results_bars(self, resultados):
        """
        Plota os resultados do clustering usando Plotly, exibindo gráficos de barras
        para comparar o desempenho de cada modelo em cada métrica para cada algoritmo.
        """
        metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        algorithms = ['KMeans', 'DBSCAN', 'HDBSCAN']
        colors = ['blue', 'green', 'yellow', 'cyan', 'orange', 'purple']

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

                # Adicionar gráfico de barras para métrica e algoritmo com subplots
                for k, model_name in enumerate(model_values.keys()):
                    fig.add_trace(go.Bar(
                        x=[algorithm_name],
                        y=[model_values[model_name]],
                        name=model_name,  # Usar o nome do modelo na legenda
                        showlegend=i == 0 and j == 0,  # Exibe a legenda apenas no primeiro subplot
                        legendgroup=model_name,  # Agrupa as barras por modelo
                        offsetgroup=k,  # Define o offset para agrupar as barras por modelo
                        marker_color=colors[k]  # Cores para os modelos
                    ), row=1, col=i+1)  # Especificar a linha e coluna do subplot

        # Configura o layout do gráfico
        fig.update_layout(
            title="Comparação da qualidade de clustering com embeedings gerados em cada um dos modelos",
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

    def transform_df_report(self, df_report):
        """
        Transforms the df_report DataFrame:

        * Removes the "±" and everything after it from the "Silhouette", "Calinski-Harabasz", and "Davies-Bouldin" columns,
        keeping only the first value.
        * Converts the "Silhouette", "Calinski-Harabasz", "Davies-Bouldin", and "Tempo (s)" columns to numeric.
        """

        for col in ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]:
            df_report[col] = df_report[col].str.split("±").str[0]
            df_report[col] = pd.to_numeric(df_report[col])

        df_report["Tempo (s)"] = pd.to_numeric(df_report["Tempo (s)"])

        return df_report


    def generate_report_charts(self, df_report):
        """
        Autor: Marcos Aires (Nov.2024)
        Gera gráficos para o relatório de benchmarking usando Plotly.

        Args:
            df_report: O dataframe contendo os dados do relatório.
        """
        try:
            metrics = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Tempo (s)']
            criterio = ['(Maior melhor)', '(Maior melhor)', '(Menor melhor)', '(Menor melhor)']
            algorithms = df_report['Algoritmo'].unique()
            modelos = df_report['Modelo'].unique()

            # Ajustar o número de subplots de acordo com o número de métricas
            fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics, horizontal_spacing=0.15)

            # Cores para os modelos
            colors = ['blue', 'green', 'yellow', 'cyan', 'orange', 'purple']

            # Loop pelas métricas
            for i, metric_name in enumerate(metrics):
                # Loop pelos algoritmos
                for j, algorithm_name in enumerate(algorithms):
                    # Obter os valores da métrica para cada modelo
                    for k, modelo in enumerate(modelos):
                        df_filtered = df_report[(df_report['Modelo'] == modelo) & (df_report['Algoritmo'] == algorithm_name)]
                        valor = df_filtered[metric_name].values[0]

                        # Gráfico de barras para métrica e algoritmo em subplots
                        fig.add_trace(go.Bar(
                            x=[algorithm_name],
                            y=[valor],
                            name=modelo,  # Usar o nome do modelo na legenda
                            showlegend=i == 0 and j == 0,  # Exibe a legenda apenas no primeiro subplot
                            legendgroup=modelo,  # Agrupa as barras por modelo
                            offsetgroup=k,  # Define o offset para agrupar as barras por modelo
                            marker_color=colors[k]  # Cores para os modelos
                        ), row=1, col=i+1)  # Especificar a linha e coluna do subplot

            # Configura o layout do gráfico
            fig.update_layout(
                title="Comparação da qualidade de clustering com embeddings gerados em cada um dos modelos",
                height=600,
                width=1600
            )

            # Ajustar os títulos dos eixos
            for i, metric_name in enumerate(metrics):
                fig.update_xaxes(title_text="Algoritmo", row=1, col=i+1)
                fig.update_yaxes(title_text=f"{metric_name}", row=1, col=i+1)

            fig.show()

        except Exception as e:
            print(f"Erro ao gerar os gráficos do relatório: {e}")
            traceback.print_exc()


    def generate_seaborn_report_charts(self, df_report):
        """
        Gera gráficos para o relatório de benchmarking utilizando bilbioteca Seaborn.

        Args:
        df_report: O dataframe contendo os dados do relatório.
        """
        df_report = self.transform_df_report(df_report.copy())
        try:
            # 1. Configurar o estilo dos gráficos
            sns.set_theme(style="whitegrid")

            # 2. Criar os gráficos
            # Gráfico de barras para Silhouette Score
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Modelo", y="Silhouette", hue="Algoritmo", data=df_report)
            plt.title("Silhouette Score (Maior melhor)")
            plt.ylabel("Silhouette Score")
            plt.xticks(rotation=0)
            plt.show()

            # Gráfico de barras para Calinski-Harabasz Index
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Modelo", y="Calinski-Harabasz", hue="Algoritmo", data=df_report)
            plt.title("Calinski-Harabasz Index (Maior melhor)")
            plt.ylabel("Calinski-Harabasz Index")
            plt.xticks(rotation=0)
            plt.show()

            # Gráfico de barras para Davies-Bouldin Index
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Modelo", y="Davies-Bouldin", hue="Algoritmo", data=df_report)
            plt.title("Davies-Bouldin Index (Menor melhor)")
            plt.ylabel("Davies-Bouldin Index")
            plt.xticks(rotation=0)
            plt.show()

            # Gráfico de barras para Tempo de Execução
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Modelo", y="Tempo (s)", hue="Algoritmo", data=df_report)
            plt.title("Tempo de Execução (s) (Menor melhor)")
            plt.ylabel("Tempo (s)")
            plt.xticks(rotation=0)
            plt.show()

        except Exception as e:
            print(f"Erro ao gerar os gráficos do relatório: {e}")
            traceback.print_exc()

    def generate_report(self, resultados):
        """
        Autor: Marcos Aires (Nov.2024)
        Generates a report with the benchmarking results.
        """
        try:
            # 1. Preparar os dados para o relatório
            report_data = []
            for model_name, model_results in resultados.items():
                for algorithm_name, algorithm_results in model_results.items():

                    # Acessa as pontuações das métricas diretamente
                    pontuacoes_metricas = {metrica: [] for metrica in ['silhouette', 'calinski_harabasz', 'davies_bouldin']}
                    for resultado in algorithm_results['resultados']:
                        for metrica in pontuacoes_metricas:
                            pontuacoes_metricas[metrica].append(resultado[metrica])

                    # Calcula as médias e desvios das métricas
                    medias = [np.mean(pontuacoes_metricas[metrica]) for metrica in pontuacoes_metricas]
                    desvios = [np.std(pontuacoes_metricas[metrica]) for metrica in pontuacoes_metricas]

                    tempo = algorithm_results["tempo"]
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
            folder_data_output = os.path.join(str(os.getcwd()), '_reports')
            os.makedirs(folder_data_output, exist_ok=True)  # Criar a pasta se não existir
            report_filename = os.path.join(folder_data_output, 'benchmark_report.csv')
            df_report.to_csv(report_filename, index=False)

            # 4. Gerar os gráficos do relatório
            self.plot_clustering_results_bars(resultados)
            self.generate_report_charts(df_report)

        except Exception as e:
            print(f"Erro ao gerar o relatório: {e}")
            traceback.print_exc()