import os
import cudf
import json
import spacy
import cugraph
import numpy as np
import networkx as nx
import contextualSpellCheck

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.utils import to_dense_adj

from git import Repo
from scipy.fftpack import dct
from langdetect import detect
from tqdm.notebook import tqdm
from wordcloud import WordCloud
from nltk.corpus import stopwords
from pyvis.network import Network
from pytorch_metric_learning import losses
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

class RedeNeuralHibrida(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(RedeNeuralHibrida, self).__init__()
        self.device = device
        self.conv1 = gnn.GCNConv(num_features, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.to(self.device)

    def forward(self, x, edge_index, edge_weight=None):
        try:
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            if edge_weight is not None:
                edge_weight = edge_weight.to(self.device)

            x = self.conv1(x, edge_index, edge_weight)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index, edge_weight)

            # Aplicando a dinâmica de sincronização
            adj_matrix = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze(0)
            degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
            laplacian_matrix = degree_matrix - adj_matrix
            eigenvalues = torch.linalg.eigvals(laplacian_matrix)
            
            # 1. Calcular o segundo menor autovalor (valor de Fiedler)
            sorted_eigenvalues = torch.sort(eigenvalues.real)[0]  # Ordenar os autovalores
            fiedler_value = sorted_eigenvalues[1]  # Obter o segundo menor valor

            # 2. Ajustar os embeddings com base no valor de Fiedler
            # (exemplo de ajuste - ponderar os embeddings pelo inverso do valor de Fiedler)
            # (ajustes mais elaborados pode ser feito com base na estratégia de controle)
            embeddings = embeddings * (1 / (fiedler_value + 1e-6))  # Evitar divisão por zero

            return embeddings

        except Exception as e:
            print(f"Erro na RedeNeuralHibrida: {e}")
            raise e

    def inferir(self, grafo):
        try:
            x = torch.tensor(list(nx.get_node_attributes(grafo, 'features').values()), dtype=torch.float)
            edge_index = torch.tensor(list(grafo.edges()), dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(list(nx.get_edge_attributes(grafo, 'weight').values()), dtype=torch.float)
            
            self.eval()  # Mudar para o modo de avaliação
            with torch.no_grad():  # Desabilitar o cálculo de gradientes
                embeddings = self(x, edge_index, edge_weight)
            
            # 1. Aplicar um algoritmo de clustering não-supervisionado
            from sklearn.cluster import KMeans # (Exemplo com k-means)
            kmeans = KMeans(n_clusters=5)  # Definir o número de clusters
            clusters = kmeans.fit_predict(embeddings.cpu().detach().numpy())  # Executar o k-means

            return clusters, embeddings

        except Exception as e:
            print(f"Erro na inferência da RedeNeuralHibrida: {e}")
            raise e

class RedeNeuralKAN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers=3, dropout=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(RedeNeuralKAN, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, num_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.gnn_layer = gnn.GCNConv(num_classes, num_classes)  # Camada GNN para capturar interações
        self.to(self.device)

    def forward(self, x, edge_index, edge_weight=None):
        try:
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            if edge_weight is not None:
                edge_weight = edge_weight.to(self.device)

            # Aplicando a KAN (f(⋅))
            for layer in self.layers[:-1]:
                x = layer(x)
                x = self.relu(x)
                x = self.dropout(x)
            x = self.layers[-1](x)

            # Aplicando a GNN para capturar interações entre os nós
            x = self.gnn_layer(x, edge_index, edge_weight)

            return x

        except Exception as e:
            print(f"Erro na RedeNeuralKAN: {e}")
            raise e

def inferir(self, grafo):
        '''
        Explicação do código:

        Preparação dos dados:
        x: Tensor com as features dos nós do grafo.
        edge_index: Tensor com os índices das arestas do grafo.
        edge_weight: Tensor com os pesos das arestas do grafo.

        Inferência:
        self.eval(): Coloca o modelo em modo de avaliação.
        with torch.no_grad(): Desabilita o cálculo de gradientes, economizando memória e tempo de processamento.
        embeddings = self(x, edge_index, edge_weight): Obtém os embeddings dos nós utilizando o modelo.

        Clustering:
        from sklearn.cluster import KMeans: Importa o algoritmo k-means.
        kmeans = KMeans(n_clusters=5): Cria um objeto k-means com 5 clusters.
        clusters = kmeans.fit_predict(embeddings.cpu().detach().numpy()): Executa o k-means nos embeddings.
        cpu(): Move os embeddings para a CPU, pois o k-means do scikit-learn não funciona diretamente com tensores na GPU.
        detach(): Remove os embeddings do grafo computacional do PyTorch.
        numpy(): Converte os embeddings em um array NumPy.
        '''
        try:
            x = torch.tensor(list(nx.get_node_attributes(grafo, 'features').values()), dtype=torch.float)
            edge_index = torch.tensor(list(grafo.edges()), dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(list(nx.get_edge_attributes(grafo, 'weight').values()), dtype=torch.float)
            
            self.eval()  # Mudar para o modo de avaliação
            with torch.no_grad():  # Desabilitar o cálculo de gradientes
                embeddings = self(x, edge_index, edge_weight)
            
            # 1. Aplicar um algoritmo de clustering não-supervisionado
            from sklearn.cluster import KMeans #    (Exemplo com k-means)
            kmeans = KMeans(n_clusters=5)  # Definir o número de clusters
            clusters = kmeans.fit_predict(embeddings.cpu().detach().numpy())  # Executar o k-means

            return clusters, embeddings

        except Exception as e:
            print(f"Erro na inferência da RedeNeuralKAN: {e}")
            raise e

class RedeNeuralFourier(nn.Module):
    '''
    Observações:

    Transformada de Fourier: A Transformada Discreta do Cosseno (DCT) é utilizada para extrair características estruturais dos features dos nós. 
    A biblioteca scipy.fftpack é utilizada para calcular a DCT.
    
    Concatenação de Features: Os features originais e os features transformados são concatenados para fornecer mais informações para as camadas GCN.
    
    GCN: Duas camadas GCNConv (gnn.GCNConv) são utilizadas para processar os features e gerar os embeddings.
    
    Controle de Exceções: O código inclui blocos try-except para capturar e tratar exceções.
    
    Aceleração por GPU: O código utiliza o parâmetro device para executar o modelo na GPU, se disponível.
    
    Clustering: O código inclui um exemplo de como realizar o clustering com k-means, mas você pode utilizar outro algoritmo de clustering.
    '''
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(RedeNeuralFourier, self).__init__()
        self.device = device
        self.conv1 = gnn.GCNConv(num_features, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.to(self.device)

    def forward(self, x, edge_index, edge_weight=None):
        try:
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            if edge_weight is not None:
                edge_weight = edge_weight.to(self.device)

            # 1. Aplicar a Transformada de Fourier (DCT) nos features dos nós
            x_fft = torch.tensor(dct(x.cpu().detach().numpy(), type=2, norm='ortho'), dtype=torch.float).to(self.device)

            # 2. Concatenar os features originais com os features transformados
            x = torch.cat((x, x_fft), dim=1)

            # 3. Aplicar as camadas GCN
            x = self.conv1(x, edge_index, edge_weight)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index, edge_weight)

            return x

        except Exception as e:
            print(f"Erro na RedeNeuralFourier: {e}")
            raise e

    def inferir(self, grafo):
        try:
            x = torch.tensor(list(nx.get_node_attributes(grafo, 'features').values()), dtype=torch.float)
            edge_index = torch.tensor(list(grafo.edges()), dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(list(nx.get_edge_attributes(grafo, 'weight').values()), dtype=torch.float)
            
            self.eval()  # Mudar para o modo de avaliação
            with torch.no_grad():  # Desabilitar o cálculo de gradientes
                embeddings = self(x, edge_index, edge_weight)
            
            # 1. Aplicar um algoritmo de clustering não-supervisionado
            #    (Exemplo com k-means)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5)  # Definir o número de clusters
            clusters = kmeans.fit_predict(embeddings.cpu().detach().numpy())  # Executar o k-means

            return clusters, embeddings

        except Exception as e:
            print(f"Erro na inferência da RedeNeuralFourier: {e}")
            raise e

## Funções para avaliar desempenho
def calcular_coeficiente_silhueta(embeddings, clusters):
    """
    Calcula o Coeficiente de Silhueta para os clusters gerados.

    Args:
      embeddings: Uma lista de embeddings dos nós.
      clusters: Uma lista de clusters aos quais os nós pertencem.

    Returns:
      O valor do Coeficiente de Silhueta.
    """
    return silhouette_score(embeddings, clusters)

def calcular_indice_davies_bouldin(embeddings, clusters):
    """
    Calcula o Índice de Davies-Bouldin para os clusters gerados.

    Args:
        embeddings: Uma lista de embeddings dos nós.
        clusters: Uma lista de clusters aos quais os nós pertencem.

    Returns:
        O valor do Índice de Davies-Bouldin.
    """
    return davies_bouldin_score(embeddings, clusters)

def calcular_modularidade(grafo, clusters):
    """
    Calcula a Modularidade para os clusters gerados.

    Args:
        grafo: O grafo de conhecimento (objeto NetworkX).
        clusters: Uma lista de clusters aos quais os nós pertencem.

    Returns:
        O valor da Modularidade.
    """
    particao = {}
    for i, no in enumerate(grafo.nodes()):
        particao[no] = clusters[i]
    return nx.community.modularity(grafo, particao.values())

def calcular_condutividade(grafo, clusters):
    """
    Calcula a Condutividade para os clusters gerados.
    
    Args:
        grafo: O grafo de conhecimento (objeto NetworkX).
        clusters: Uma lista de clusters aos quais os nós pertencem.

    Returns:
        O valor médio da Condutividade dos clusters.
    """
    condutividades = []
    for cluster_id in set(clusters):
        nos_cluster = [no for i, no in enumerate(grafo.nodes()) if clusters[i] == cluster_id]
        subgrafo = grafo.subgraph(nos_cluster)
        peso_total = subgrafo.size(weight='weight')  # Soma dos pesos das arestas
        condutividade = peso_total / len(nos_cluster)
        condutividades.append(condutividade)
    return np.mean(condutividades)

class TesteRedeNeural:
    def __init__(self, grafo, parametros_modelo):
        """
        Inicializa a classe de testes.

        Args:
          grafo: O grafo de conhecimento (objeto NetworkX).
          parametros_modelo: Os parâmetros da rede neural.
        """
        self.grafo = grafo
        self.parametros_modelo = parametros_modelo

    def avaliar_desempenho(self, clusters):
        """
        Avalia o desempenho da rede neural.

        Args:
          clusters: Os clusters gerados pela rede neural.

        Returns:
          Um dicionário com as métricas de avaliação calculadas.
        """
        metricas = {}
        # Calcula as métricas de avaliação
        metricas['coeficiente_silhueta'] = calcular_coeficiente_silhueta(self.grafo.nodes(), clusters)
        metricas['indice_davies_bouldin'] = calcular_indice_davies_bouldin(self.grafo.nodes(), clusters)
        metricas['modularidade'] = calcular_modularidade(self.grafo, clusters)
        metricas['condutividade'] = calcular_condutividade(self.grafo, clusters)
        return metricas

    def testar_inferencia(self, clusters, embeddings):
        """
        Testa a inferência da rede neural.

        Args:
          clusters: Os clusters gerados pela rede neural.
          embeddings: Os embeddings dos nós.

        Returns:
          As análises de cluster geradas.
        
        a) Identificar as competências existentes:
        Cria um dicionário competencias_por_cluster para armazenar as competências de cada cluster.
        Itera sobre os nós do grafo, obtendo o cluster ao qual cada nó pertence e as suas competências (atributo competencias).
        Adiciona as competências do nó ao conjunto de competências do seu cluster.
        
        b) Identificar as lacunas de competências:
        Cria um dicionário lacunas_por_produto para armazenar as lacunas de competências de cada produto.
        Itera sobre os nós do grafo, verificando se o nó é um produto estratégico (atributo tipo igual a 'produto').
        Para cada produto, obtém as competências necessárias (atributo competencias).
        Compara as competências necessárias com as competências de cada cluster, identificando as lacunas.
        Armazena as lacunas no dicionário lacunas_por_produto.

        c) Gerar as análises de cluster:

        Chama o método avaliar_desempenho para calcular as métricas de avaliação.
        Cria um dicionário analises contendo as métricas, os clusters, as competências por cluster e as lacunas por produto.
        """

        # 1. Identificar as competências existentes
        competencias_por_cluster = {}
        for i, no in enumerate(self.grafo.nodes()):
            cluster_id = clusters[i]
            if cluster_id not in competencias_por_cluster:
                competencias_por_cluster[cluster_id] = set()
            competencias_por_cluster[cluster_id].update(self.grafo.nodes[no].get('competencias', []))

        # 2. Identificar as lacunas de competências
        lacunas_por_produto = {}
        for no in self.grafo.nodes():
            if self.grafo.nodes[no].get('tipo') == 'produto':
                competencias_necessarias = self.grafo.nodes[no].get('competencias', [])
                for cluster_id, competencias_cluster in competencias_por_cluster.items():
                    lacunas = set(competencias_necessarias) - competencias_cluster
                    if lacunas:
                        if no not in lacunas_por_produto:
                            lacunas_por_produto[no] = {}
                        lacunas_por_produto[no][cluster_id] = lacunas

        # 3. Gerar as análises de cluster (incluindo as métricas)
        analises = self.avaliar_desempenho(clusters)
        analises['clusters'] = clusters
        analises['competencias_por_cluster'] = competencias_por_cluster
        analises['lacunas_por_produto'] = lacunas_por_produto

        return analises

# Classe para pré-processamento com tradução de português para inglês e correção ortográfica
class ENPreprocessor:
    def __init__(self):
        # Carregar o modelo de tradução e o tokenizador do Hugging Face
        self.model_name = "unicamp-dl/translation-pt-en-t5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_tr = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_tr.to(self.device)

        # Carregar o modelo transformer para o Inglês
        self.nlp_en = spacy.load("en_core_web_trf")

        # Adicionar ao pipeline correção ortográfica
        contextualSpellCheck.add_to_pipe(self.nlp_en)

        # Carregar as stopwords em inglês
        self.stop_words_en = set(stopwords.words('english'))

        # Adicionar as stopwords personalizadas em inglês
        self.stop_words_en.update(["must", "due", "track", "may", "non", "year", "apply", "prepare", "era", "eligibility",
                              "funded value", "deadline", "application form", "description", "name", "address", "phone",
                              "Fax", "e-mail", "email", "contact", "homepage", "home page", "home", "page"])

    def translate_to_en(self, nome):
        try:
            # Traduzir usando o modelo Hugging Face pré-treinado em tradução pt/en
            inputs = self.tokenizer(nome, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model_tr.generate(**inputs, max_new_tokens=512)
            translation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return translation
        except Exception as e:
            print(f"Erro na tradução: {e}")
            return nome

    def detect_language(self, text):
        try:
            return detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            logging.error(f"Erro ao identificar linguagem")
            return 'unknown'

    def preprocess_text(self, text):
        # Traduzir o texto para inglês (se necessário) em lote
        try:
            # logging.info("Traduzindo texto para o inglês (se necessário)...")
            text_translated = self.translate_to_en([text])[0] if self.detect_language(text) != 'en' else text
        except Exception as e:
            logging.error(f"Erro na tradução: {e}")
            return []

        # Converter para minúsculas e remover pontuação
        # logging.info("Limpando e normalizando o texto...")
        text_translated = text_translated.lower().translate(str.maketrans('', '', string.punctuation))

        # Truncar o texto traduzido se for muito longo
        max_length = 512  # Ajuste conforme necessário
        text_translated = text_translated[:max_length]

        # Aplicar o corretor ortográfico e lematizar em inglês em lote (usando pipe do spaCy)
        # logging.info("Processando o texto com spaCy...")
        with self.nlp_en.disable_pipes('ner'):  # Desabilitar NER para economizar memória da GPU
            # Definir o tamanho do lote para cada envio de entradas para a GPU
            docs = self.nlp_en.pipe([text_translated], batch_size=64)

        for doc in docs:
            words_en = [token.lemma_.lower() if token.text.lower() not in ["institute", "institution", "institutional"] else "institution"
                        for token in doc
                        if token.is_alpha and not token.is_stop and token.lemma_.lower() not in self.stop_words_en]

        return words_en


# Classe para pré-processamento com tradução de inglês para português
class BRPreprocessor:
    def __init__(self):
        # Carregar o modelo de tradução e o tokenizador do Hugging Face
        self.model_name = "unicamp-dl/translation-en-pt-t5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_tr = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_tr.to(self.device)

        # Carregar o modelo de linguagem em português do Spacy
        self.nlp_pt = spacy.load("pt_core_news_sm")

        # Adicionar o corretor ortográfico contextual ao pipeline do spaCy (se disponível para português)
        contextualSpellCheck.add_to_pipe(self.nlp_pt)

        # Carregar as stopwords em português
        self.stop_words_pt = set(stopwords.words('portuguese'))

        # Adicionar as stopwords personalizadas em português
        self.stop_words_pt.update(["deve", "devido", "acompanhar", "pode", "não", "ano", "aplicar", "preparar", "era", "elegibilidade",
                              "valorfinanciado", "datalimite", "formuláriodesolicitacao", "descrição", "homepage", "nome",
                              "endereço", "telefone", "fax", "e-mail", "contato", "home page", "casa", "página"])

    def translate_to_pt(self, texts):
        try:
            # Traduzir usando o modelo Hugging Face pré-treinado em tradução en/pt
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            outputs = self.model_tr.generate(**inputs)
            translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return translations
        except Exception as e:
            print(f"Erro na tradução: {e}")
            return texts

    def detect_language(self, text):
        try:
            return detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            logging.error(f"Erro ao identificar linguagem")
            return 'unknown'

    def preprocess_text(self, text):
        # Traduzir o texto para português (se necessário)
        try:
            # logging.info("Traduzindo texto para o português (se necessário)...")
            text_translated = self.translate_to_pt([text])[0] if self.detect_language(text) != 'pt' else text
        except Exception as e:
            logging.error(f"Erro na tradução: {e}")
            return []

        # Converter para minúsculas e remover pontuação
        # logging.info("Limpando e normalizando o texto...")
        text_translated = text_translated.lower().translate(str.maketrans('', '', string.punctuation))

        # Truncar o texto traduzido se for muito longo
        max_length = 512
        text_translated = text_translated[:max_length]

        # Lematizar em português
        # logging.info("Lematizando o texto...")
        doc_pt = self.nlp_pt(text_translated)
        words_pt = [token.lemma_.lower()
                    for token in doc_pt
                    if token.is_alpha and not token.is_stop and token.lemma_.lower() not in self.stop_words_pt
                    and not (token.pos_ == "PROPN" and token.text.lower() not in self.stop_words_pt)]

        return words_pt

class SemanticMatcher:
    def __init__(self, curriculos_data, matriz_ceis_data, relacoes_biologicos,
                 relacoes_pequenas_moleculas):
        self.curriculos_data = curriculos_data
        self.matriz_ceis_data = matriz_ceis_data
        self.relacoes_biologicos = relacoes_biologicos
        self.relacoes_pequenas_moleculas = relacoes_pequenas_moleculas

        # Inicializar o modelo de tradução e o tokenizador
        self.model_name = "unicamp-dl/translation-pt-en-t5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_tr = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_tr.to(self.device)

        self.produtos_df = self.criar_dataframe_produtos()
        self.biologicos, self.pequenas_moleculas = self.criar_grafos_relacionamentos()

        # Carregar o modelo transformer para o Inglês
        self.nlp_en = spacy.load("en_core_web_trf")

        # Adicionar ao pipeline correção ortográfica
        contextualSpellCheck.add_to_pipe(self.nlp_en)

        # Carregar as stopwords em inglês
        self.stop_words_en = set(stopwords.words('english'))

        # Adicionar as stopwords personalizadas em inglês
        self.stop_words_en.update(["must", "due", "track", "may", "non", "year", "apply", "prepare", "era", "eligibility",
                                        "funded value", "deadline", "application form", "description", "name", "address", "phone",
                                        "Fax", "e-mail", "email", "contact", "homepage", "home page", "home", "page"])

    def traduzir_nomes_produtos(self):
        # Função para traduzir um lote de nomes de produtos
        def batch_translate(nomes):
            try:
                inputs = self.tokenizer(nomes.to_arrow().to_pylist(), return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model_tr.generate(**inputs, max_new_tokens=512)
                translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                return translations
            except Exception as e:
                print(f"Erro na tradução: {e}")
                return nomes

        # Aplicar a tradução em lotes
        self.produtos_df['nome_en'] = self.produtos_df['nome'].map_partitions(batch_translate, meta="str")

    def criar_dataframe_produtos(self):
        # Informar caminho para arquivo CSV usando raiz do repositório Git como referência
        repo = Repo(search_parent_directories=True)
        root_folder = repo.working_tree_dir
        json_folder = os.path.join(root_folder, '_data', 'out_json')

        # Carregar a Matriz CEIS
        with open(os.path.join(json_folder,'matriz_ceis.json'), 'r') as f:
            matriz_ceis_data = json.load(f)

        # Extrair os dados dos produtos
        produtos = []
        for bloco in matriz_ceis_data['blocos']:
            for produto in bloco['produtos']:
                produto['bloco_id'] = bloco['id']
                produto['bloco_nome'] = bloco['titulo']
                produtos.append(produto)

        # Retornar o DataFrame cuDF
        return cudf.DataFrame(produtos)

    def criar_grafos_relacionamentos(self):
        # ... (código para criar os grafos de biológicos e pequenas moléculas) ...
        biologicos = nx.DiGraph()
        for node in self.relacoes_biologicos["nodes"]:
            biologicos.add_node(node['id'], **node)
        for edge in self.relacoes_biologicos["edges"]:
            biologicos.add_edge(edge['from'], edge['to'])

        pequenas_moleculas = nx.DiGraph()
        for node in self.relacoes_pequenas_moleculas["nodes"]:
            pequenas_moleculas.add_node(node['id'], **node)
        for edge in self.relacoes_pequenas_moleculas["edges"]:
            pequenas_moleculas.add_edge(edge['from'], edge['to'])
        return biologicos, pequenas_moleculas

    def traduzir_nomes_produtos(self):
        self.produtos_df['nome_en'] = self.produtos_df['nome'].apply(lambda x: ENPreprocessor.translate_to_en([x])[0])

    def extrair_caracteristicas(self):
        # ... (código para extrair características semânticas) ...
        pass  # Implemente a extração de características aqui

    def classificar_produtos(self):
        # ... (código para classificar os produtos) ...
        pass  # Implemente a classificação dos produtos aqui

    def calcular_similaridade(self, produto, grafo, tipo_transformada):
        # ... (código para calcular similaridade usando a abordagem especificada) ...
        pass  # Implemente o cálculo de similaridade aqui

    def conectar_produtos_grafo(self):
        # ... (código para conectar os produtos aos grafos) ...
        pass  # Implemente a conexão dos produtos aos grafos aqui

    def avaliar_desempenho(self):
        # ... (código para avaliar o desempenho das abordagens) ...
        pass  # Implemente a avaliação de desempenho aqui