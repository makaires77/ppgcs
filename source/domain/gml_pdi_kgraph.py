import math
import os
import json
import unicodedata
import networkx as nx
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain

from tqdm import tqdm
from numpy import False_
from pathlib import Path
from tabnanny import verbose
from pyvis.network import Network
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util


class Utils:
    @staticmethod
    def primeira_letra_maiuscula(texto):
        """
        Converter a primeira letra de uma string para maiúscula.
        """
        if not texto:
            return texto
        return texto[0].upper() + texto[1:]

class GraphDataPreprocessor:
    """
    Classe se pré-processamento para treinamento de layout dinâmico de acordo com dados do KG
    Extrai features estruturais dos nós, cria mapeamentos para tipos de nós e arestas e gera representações numéricas das features que serão utilizadas para gerar o layout dinâmico.   
    """
    def __init__(self, graph):
        self.graph = graph
        self.node_features = {}
        self.edge_features = {}
        self.node_mappings = {}
        self.edge_type_mappings = {}
        
    def extract_node_features(self):
        """Extrai features dos nós e cria mapeamentos para embeddings"""
        node_types = set()
        for node, data in self.graph.nodes(data=True):
            node_types.add(data.get('tipo', 'unknown'))
        
        # Criar mapeamento one-hot para tipos de nós
        self.node_mappings = {t: i for i, t in enumerate(sorted(node_types))}
        
        # Gerar features para cada nó
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('tipo', 'unknown')
            degree = self.graph.degree(node)
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            
            self.node_features[node] = {
                'type_idx': self.node_mappings[node_type],
                'degree': degree,
                'in_degree': in_degree,
                'out_degree': out_degree
            }
    
    def extract_edge_features(self):
        """Extrai features das arestas e cria mapeamentos para embeddings"""
        edge_types = set()
        for _, _, data in self.graph.edges(data=True):
            edge_types.add(data.get('relation', 'unknown'))
        
        # Criar mapeamento one-hot para tipos de arestas
        self.edge_type_mappings = {t: i for i, t in enumerate(sorted(edge_types))}
        
        # Gerar features para cada aresta
        for source, target, data in self.graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            self.edge_features[(source, target)] = {
                'type_idx': self.edge_type_mappings[relation]
            }

## PyTorchGraphDataset
import torch

class PyTorchGraphDataset:
    """
    Classe para converter dados do grafo para tensores PyTorch, suporta processamento em GPU quando disponível, gerando as estruturas de dados otimizadas para treinamento.
    """
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_tensors(self):
        """Converte features em tensores PyTorch"""
        # Nós
        node_ids = list(self.preprocessor.graph.nodes())
        node_features = torch.zeros(
            (len(node_ids), len(self.preprocessor.node_mappings)), 
            device=self.device
        )
        
        for idx, node in enumerate(node_ids):
            type_idx = self.preprocessor.node_features[node]['type_idx']
            node_features[idx, type_idx] = 1
            
        # Arestas
        edge_index = []
        edge_type = []
        for source, target in self.preprocessor.graph.edges():
            source_idx = node_ids.index(source)
            target_idx = node_ids.index(target)
            edge_index.append([source_idx, target_idx])
            edge_type.append(
                self.preprocessor.edge_features[(source, target)]['type_idx']
            )
            
        edge_index = torch.tensor(edge_index, device=self.device).t()
        edge_type = torch.tensor(edge_type, device=self.device)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'node_ids': node_ids
        }


class EdgeStyleManager:
    def __init__(self):
        # Dicionário para mapear espessuras de acordo com os tipos de relacionamentos
        self.edge_widths = {
            'TEM_INTERESSE': 4.0,
            'TEM_SIMILARIDADE': 1.0,
            'SEGUIDO_POR': 2.0,
            'PRODUZ': 3.0,
            'FORNECE': 3.0,
            'TEM_BLOCO': 2.5,
            'TEM_PESQUISADOR': 2.5
        }
        
        # Configurações de curva por tipo de relação
        self.edge_smoothness = {
            'default': {
                'type': 'curvedCW',
                'roundness': 0.2
            }
        }
    
    def get_edge_style(self, edge_type, source_type, target_type):
        return {
            'width': self.edge_widths.get(edge_type, 1.0),
            'smooth': self.edge_smoothness.get(edge_type, self.edge_smoothness['default'])
        }

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import networkx as nx

class EdgeLengthPredictor:
    """
    Classe para predizer o comprimento de arestas de acordo com aprendizado de máquina no grafo
    Usa mapeamentos para codificar tipos de nós e relações e extrai features relevantes das arestas incluindo:
    - Tipos dos nós fonte e destino
    - Tipo da relação
    - Grau dos nós
    - Centralidade dos nós
    - Informação de comunidade
    
    Calcula os comprimentos iniciais baseados nos tipos dos nós e treina um modelo RandomForest para prever comprimentos ideais, com métodos para predição de comprimentos das novas arestas
    """
    def __init__(self, graph):
        self.graph = graph
        self.model = None
        self.node_type_mapping = {
            'produto': 0,
            'pesquisador': 1,
            'plataforma': 2,
            'desafio': 3,
            'bloco': 4,
            'area': 5,
            'instituicao': 6,
            'processo_biologico': 7,
            'processo_smallmolecule': 8
        }
        self.relation_type_mapping = {
            'TEM_INTERESSE': 0,
            'TEM_SIMILARIDADE': 1,
            'SEGUIDO_POR': 2,
            'PRODUZ': 3,
            'FORNECE': 4,
            'TEM_BLOCO': 5,
            'TEM_PESQUISADOR': 6
        }
    
    def _encode_node_type(self, node_type):
        return self.node_type_mapping.get(node_type, -1)
    
    def _encode_relation_type(self, relation_type):
        return self.relation_type_mapping.get(relation_type, -1)
    
    def extract_edge_features(self):
        features = []
        lengths = []
        
        for source, target, data in self.graph.edges(data=True):
            # Features dos nós
            source_type = self.graph.nodes[source].get('tipo')
            target_type = self.graph.nodes[target].get('tipo')
            relation_type = data.get('relation')
            
            # Métricas do grafo
            source_degree = self.graph.degree(source)
            target_degree = self.graph.degree(target)
            source_centrality = nx.degree_centrality(self.graph)[source]
            target_centrality = nx.degree_centrality(self.graph)[target]
            
            # Métricas de comunidade
            try:
                # communities = nx.community.louvain_communities(self.graph.to_undirected())
                import community.community_louvain as community_louvain
                partition = community_louvain.best_partition(self.graph.to_undirected())
                same_community = partition[source] == partition[target]
            except (ImportError, Exception) as e:
                print(f"  Erro ao calcular comunidades: {e}")
                same_community = False

            # Criar vetor de features
            edge_features = [
                self._encode_node_type(source_type),
                self._encode_node_type(target_type),
                self._encode_relation_type(relation_type),
                source_degree,
                target_degree,
                source_centrality,
                target_centrality,
                int(same_community)
            ]
            
            # Calcular comprimento inicial baseado nos tipos
            initial_length = self._calculate_initial_length(source_type, target_type)
            
            features.append(edge_features)
            lengths.append(initial_length)
            
        return np.array(features), np.array(lengths)
    
    def _calculate_initial_length(self, source_type, target_type):
        """Calcula comprimento inicial baseado nos tipos dos nós"""
        base_length = 200
        
        # Ajustar comprimento baseado nos tipos
        if source_type == target_type:
            return base_length * 0.8
        elif source_type == 'instituicao' or target_type == 'instituicao':
            return base_length * 1.5
        elif 'processo' in str(source_type) or 'processo' in str(target_type):
            return base_length * 1.2
        
        return base_length
    
    def train_model(self):
        """Treina o modelo de predição de comprimento de arestas"""
        X, y = self.extract_edge_features()
        
        if len(X) == 0:
            raise ValueError("Não há dados suficientes para treinar o modelo")
        
        # Dividir dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Treinar modelo
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Avaliar modelo
        score = self.model.score(X_test, y_test)
        print(f"  Atualizar layout (Avaliação de treino com R² Score: {score:.4f})")
        
        return score
    
    def predict_edge_length(self, source, target, relation):
        """Prediz o comprimento ideal para uma aresta"""
        if not self.model:
            raise ValueError("Modelo não treinado. Execute train_model() primeiro.")
        
        features = self._get_edge_features(source, target, relation)
        return self.model.predict([features])[0]
    
    def _get_edge_features(self, source, target, relation):
        """Extrai features para uma aresta específica"""
        source_type = self.graph.nodes[source].get('tipo')
        target_type = self.graph.nodes[target].get('tipo')
        
        source_degree = self.graph.degree(source)
        target_degree = self.graph.degree(target)
        source_centrality = nx.degree_centrality(self.graph)[source]
        target_centrality = nx.degree_centrality(self.graph)[target]
        
        # Métricas de comunidade
        try:
            # communities = nx.community.louvain_communities(self.graph.to_undirected())
            import community.community_louvain as community_louvain
            partition = community_louvain.best_partition(self.graph.to_undirected())
            same_community = partition[source] == partition[target]
        except (ImportError, Exception) as e:
            print(f"  Erro ao calcular comunidades: {e}")
            same_community = False

        return [
            self._encode_node_type(source_type),
            self._encode_node_type(target_type),
            self._encode_relation_type(relation),
            source_degree,
            target_degree,
            source_centrality,
            target_centrality,
            int(same_community)
        ]

class NodeMetricsCalculator:
    """
    Classe para calcular os parâmetros dos subgrafos que serão utilizando no treinamento e na gerção de layout para visualização dinâmica.
    Calcula métricas básicas que funcionam em grafos desconexos normalmente, trata a centralidade de autovetor por componente conectado e fornece fallbacks para quando o cálculo falha em algum componente. Mantém valores zero para nós em componentes isolados, suportando tanto grafos direcionados quanto não direcionados
    """
    def __init__(self, graph):
        self.graph = graph
        self.metrics = {}

    def get_node_metrics(self, node_id):
        """
        Retorna todas as métricas calculadas para um nó específico.
        
        Args:
            node_id: Identificador do nó
            
        Returns:
            dict: Dicionário com todas as métricas do nó
        """
        return {
            metric_name: metric_values[node_id]
            for metric_name, metric_values in self.metrics.items()
            if node_id in metric_values
        }

    def calculate_all_metrics(self):
        """Calcula todas as métricas de centralidade disponíveis"""
        # Métricas que funcionam em grafos desconexos
        self.metrics['degree'] = nx.degree_centrality(self.graph)
        self.metrics['betweenness'] = nx.betweenness_centrality(self.graph)
        
        # Tratar centralidade de proximidade para grafos desconexos
        self.metrics['closeness'] = nx.closeness_centrality(self.graph)
        
        # Calcular centralidade de autovetor por componente
        self.metrics['eigenvector'] = self._calculate_eigenvector_by_component()
        
        return self.metrics
    
    def _calculate_eigenvector_by_component(self):
        eigenvector_centrality = {node: 0.0 for node in self.graph.nodes()}
        
        # Identificar componentes conectados
        if self.graph.is_directed():
            components = nx.strongly_connected_components(self.graph)
        else:
            components = nx.connected_components(self.graph)
        
        for component in components:
            if len(component) > 1:
                subgraph = self.graph.subgraph(component)
                try:
                    # Para componentes pequenos, usar método power iteration
                    if len(component) < 50:
                        comp_centrality = nx.eigenvector_centrality(
                            subgraph,
                            max_iter=1000,
                            tol=1e-6
                        )
                    # Para componentes maiores, usar método numpy
                    else:
                        comp_centrality = nx.eigenvector_centrality_numpy(subgraph)
                        
                    # Atualizar valores no dicionário principal
                    for node, value in comp_centrality.items():
                        eigenvector_centrality[node] = value
                except:
                    # Fallback para centralidade de grau se ambos métodos falharem
                    comp_centrality = nx.degree_centrality(subgraph)
                    for node, value in comp_centrality.items():
                        eigenvector_centrality[node] = value
        
        return eigenvector_centrality



class NodeStyleManager:
    def __init__(self):
        self.node_styles = {
            'produto': {
                'shape': 'dot',
                'color': '#90EE90',  # verde claro
                'base_size': 20
            },
            'pesquisador': {
                'shape': 'star',
                'color': '#FFFF00',  # amarelo claro
                'base_size': 20
            },
            'plataforma': {
                'shape': 'triangle',
                'color': '#A9A9A9',  # cinza
                'base_size': 15
            },
            'desafio': {
                'shape': 'diamond',
                'color': '#FFB6C1',  # rosa claro
                'base_size': 30
            },
            'bloco': {
                'shape': 'box',
                'color': '#87CEEB',  # azul claro
                'base_size': 40
            },
            'instituicao': {
                'shape': 'dot',
                'color': '#FFA500',  # laranja
                'base_size': 100
            },
            'processo_biologico': {
                'shape': 'square',
                'color': '#008000',  # verde escuro
                'base_size': 20
            },
            'processo_smallmolecule': {
                'shape': 'square',
                'color': '#90EE90',  # verde claro
                'base_size': 20
            }
        }

        # '#4169E1',  # azul royal
        # '#8B008B',  # roxo escuro
        # '#FFA500'  # laranja
        # '#FFD900'  # amarelo alaranjado
        # '#FFD500'  # amarelo 

    def get_node_style(self, node_type, metrics=None, canal_alpha=False):
        """
        Retorna estilo do nó baseado no tipo e métricas
        """
        base_style = self.node_styles.get(node_type, {
            'shape': 'dot',
            'color': '#808080',
            'base_size': 10
        })
        
        if metrics:
            # Ajusta tamanho baseado na centralidade de grau
            size_multiplier = 1 + (metrics.get('degree', 0) * 2)
            # Ajusta cor baseado na centralidade de intermediação
            if canal_alpha:
                color_alpha = 0.3 + (metrics.get('betweenness', 0) * 0.7)
            else:
                color_alpha = 1
            
            base_style['size'] = base_style['base_size'] * size_multiplier
            base_style['color'] = self._adjust_color_alpha(
                base_style['color'], 
                color_alpha
            )
        
        return base_style
    
    def _adjust_color_alpha(self, hex_color, alpha):
        """Ajusta a transparência de uma cor"""
        try:
            # Se a cor já estiver em formato rgba
            if hex_color.startswith('rgba'):
                return hex_color
                
            # Remover # e validar formato hex
            hex_str = hex_color.lstrip('#')
            if len(hex_str) != 6:
                return f'rgba(128, 128, 128, {alpha})'  # fallback para cinza
                
            # Converter hex para RGB
            rgb = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba{rgb + (alpha,)}'
        except (ValueError, AttributeError):
            # Retornar cor padrão em caso de erro
            return f'rgba(128, 128, 128, {alpha})'


class GraphLayoutLearner:
    def __init__(self, graph):
        self.graph = graph
        self.edge_lengths = {}
        self.node_positions = {}
        
    def learn_edge_lengths(self):
        """
        Aprende comprimentos ideais das arestas baseado em:
        - Tipos dos nós de origem e destino
        - Estrutura do grafo
        - Centralidade dos nós
        - Grau dos nós
        """
        for edge in self.graph.edges():
            source, target = edge
            source_type = self.graph.nodes[source].get('tipo')
            target_type = self.graph.nodes[target].get('tipo')
            
            # Calcular comprimento baseado nos tipos dos nós e métricas do grafo
            length = self._calculate_optimal_length(source, target, source_type, target_type)
            self.edge_lengths[edge] = length
    
    def _calculate_optimal_length(self, source, target, source_type, target_type):
        """
        Calcula comprimento ótimo baseado em:
        - Centralidade dos nós
        - Grau dos nós
        - Tipos dos nós
        - Comunidades do grafo
        """
        base_length = 200
        multiplier = 1.0
        
        # Ajustar baseado nos tipos dos nós
        if source_type == target_type:
            multiplier *= 0.8
        
        # Ajustar baseado na centralidade
        source_centrality = nx.degree_centrality(self.graph)[source]
        target_centrality = nx.degree_centrality(self.graph)[target]
        centrality_factor = (source_centrality + target_centrality) / 2
        multiplier *= (1 + centrality_factor)
        
        return base_length * multiplier


class GrafoConhecimento:
    def __init__(self, dict_list, dados_demanda):
        self.dict_list = dict_list
        self.dados_demanda = dados_demanda
        self.grafo = nx.DiGraph()

    def conectar_nos_similares(self, subgrafo1, subgrafo2):
        """
        Busca similaridades entre labels de nós de dois subgrafos e cria arestas
        baseadas no grau de similaridade.
        
        Args:
            subgrafo1 (str): Nome do primeiro subgrafo para comparação
            subgrafo2 (str): Nome do segundo subgrafo para comparação
        """
        import re
        from difflib import SequenceMatcher
        
        def calcular_similaridade_ratclif(str1, str2):
            """Calcula a similaridade entre duas strings com algoritmo de Ratcliff/Obershelp através da classe SequenceMatcher do módulo difflib do Python. Método mais sofisticado que uma simples comparação caractere a caractere, pois considera sequências comuns de caracteres em qualquer posição das strings.

                - Compara duas strings convertidas para minúsculas (usando lower())
                - Encontra a maior subsequência comum entre elas
                - Recursivamente encontra subsequências comuns nas partes não correspondidas
            
            Retorna um valor entre 0 e 1 que representa o grau de similaridade:
                - 1.0 significa strings idênticas
                - 0.0 significa strings completamente diferentes
                - Valores intermediários indicam graus parciais de similaridade
            
            O algoritmo é especialmente útil para:
                Detecção de strings similares com pequenas diferenças
                Comparação de textos ignorando capitalização
                Identificação de variações de palavras ou frases
            """
            return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

        def calcular_similaridade(str1, str2, verbose=True):
            """Calcula a similaridade entre duas strings com pré-processamento"""
            # Normalizar strings
            str1 = str1.lower().strip()
            str2 = str2.lower().strip()
            
            # Remover caracteres especiais e pontuação
            str1 = re.sub(r'[^\w\s]', '', str1)
            str2 = re.sub(r'[^\w\s]', '', str2)
            
            # Dividir em tokens e remover stopwords
            tokens1 = set(str1.split())
            tokens2 = set(str2.split())
            
            # Calcular similaridade usando coeficiente de Jaccard
            intersecao = len(tokens1.intersection(tokens2))
            uniao = len(tokens1.union(tokens2))
            
            return intersecao / uniao if uniao > 0 else 0.0

        # Coletar nós de cada subgrafo
        nos_subgrafo1 = [(node, data) for node, data in self.grafo.nodes(data=True) 
                        if data.get('tipo', '') == subgrafo1]
        nos_subgrafo2 = [(node, data) for node, data in self.grafo.nodes(data=True) 
                        if data.get('tipo', '') == subgrafo2]
               
        # Contador para arestas criadas
        arestas_interesse = 0
        arestas_similaridade = 0
        
        # Comparar labels dos nós
        for no1, dados1 in nos_subgrafo1:
            label1 = dados1.get('label', no1)
            for no2, dados2 in nos_subgrafo2:
                label2 = dados2.get('label', no2)
                
                # Calcular similaridade entre os labels
                similaridade = calcular_similaridade(str(label1), str(label2))
                if verbose:
                    print(f"{np.round(similaridade,4)} | {str(label1)} | {str(label2)}")
                
                # Criar arestas baseadas na similaridade
                if similaridade > 0.95:
                    self.grafo.add_edge(
                        no1, 
                        no2, 
                        relation='TEM_INTERESSE',
                        color='#808080',  # cinza
                        width=2.0,
                        title=f'Similaridade: {similaridade:.2f}'
                    )
                    arestas_interesse += 1
                elif 0.7 <= similaridade <= 0.95:
                    self.grafo.add_edge(
                        no1, 
                        no2, 
                        relation='TEM_SIMILARIDADE',
                        color='#808080',  # cinza
                        width=1.0,
                        title=f'Similaridade: {similaridade:.2f}'
                    )
                    arestas_similaridade += 1
        
        print(f"\nConexões criadas entre {subgrafo1} e {subgrafo2}:")
        print(f"- TEM_INTERESSE: {arestas_interesse} arestas")
        print(f"- TEM_SIMILARIDADE: {arestas_similaridade} arestas")


    def info_subgrafo(self, nome_subgrafo, subgrafo):
        """
        Imprime informações sobre o subgrafo, incluindo o número de nós, 
        arestas e a contagem de cada tipo de nó e aresta.

        Args:
            nome_subgrafo (str): Nome do subgrafo
            subgrafo (nx.DiGraph): O subgrafo a ser analisado
        """
        num_nos = subgrafo.number_of_nodes()
        num_arestas = subgrafo.number_of_edges()

        # Contar nós de cada tipo
        tipos_nos = defaultdict(int)
        for _, dados in subgrafo.nodes(data=True):
            tipos_nos[dados['tipo']] += 1

        # Contar arestas de cada tipo
        tipos_arestas = defaultdict(int)
        for _, _, dados in subgrafo.edges(data=True):
            tipos_arestas[dados['relation']] += 1

        # Imprimir a mensagem com as informações adicionais
        print(f"\nSUBGRAFO {nome_subgrafo.upper()} criado com {num_nos} nós e {num_arestas} arestas.")
        print("  Nós por tipo:")
        for tipo, quantidade in tipos_nos.items():
            print(f"  - {tipo}: {quantidade}")
        print("  Arestas por tipo:")
        for tipo, quantidade in tipos_arestas.items():
            print(f"  - {tipo}: {quantidade}")


    def extrair_caracteristicas(self, no):
        """
        Extrair as características relevantes de um nó.
        """
        caracteristicas = []
        try:
            tipo_no = self.grafo.nodes[no]['tipo']

            if tipo_no == 'pesquisador':
                # 1. Número de áreas de atuação
                num_areas = len(list(self.grafo.neighbors(no)))
                caracteristicas.append(num_areas)

                # 2. Número de subáreas de atuação
                num_subareas = 0
                for area in self.grafo.neighbors(no):
                    if self.grafo.nodes[area].get('tipo', 0) == 'area':
                        num_subareas += len(list(self.grafo.neighbors(area)))
                caracteristicas.append(num_subareas)

                # 3. Número de projetos de pesquisa
                num_projetos = 0
                for vizinho, attrs in self.grafo.adj[no].items():
                    if attrs['relation'] == 'PARTICIPOU_DO_PROJETO':
                        num_projetos += 1
                caracteristicas.append(num_projetos)

                # 4. Número de competências declaradas
                num_competencias_declaradas = 0
                for vizinho, attrs in self.grafo.adj[no].items():
                    if attrs['relation'] == 'COMPETENCIA_DECLARADA':
                        num_competencias_declaradas += 1
                caracteristicas.append(num_competencias_declaradas)

                # 5. Número de competências a desenvolver
                num_competencias_desenvolver = 0
                for vizinho, attrs in self.grafo.adj[no].items():
                    if attrs['relation'] == 'COMPETENCIA_DESEJADA':
                        num_competencias_desenvolver += 1
                caracteristicas.append(num_competencias_desenvolver)

                # 6. Presença de intenção de desenvolvimento (booleano)
                tem_intencao = False
                for vizinho, attrs in self.grafo.adj[no].items():
                    if attrs['relation'] == 'POSSUI_INTENCAO':
                        tem_intencao = True
                        break
                caracteristicas.append(int(tem_intencao))  # Converter booleano para inteiro (0 ou 1)

            elif tipo_no == 'area_subarea':
                # 1. Número de pesquisadores na área/subárea
                num_pesquisadores = len(list(self.grafo.predecessors(no)))
                caracteristicas.append(num_pesquisadores)

            elif tipo_no == 'produto':
                # 1. Demanda do produto
                demanda = self.grafo.nodes[no].get('demanda', 0)
                caracteristicas.append(demanda)

            elif tipo_no == 'desafio':
                # 1. Número de plataformas relacionadas ao desafio
                num_plataformas = len(list(self.grafo.neighbors(no)))
                caracteristicas.append(num_plataformas)

            elif tipo_no == 'plataforma':
                # 1. Número de desafios que requerem a plataforma
                num_desafios = len(list(self.grafo.predecessors(no)))
                caracteristicas.append(num_desafios)

            return caracteristicas

        except KeyError as e:
            print(f"Erro ao extrair características do nó {no}: {e}")
            return []
        except Exception as e:
            print(f"Erro inesperado ao extrair características do nó {no}: {e}")
            return []

    # def adicionar_subgrafo_cnpq(self, camada):
    #     """
    #     Agregar a camada especificada no grafo.
    #     """
    #     if camada == 'area':
    #         # Agregar áreas e subáreas em um único nó "area_subarea"
    #         areas_subareas = defaultdict(list)
    #         for no, dados in self.grafo.nodes(data=True):
    #             if dados['tipo'] == 'subarea':
    #                 no = no.replace("Subárea: ", "")
    #                 for vizinho, attrs in self.grafo.adj[no].items():
    #                     if attrs['relation'] == 'CONTEM_SUBAREA':
    #                         areas_subareas[vizinho].append(no)

    #         for area, subareas in areas_subareas.items():
    #             novo_no = f"{area} - {', '.join(subareas)}"
    #             self.grafo.add_node(novo_no, tipo='area_subarea')
    #             for subarea in subareas:
    #                 for vizinho, attrs in self.grafo.adj[subarea].items():
    #                     if attrs['relation'] == 'ATUA_NA_AREA' and vizinho != area:
    #                         self.grafo.add_edge(novo_no, vizinho, relation='ATUA_NA_AREA_SUBAREA')
    #                 self.grafo.remove_node(subarea)
    #             self.grafo.remove_node(area)
        
    #     elif camada == 'outra_camada':
    #         # Acrescentar implementação para agregar outras camada, quando necessário no âmbito CNPQ
    #         # Para outras camadas de outras fontes usar outra função modular a ser chamada na integração
    #         pass

    # def ajustar_posicoes_nos_processo(self, fases_horizontal=False):
    #     """
    #     Ajusta as posições dos nós que começam com letras de A a H seguidas de underscore.
    #     """
    #     offset = 500
    #     posicoes_x = {
    #         'A': -8 * offset,
    #         'B': -6 * offset,
    #         'C': -4 * offset,
    #         'D': -2 * offset,
    #         'E': 2 * offset,
    #         'F': 4 * offset,
    #         'G': 6 * offset,
    #         'H': 8 * offset
    #     }

    #     # Primeiro identificar os nós de fase
    #     nos_fase = {}
    #     for node_id, node_data in self.grafo.nodes(data=True):
    #         if isinstance(node_id, str) and len(node_id) >= 2:
    #             # Remover sufixos para verificar a letra inicial
    #             base_id = node_id.replace('_bio', '').replace('_sm', '')
    #             primeira_letra = base_id[0]
                
    #             if primeira_letra in posicoes_x and base_id[1] == '_':
    #                 # Guardar referência do nó de fase e seu tipo
    #                 tipo_processo = 'processo_biologico' if '_bio' in node_id else 'processo_smallmolecule'
    #                 nos_fase[node_id] = tipo_processo

    #     # Posicionar nós de fase e conectar atividades
    #     for fase_id, tipo_processo in nos_fase.items():
    #         base_id = fase_id.replace('_bio', '').replace('_sm', '')
    #         primeira_letra = base_id[0]
            
    #         # Definir posição y baseada no tipo do processo
    #         y_pos = -4*offset if tipo_processo == 'processo_biologico' else 4*offset
            
    #         # Atualizar atributos do nó de fase
    #         self.grafo.nodes[fase_id].update({
    #             'x': posicoes_x[primeira_letra],
    #             'y': y_pos,
    #             'physics': fases_horizontal,
    #             'fixed': False,
    #             'size': 50,
    #             'font': {'size': 180}
    #         })
            
    #         # Conectar ao nó principal do tipo de processo
    #         processo_principal = 'BIOLOGICOS' if tipo_processo == 'processo_biologico' else 'SMALLMOLECULE'
    #         self.grafo.add_edge(processo_principal, fase_id, relation='TEM_FASE')
            
    #         # Procurar atividades desta fase no mesmo subgrafo
    #         for node_id, node_data in self.grafo.nodes(data=True):
    #             if 'phase' in node_data:
    #                 fase_id = node_data['phase']
    #                 if self.grafo.has_node(fase_id):
    #                     # Verificar se os nós pertencem ao mesmo tipo de processo
    #                     node_is_bio = '_bio' in node_id
    #                     fase_is_bio = '_bio' in fase_id
                        
    #                     if node_is_bio == fase_is_bio:
    #                         self.grafo.add_edge(
    #                             node_id,
    #                             fase_id,
    #                             relation='PERTENCE_A_FASE',
    #                             color='#D3D3D3',
    #                             width=0.5
    #                         )

    def ajustar_posicoes_nos_processo(self, fases_horizontal=False, verbose=False):
        offset = 500
        posicoes_x = {
            'A': -8 * offset,
            'B': -6 * offset,
            'C': -4 * offset,
            'D': -2 * offset,
            'E': 2 * offset,
            'F': 4 * offset,
            'G': 6 * offset,
            'H': 8 * offset
        }

        if verbose:
            print("\nIniciando ajuste de posições dos nós de processo...")

        # Primeiro posicionar os nós de fase
        for node_id, node_data in self.grafo.nodes(data=True):
            if isinstance(node_id, str) and len(node_id) >= 2:
                base_id = node_id.replace('_bio', '').replace('_sm', '')
                primeira_letra = base_id[0]
                
                if primeira_letra in posicoes_x and base_id[1] == '_':
                    if verbose:
                        print(f"\nProcessando nó de fase: {node_id}")
                        print(f"Tipo: {node_data.get('tipo')}")
                        print(f"Label: {node_data.get('label')}")
                    
                    # Determinar tipo de processo
                    tipo_processo = 'processo_biologico' if '_bio' in node_id else 'processo_smallmolecule'
                    y_pos = -4*offset if tipo_processo == 'processo_biologico' else 4*offset
                    
                    if verbose:
                        print(f"Posição calculada: x={posicoes_x[primeira_letra]}, y={y_pos}")
                    
                    # Atualizar atributos do nó de fase
                    self.grafo.nodes[node_id].update({
                        'x': posicoes_x[primeira_letra],
                        'y': y_pos,
                        'physics': fases_horizontal,
                        'fixed': False,
                        'size': 50,
                        'font': {'size': 180}
                    })

                    # Conectar ao nó principal do tipo de processo
                    processo_principal = 'BIOLOGICOS' if tipo_processo == 'processo_biologico' else 'SMALLMOLECULE'
                    self.grafo.add_edge(processo_principal, node_id, relation='TEM_FASE')

        if verbose:
            print(f"\nConectando atividades às fases do processo produtivo...")

        # Depois conectar atividades às suas fases
        qte_ativ = 0
        for node_id, node_data in self.grafo.nodes(data=True):
            if verbose:
                print(f"{node_id}: {node_data}")
            if 'phase' in node_data:
                qte_ativ+=1
                fase_id = node_data['phase']
                
                # Determinar o sufixo correto baseado no tipo de processo
                sufixo = '_bio' if '_bio' in node_id else '_sm' if '_sm' in node_id else ''
                
                # Adicionar o sufixo ao ID da fase
                fase_id_completo = f"{fase_id}{sufixo}"
                
                if verbose:
                    print(f"\nProcessando atividade: {node_id}")
                    print(f"Fase relacionada: {fase_id_completo}")
                
                if self.grafo.has_node(fase_id_completo):
                    if verbose:
                        print(f"Criando aresta entre {node_id} e {fase_id_completo}")
                    
                    self.grafo.add_edge(
                        node_id,
                        fase_id_completo,
                        relation='PERTENCE_A_FASE',
                        color='#D3D3D3',
                        width=0.5
                    )
                elif verbose:
                    print(f"AVISO: Fase {fase_id_completo} não encontrada no grafo")
        
        print(f"  {qte_ativ} atividades produtivas processadas")

        if verbose:
            print("\nAjuste de posições concluído")



    ## PIPELINE DE CONSTRUÇÃO DO GRAFO DE CONHECIMENTO
    def construir_grafo_base(self, verbose=False):
        """
        Construir a estrutura base do grafo de conhecimento com as relações diretas.
        """
        # Criar índices
        self.indice_interesses = self.criar_indice_interesses()
        self.indice_produtos_desafios = self.criar_indice_produtos_desafios()
        
        # Criar subgrafos de oferta e demanda
        self.oferta = GrafoOferta(
            self.dict_list,
            self.indice_interesses,
            self.indice_produtos_desafios,
            self  # Passando a referência do próprio GrafoConhecimento
        )
        self.demanda = GrafoDemanda(
            self.dados_demanda,
            self  # Passar a referência da própria instância
        )
        
        # Construir o grafo de conhecimento por subgrafos em camadas
        self.demanda.subgrafo_demanda_pdi()
        self.demanda.adicionar_subgrafo_processos(
            'input_process_biologics.json',
            'input_process_smallmolecules.json'
        )
        self.oferta.subgrafo_oferta_pdi()
        self.oferta.subgrafo_intencoes(verbose)
        self.oferta.adicionar_projetos()

        # Integrar os subgrafos de oferta e demanda ao grafo principal
        self.integrar_subgrafos()

        # Adicionar nó CEIS e conectar aos blocos
        self.grafo.add_node('CEIS', 
                            tipo='instituicao',
                            label='CEIS Ecosystem Strategy',)
        for no, dados in self.grafo.nodes(data=True):
            if dados.get('tipo') == 'bloco':
                self.grafo.add_edge('CEIS', no, relation='TEM_BLOCO')

        # Adicionar nó ICT_Competencies e conectar aos pesquisadores
        self.grafo.add_node('ICT', 
                            tipo='instituicao',
                            label='STI Competencies',)
        for no, dados in self.grafo.nodes(data=True):
            if dados.get('tipo') == 'pesquisador':
                self.grafo.add_edge('ICT', no, relation='TEM_PESQUISADOR')

        # Ajustar posições dos nós de processo
        print(f"\nAtribuindo atividades às fases dos processos produtivos...")
        self.ajustar_posicoes_nos_processo()

        self.visualizar()

    def analisar_similaridades(self):
        """
        Realizar análises de similaridade e estabelecer novas arestas baseadas em métricas.
        """
        # Preparar dados para análise
        self.preprocessor = GraphDataPreprocessor(self.grafo)
        self.preprocessor.extract_node_features()
        self.preprocessor.extract_edge_features()
        
        # Treinar modelo para predição de comprimentos
        print(f"\nTreinando modelo aprender comprimento de arestas com base semântica...")
        self.length_predictor = EdgeLengthPredictor(self.grafo)
        self.length_predictor.train_model()
        
        # Calcular métricas de centralidade
        print(f"  Calculando centralidades...")
        self.metrics_calculator = NodeMetricsCalculator(self.grafo)
        self.metrics_calculator.calculate_all_metrics()
        
        # Calcular similaridades entre entidades
        print(f"  Calculando similaridades...")
        self.calcular_similaridade()
        
        # Identificar lacunas de competências
        self.lacunas = self.identificar_lacunas()


    def construir_grafo_completo(self, verbose=True):
        """
        Pipeline para construir o grafo de conhecimento com os subgrafos de demanda e oferta e demais subgrafos necessários de acordo com cada análise desejada.
        """      
        # Criar índices
        self.indice_interesses = self.criar_indice_interesses()
        self.indice_produtos_desafios = self.criar_indice_produtos_desafios()

        # Criar subgrafos de oferta e demanda passando self como referência
        self.oferta = GrafoOferta(
            self.dict_list, 
            self.indice_interesses, 
            self.indice_produtos_desafios,
            self  # Passar a referência da própria instância
        )
        self.demanda = GrafoDemanda(
            self.dados_demanda,
            self  # Passar a referência da própria instância
        )
        
        # Construir o grafo de conhecimento por subgrafos em camadas
        self.demanda.subgrafo_demanda_pdi()
        self.demanda.adicionar_subgrafo_processos(
            'input_process_biologics.json',
            'input_process_smallmolecules.json'
        )
        self.oferta.subgrafo_oferta_pdi()
        self.oferta.subgrafo_intencoes(verbose)
        self.oferta.adicionar_projetos()

        # Integrar os subgrafos de oferta e demanda ao grafo principal
        self.integrar_subgrafos()

        # Adicionar nó CEIS e conectar aos blocos
        self.grafo.add_node('CEIS', 
                            tipo='instituicao',
                            label='CEIS Ecosystem Strategy',)
        for no, dados in self.grafo.nodes(data=True):
            if dados.get('tipo') == 'bloco':
                self.grafo.add_edge('CEIS', no, relation='TEM_BLOCO')

        # Adicionar nó ICT_Competencies e conectar aos pesquisadores
        self.grafo.add_node('ICT', 
                            tipo='instituicao',
                            label='STI Competencies',)
        for no, dados in self.grafo.nodes(data=True):
            if dados.get('tipo') == 'pesquisador':
                self.grafo.add_edge('ICT', no, relation='TEM_PESQUISADOR')

        # Ajustar posições dos nós de processo
        self.ajustar_posicoes_nos_processo()

        ## Estabelecer arestas com base em similaridade dos rótulos de nós
        # self.conectar_nos_similares('processo_biologico', 'processo_smallmolecule')
        self.conectar_nos_similares('ceis_produto_emergencial', 'produto')
        self.conectar_nos_similares('ceis_produto_agravo', 'produto')
        self.conectar_nos_similares('ceis_desafio', 'desafio')

        # Calcular parâmetros de layout com base no grafo de conhecimento
        self.preprocessor = GraphDataPreprocessor(self.grafo)
        self.preprocessor.extract_node_features()
        self.preprocessor.extract_edge_features()
        
        # Preparar dataset PyTorch
        self.pytorch_dataset = PyTorchGraphDataset(self.preprocessor)
        self.graph_tensors = self.pytorch_dataset.prepare_tensors()

        # Gerar visualização do grafo com pyvis
        arquivo_html = self.visualizar()

        # Gerar visualização do grafo com matplotlib
        # arquivo_html = self.visualizar_grafo_matplotlib() 


    def integrar_subgrafos(self):
        """
        Integrar os subgrafos de oferta e demanda ao grafo principal.
        """
        self.grafo.add_nodes_from(self.demanda.grafo.nodes(data=True))
        self.grafo.add_edges_from(self.demanda.grafo.edges(data=True))

        self.grafo.add_nodes_from(self.oferta.grafo.nodes(data=True))
        self.grafo.add_edges_from(self.oferta.grafo.edges(data=True))

        self.grafo.add_nodes_from(self.oferta.grafo.nodes(data=True))
        self.grafo.add_edges_from(self.oferta.grafo.edges(data=True))        
        
        # Treinar modelo de comprimento após integração
        length_predictor = EdgeLengthPredictor(self.grafo)
        length_predictor.train_model()


    ## FUNÇÕES PARA OTIMIZAR BUSCA VETORIAL NO GRAFO DE CONHECIMENTO
    def criar_indice_interesses(self):
        """
        Criar um índice para os interesses dos pesquisadores, 
        mapeando cada interesse a um ID numérico.
        """
        indice_interesses = {}
        id_numerico = 0
        for dicionario in self.dict_list:
            if isinstance(dicionario, dict):
                interesses = dicionario.get('Interesses', [])
                for interesse in interesses:
                    if interesse not in indice_interesses:
                        indice_interesses[interesse] = id_numerico
                        id_numerico += 1
        return indice_interesses


    def criar_indice_produtos_desafios(self):
        """
        Criar um índice para os produtos e desafios do CEIS, 
        mapeando cada item a um ID numérico.
        """
        indice_produtos_desafios = {}
        id_numerico = 0
        try:
            # Usar self.dados_demanda para abrir o arquivo
            with open(self.dados_demanda, 'r') as f:  
                dados_ceis = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Erro ao ler o arquivo matriz_ceis.json: {e}")
            return {}

        for bloco in dados_ceis['blocos']:
            for produto in bloco['produtos']:
                if produto['id'] not in indice_produtos_desafios:
                    indice_produtos_desafios[produto['id']] = id_numerico
                    id_numerico += 1
            for desafio in bloco['desafios']:
                if desafio['id'] not in indice_produtos_desafios:
                    indice_produtos_desafios[desafio['id']] = id_numerico
                    id_numerico += 1
        return indice_produtos_desafios


    def calcular_similaridade(self):
        """Calcular a similaridade entre as entidades do grafo."""
        caracteristicas = []
        nos = []
        
        # Coletar características e padronizar tamanho
        max_features = 0
        temp_caracteristicas = []
        
        # Primeira passagem para determinar o tamanho máximo
        for no, dados in tqdm(self.grafo.nodes(data=True), desc="  Processando nós", total=self.grafo.number_of_nodes()):
            tipo_no = dados.get('tipo')
            if tipo_no in ['pesquisador', 'area_subarea', 'produto', 'desafio', 'plataforma']:
                nos.append(no)
                features = self.extrair_caracteristicas(no)
                temp_caracteristicas.append(features)
                max_features = max(max_features, len(features))
        
        # Segunda passagem para padronizar os vetores
        for features in tqdm(temp_caracteristicas, desc="  Ajustar vetores", total=len(temp_caracteristicas)):
            # Preencher com zeros até atingir max_features
            padded_features = features + [0] * (max_features - len(features))
            caracteristicas.append(padded_features)
        
        # Converter para array numpy e reshape para 2D
        caracteristicas = np.array(caracteristicas).reshape(len(caracteristicas), -1)
        
        # Calcular similaridade
        similaridade = cosine_similarity(caracteristicas)
        
        # Adicionar arestas baseadas na similaridade
        for i in range(len(nos)):
            for j in range(i + 1, len(nos)):
                if similaridade[i, j] > 0:
                    # Definir o tipo de relação baseado no valor de similaridade
                    if similaridade[i, j] > 0.95:
                        relation = 'TEM_INTERESSE'
                    else:
                        relation = 'TEM_SIMILARIDADE'
                        
                    # Adicionar aresta com todos os atributos necessários
                    self.grafo.add_edge(
                        nos[i], 
                        nos[j], 
                        similaridade=float(similaridade[i, j]),
                        relation=relation,
                        width=4.0 if relation == 'TEM_INTERESSE' else 1.0
                    )


    def identificar_lacunas(self):
        """
        Identificar lacunas de competências para a produção de produtos demandados.
        """
        lacunas = {}
        for produto in tqdm(self.grafo.nodes(data=True), 
                desc="  Cálculo lacunas", 
                total=self.grafo.number_of_nodes()):
            try:
                if produto[1]['tipo'] == 'produto':
                    nome_produto = produto[0]
                    similaridades_com_areas = []
                    for vizinho, attrs in self.grafo.adj[nome_produto].items():
                        if self.grafo.nodes[vizinho]['tipo'] == 'area_subarea':
                            similaridades_com_areas.append(attrs['similaridade'])
                    if similaridades_com_areas:
                        similaridade_media = sum(similaridades_com_areas) / len(similaridades_com_areas)
                        if similaridade_media < 0.5:
                            lacunas[nome_produto] = 1 - similaridade_media
                    else:
                        lacunas[nome_produto] = 1
            except Exception as e:
                print(f"Erro ao calcular lacunas: {e} em produto {produto}")
        return lacunas


    def adicionar_legenda_interativa(self, net):
        """
        Gera o HTML da legenda interativa.
        Returns:
            str: HTML da legenda
        """
        legend_html = """
        <div id="graph-legend" style="position: absolute; top: 10px; right: 10px; 
            background-color: white; padding: 15px; border: 1px solid #ccc; 
            border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); z-index: 1000;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <h3 style="margin: 0;">Legenda</h3>
                <button onclick="toggleLegend()" style="border: none; background: none; cursor: pointer;">▼</button>
            </div>
            <div id="legend-content">
                <div style="margin-bottom: 15px;">
                    <h4 style="margin: 5px 0;">Tipos de Vértices</h4>

                    <div class="legend-item" onclick="toggleNodeType('instituicao')">
                        <div style="width: 20px; height: 20px; background-color: orange; border-radius: 50%;"></div>
                        <span>Instituição</span>
                    </div>
                    
                    <b>Entidades do Ecossistema (CEIS)</b>
                    <div class="legend-item" onclick="toggleNodeType('bloco')">
                        <div style="width: 20px; height: 20px; background-color: cyan; border-radius: 50%;"></div>
                        <span>Bloco</span>
                    </div>
                    <div class="legend-item" onclick="toggleNodeType('desafio')">
                        <div style="width: 20px; height: 20px; background-color: pink; clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);"></div>
                        <span>Desafio</span>
                    </div>
                    <div class="legend-item" onclick="toggleNodeType('plataforma')">
                        <div style="width: 20px; height: 20px; background-color: gray; clip-path: polygon(50% 0%, 0% 100%, 100% 100%);"></div>
                        <span>Plataforma</span>
                    </div>
                    <div class="legend-item" onclick="toggleNodeType('produto')">
                        <div style="width: 20px; height: 20px; background-color: green; clip-path: polygon(0% 0%, 100% 0%, 100% 100%, 0% 100%);"></div>
                        <span>Produto Demandado</span>
                    </div>
                    <div class="legend-item" onclick="toggleNodeType('processo_biologico')">
                        <div style="width: 20px; height: 20px; background-color: green; clip-path: polygon(0% 0%, 100% 0%, 100% 100%, 0% 100%);"></div>
                        <span>Processo Biológico</span>
                    </div>
                    <div class="legend-item" onclick="toggleNodeType('processo_smallmolecule')">
                        <div style="width: 20px; height: 20px; background-color: lightgreen; clip-path: polygon(0% 0%, 100% 0%, 100% 100%, 0% 100%);"></div>
                        <span>Processo PequenaMolécula</span>
                    </div>
                    
                    <b>Entidades da ICT</b>
                    <div class="legend-item" onclick="toggleNodeType('pesquisador')">
                        <div style="width: 20px; height: 20px; background-color: yellow; border-radius: 50%;"></div>
                        <span>Pesquisador</span>
                    </div>
                    <div class="legend-item" onclick="toggleNodeType('questoes_pesquisa')">
                        <div style="width: 15px; height: 15px; background-color: gray; border-radius: 50%;"></div>
                        <span>Questões de Pesquisa</span>
                    </div>
                    <div class="legend-item" onclick="toggleNodeType('palavras_chave')">
                        <div style="width: 15px; height: 15px; background-color: gray; border-radius: 50%;"></div>
                        <span>Palavras-chave</span>
                    </div>
                    <div class="legend-item" onclick="toggleNodeType('competencia_declarada')">
                        <div style="width: 15px; height: 15px; background-color: gray; border-radius: 50%;"></div>
                        <span>Competência Declarada</span>
                    </div>
                    <div class="legend-item" onclick="toggleNodeType('competencia_desejada')">
                        <div style="width: 15px; height: 15px; background-color: gray; border-radius: 50%;"></div>
                        <span>Competência Desejada</span>
                    </div>
                    <div class="legend-item" onclick="toggleNodeType('ceis_desafio')">
                        <div style="width: 15px; height: 15px; background-color: gray; border-radius: 50%;"></div>
                        <span>Interesse Desafio CEIS</span>
                    </div>
                    <div class="legend-item" onclick="toggleNodeType('desenvolvimento')">
                        <div style="width: 15px; height: 15px; background-color: gray; border-radius: 50%;"></div>
                        <span>Interesse Desenvolvimento</span>
                    </div>                    
                    <div class="legend-item" onclick="toggleNodeType('ceis_produto_emergencial')">
                        <div style="width: 15px; height: 15px; background-color: gray; border-radius: 50%;"></div>
                        <span>Interesse Produto B.Emergências</span>
                    </div>
                    <div class="legend-item" onclick="toggleNodeType('ceis_produto_agravo')">
                        <div style="width: 15px; height: 15px; background-color: gray; border-radius: 50%;"></div>
                        <span>Interesse Produto B.Agravos</span>
                    </div>
                </div>
                <div>
                    <h4 style="margin: 5px 0;">Tipos de Relações</h4>
                    <div class="legend-item" onclick="toggleEdgeType('TEM_INTERESSE')">
                        <div style="width: 30px; height: 4px; background-color: #808080;"></div>
                        <span>TEM_INTERESSE</span>
                    </div>
                    <div class="legend-item" onclick="toggleEdgeType('TEM_SIMILARIDADE')">
                        <div style="width: 30px; height: 1px; background-color: #808080;"></div>
                        <span>TEM_SIMILARIDADE</span>
                    </div>
                    <div class="legend-item" onclick="toggleEdgeType('COMPETENCIA_DECLARADA')">
                        <div style="width: 30px; height: 3px; background-color: magenta;"></div>
                        <span>COMPETENCIA_DECLARADA</span>
                    </div>
                </div>
            </div>
        </div>

        <style>
            .legend-item {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 5px;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            .legend-item:hover {
                background-color: #f0f0f0;
            }
            #graph-legend {
                font-family: Arial, sans-serif;
                font-size: 14px;
            }
        </style>

        <script>
            function toggleLegend() {
                const content = document.getElementById('legend-content');
                const button = document.querySelector('#graph-legend button');
                if (content.style.display === 'none') {
                    content.style.display = 'block';
                    button.textContent = '▼';
                } else {
                    content.style.display = 'none';
                    button.textContent = '▲';
                }
            }

            function toggleNodeType(type) {
                const nodes = network.body.data.nodes.get();
                const filtered = nodes.filter(node => node.tipo === type);
                filtered.forEach(node => {
                    node.hidden = !node.hidden;
                    network.body.data.nodes.update(node);
                });
            }

            function toggleEdgeType(type) {
                const edges = network.body.data.edges.get();
                const filtered = edges.filter(edge => edge.relation === type);
                filtered.forEach(edge => {
                    edge.hidden = !edge.hidden;
                    network.body.data.edges.update(edge);
                });
            }
        </script>
        """
        return legend_html

       
    def visualizar(self, nome_arquivo="grafo_conhecimento.html", use_metrics=True):
        """
        Gerar visualização interativa do grafo usando pyvis com métricas e estilos customizados.
        """
        import math
        import os
        from pyvis.network import Network

        net = Network(
            height='1200',
            bgcolor='#ffffff',
            font_color='#000000', # type: ignore
            directed=True
        )
        
        net.from_nx(self.grafo)

        # Calcular métricas se solicitado
        metrics_calculator = None
        if use_metrics:
            metrics_calculator = NodeMetricsCalculator(self.grafo)
            metrics_calculator.calculate_all_metrics()
        
        # Inicializar gerenciadores
        edge_manager = EdgeStyleManager()
        style_manager = NodeStyleManager()
        length_predictor = EdgeLengthPredictor(self.grafo)
        length_predictor.train_model()

        # Encontrar pesquisadores ativos (com mais de uma aresta)
        pesquisadores_ativos = []
        for node in net.nodes:
            if node.get('tipo') == 'pesquisador':
                num_arestas = len(list(self.grafo.in_edges(node['id']))) + len(list(self.grafo.out_edges(node['id'])))
                if num_arestas > 1:
                    pesquisadores_ativos.append(node)

        # Antes do loop de posicionamento
        if 'ICT' not in self.grafo.nodes or 'CEIS' not in self.grafo.nodes:
            print("Aviso: Nós ICT ou CEIS não encontrados no grafo")

        blocos=[]
        # Posicionar nós principais e pesquisadores ativos
        for node in net.nodes:
            distance = 4000
            if node['id'] == 'CEIS':
                node.update({
                    'x': distance,
                    'y': 0,
                    'physics': False,
                    'fixed': True
                })
            elif node['id'] == 'ICT':
                node.update({
                    'x': -distance,
                    'y': 0,
                    'physics': False,
                    'fixed': True
                })

            # Posicionar os nós dos blocos do CEIS
            if node.get('tipo') == 'bloco':
                blocos.append(node)

        # Distribuir pesquisadores ativos em semicírculo à esquerda do ICT
        x_pos_icts = distance*(-1)
        x_pos_ceis = -x_pos_icts
        raio = 500
        for i, node in enumerate(pesquisadores_ativos):
            
            # Calcular ângulo entre: π/2 e -π/2 (90° a -90°)
            angulo = (math.pi/2) - (math.pi * i / (len(pesquisadores_ativos) - 1)) if len(pesquisadores_ativos) > 1 else 0
            
            # Usar seno para x (subtrair o raio para posicionar à esquerda) e cosseno para y
            # x = x_pos_icts - raio * math.cos(angulo)  # Posicionar relativo ao nó da ICT
            # y = raio * math.sin(angulo)
            
            # Usar seno para x (adicionar o raio para posicionar à esquerda) e cosseno para y
            x = x_pos_icts + raio * math.cos(angulo)  # Posicionar relativo ao nó da ICT
            y = raio * math.sin(angulo)

            node.update({
                'x': x,
                'y': y,
                'physics': False,
                'fixed': True
            })

        # Aplicar estilos aos nós
        for node in net.nodes:
            try:
                node_type = node.get('tipo')

                # Obter métricas do nó se disponíveis
                node_metrics = None
                if metrics_calculator:
                    node_metrics = metrics_calculator.get_node_metrics(node['id'])
                
                # Aplicar estilo
                style = style_manager.get_node_style(node_type, node_metrics)
                node.update(style)
            except:
                print(f"Nó não possui propriedade 'tipo':")
                print(type(node), node.keys())
                print(node)

        # Configurar nós com rótulos em tamanho maior
        for node in net.nodes:
            fontsize_big = 180
            try:
                # Customizar nós por id específico
                if node['id'] == 'CEIS':
                    node['font'] = {'size': fontsize_big}                                       
                elif node['id'] == 'ICT':
                    node['font'] = {'size': fontsize_big}
                elif node['id'] == 'BIOLOGICOS':
                    node['font'] = {'size': fontsize_big}
                elif node['id'] == 'SMALLMOLECULE':
                    node['font'] = {'size': fontsize_big}
            except KeyError:
                continue 

        # Posicionar os dois blocos
        if len(blocos) == 2:
            blocos[0].update({
                'x': x_pos_ceis+500,
                'y': -500,
                'physics': False,
                'fixed': True
            })
            blocos[1].update({
                'x': x_pos_ceis+500,
                'y': 500,
                'physics': False,
                'fixed': True
            })
        else:
            print('  Erro!! Ao encontrar os blocos do CEIS')

        # Aplicar estilos e comprimentos às arestas
        print(f"Aplicando parâmetros aprendidos à visualização do grafo de conhecimento...")
        for edge in net.edges:
            source = edge['from']
            target = edge['to']
            try:
                relation = edge['relation']
                
                # Obter estilo da aresta
                style = edge_manager.get_edge_style(
                    relation,
                    self.grafo.nodes[source].get('tipo'),
                    self.grafo.nodes[target].get('tipo')
                )
                
                # Obter comprimento predito
                length = length_predictor.predict_edge_length(source, target, relation)
                
                # Aplicar configurações
                edge.update({
                    'width': style['width'],
                    'length': length,
                    'smooth': style['smooth'],
                    # 'color': style.get('color', '#808080'),
                    'title': relation,
                    'opacity': 0.3  # 30% de opacidade
                })
                # print(f"Relação {relation} obtida na aresta {edge}")
            except Exception as e:
                print(f"Erro ao obter relação {e} na aresta {edge}")        
        
        
        
        # Configurações de física e visualização
        net.set_options("""
        {
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -2000,
                    "centralGravity": 0.3,
                    "springLength": 750,
                    "springConstant": 0.02,
                    "damping": 0.1,
                    "avoidOverlap": 1
                },
                "minVelocity": 0.75
            },
            "nodes": {
                "font": {
                    "size": 12
                }
            },
            "edges": {
                "smooth": {
                    "type": "curvedCW",
                    "roundness": 0.2
                }
            }
        }
        """)
      
    #     # # Configurações de física otimizadas com avoidOverlap
    #     # net.set_options("""
    #     # {
    #     #     "physics": {
    #     #         "barnesHut": {
    #     #             "gravitationalConstant": -2000,
    #     #             "centralGravity": 0.25,
    #     #             "springLength": 200,
    #     #             "springConstant": 0.15,
    #     #             "damping": 0.09
    #     #         },
    #     #         "minVelocity": 0.75,
    #     #         "solver": "barnesHut",
    #     #         "avoidOverlap": 1
    #     #     }
    #     # }
    #     # """)

    #     # # Configurações de física mais leves
    #     # net.set_options("""
    #     # {
    #     #     "physics": {
    #     #         "barnesHut": {
    #     #             "gravitationalConstant": -2000,
    #     #             "centralGravity": 0.3,
    #     #             "springLength": 95,
    #     #             "springConstant": 0.04,
    #     #             "damping": 0.09
    #     #         },
    #     #         "minVelocity": 0.75
    #     #     }
    #     # }
    #     # """)
        
    #     try:
    #         net.write_html(nome_arquivo)
    #         print(f"\nArquivo HTML gerado com sucesso: {nome_arquivo}")
    #         return True
    #     except Exception as e:
    #         print(f"Erro ao gerar arquivo HTML: {e}")
    #         return False

        try:
            # Adicionar legenda interativa e obter o HTML
            legend_html = self.adicionar_legenda_interativa(net)

            # Gerar o HTML base
            html_string = net.generate_html()
            
            # Inserir a legenda antes do fechamento do body
            html_string = html_string.replace('</body>', f'{legend_html}</body>')

            # Escrever o arquivo HTML modificado
            with open(nome_arquivo, 'w', encoding='utf-8') as f:
                f.write(html_string)
                
            print(f"\nArquivo HTML do grafo de conhecimento gerado com sucesso:")
            print(f"  Local: {os.path.abspath(nome_arquivo)}")
            return True
        except Exception as e:
            print(f"Erro ao gerar arquivo HTML: {e}")
            return False


    def visualizar_grafo_matplotlib(self):
        """
        Gera uma visualização do grafo usando networkx e matplotlib.
        """
        print(f"\n Gerando figura com matplotlib e netowrkx...")
        plt.figure(figsize=(80, 40))  # Aumentar o tamanho da figura (largura, altura) em polegadas

        # Definir layout e parâmetros de visualização
        pos = nx.spring_layout(self.grafo, k=0.3, iterations=50)  # Ajustar layout
        nx.draw(self.grafo, pos, 
                with_labels=True, 
                node_size=500,  # Ajustar node_size para evitar sobreposição
                font_size=14,
                font_family='FreeSans',
                node_color="skyblue", 
                edge_color="gray", 
                width=0.5,  # Ajustar width para melhor visualização das arestas
                alpha=0.7)  # Ajustar alpha para melhor visualização das arestas

        # Desenhar rótulos das arestas
        labels = nx.get_edge_attributes(self.grafo, 'relation')
        nx.draw_networkx_edge_labels(self.grafo, pos, edge_labels=labels, font_size=8)

        # Ajustar os limites do gráfico para evitar cortes
        plt.xlim(-1.5, 1.5)  # Ajustar os limites do eixo x, se necessário
        plt.ylim(-1.5, 1.5)  # Ajustar os limites do eixo y, se necessário

        # Salvar a figura com resolução de 600 pontos por polegada
        plt.savefig("grafo_conhecimento.png", dpi=150)

        return plt.show()


## Classes para tratar cada subgrafo de Demanda e Oferta separadamente
import os
import json
import networkx as nx
from pathlib import Path
from collections import defaultdict

class GrafoDemanda:
    def __init__(self, dados_demanda, grafo_conhecimento):
        self.grafo = nx.DiGraph()
        self.dados_demanda = dados_demanda
        self.base_repo_dir = self.find_repo_root()
        self.in_json = os.path.join(str(self.base_repo_dir), '_data', 'in_json')
        self.grafo_conhecimento = grafo_conhecimento  # Referência para a classe pai
        self.edge_manager = EdgeStyleManager()

    def find_repo_root(self, path='.', depth=10):
        ''' 
        Busca o arquivo .git e retorna string com a pasta raiz do repositório
        '''
        # Prevenir recursão infinita limitanto a profundidade da busca
        if depth < 0:
            return None
        path = Path(path).absolute()
        if (path / '.git').is_dir():
            return path
        return self.find_repo_root(str(path.parent), depth-1)
    
    def subgrafo_demanda_pdi(self):
        """
        Constrói o subgrafo de demanda com base nos dados da matriz CEIS.
        """
        # Adicionar nó CEIS e conectar aos blocos
        nome_instituicao = 'CEIS'
        self.grafo.add_node(nome_instituicao, tipo='instituicao')
        print(f" Criado nó instituição: {nome_instituicao}")

        try:
            pathfilename = os.path.join(self.in_json, 'matriz_ceis.json')
            with open(pathfilename, 'r') as f:  
                dados_ceis = json.load(f)
                print(f"    {len(dados_ceis)} subdicionários da matriz_ceis carregados...")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Erro ao ler o arquivo matriz_ceis.json: {e}")
            return

        for bloco in dados_ceis['blocos']:
            # Adicionar bloco como nó
            self.grafo.add_node(bloco['id'], tipo='bloco', label=bloco['nome'])
            # Conectar produtos ao bloco
            for produto in bloco['produtos']:  # Iterar sobre a lista de produtos
                self.grafo.add_node(produto['id'], tipo='produto', label=produto['nome'])
                self.grafo.add_edge(bloco['id'], produto['id'], relation='CONTEM_PRODUTO')

            # Conectar desafios ao bloco e, em seguida, plataformas aos desafios
            for desafio in bloco['desafios']:  # Iterar sobre os desafios
                self.grafo.add_node(desafio['id'], tipo='desafio', label=desafio['nome'])
                self.grafo.add_edge(bloco['id'], desafio['id'], relation='CONTEM_DESAFIO')

                for plataforma in desafio['plataformas']:  # Iterar sobre as plataformas
                    self.grafo.add_node(plataforma['id'], tipo='plataforma', label=plataforma['nome'])
                    self.grafo.add_edge(desafio['id'], plataforma['id'], relation='REQUER_PLATAFORMA')
                    
        # Após criar o subgrafo, chamar info_subgrafo
        self.grafo_conhecimento.info_subgrafo("demanda_pdi", self.grafo)


    def adicionar_subgrafo_processos(self, biologics_file, smallmolecules_file):
        def carregar_json(arquivo):
            try:
                pathfilename = os.path.join(self.in_json, arquivo)
                with open(pathfilename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Erro ao carregar {arquivo}: {e}")
                return None

        biologics_data = carregar_json(biologics_file)
        smallmolecules_data = carregar_json(smallmolecules_file)
        
        if not biologics_data or not smallmolecules_data:
            return False
        
        # Adicionar nós de nomes dos processos produtivos
        y_pos_processo=2500
        self.grafo.add_node(
            'BIOLOGICOS',
            tipo='processo_principal',
            label='Biological Products Processes',
            # color = '#4169E1', # azul marinho
            # color = '#013220', # verde muito escuro
            # color = '#008000', # verde escuro
            color = '#555555', # cinza
            size=75,
            x=0,
            y=-y_pos_processo,
            physics=False,
            fixed=True
        )
        
        self.grafo.add_node(
            'SMALLMOLECULE',
            tipo='processo_principal',
            label='SmallMolecule Products Processes',
            # color = '#8B008B', # roxo
            # color = '#00FF00', # verde limão
            color = '#555555', # cinza
            size=50,
            x=0,
            y=y_pos_processo,
            physics=False,
            fixed=True
        )
        
        # Adicionar nós das etapas dos processos para biológicos
        for node in biologics_data['nodes']:
            node_id = f"{node['id']}_bio"  # Mantém ID original e adiciona sufixo
            self.grafo.add_node(
                node_id,
                label=node['label'],
                tipo='processo_biologico',
                phase=node.get('phase'), 
                # color = '#87CEEB', # azul claro
                color = '#9ACD32', # yelowgreen
                y=-y_pos_processo+250,
                size=20,
                physics=True
            )
        
        # Adicionar nós das etapas dos processos para pequenas moléculas
        for node in smallmolecules_data['nodes']:
            node_id = f"{node['id']}_sm"  # Mantém ID original e adiciona sufixo
            self.grafo.add_node(
                node_id,
                label=node['label'],
                tipo='processo_smallmolecule',
                phase=node.get('phase'),
                # color = '#DDA0DD', # roxo claro
                color = '#90EE90', # verde claro
                size=20,
                y=y_pos_processo-250,
                physics=True
            )
        
        # Adicionar arestas dos processos biológicos
        for edge in biologics_data['edges']:
            self.grafo.add_edge(
                f"{edge['from']}_bio",
                f"{edge['to']}_bio",
                relation='SEGUIDO_POR'
            )
        
        # Adicionar arestas das pequenas moléculas
        for edge in smallmolecules_data['edges']:
            self.grafo.add_edge(
                f"{edge['from']}_sm",
                f"{edge['to']}_sm",
                relation='SEGUIDO_POR'
            )

        return True


class GrafoOferta:
    def __init__(self, dict_list, indice_interesses, indice_produtos_desafios, grafo_conhecimento):
        # self.grafo = nx.Graph()  # Usar Graph para relaxar restrição de direcionamento
        self.grafo = nx.DiGraph()  # Usar DiGraph para obrigar uso de grafo direcionado
        self.grafo_conhecimento = grafo_conhecimento  # Referência para a classe pai
        self.dict_list = dict_list
        self.indice_interesses = indice_interesses
        self.indice_produtos_desafios = indice_produtos_desafios
        self.base_repo_dir = self.find_repo_root()
        self.in_json = os.path.join(str(self.base_repo_dir), '_data', 'in_json')
        self.qte_respostas_registradas = 0
        self.tipos_nos = defaultdict(int)
        self.tipos_arestas = defaultdict(int)
        self.model = SentenceTransformer('sentence-t5-base')
        self.edge_manager = EdgeStyleManager()


    def find_repo_root(self, path='.', depth=10):
        ''' 
        Busca o arquivo .git e retorna string com a pasta raiz do repositório
        '''
        if depth < 0:
            return None
        path = Path(path).absolute()
        if (path / '.git').is_dir():
            return path
        return self.find_repo_root(str(path.parent), depth-1)
    

    def extrair_areas(self, dicionario_areas):
        """
        Extrai a lista de áreas de um dicionário de áreas.
        """
        lista_areas = []
        if isinstance(dicionario_areas, dict):
            for chave, valor in dicionario_areas.items():
                partes = valor.split(' / ')
                if len(partes) >= 2:
                    area = partes[1].replace('.', '')
                    lista_areas.append(area)
        elif isinstance(dicionario_areas, str):
            partes = dicionario_areas.split(' / ')
            if len(partes) >= 2:
                area = partes[1].replace('.', '')
                lista_areas.append(area)
        return lista_areas


    def extrair_subareas(self, area):
        """
        Extrai a lista de subáreas de uma área.
        """
        partes = area.split(' / ')
        if len(partes) >= 3:
            subarea = partes[2].replace('.', '')
            return [subarea]
        return []


    def encontrar_id_lattes(self, nome):
        """
        Encontra o ID Lattes correspondente ao nome do pesquisador no grafo,
        considerando nomes abreviados e normalização.

        Args:
            nome (str): O nome do pesquisador.

        Returns:
            str: O ID Lattes do pesquisador, ou '9999999999999999' se não for encontrado.
        """
        if nome.lower().split()[0] == 'não':
            # print(f"  Aviso: Nome não informado. Usando ID Lattes genérico.")
            id_lattes = '9999999999999999'
            self.grafo.add_node(id_lattes, tipo='pesquisador', nome='Anonimo')
            return id_lattes

        # Normalizar o nome para minúsculas e sem acentos
        nome_normalizado = unicodedata.normalize('NFKD', nome).encode('ASCII', 'ignore').decode('ASCII').lower()
        partes_nome_normalizado = nome_normalizado.split()

        for no, dados in self.grafo.nodes(data=True):
            if dados.get('tipo') == 'pesquisador' and dados.get('nome'):
                nome_pesquisador_normalizado = unicodedata.normalize('NFKD', dados.get('nome')).encode('ASCII', 'ignore').decode('ASCII').lower()
                # Verificar se todas as partes do nome abreviado estão presentes no nome completo
                if all(parte in nome_pesquisador_normalizado for parte in partes_nome_normalizado):
                    return no

        # Se não encontrar, imprimir aviso
        print(f"  Aviso: Nome '{nome}' não encontrado no grafo.")
        return '9999999999999999'  # ID Lattes padrão se não encontrar


    def encontrar_produto(self, nome):
        """
        Encontra o Produto no subgrafo da demanda do CEIS.

        Args:
            nome (str): O nome do produto.

        Returns:
            str: O ID do produto, ou None se não for encontrado.
        """

        # Normalizar o nome do produto para minúsculas e sem acentos
        nome_normalizado = unicodedata.normalize('NFKD', nome).encode('ASCII', 'ignore').decode('ASCII').lower()
        partes_nome_normalizado = nome_normalizado.split()

        for no, dados in self.grafo.nodes(data=True):
            if dados.get('tipo') == 'produto' and dados.get('nome'):
                nome_produto_normalizado = unicodedata.normalize('NFKD', dados.get('nome')).encode('ASCII', 'ignore').decode('ASCII').lower()
                # Verificar se todas as partes do nome abreviado estão presentes no nome completo
                if all(parte in nome_produto_normalizado for parte in partes_nome_normalizado):
                    return no


    def encontrar_desafio(self, nome):
        """
        Encontra o desafio no subgrafo da demanda do CEIS.

        Args:
            nome (str): O nome do desafio.

        Returns:
            str: O ID do desafio, ou None se não for encontrado.
        """

        # Normalizar o nome do desafio para minúsculas e sem acentos
        nome_normalizado = unicodedata.normalize('NFKD', nome).encode('ASCII', 'ignore').decode('ASCII').lower()
        partes_nome_normalizado = nome_normalizado.split()

        for no, dados in self.grafo.nodes(data=True):
            if dados.get('tipo') == 'desafio' and dados.get('nome'):
                nome_desafio_normalizado = unicodedata.normalize('NFKD', dados.get('nome')).encode('ASCII', 'ignore').decode('ASCII').lower()
                # Verificar se todas as partes do nome abreviado estão presentes no nome completo
                if all(parte in nome_desafio_normalizado for parte in partes_nome_normalizado):
                    return no

        # Se não encontrar, imprimir aviso
        print(f"  Aviso: Desafio '{nome}' não encontrado no grafo.")
        return None


    def subgrafo_oferta_pdi(self, verbose=False):
        """
        Constrói o subgrafo de oferta com base nos dados dos pesquisadores.
        """
        if self.dict_list:
            for dicionario in self.dict_list:
                if isinstance(dicionario, dict):
                    id_lattes = dicionario.get('Identificação', {}).get('ID Lattes')
                    nome = dicionario.get('Identificação', {}).get('Nome')

                    if verbose:
                        print(f"ID Lattes: {id_lattes}, Nome: {nome}")
                        self.grafo.add_node(id_lattes, tipo='pesquisador', nome=nome)
                        print(f"Nó criado: {id_lattes}, Atributos: {self.grafo.nodes[id_lattes]}")

                    if id_lattes and nome:
                        self.grafo.add_node(id_lattes, tipo='pesquisador', nome=nome)

                        # areas = self.extrair_areas(dicionario.get('Áreas', {}))
                        # for area in areas:
                        #     self.grafo.add_node(area, tipo='area')
                        #     self.grafo.add_edge(id_lattes, area, relation='ATUA_NA_AREA')

                        #     subareas = self.extrair_subareas(area)
                        #     for subarea in subareas:
                        #         self.grafo.add_node(subarea, tipo='subarea')
                        #         self.grafo.add_edge(area, subarea, relation='CONTEM_SUBAREA')

                else:
                    print(f"Erro com objeto dicionário: {type(dicionario)}")

            # # Adicionar checagem de tipos e quantidades
            # self.grafo_conhecimento.info_subgrafo("oferta_pdi", self.grafo)

            # num_nos = self.grafo.number_of_nodes()
            # num_arestas = self.grafo.number_of_edges()

            # # Contar nós de cada tipo
            # tipos_nos = defaultdict(int)
            # for _, dados in self.grafo.nodes(data=True):
            #     tipos_nos[dados['tipo']] += 1

            # # Contar arestas de cada tipo
            # tipos_arestas = defaultdict(int)
            # for _, _, dados in self.grafo.edges(data=True):
            #     tipos_arestas[dados['relation']] += 1

            # # Imprimir a mensagem com as informações adicionais
            # print(f"\nSUBGRAFO DE OFERTA criado com {num_nos} nós e {num_arestas} arestas.")
            # print("  Nós por tipo:")
            # for tipo, quantidade in tipos_nos.items():
            #     print(f"  - {tipo}: {quantidade}")
            # print("  Arestas por tipo:")
            # for tipo, quantidade in tipos_arestas.items():
            #     print(f"  - {tipo}: {quantidade}")


    def subgrafo_intencoes(self, verbose=True):
        """
        Adiciona dados do levantamento das intenções junto aos pesquiadores ao grafo,
        relacionando intenções aos id_lattes e ao nó da ICT_Competencies.
        """
        try:
            pathfilename = os.path.join(self.in_json, 'input_interesses_pesquisadores.json')
            with open(pathfilename, 'r') as f:  
                respostas_pesquisadores = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Erro ao ler o arquivo input_interesses_pesquisadores.json: {e}")
            return

        # print(f"\nPopulando subgrafo de oferta com as intenções dos pesquisadores da ICT_Competencies:")
        
        # Contadores
        num_arestas_interesse = 0
        tipos_nos = defaultdict(int)
        tipos_arestas = defaultdict(int)
        respostas_nao_associadas = []

        for i, pesquisador in enumerate(respostas_pesquisadores):
            # Ignorar o primeiro dicionário (referente às perguntas do questionário)
            if i == 0:
                continue

            try:
                nome = pesquisador.get('nome_pesquisador')
                if nome is None or not isinstance(nome, str) or nome.strip() in ('', 'Não desejo.'):
                    nome = 'Não Informado'

                # Encontrar o id_lattes correspondente ao nome do pesquisador
                id_lattes = self.encontrar_id_lattes(nome)

                # --- Competências Possuídas ---
                competencias_presentes = pesquisador.get("competencias_possuidas")
                if competencias_presentes and isinstance(competencias_presentes, list):
                    competencias_presentes = self.limpar_lista(competencias_presentes)
                    for competencia_declarada in competencias_presentes:
                        if isinstance(competencia_declarada, str):
                            # Criar nós de competencia_declarada no grafo para objeto salvo como competencias_possuidas nas respostas
                            self.grafo.add_node(competencia_declarada, tipo='competencia_declarada')
                            tipos_nos['competencia_declarada'] += 1

                            # Adicionar arestas COMPETENCIA_DECLARADA para id_lattes
                            if id_lattes:
                                self.grafo.add_edge(id_lattes, competencia_declarada, relation='COMPETENCIA_DECLARADA')
                                tipos_arestas['COMPETENCIA_DECLARADA'] += 1

                # --- Competências a Desenvolver ---
                competencias_desenvolver = pesquisador.get("competencias_desenvolver")
                if competencias_desenvolver and isinstance(competencias_desenvolver, list):
                    competencias_desenvolver = self.limpar_lista(competencias_desenvolver)
                    for competencia_desejada in competencias_desenvolver:
                        if isinstance(competencia_desejada, str):
                            # Criar nós de competencias_desenvolver
                            self.grafo.add_node(competencia_desejada, tipo='competencia_desejada')
                            tipos_nos['competencia_desejada'] += 1

                            # Adicionar arestas COMPETENCIA_DESEJADA para id_lattes
                            if id_lattes:
                                self.grafo.add_edge(id_lattes, competencia_desejada, relation='COMPETENCIA_DESEJADA')
                                tipos_arestas['COMPETENCIA_DESEJADA'] += 1

                # --- Intenções ---
                intencoes = {
                    "questoes_pesquisa": [],
                    "palavras_chave": [],
                    "desenvolvimento": [],
                    "desafios_ceis": [],
                    "produtos_emergenciais": [],
                    "produtos_agravos": []
                }

                string_questoes = pesquisador.get("questoes_interesse")
                if string_questoes and isinstance(string_questoes, str):
                    lista_questoes = self.limpar_questoes(string_questoes)
                    intencoes["questoes_pesquisa"].extend(lista_questoes)

                    # Criar aresta do id_lattes para interesse em questão de pesquisa
                    for questao_pesquisa in lista_questoes:
                        if isinstance(questao_pesquisa, str):
                            self.grafo.add_node(questao_pesquisa, tipo='questao_pesquisa')
                            if id_lattes:
                                self.grafo.add_edge(id_lattes, questao_pesquisa, relation='TEM_QUESTAO')

                palavras_chave = pesquisador.get("palavras_chave")
                if palavras_chave and isinstance(palavras_chave, list):
                    palavras_chave = self.limpar_lista(palavras_chave)
                    intencoes["palavras_chave"].extend(palavras_chave)

                    # Criar aresta do id_lattes para interesse em palavra_chave
                    for palavra_chave in palavras_chave:
                        if isinstance(palavra_chave, str):
                            self.grafo.add_node(palavra_chave, tipo='palavra_chave')
                            if id_lattes:
                                self.grafo.add_edge(id_lattes, palavra_chave, relation='TEM_PALAVRA_CHAVE')

                pretende_desenvolver = pesquisador.get("intencao_desenvolvimento")
                if pretende_desenvolver and isinstance(pretende_desenvolver, str):
                    intencoes["desenvolvimento"].append(pretende_desenvolver.strip())

                ceis_interesse_desafios = pesquisador.get("ceis_interesse_desafios")
                if ceis_interesse_desafios and isinstance(ceis_interesse_desafios, str):
                    lista_desafios = [x.strip() for x in ceis_interesse_desafios.split(';')]
                    lista_desafios = self.limpar_lista(lista_desafios)
                    intencoes["desafios_ceis"].extend(lista_desafios)

                    # Criar aresta do id_lattes para interesse em desafio do CEIS
                    for desafio in lista_desafios:
                        if isinstance(desafio, str):
                            self.grafo.add_node(desafio, tipo='ceis_desafio')
                            if id_lattes:
                                self.grafo.add_edge(id_lattes, desafio, relation='INTERESSA_DESAFIO')

                ceis_interesse_produtos_emergencias = pesquisador.get("ceis_interesse_produtos_emergencias")
                if ceis_interesse_produtos_emergencias and isinstance(ceis_interesse_produtos_emergencias, list):
                    for produto_emergencial in ceis_interesse_produtos_emergencias:
                        if isinstance(produto_emergencial, str):
                            if ";" in produto_emergencial:
                                lista_produtos = produto_emergencial.split(';')
                                for produto in lista_produtos:
                                    if produto != '':
                                        intencoes["produtos_emergenciais"].append(produto.strip())
                                    
                                        # Criar aresta id_lattes para produto bloco emergências do CEIS
                                        if isinstance(produto, str):
                                            self.grafo.add_node(produto, tipo='ceis_produto_emergencial')
                                            if id_lattes:
                                                self.grafo.add_edge(id_lattes, produto, relation='INTERESSA_PRODUTO')

                        elif isinstance(produto_emergencial, list):
                            for produto in produto_emergencial:
                                if produto != '':
                                    intencoes["produtos_emergenciais"].append(produto.strip())

                                    # Criar aresta id_lattes para produto bloco emergências do CEIS
                                    if isinstance(produto, str):
                                        self.grafo.add_node(produto, tipo='ceis_produto_emergencial')
                                        if id_lattes:
                                            self.grafo.add_edge(id_lattes, produto, relation='INTERESSA_PRODUTO')


                ceis_interesse_produtos_agravos = pesquisador.get("ceis_interesse_produtos_agravos")
                if ceis_interesse_produtos_agravos and isinstance(ceis_interesse_produtos_agravos, list):
                    for produto_agravo in ceis_interesse_produtos_agravos:
                        if isinstance(produto_agravo, str):
                            if ";" in produto_agravo:
                                lista_produtos = produto_agravo.split(';')
                                for produto in lista_produtos:
                                    if produto != '':
                                        intencoes["produtos_agravos"].append(produto.strip())

                                        # Criar aresta id_lattes para produto bloco agravos do CEIS
                                        if isinstance(produto, str):
                                            self.grafo.add_node(produto, tipo='ceis_produto_agravo')
                                            if id_lattes:
                                                self.grafo.add_edge(id_lattes, produto, relation='INTERESSA_PRODUTO')

                        if isinstance(produto_agravo, list):
                            for produto in produto_agravo:
                                if produto != '':
                                    intencoes["produtos_agravos"].append(produto.strip())

                                    # Criar aresta id_lattes para produto bloco agravos do CEIS
                                    if isinstance(produto, str):
                                        self.grafo.add_node(produto, tipo='ceis_produto_agravo')
                                        if id_lattes:
                                            self.grafo.add_edge(id_lattes, produto, relation='INTERESSA_PRODUTO')

                if verbose:
                    print(f"    Objeto de intenções tipo: {type(intencoes)} com {len(intencoes)} instancias")

                if id_lattes and any(intencoes.values()):
                    # Adicionar nós e arestas para cada tipo de intenção
                    for tipo_intencao, lista_intencoes in intencoes.items():
                        for intencao in lista_intencoes:
                            if tipo_intencao in ['produtos_emergenciais', 'produtos_agravos', 'desafios_ceis']:
                                if verbose:
                                    print(tipo_intencao)
                                # Verificar se o nó já existe no grafo
                                for no, dados in self.grafo.nodes(data=True):
                                    if dados.get('nome') == intencao:
                                        self.grafo.add_edge(id_lattes, no, relation='INTERESSE_' + tipo_intencao.upper())
                                        self.tipos_arestas['INTERESSE_' + tipo_intencao.upper()] += 1
                                        break
                            else:
                                # Criar um novo nó para a intenção
                                self.grafo.add_node(intencao, tipo=tipo_intencao)
                                self.grafo.add_edge(id_lattes, intencao, relation='POSSUI_' + tipo_intencao.upper())
                                self.tipos_nos[tipo_intencao] += 1
                                self.tipos_arestas['POSSUI_' + tipo_intencao.upper()] += 1

                    # Adicionar arestas para produtos e desafios do CEIS
                    self.adicionar_interesses_declarados_ceis(intencoes, pesquisador, respostas_nao_associadas)
                    num_arestas_interesse += 1

                    # Adicionar arestas por similaridade
                    # self.adicionar_interesses_por_similaridade(intencoes, pesquisador)
                else:
                    print("Nenhum valor no dicionário de intenções")

            except Exception as e:
                print(f"    Erro ao processar pesquisador {i}: {e}")

        # Adicionar checagem de tipos e quantidades
        self.grafo_conhecimento.info_subgrafo("oferta_pdi", self.grafo)


    def adicionar_projetos(self):
        """
        Adiciona os projetos de pesquisa ao grafo.
        """
        if self.dict_list:               
            for dicionario in self.dict_list:
                if isinstance(dicionario, dict):
                    id_lattes = dicionario.get('Identificação', {}).get('ID Lattes')
                    projetos = dicionario.get('Projetos de pesquisa', [])
                    for projeto in projetos:
                        self.grafo.add_node(projeto, tipo='projeto')
                        self.grafo.add_edge(id_lattes, projeto, relation='PARTICIPOU_DO_PROJETO')


    def listar_nomes_pesquisador(self, dados):
        print(f"    Estrutura de dados objeto tipo 'pesquisador' no grafo:")
        for no, dados in self.grafo.nodes(data=True):
            if dados.get('tipo') == 'pesquisador':
                print(f"    - {dados}")
        if dados.get('nome'):
            print(f"    Lista de todos pesquisadores no grafo atualmente:")
            print(f"    - {dados.get('nome')}")


    def comparar_nomes(self, nome_completo, nomes_parciais):
        """
        Compara duas strings de nomes, uma com o nome completo e outra com nomes parciais, 
        e retorna True se todos os nomes parciais estiverem presentes no nome completo.
        Ignora diferenças em acentuação gráfica.

        Args:
            nome_completo: Uma string com o nome completo.
            nomes_parciais: Uma string com os nomes parciais a serem verificados.

        Returns:
            True se todos os nomes parciais estiverem presentes no nome completo, False caso contrário.
        """
        try:
            # Normalizar os nomes para remover acentos
            nome_completo = unicodedata.normalize('NFKD', nome_completo).encode('ASCII', 'ignore').decode('ASCII')
            nomes_parciais = unicodedata.normalize('NFKD', nomes_parciais).encode('ASCII', 'ignore').decode('ASCII')

            # Converter os nomes para minúsculas e dividir em listas de palavras
            lista_nome_completo = nome_completo.lower().split()
            lista_nomes_parciais = nomes_parciais.lower().split()

            # Verificar se todos os nomes parciais estão na lista do nome completo
            for nome_parcial in lista_nomes_parciais:
                if nome_parcial not in lista_nome_completo:
                    return False

            return True
        except Exception as e:
            print(f"Erro ao comparar nomes: {e}")
            return False


    def primeira_letra_maiuscula(self, texto):
        """
        Converte a primeira letra de uma string para maiúscula.

        Args:
            texto: A string que você deseja modificar.

        Returns:
            A string com a primeira letra em maiúscula.
        """
        texto=texto.lower()
        return texto[0].upper() + texto[1:]


    def limpar_questoes(self, string_questoes):
        """
        Transforma uma string de questões em uma lista de questões, 
        considerando diferentes separadores.

        Args:
            string_questoes: A string contendo as questões, 
                            separadas por '\n' e/ou ';'.

        Returns:
            Uma lista de strings, onde cada string representa uma questão.
        """
        questoes = []
        for questao in string_questoes.split('\n'):
            for subquestao in questao.split(';'):
                subquestao = subquestao.strip()
                if subquestao:
                    questoes.append(subquestao.strip())
        return questoes


    def limpar_lista(self, lista):
        """
        Remove strings vazias, conjuntos vazios e itens especificados de uma lista.

        Args:
            lista: A lista que você deseja limpar.

        Returns:
            Uma nova lista sem as strings vazias, conjuntos vazios e itens especificados.
        """
        ignorar = [
            '',
            'As principais questões científicas que norteiam minhas pesquisas na Fiocruz Ceará, relacionadas ao enfrentamento dos desafios em saúde, envolvem:',
            'As principais palavras-chave que podem associar meus temas de pesquisa com oportunidades de fomento que desejo monitorar são:',
            'Competências científicas: ',
            'Competências Tecnológicas: ', 
            'As principais competências científicas e tecnológicas do grupo de pesquisa em que atuo, que podem contribuir para a implementação da Estratégia Nacional de Desenvolvimento do Complexo Econômico-Industrial da Saúde (CEIS), incluem:', 
            'Competências científicas: '
        ]
        return [item for item in lista if item not in ignorar]


    def adicionar_interesses_declarados_ceis(self, dic_intencoes, pesquisador, respostas_nao_associadas, verbose=False):
        """
        Adiciona arestas entre as intenções dos pesquisadores e os 
        produtos e desafios do CEIS, por escolha ou similaridade, 
        considerando a normalização dos nomes e a estrutura de listas.
        """

        # 1. Obter os IDs dos produtos e desafios de interesse
        produtos_emergenciais = pesquisador.get('ceis_interesse_produtos_emergencias', [])
        if verbose:
            print(f"  Função: adicionar_interesses_declarados_ceis")
            print(f"  {type(produtos_emergenciais)} {len(produtos_emergenciais)} Produtos_emergenciais:")
            print(f"  {produtos_emergenciais}")
        
        produtos_agravos = pesquisador.get('ceis_interesse_produtos_agravos', [])
        if verbose:
            print(f"  {type(produtos_agravos)} {len(produtos_agravos)} Produtos_agravos:")
            print(f"  {produtos_agravos}")
        
        desafios_interesse = pesquisador.get('ceis_interesse_desafios', "").split(';')
        if verbose:
            print(f"  {type(desafios_interesse)} {len(desafios_interesse)} Desafios:")
            print(f"  {desafios_interesse}")

        # 2. Normalizar os nomes dos produtos e desafios
        produtos_emergenciais_normalizados = [
            unicodedata.normalize('NFKD', p.strip())
            .encode('ASCII', 'ignore')
            .decode('ASCII')
            .lower() 
            for sublista in produtos_emergenciais for p in sublista
        ]
        produtos_agravos_normalizados = [
            unicodedata.normalize('NFKD', p.strip())
            .encode('ASCII', 'ignore')
            .decode('ASCII')
            .lower() 
            for sublista in produtos_agravos for p in sublista
        ]
        desafios_interesse_normalizados = [
            unicodedata.normalize('NFKD', d.strip())
            .encode('ASCII', 'ignore')
            .decode('ASCII')
            .lower() 
            for d in desafios_interesse
        ]

        # 3. Criar arestas para os produtos e desafios de interesse
        for intencao in dic_intencoes:
            if verbose:
                print(f"\nIntenção em intenções: {intencao}")
            for no, dados in self.grafo.nodes(data=True):
                nome_no_normalizado = unicodedata.normalize('NFKD', dados.get('nome', ''))
                nome_no_normalizado = nome_no_normalizado.encode('ASCII', 'ignore').decode('ASCII').lower()

                tipo = 'produto'
                if dados.get('tipo') == tipo and nome_no_normalizado in produtos_emergenciais_normalizados:
                    self.grafo.add_edge(intencao, no, relation='INTERESSE_PRODUTOS_EMERGENCIAIS')
                    self.tipos_arestas['INTERESSE_PRODUTOS_EMERGENCIAIS'] += 1
                else:
                    if verbose:
                        print(f"  Não foram encontrados nó com tipo '{tipo}' no grafo")

                if dados.get('tipo') == tipo and nome_no_normalizado in produtos_agravos_normalizados:
                    self.grafo.add_edge(intencao, no, relation='INTERESSE_AGRAVOS_CRITICOS')
                    self.tipos_arestas['INTERESSE_AGRAVOS_CRITICOS'] += 1
                else:
                    if verbose:
                        print(f"  Não foram encontrados nó com tipo '{tipo}' no grafo")

                tipo = 'desafio'
                if dados.get('tipo') == tipo and nome_no_normalizado in desafios_interesse_normalizados:
                    self.grafo.add_edge(intencao, no, relation='TEM_INTERESSE_EM_DESAFIO')
                    self.tipos_arestas['TEM_INTERESSE_EM_DESAFIO'] += 1
                else:
                    if verbose:
                        print(f"  Não foram encontrados nó com tipo '{tipo}' no grafo")


        ## 4. Criar arestas SIMILAR por aproximação semântica
        # self.adicionar_interesses_por_similaridade(dic_intencoes)

        ## Mostrar acumulado de arestas à medida que processa cada currículo
        # if pesquisador.get('nome_pesquisador') in ['Não desejo.','']:
        #     nome_pesquisador = 'Anônimo'
        # else:
        #     nome_pesquisador = pesquisador.get('nome_pesquisador')

        # Imprimir a quantidade de relações por tipo criadas com sucesso
        # print(f"  Quantidade acumulada de Relações criadas no grafo de conhecimento:")
        # for relation, count in self.tipos_arestas.items():
        #     print(f"  - {relation}: {count}")
        # print()

    def calcular_similaridade_semantica(self, texto1, texto2):
        """
        Calcula a similaridade semântica entre dois textos.
        """
        embeddings1 = self.model.encode(texto1, convert_to_tensor=True)
        embeddings2 = self.model.encode(texto2, convert_to_tensor=True)
        cosine_sim = util.cos_sim(embeddings1, embeddings2)
        return cosine_sim.item()


    def adicionar_interesses_por_similaridade(self, intencoes, verbose=True):
        """
        Adiciona arestas SIMILAR entre as intenções dos pesquisadores 
        e as áreas de pesquisa, produtos e desafios do CEIS, 
        por aproximação semântica.
        """
        threshold_similaridade = 0.7  # Defina o limiar de similaridade
        from sentence_transformers import SentenceTransformer

        if verbose:
            print("="*175)
            print(f"Verificar dicionário de intenções: {type(intencoes)}")
            for i in [(x,y) for (x,y) in intencoes.items()]:
                print(i)
            print("="*175)

        # 1. Obter o texto das intenções
        textos_intencoes = []
        for x,y in intencoes.items():
            if isinstance(y, str):
                if ';' in y:
                    intencoes_produtos = y.split(";")
                    for produto in intencoes_produtos:
                        textos_intencoes.append(produto)
            elif isinstance(y, list):
                for k in y:
                    textos_intencoes.append(k)
            else:
                print(f"Objeto não é um string é: {type(y)}")

        # 2. Obter o texto das áreas de pesquisa, produtos e desafios
        textos_produtos_desafios = []
        for no, dados in self.grafo.nodes(data=True):
            if verbose:
                print(f"Dados do nó: {dados}")
            if dados.get('tipo') in ['produto', 'desafio']:
                texto_demanda = dados.get('nome', '')
                textos_produtos_desafios.append(texto_demanda)
                print(f"Texto: {texto_demanda}")

        # 3. Calcular a similaridade de cosseno para cada intenção
        for intencao in textos_intencoes:
            if verbose:
                print('-'*125)
                print(f"Intenção: {intencao}")
            for texto_area_produto_desafio in textos_produtos_desafios:
                print(f"Texto: {texto_area_produto_desafio}")
                similaridade = self.calcular_similaridade_semantica(intencao, texto_area_produto_desafio)
                if verbose:
                    print(f"{similaridade} | {intencao} | {texto_area_produto_desafio}")
                if similaridade >= threshold_similaridade:
                    # Encontrar o nó correspondente ao texto_area_produto_desafio
                    for no, dados in self.grafo.nodes(data=True):
                        if dados.get('nome') == texto_area_produto_desafio:
                            self.grafo.add_edge(no, intencao, relation='SIMILAR_A_PRODUTO_CEIS', similaridade=similaridade)
                            self.tipos_arestas['SIMILAR_A_PRODUTO_CEIS'] += 1
                            break  # Interromper o loop após encontrar o nó

    # def adicionar_competencias(self, id_lattes, nome, pesquisador, tipos_nos):
    #     """
    #     Adiciona camada de nós de competências ao grafo, relacionando-as aos pesquisadores.
    #     Baseada em dados das respostas dos pesquisadores aos levantamentos e questionários
    #     """
    #     competencias_possuidas = pesquisador.get('competencias_possuidas', [])
    #     competencias_desenvolver = pesquisador.get('competencias_desenvolver', [])

    #     for competencias in competencias_possuidas:
    #         # Criar um nó para a competência, se ele ainda não existir
    #         if not self.grafo.has_node(competencias):
    #             self.grafo.add_node(competencias, tipo='competencia_possuida')

    #         # Criar uma aresta entre o pesquisador e a competência
    #         if id_lattes:
    #             self.grafo.add_edge(id_lattes, competencias, relation='POSSUI_COMPETENCIA')
    #             tipos_nos['competencia_possuida'] += 1

    #     for competencias in competencias_desenvolver:
    #         # Criar um nó para a competência, se ele ainda não existir
    #         if not self.grafo.has_node(competencias):
    #             self.grafo.add_node(competencias, tipo='competencia_desenvolver')

    #         # Criar uma aresta entre o pesquisador e a competência
    #         if id_lattes:
    #             self.grafo.add_edge(id_lattes, competencias, relation='DESEJA_DESENVOLVER_COMPETENCIA')
    #             tipos_nos['competencia_desenvolver'] += 1


## CLASSIFICADOR POR ZEROSHOT
from transformers import pipeline

class ZeroShotClassifier:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "zero-shot-classification", 
            model=model_name,
            device=device,
            output_scores=True  # Adicionar este parâmetro
        )
        self.labels = ["biological", "small molecule"]

    def classify(self, text):
        result = self.classifier(text, self.labels)
        return {
            "label": result["labels"][0], # type: ignore
            "score": np.round(float(result["scores"][0]),4),  # type: ignore # float para serializar 
            "scores": {
                label: float(np.round(score,4)) 
                for label, score in zip(result["labels"], result["scores"]) # type: ignore
            }
        }


## CLASSIFICADOR POR FEWSHOT
from sentence_transformers import SentenceTransformer, util

class FewShotClassifier:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name).to(self.device)
        self.examples = {
            "biological": [],
            "small molecule": []
        }

    def add_example(self, text, label):
        self.examples[label].append(text)

    def classify(self, text):
        text_embedding = self.model.encode(text, convert_to_tensor=True).to(self.device)
        
        best_score = -1
        best_label = None

        for label, examples in self.examples.items():
            example_embeddings = self.model.encode(examples, convert_to_tensor=True).to(self.device)
            cosine_scores = util.pytorch_cos_sim(text_embedding, example_embeddings)
            max_score = cosine_scores.max().item()
            
            if max_score > best_score:
                best_score = max_score
                best_label = label

        return {
            "label": best_label,
            "score": np.round(best_score,4)
        }


## CLASSIFICADOR POR SIMILARIDADE
from sentence_transformers import SentenceTransformer, util
import numpy as np

class SimilarityBasedClassifier:
    def __init__(self, model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name).to(self.device)
        self.biological_keywords = [
            "antibody", "protein", "vaccine", "monoclonal",
            "enzyme", "hormone", "peptide", "immunoglobulin",
            "antiserum", "cellular", "recombinant", "biological",
            "therapeutic protein", "biomolecule"
        ]
        self.small_molecule_keywords = [
            "chemical", "synthetic", "compound", "inhibitor",
            "small molecule", "drug", "organic", "synthesis",
            "molecular weight", "crystalline", "salt", "tablet",
            "oral", "chemical synthesis"
        ]
        
        # Criar embeddings para as palavras-chave
        self.bio_embeddings = self.model.encode(self.biological_keywords, convert_to_tensor=True).to(self.device)
        self.sm_embeddings = self.model.encode(self.small_molecule_keywords, convert_to_tensor=True).to(self.device)
    
    def classify(self, text):
        # Gerar embedding para o texto
        text_embedding = self.model.encode(text, convert_to_tensor=True).to(self.device)
        
        # Calcular similaridade média com cada conjunto de keywords
        bio_similarities = util.pytorch_cos_sim(
            text_embedding.unsqueeze(0), 
            self.bio_embeddings
        ).mean()
        
        sm_similarities = util.pytorch_cos_sim(
            text_embedding.unsqueeze(0), 
            self.sm_embeddings
        ).mean()
        
        # Determinar classificação baseada na maior similaridade
        if bio_similarities > sm_similarities:
            return {
                "label": "biological",
                "score": np.round(float(bio_similarities.cpu()),)
            }
        else:
            return {
                "label": "small molecule",
                "score": np.round(float(sm_similarities.cpu()))
            }


## MODELO COMBINADO DE CLASSIFICAÇÃO DO TIPO DE ROTA TECNOLÓGICA
class CombinedClassifier:
    def __init__(self, zero_shot_classifier, few_shot_classifier, similarity_classifier):
        self.zero_shot = zero_shot_classifier
        self.few_shot = few_shot_classifier
        self.similarity = similarity_classifier
    
    def classify(self, text):
        # Obter resultados dos três classificadores
        zero_shot_result = self.zero_shot.classify(text)
        few_shot_result = self.few_shot.classify(text)
        similarity_result = self.similarity.classify(text)
        
        # Comparar scores e escolher o melhor resultado
        results = [
            (zero_shot_result, "zero_shot"),
            (few_shot_result, "few_shot"),
            (similarity_result, "similarity")
        ]
        
        best_result = max(results, key=lambda x: x[0]["score"])
        return best_result[0]
    
    def explain(self, text):
        zero_shot_result = self.zero_shot.classify(text)
        few_shot_result = self.few_shot.classify(text)
        similarity_result = self.similarity.classify(text)
        
        final_classification = self.classify(text)
        
        return {
            "zero_shot": zero_shot_result,
            "few_shot": few_shot_result,
            "similarity": similarity_result,
            "final_classification": final_classification
        }



## MODELO CLASSIFICADOR DA ROTA TECNOLÓGICA BIOLÓGICA OU SINTÉTICA
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

class ProductTypeClassifier:
    def __init__(self):
        # Carregar BioBERT, um BERT pré-treinado em textos biomédicos
        self.model_name = "dmis-lab/biobert-v1.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Classificador final
        self.classifier = None
        
    def prepare_training_data(self, grafo):
        """Prepara dados de treinamento a partir do grafo"""
        texts = []
        labels = []
        
        for node, data in grafo.nodes(data=True):
            if data.get('tipo') in ['produto']:
                label = data.get('label', '')
                texts.append(label)
                
                # Determinar classe baseado nas características do produto
                is_biological = self._check_biological_indicators(label)
                labels.append(1 if is_biological else 0)
        
        print(f"{len(texts)} rótulos de produto lidos no grafo de conhecimento")        
        return texts, labels
    
    def _check_biological_indicators(self, text):
        """
        Verifica indicadores de tipo de produto biológico ou sintético no texto
        Baseado em palavras-chave comuns em produtos biológicos
        """
        bio_keywords = [
            'antibody', 'protein', 'vaccine', 'monoclonal',
            'enzyme', 'hormone', 'peptide', 'immunoglobulin',
            'antiserum', 'cellular', 'recombinant'
        ]
        
        text = text.lower()
        return any(keyword in text for keyword in bio_keywords)
    
    def get_embeddings(self, texts):
        """Gera embeddings usando BioBERT"""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding.cpu().numpy().squeeze())
                
        return np.array(embeddings)
    
    def train(self, grafo, test_size=0.2):
        """Treina o classificador"""
        texts, labels = self.prepare_training_data(grafo)
        
        # Verificar quantidade mínima de amostras
        n_samples = len(texts)
        if n_samples < 2:
            raise ValueError(f"Número insuficiente de amostras para treino: {n_samples}. Mínimo necessário: 2")
        
        # Verificar classes presentes
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError(f"Necessário ter amostras de ambas as classes (biological e small molecule). Encontrado apenas: {len(unique_labels)} classe(s)")
        
        # Gerar embeddings e treinar
        embeddings = self.get_embeddings(texts)
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=42
        )
        
        # Treinar classificador XGBoost
        from xgboost import XGBClassifier
        self.classifier = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.classifier.fit(X_train, y_train)
        
        # Avaliar modelo
        y_pred = self.classifier.predict(X_test)
        
        # Identificar classes presentes para o relatório
        present_classes = sorted(list(set(y_test)))
        target_names = ['Small Molecule' if i == 0 else 'Biological' for i in present_classes]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        
    def predict(self, text):
        """Prediz o tipo de um novo produto"""
        embedding = self.get_embeddings([text])
        prob = self.classifier.predict_proba(embedding)[0]
        return {
            'type': 'biological' if prob[1] > 0.5 else 'small_molecule',
            'confidence': float(max(prob))
        }
    
    def save_model(self, filename):
        """Salva o modelo treinado"""
        with open(filename, 'wb') as f:
            pickle.dump(self.classifier, f)
    
    def load_model(self, filename):
        """Carrega um modelo treinado"""
        with open(filename, 'rb') as f:
            self.classifier = pickle.load(f)

class ProductClassifier:
    def __init__(self, grafo, classifier):
        self.grafo = grafo
        self.classifier = classifier
    
    def classify_products(self):
        for node_id, node_data in self.grafo.nodes(data=True):
            if node_data.get('tipo') == 'produto':
                label = node_data.get('label', '')
                if label:
                    classification = self.classifier.classify(label)
                    
                    # Atualizar o nó com a classificação
                    self.grafo.nodes[node_id]['categoria'] = classification['label']
                    self.grafo.nodes[node_id]['categoria_score'] = classification['score']
                    
                    print(f"Produto: {label}")
                    print(f"Classificação: {classification['label']}")
                    print(f"Score: {classification['score']}")
                    print("---")
    
    def get_classification_summary(self):
        biological_count = 0
        small_molecule_count = 0
        
        for _, node_data in self.grafo.nodes(data=True):
            if node_data.get('tipo') == 'produto':
                if node_data.get('categoria') == 'biological':
                    biological_count += 1
                elif node_data.get('categoria') == 'small molecule':
                    small_molecule_count += 1
        
        total = biological_count + small_molecule_count
        
        return {
            'total_produtos': total,
            'biological': biological_count,
            'small_molecule': small_molecule_count,
            'biological_percentage': (biological_count / total) * 100 if total > 0 else 0,
            'small_molecule_percentage': (small_molecule_count / total) * 100 if total > 0 else 0
        }

class UserFeedbackClassifierNewEntries:
    def __init__(self, combined_classifier):
        self.combined_classifier = combined_classifier
        self.feedback_data = []

    def classify_with_feedback(self, text):
        # Classificação inicial
        result = self.combined_classifier.classify(text)
        
        # Solicitar feedback do usuário
        print(f"Classificação inicial: {result['label']} (confiança: {result['score']:.2f})")
        user_feedback = input("Esta classificação está correta? (s/n): ").lower()
        
        if user_feedback == 'n':
            correct_label = input("Qual é a classificação correta? (biological/small molecule): ").lower()
            self.feedback_data.append((text, correct_label))
            print("Feedback registrado. Obrigado!")
            
            # Atualizar o classificador few-shot
            self.combined_classifier.few_shot.add_example(text, correct_label)
        
        return result

    def train_from_feedback(self):
        # Treinar o classificador com os dados de feedback
        for text, label in self.feedback_data:
            self.combined_classifier.few_shot.add_example(text, label)
        
        print(f"Classificador atualizado com {len(self.feedback_data)} novos exemplos.")
        self.feedback_data = []  # Limpar dados de feedback após o treinamento

    def evaluate_performance(self, test_data):
        correct = 0
        total = len(test_data)
        
        for text, true_label in test_data:
            prediction = self.combined_classifier.classify(text)
            if prediction['label'] == true_label:
                correct += 1
        
        accuracy = correct / total
        print(f"Acurácia atual: {accuracy:.2f}")

    def interactive_classification_session(self):
        while True:
            text = input("Digite o texto para classificar (ou 'sair' para encerrar): ")
            if text.lower() == 'sair':
                break
            
            self.classify_with_feedback(text)
            
        self.train_from_feedback()

class UserFeedbackClassifier:
    def __init__(self, combined_classifier, grafo):
        self.combined_classifier = combined_classifier
        self.grafo = grafo
        self.feedback_data = []

    def classify_nodes_with_feedback(self, tipo_no='produto'):
        """Classifica nós do tipo especificado com feedback do usuário"""
        for node_id, node_data in self.grafo.nodes(data=True):
            if node_data.get('tipo') == tipo_no:
                label = node_data.get('label', '')
                if label:
                    # Classificação inicial
                    result = self.combined_classifier.classify(label)
                    
                    print(f"\nNó: {node_id}")
                    print(f"Label: {label}")
                    print(f"Classificação inicial: {result['label']} (confiança: {result['score']:.2f})")
                    
                    user_feedback = input("Esta classificação está correta? (s/n): ").lower()
                    
                    if user_feedback == 'n':
                        correct_label = input("Qual é a classificação correta? (biological/small molecule): ").lower()
                        self.feedback_data.append((label, correct_label))
                        
                        # Atualizar o nó com a classificação correta
                        self.grafo.nodes[node_id]['categoria'] = correct_label
                        print("Feedback registrado e nó atualizado.")
                    else:
                        # Atualizar o nó com a classificação do modelo
                        self.grafo.nodes[node_id]['categoria'] = result['label']
                        self.grafo.nodes[node_id]['categoria_score'] = result['score']

    def get_classification_summary(self):
        """Retorna um resumo das classificações no grafo"""
        biological_count = 0
        small_molecule_count = 0
        
        for _, node_data in self.grafo.nodes(data=True):
            if node_data.get('tipo') == 'produto':
                categoria = node_data.get('categoria')
                if categoria == 'biological':
                    biological_count += 1
                elif categoria == 'small molecule':
                    small_molecule_count += 1
        
        total = biological_count + small_molecule_count
        
        return {
            'total_produtos': total,
            'biological': biological_count,
            'small_molecule': small_molecule_count,
            'biological_percentage': (biological_count / total) * 100 if total > 0 else 0,
            'small_molecule_percentage': (small_molecule_count / total) * 100 if total > 0 else 0
        }

    def train_from_feedback(self):
        """Treina o classificador com os dados de feedback coletados"""
        for text, label in self.feedback_data:
            self.combined_classifier.few_shot.add_example(text, label)
        
        print(f"Classificador atualizado com {len(self.feedback_data)} novos exemplos.")
        self.feedback_data = []
