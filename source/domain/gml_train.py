import torch
import networkx as nx
from torch_geometric.data import Data
from sklearn.metrics import adjusted_mutual_info_score

class AprendizadoGrafo:

    def __init__(self, grafo_conhecimento, modelos, arquiteturas, 
                 epocas=100, taxa_aprendizado=0.01):
        """
        Inicializa a classe AprendizadoGrafo.

        Args:
            grafo_conhecimento: Objeto GrafoConhecimento contendo o grafo de conhecimento.
            modelos (list): Lista de modelos de embedding a serem testados.
            arquiteturas (list): Lista de arquiteturas de GNN a serem testadas.
            epocas (int): Número de épocas de treinamento.
            taxa_aprendizado (float): Taxa de aprendizado do otimizador.
        """
        self.grafo_conhecimento = grafo_conhecimento
        self.modelos = modelos
        self.arquiteturas = arquiteturas
        self.epocas = epocas
        self.taxa_aprendizado = taxa_aprendizado
        self.resultados = []

    def converter_para_pytorch_geometric(self):
        """
        Converte o grafo NetworkX para o formato PyTorch Geometric.
        """
        # 1. Converter nós para tensores PyTorch
        caracteristicas_nos = []
        for no in self.grafo_conhecimento.grafo.nodes():
            caracteristicas = self.grafo_conhecimento.extrair_caracteristicas(no)
            caracteristicas_nos.append(caracteristicas)
        x = torch.tensor(caracteristicas_nos, dtype=torch.float)  # Converter para tensor PyTorch

        # 2. Converter arestas para tensores PyTorch
        arestas = list(self.grafo_conhecimento.grafo.edges())
        edge_index = torch.tensor(arestas, dtype=torch.long).t().contiguous()  # Converter para tensor PyTorch

        # 3. Criar objeto Data do PyTorch Geometric
        data = Data(x=x, edge_index=edge_index)

        return data

    def treinar(self):
        """
        Treina os modelos de GNN com diferentes modelos de embedding e arquiteturas.
        """
        for modelo_embedding in self.modelos:
            for arquitetura in self.arquiteturas:
                # Criar o modelo GNN
                # ... (implementação da criação do modelo GNN) ...

                # Converter o grafo para PyTorch Geometric
                data = self.converter_para_pytorch_geometric()

                # Otimizador
                otimizador = torch.optim.Adam(modelo.parameters(), lr=self.taxa_aprendizado)

                # Loop de treinamento
                for epoca in range(self.epocas):
                    # Forward pass
                    # ... (implementação do forward pass) ...

                    # Calcular a perda
                    # ... (implementação do cálculo da perda) ...

                    # Backward pass e otimização
                    # ... (implementação do backward pass e otimização) ...

                    # Calcular métricas (modularidade, NMI, etc.)
                    # ... (implementação do cálculo das métricas) ...

                # Salvar os resultados
                self.resultados.append({
                    'modelo_embedding': modelo_embedding,
                    'arquitetura': arquitetura,
                    'metricas': metricas  # Dicionário com as métricas calculadas
                })

    def gerar_tabela_comparativa(self):
        """
        Gera uma tabela comparativa dos resultados do treinamento em LaTeX.
        """
        # Criar um DataFrame pandas com os resultados
        # ... (implementação da criação do DataFrame) ...

        # Gerar a tabela em LaTeX
        tabela_latex = df.to_latex(index=False, float_format="{:.4f}".format)

        # Exibir a tabela
        print(tabela_latex)

    def gerar_arestas_similaridade(self, modelo, arquitetura):
        """
        Gera arestas de similaridade com base nos embeddings 
        gerados pelo modelo GNN.
        """
        # Carregar o modelo treinado
        # ... (implementação do carregamento do modelo) ...

        # Gerar embeddings para os nós
        # ... (implementação da geração de embeddings) ...

        # Calcular a similaridade entre os nós
        # ... (implementação do cálculo de similaridade) ...

        # Adicionar arestas ao grafo com base na similaridade
        # ... (implementação da adição de arestas) ...

import pandas as pd

class GeradorTabelas:

    def __init__(self, resultados):
        """
        Inicializa a classe GeradorTabelas.

        Args:
            resultados (list): Lista de dicionários com os resultados do treinamento.
        """
        self.resultados = resultados

    def gerar_tabela_latex(self):
        """
        Gera uma tabela em LaTeX com os resultados do treinamento.
        """
        df = pd.DataFrame(self.resultados)
        tabela_latex = df.to_latex(index=False, float_format="{:.4f}".format)
        return tabela_latex        