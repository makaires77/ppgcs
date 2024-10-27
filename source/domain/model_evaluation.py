import gc
import os
import json
import time
import subprocess
from git import Repo

import pandas as pd
import plotly.graph_objects as go
from validclust import dunn
from scipy.spatial.distance import pdist
from jsonschema import validate, ValidationError
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader


class GraphDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GKANModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation_function):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.activation_function = activation_function

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.activation_function(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class ModelPerformance:
    def __init__(self, models, data, batch_size=32, epochs=10):
        self.models = models
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

    def train_and_evaluate(self):
        for model_name, model_params in self.models.items():
            model = model_params["model"]
            model.to(self.device)
            optimizer = torch.optim.Adam(model.parameters())

            dataset = GraphDataset([self.data])  # Encapsula os dados do grafo em um Dataset
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            start_time = time.time()
            for epoch in range(self.epochs):
                for batch in dataloader:  # Itera sobre os batches do grafo
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    out = model(batch)
                    loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
                    loss.backward()
                    optimizer.step()
            end_time = time.time()

            _, predicted = torch.max(out, 1)

            # accuracy = self.calculate_accuracy(model, batch)  # Calcula a acurácia para aprendizagem supervisionada
            gpu_utilization = self.get_gpu_utilization()  # Obtém a utilização da GPU
           
            self.results[model_name] = {
                "tempo_epoca": (end_time - start_time) / self.epochs,
                # "acuracia": accuracy  # Manter comentado para os problemas não-supervisionados
                "utilizacao_gpu": self.get_gpu_utilization(),  # Função para obter a utilização da GPU
                "silhouette": silhouette_score(data.x.cpu().numpy(), predicted.cpu().numpy()),
                "davies_bouldin": davies_bouldin_score(data.x.cpu().numpy(), predicted.cpu().numpy()),
                "calinski_harabasz": calinski_harabasz_score(data.x.cpu().numpy(), predicted.cpu().numpy()),
                "dunn": dunn(dist.pdist(data.x.cpu().numpy()), predicted.cpu().numpy()),
            }

    def calculate_accuracy(self, model, data):
        """
        A função coloca o modelo em modo de avaliação (model.eval()) para desativar recursos
          como dropout e batch normalization, que são usados apenas durante o treinamento.
        O bloco with torch.no_grad(): desativa o cálculo de gradientes, o que economiza 
          memória e acelera o processo de inferência.
        A função assume que os dados possuem uma máscara test_mask que indica quais nós 
          pertencem ao conjunto de teste.
        A acurácia é calculada como a proporção de previsões corretas em relação ao 
          número total de amostras no conjunto de teste.
        """
        model.eval()  # Coloca o modelo em modo de avaliação
        with torch.no_grad():  # Desativa o cálculo de gradientes
            out = model(data)
            _, pred = out.max(dim=1)  # Obtém as previsões do modelo
            correct = pred[data.test_mask] == data.y[data.test_mask]  # Compara com os rótulos reais
            accuracy = int(correct.sum()) / int(data.test_mask.sum())  # Calcula a acurácia
        return accuracy

    def get_gpu_utilization(self):
        """
        A função utiliza o comando nvidia-smi para obter a utilização da GPU.
          O comando nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader 
          retorna a utilização da GPU como um valor inteiro (em porcentagem).
        A função inclui um tratamento de erro (try-except) para o caso de o comando nvidia-smi 
          não ser encontrado, o que pode acontecer se os drivers da NVIDIA não estiverem instalados.
        """
        try:
            result = subprocess.check_output(
                [
                    'nvidia-smi', '--query-gpu=utilization.gpu',
                    '--format=csv,nounits,noheader'
                ], encoding='utf-8')
            utilization = int(result.strip())
            return utilization
        except FileNotFoundError:
            print("nvidia-smi não encontrado. Certifique-se de ter os drivers da NVIDIA instalados.")
            return None

    def plot_results(self):
        model_names = list(self.results.keys())
        tempos_epoca = [result["tempo_epoca"] for result in self.results.values()]
        utilizacao_gpu = [result["utilizacao_gpu"] for result in self.results.values()]
        silhouette_scores = [result["silhouette"] for result in self.results.values()]
        davies_bouldin_scores = [result["davies_bouldin"] for result in self.results.values()]
        calinski_harabasz_scores = [result["calinski_harabasz"] for result in self.results.values()]
        dunn_scores = [result["dunn"] for result in self.results.values()]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=model_names, y=tempos_epoca, name="Tempo por Época"))
        fig.add_trace(go.Scatter(x=model_names, y=silhouette_scores, name="Silhouette Score", yaxis="y2"))
        fig.add_trace(go.Scatter(x=model_names, y=davies_bouldin_scores, name="Davies-Bouldin Index", yaxis="y2"))
        fig.add_trace(go.Scatter(x=model_names, y=calinski_harabasz_scores, name="Calinski-Harabasz Index", yaxis="y2"))
        fig.add_trace(go.Scatter(x=model_names, y=dunn_scores, name="Dunn Index", yaxis="y2"))

        fig.update_layout(
            title="Desempenho dos Modelos",
            xaxis_title="Modelo",
            yaxis_title="Tempo por Época (s)",
            yaxis2=dict(title="Métricas de Avaliação", overlaying="y", side="right"),
            yaxis3=dict(title="Utilização da GPU (%)", overlaying="y", side="left", anchor="free", position=0.02),
            legend=dict(x=0, y=1.1),
        )
        fig.show()

    def clear_gpu():
        torch.cuda.empty_cache()
        gc.collect()

    def create_and_train_models(self, data, epochs):  # Movido para dentro da classe
        # Modelos de aprendizado em grafos
        models = {
            "GNN": {
                "model": GNNModel(data.num_node_features, 16, 2),
            },
            "KAN_ReLU": {
                "model": GKANModel(data.num_node_features, 16, 2, torch.relu),
            },
            "KAN_Sigmoid": {
                "model": GKANModel(data.num_node_features, 16, 2, torch.sigmoid),
            },
            # ... outros modelos KAN com diferentes funções de ativação
            "GNN_GFT": {
                "model": GNNModel(data.num_node_features, 16, 2),  # Substitua pela GCN com GFT
            },
            # ... outros modelos GNN com diferentes transformadas
        }

        self.models = models  # Atribui os modelos ao atributo self.models
        self.data = data  # Atribui os dados ao atributo self.data
        self.epochs = epochs  # Atribui as épocas ao atributo self.epochs
        self.train_and_evaluate()  # Chama o método train_and_evaluate
        return self.results  # Retorna os resultados

    def generate_latex_tables(results):
        df = pd.DataFrame(results).transpose()
        print(df.to_latex(index=True, float_format="%.4f"))

    def generate_performance_graphs(results):
        model_names = list(results.keys())
        metrics = ["acuracia", "silhouette", "davies_bouldin", "calinski_harabasz", "dunn"]
        
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Bar(
                x=model_names,
                y=[result[metric] for result in results.values()],
                name=metric
            ))

        fig.update_layout(
            title="Desempenho dos Modelos",
            xaxis_title="Modelo",
            yaxis_title="Valor da Métrica",
            barmode='group',
            legend=dict(x=0, y=1.1),
        )
        fig.show()

    if __name__ == "__main__":
        clear_gpu()

        # Dados do grafo (exemplo)
        edge_index = torch.tensor([[0, 1, 1, 2],
                                [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)

        epochs = 10  # Defina o número de épocas

        performance = ModelPerformance()  # Cria uma instância da classe
        performance_results = performance.create_and_train_models(data, epochs)  # Chama o método na instância

        generate_latex_tables(performance_results)
        generate_performance_graphs(performance_results)


class JSONValidator:
    def __init__(self):
        pass  # Não precisa de construtor, pois o schema será determinado dinamicamente

    def validate_json(self, json_file, schema_file):  # Adiciona o argumento schema_file
        try:
            with open(schema_file, 'r') as f:
                schema = json.load(f)
            with open(json_file, 'r') as f:
                data = json.load(f)
            validate(instance=data, schema=schema)
            print(f"Arquivo validado com sucesso: {os.path.basename(json_file)}")
            return True
        except ValidationError as e:
            print(f"Erro de validação no arquivo {json_file}: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar o arquivo {json_file}: {e}")
            return False

    def validate_input(self, filename):
        try:
            repo = Repo(search_parent_directories=True)
            root_folder = repo.working_tree_dir
            pathfilename = os.path.join(root_folder,'_data','out_json',filename)

            schema_filename = 'schema_' + os.path.splitext(filename)[0].split('_')[1] + '.json'
            # print(f"Utilizar esquema: {schema_filename}")
            schema_path = os.path.join(root_folder,'_data','out_json', schema_filename)  # Caminho completo para o schema

            validator = JSONValidator()  # Instancia a classe sem o schema
            validator.validate_json(pathfilename, schema_path)  # Chama validate_json com o schema
        except Exception as e:
            print(e)


    def extract_competencies(self, json_file):
        if self.validate_json(json_file):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Extrair competências da formação acadêmica e complementar
                competencias = []
                for pesquisador in data:
                    for formacao in pesquisador['Formação']['Acadêmica']:
                        competencias.append(formacao['Descrição'])
                    for formacao in pesquisador['Formação']['Complementar']:
                        competencias.append(formacao['Descrição'])
                    if 'Bancas' in pesquisador:
                        if 'Participação em bancas de trabalhos de conclusão' in pesquisador['Bancas']:
                            for participacao in pesquisador['Bancas']['Participação em bancas de trabalhos de conclusão'].values():
                                competencias.append(participacao)
                        if 'Participação em bancas de comissões julgadoras' in pesquisador['Bancas']:
                            for participacao in pesquisador['Bancas']['Participação em bancas de comissões julgadoras'].values():
                                competencias.append(participacao)

                return competencias
            except KeyError as e:
                print(f"Erro ao extrair competências do arquivo {json_file}: {e}")
                return []
        else:
            return []         