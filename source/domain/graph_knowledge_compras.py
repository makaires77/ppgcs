import torch
from torch_geometric.data import Data
import networkx as nx

class Grafo:
    def __init__(self):
        self.grafo = None
        self.no_atual = None
        self.no_atual_id = None
        self.node_id_to_index = None  # Dicionário para mapear IDs para índices

    def extrair_dados_normas(self, url_norma):
        """
        Extrai informações relevantes de uma página HTML de norma.
        """
        try:
            response = requests.get(url_norma)
            response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
            soup = BeautifulSoup(response.content, 'html.parser')

            # Exemplo: extrair o título da norma e o número do artigo
            titulo_norma = soup.find('h1', class_='titulo-norma').text.strip()
            artigo = soup.find('p', class_='artigo-norma').text.strip()

            informacoes_relevantes = {
                'titulo': titulo_norma,
                'artigo': artigo
                # ... outras informações relevantes
            }
            return informacoes_relevantes
        except requests.exceptions.RequestException as e:
            print(f"Erro ao acessar a página: {e}")
            return None

    def carregar_grafo(self, json_data):
        """
        Carrega o grafo a partir dos dados JSON e converte para PyTorch Geometric Data.
        """
        nodes = json_data['nodes']
        edges = json_data['edges']

        node_features = []
        edge_index = [[], []]

        self.node_id_to_index = {node['id']: i for i, node in enumerate(nodes)}

        for node in nodes:
            features = self._extrair_features_do_no(node)
            node_features.append(features)

        for edge in edges:
            source_index = self.node_id_to_index.get(edge['source'])
            target_index = self.node_id_to_index.get(edge['target'])
            if source_index is not None and target_index is not None:  # Verifica se os nós existem
                edge_index[0].append(source_index)
                edge_index[1].append(target_index)
            else:
                print(f"Erro: Nó {edge['source']} ou {edge['target']} não encontrado.")

        # Converter para tensores do PyTorch
        if node_features:  # Verifica se há features
            x = torch.tensor(node_features, dtype=torch.float)
        else:
            x = torch.empty((len(nodes), 0))  # Cria um tensor vazio se não houver features

        edge_index = torch.tensor(edge_index, dtype=torch.long)

        self.grafo = Data(x=x, edge_index=edge_index)
        self.no_atual_id = self._obter_no_inicial(nodes)
        self.no_atual = self.node_id_to_index[self.no_atual_id]

    def adicionar_dados_no(self, no_id, dados_formulario):
        """
        Adiciona os dados do formulário aos atributos do nó.
        """
        no_index = self.node_id_to_index[no_id]
        for chave, valor in dados_formulario.items():
            #  Verifica se o atributo já existe no nó
            if hasattr(self.grafo, chave):
                #  Adiciona o valor ao atributo existente (se for uma lista)
                if isinstance(getattr(self.grafo, chave), list):
                    getattr(self.grafo, chave).append(valor)
                else:
                    #  Converte o atributo existente em uma lista e adiciona o valor
                    setattr(self.grafo, chave, [getattr(self.grafo, chave), valor])
            else:
                #  Cria um novo atributo no nó com o valor
                setattr(self.grafo, chave, valor)

    def _extrair_features_do_no(self, node):
        """
        Extrai as features relevantes de um nó.
        """
        features = []
        # Exemplo: codificar o tipo de nó e a cor
        tipo_no = 0 if node['shape'] == 'rectangle' else 1 if node['shape'] == 'diamond' else 2
        features.append(tipo_no)

        # Codificar a cor (exemplo simples)
        if 'color' in node:
            cor = node['color']
            if cor == 'blue':
                features.append(0)
            elif cor == 'green':
                features.append(1)
            # ... adicionar outras cores
        return features

    def _obter_no_inicial(self, nodes):
        """
        Determina o nó inicial do processo.
        """
        # Busca pelo nó com o label "Compreender o problema"
        for node in nodes:
            if node['label'] == "Compreender o problema":
                return node['id']
        # Se não encontrar, retorna o primeiro nó
        return nodes[0]['id']

    def obter_no_atual(self):
        """
        Retorna as features do nó atual.
        """
        return self.grafo.x[self.no_atual]

    def _determinar_proximo_no(self, decisao, dados_processo):
        """
        Determina o próximo nó com base na decisão do usuário, 
        nas regras do processo, nas informações das normas e no grafo.
        """
        # Obter as arestas que saem do nó atual
        arestas_saindo = [
            (self.grafo.edge_index[0][i], self.grafo.edge_index[1][i])
            for i in range(self.grafo.edge_index.size(1))
            if self.grafo.edge_index[0][i] == self.no_atual
        ]

        # Filtrar as arestas pela decisão
        for source_index, target_index in arestas_saindo:
            source_id = list(self.node_id_to_index.keys())[list(self.node_id_to_index.values()).index(source_index)]
            target_id = list(self.node_id_to_index.keys())[list(self.node_id_to_index.values()).index(target_index)]
            aresta = next((aresta for aresta in self.grafo.json_data['edges'] if aresta['source'] == source_id and aresta['target'] == target_id), None)
            if aresta and aresta['label'] == decisao:
                # Obter o nó de destino
                proximo_no_id = aresta['target']

                # Lógica adicional com base em informações das normas e dados do processo (opcional)
                # if proximo_no_id == 'C' and dados_processo['valor_estimado'] > 100000:
                #     proximo_no_id = 'D'  # Exemplo de lógica adicional

                return proximo_no_id

        return None  # Retorna None se o próximo nó não for encontrado

    def avancar_no(self, decisao):
        """
        Avança para o próximo nó com base na decisão e nas informações extraídas.
        """
        proximo_no_id = self._determinar_proximo_no(decisao, self.dados_processo)

        if proximo_no_id:
            self.no_atual_id = proximo_no_id
            self.no_atual = self.node_id_to_index[self.no_atual_id]
        else:
            print(f"Erro: Próximo nó não encontrado para a decisão {decisao}")

    def obter_opcoes(self):
        """
        Retorna as opções disponíveis no nó atual, extraindo-as dinamicamente do grafo.
        """
        # 1. Obter as arestas que saem do nó atual
        arestas_saindo = [
            (self.grafo.edge_index[0][i], self.grafo.edge_index[1][i])
            for i in range(self.grafo.edge_index.size(1))
            if self.grafo.edge_index[0][i] == self.no_atual
        ]

        # 2. Extrair os labels das arestas
        opcoes = []
        for source_index, target_index in arestas_saindo:
            source_id = list(self.node_id_to_index.keys())[list(self.node_id_to_index.values()).index(source_index)]
            target_id = list(self.node_id_to_index.keys())[list(self.node_id_to_index.values()).index(target_index)]
            aresta = next((aresta for aresta in self.grafo.json_data['edges'] if aresta['source'] == source_id and aresta['target'] == target_id), None)
            if aresta:
                opcoes.append(aresta['label'])

        return opcoes

    def obter_formulario(self):
        """
        Retorna um objeto `Formulario` para o nó atual.
        """
        return Formulario(self.no_atual_id, self.obter_no_atual())

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired

class Formulario(FlaskForm):
    def __init__(self, no_id, no_features):
        super().__init__()
        self.no_id = no_id
        self.campos = []

        # Exemplo para o nó "Definir a demanda"
        if no_id == 'C':
            self.descricao = StringField('Descrição da demanda:', validators=[DataRequired()])
            self.prioridade = SelectField('Prioridade:', choices=[('alta', 'Alta'), ('media', 'Média'), ('baixa', 'Baixa')])
            self.campos.extend([self.descricao, self.prioridade])

            # Exemplo de uso de informações extraídas da norma:
            if no_features['valor_limite'] > 10000:
                self.justificativa = StringField('Justificativa:', validators=[DataRequired()])
                self.campos.append(self.justificativa)
        # ... (adicionar lógica para outros nós)
        self.submit = SubmitField('Avançar')
        self.campos.append(self.submit)

    def gerar_html(self):
        """
        Gera o HTML do formulário.
        """
        # Exemplo usando render_template_string (Flask)
        from flask import render_template_string

    def processar_dados(self, dados, gerenciador_processo):  # Adiciona o gerenciador_processo como argumento
        """
        Processa os dados enviados pelo usuário, validando e armazenando.
        """
        if self.validate_on_submit():  # Valida os dados do formulário
            for campo in self.campos:
                if campo.name != 'submit':  # Ignora o botão de submit
                    valor = campo.data

                    # Lógica de validação específica para cada campo (opcional)
                    if campo.name == 'descricao' and len(valor) < 10:
                        # Exemplo: valida se a descrição tem pelo menos 10 caracteres
                        campo.errors.append('A descrição deve ter pelo menos 10 caracteres.')
                        return False  # Interrompe o processamento se houver erro

                    # Armazena os dados no dicionário dados_processo do GerenciadorProcesso
                    gerenciador_processo.dados_processo[campo.name] = valor
            return True  # Retorna True se a validação foi bem-sucedida
        else:
            # Lidar com erros de validação (exibir mensagens de erro, etc.)
            for campo, erros in self.errors.items():
                for erro in erros:
                    print(f"Erro no campo '{campo}': {erro}")  # Exibe as mensagens de erro
            return False  # Retorna False se houver erros de validação

    def processar_formulario(self, dados):
        """
        Processa os dados do formulário e avança no grafo.
        """
        formulario = self.grafo.obter_formulario()
        if formulario.processar_dados(dados, self):
            decisao = formulario.obter_decisao()
            self.historico_decisoes.append((self.grafo.no_atual_id, decisao))
            self.grafo.adicionar_dados_no(self.grafo.no_atual_id, self.dados_processo)  # Adiciona os dados ao nó
            self.grafo.avancar_no(decisao)

    def obter_decisao(self):
        """
        Extrai a decisão tomada pelo usuário a partir dos dados do formulário,
        lendo as opções dinamicamente do JSON do processo.
        """
        # 1. Obter o nó atual do grafo
        no_atual = gerenciador_processo.grafo.grafo.nodes[self.no_id]  # Acessa o nó pelo ID

        # 2. Verificar o tipo de nó (decisão ou atividade)
        if no_atual['shape'] == 'diamond':  # Nó de decisão
            # 3. Obter as arestas que saem do nó atual
            arestas_saindo = [
                aresta for aresta in gerenciador_processo.grafo.grafo.edges 
                if aresta['source'] == self.no_id
            ]

            # 4. Extrair as opções (rótulos das arestas)
            opcoes = [aresta['label'] for aresta in arestas_saindo]

            # 5. Criar um mapeamento entre os campos do formulário e as opções
            #    (assumindo que a ordem dos campos corresponde à ordem das opções)
            mapeamento_campos_opcoes = {campo.name: opcao for campo, opcao in zip(self.campos, opcoes)}

            # 6. Obter a decisão com base no campo selecionado no formulário
            for campo in self.campos:
                if campo.data:  # Verifica se o campo foi selecionado
                    return mapeamento_campos_opcoes[campo.name]

        return None  # Retorna None se o nó não for de decisão ou nenhuma opção for selecionada

from collections import defaultdict

class GerenciadorProcesso:
    def __init__(self, grafo):
        self.grafo = grafo
        self.dados_processo = {}
        self.modelo_gnn = GNN(...)  # Carrega o modelo treinado
        self.historico_decisoes = []  # Lista para armazenar o histórico de decisões

    def iniciar_processo(self):
        """
        Inicializa o processo no nó inicial do grafo.
        """
        self.grafo.no_atual_id = self.grafo._obter_no_inicial(self.grafo.json_data['nodes'])
        self.grafo.no_atual = self.grafo.node_id_to_index[self.grafo.no_atual_id]

    def obter_proximo_formulario(self):
        """
        Retorna o HTML do próximo formulário.
        """
        formulario = self.grafo.obter_formulario()
        # Você pode usar as opções para gerar o formulário dinamicamente aqui
        return formulario.gerar_html()

    def processar_formulario(self, dados):
        """
        Processa os dados do formulário e avança no grafo.
        """
        formulario = self.grafo.obter_formulario()
        if formulario.processar_dados(dados, self):
            decisao = formulario.obter_decisao()
            self.historico_decisoes.append((self.grafo.no_atual_id, decisao))  # Armazena a decisão
            self.grafo.avancar_no(decisao)

    def finalizar_processo(self):
        """
        Encerra o processo e gera um relatório.
        """
        # Gera um relatório com base nos dados do processo e no histórico de decisões
        relatorio = self._gerar_relatorio()
        print(relatorio)

    def _gerar_relatorio(self):
        """
        Gera um relatório do processo.
        """
        relatorio = "Relatório do Processo de Aquisição:\n\n"
        relatorio += "Dados do Processo:\n"
        for chave, valor in self.dados_processo.items():
            relatorio += f"  - {chave}: {valor}\n"

        relatorio += "\nHistórico de Decisões:\n"
        for no_id, decisao in self.historico_decisoes:
            relatorio += f"  - Nó {no_id}: {decisao}\n"

        return relatorio

    def obter_recomendacao(self):
        """
        Utiliza um modelo de aprendizado de máquina para 
        recomendar a próxima ação.
        """
        features = self._extrair_features_para_recomendacao()
        recomendacao = self._obter_recomendacao_do_modelo(features)
        return recomendacao

    def _extrair_features_para_recomendacao(self):
        """
        Extrai features relevantes do nó atual e do histórico do processo 
        para serem usadas na recomendação.
        """
        features = []
        # Adicionar features do nó atual (e.g., tipo de nó, informações extraídas)
        no_atual = self.grafo.obter_no_atual()
        features.extend(no_atual.tolist())  # Converter tensor para lista

        # Adicionar features do histórico de decisões
        # Exemplo: codificar as decisões anteriores como one-hot encoding
        decisoes_anteriores = defaultdict(lambda: 0)
        for _, decisao in self.historico_decisoes:
            decisoes_anteriores[decisao] += 1
        # ... (converter decisões_anteriores para features numéricas)
        features.extend(...)

        return features

    def _obter_recomendacao_do_modelo(self, features):
        """
        Utiliza o modelo GNN treinado para gerar a recomendação.
        """
        features_tensor = torch.tensor(features, dtype=torch.float)
        with torch.no_grad():
            saida_modelo = self.modelo_gnn(features_tensor)
        # ... (interpretar a saída do modelo e gerar a recomendação) ...
        return recomendacao

    def gerar_documento(self, tipo_documento):
        """
        Gera o documento especificado utilizando os dados do grafo.
        """
        if tipo_documento == 'edital':
            # Lógica para gerar o edital com base nos dados do grafo
            # ...
            print("Edital gerado com sucesso!")
        elif tipo_documento == 'contrato':
            # Lógica para gerar o contrato com base nos dados do grafo
            # ...
            print("Contrato gerado com sucesso!")
        # ... (adicionar lógica para outros tipos de documentos)

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv  # Exemplo de GNN: GCN
from torch_geometric.data import DataLoader

# Definir a arquitetura do modelo GNN e treinar
class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

    # gerar dados sintéticos para testes
    def gerar_dados_sinteticos(num_grafos, num_nos, num_features):
        """
        Gera dados sintéticos para treinamento.
        """
        dados = []
        for _ in range(num_grafos):
            x = torch.randn(num_nos, num_features)  # Features aleatórias
            edge_index = torch.randint(0, num_nos, (2, num_nos * 2))  # Arestas aleatórias
            y = torch.randint(0, 2, (num_nos,))  # Rótulos aleatórios (0 ou 1)
            data = Data(x=x, edge_index=edge_index, y=y)
            dados.append(data)
        return dados

    # implementar o loop de treinamento
    def treinar(modelo, loader, criterion, optimizer, num_epochs):
        modelo.train()
        for epoch in range(num_epochs):
            for data in loader:
                optimizer.zero_grad()
                out = modelo(data.x, data.edge_index)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')


# Exemplo para Iniciar o processo
# 1. Carregar o JSON do processo
# with open('processo.json', 'r') as f:
#     json_data = json.load(f)

# # Criar o grafo
# grafo = Grafo()
# grafo.carregar_grafo(json_data)

# # Criar o gerenciador de processo
# gerenciador = GerenciadorProcesso(grafo)
# gerenciador.iniciar_processo()

# # ... (lógica para interagir com o usuário, exibir formulários, etc.) ...

# 2. Preparar os dados para treinamento
# Gerar 100 grafos sintéticos com 5 nós e 3 features cada
dados_treinamento = gerar_dados_sinteticos(num_grafos=100, num_nos=5, num_features=3)

# Exemplo de dados para um grafo
x = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=torch.float)  # Features dos nós (4 nós)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                           [1, 0, 2, 1, 3, 2]], dtype=torch.long)  # Arestas
y = torch.tensor([0, 1, 1, 0], dtype=torch.long)  # Rótulos (decisões)

# Criar o objeto Data
data = Data(x=x, edge_index=edge_index, y=y)
dados_treinamento.append(data)

# ... (adicionar mais grafos ao dados_treinamento) ...

# Criar DataLoader
loader = DataLoader(dados_treinamento, batch_size=32, shuffle=True)

# 3. Definir a função de perda e o otimizador
modelo = GNN(num_features=..., hidden_dim=..., num_classes=...)
criterion = torch.nn.CrossEntropyLoss()  # Exemplo de função de perda
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.01)

# 4. Treinar o modelo
treinar(modelo, loader, criterion, optimizer, num_epochs=100)