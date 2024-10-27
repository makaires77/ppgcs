import os
import json
import traceback
import pandas as pd
from git import Repo
from torch_geometric.data import Data

class GraphKnowledge:
    def __init__(self, json_files):
        self.json_files = json_files
        self.graph_data = None

    def info_dataset():
        # 1. Recuperar dados pré-processados
        # Informar caminho para arquivo CSV usando raiz do repositório Git como referência
        repo = Repo(search_parent_directories=True)
        root_folder = repo.working_tree_dir
        xlsx_folder = os.path.join(root_folder, '_data', 'in_xls')
        json_folder = os.path.join(root_folder, '_data', 'out_json')

        # Carregar os dados dos arquivos JSON
        with open(os.path.join(json_folder,'input_interesses_pesquisadores.json'), 'r') as f:
            interesses_data = json.load(f)
        with open(os.path.join(json_folder,'input_curriculos.json'), 'r') as f:
            curriculos_data = json.load(f)
        with open(os.path.join(json_folder,'matriz_ceis.json'), 'r') as f:
            matriz_ceis_data = json.load(f)
        with open(os.path.join(json_folder,'input_process_biologics.json'), 'r') as f:
            relacoes_biologicos = json.load(f)
        with open(os.path.join(json_folder,'input_process_smallmolecules.json'), 'r') as f:
            relacoes_pequenas_moleculas = json.load(f)

        print("Síntese dos dados para criar o Grafo de Conhecimento:\n")
        print("  Relacionamentos da Cadeia de Agregação de Valor em Produtos por tipo:")
        print(f"       Biológicos: {len(relacoes_biologicos):2} chaves: {relacoes_biologicos.keys()}")
        print(f"    Peq.Moléculas: {len(relacoes_pequenas_moleculas):2} chaves: {relacoes_pequenas_moleculas.keys()}")
        print("\n  Dados das entidades em análise:")
        print(f"      Matriz_CEIS: {len(matriz_ceis_data.get('blocos')):2} blocos, contendo as chaves: {list(matriz_ceis_data.get('blocos')[0].keys())}")
        print(f"       Currículos: {len(curriculos_data):2} currículos, com {len(curriculos_data[0])} chaves em cada currículo")
        print(f"       Interesses: {len(interesses_data):2} questionários, com {len(interesses_data[0])} respostas de cada pesquisador")
        print(f"\n  Dados sobre interesses de cada pesquisador:")
        for i in list(interesses_data[0].keys()):
            print(f"     {i}")
        print(f"\n  Dados sobre cada pesquisador:")
        for i in list(curriculos_data[0].keys()):
            print(f"     {i}")    

        print('\nLista de produtos por Bloco da Matriz CEIS:')
        for n,b in enumerate(matriz_ceis_data.get('blocos')):
            print(f"  Bloco: {b.get('titulo')}")
            for p in b.get('produtos'):
                print(f"    {p.get('nome')}")
            print()

    def generate_interesses_json(excel_file, json_file):
        """
        Lê o arquivo Excel 'levantamento_interesse.xlsx' e gera o arquivo JSON 'interesses_pesquisadores.json'.

        Args:
            excel_file (str): Caminho para o arquivo Excel.
            json_file (str): Caminho para o arquivo JSON de saída.
        """
        try:
            # Nomes das colunas
            colunas = [
                "id_levantamento", "hora_inicio", "hora_conclusao", "email", "nome_vazio", "ultima_modificacao", "consentimento_livre", "questoes_interesse", "palavras_chave",
                "competencias_possuidas", "competencias_desenvolver", "intencao_desenvolvimento",
                "trl_diagnosticos", "trl_pesquisa", "trl_terapias", "trl_servicos", "trl_social",
                "trl_digital", "ceis_conhecimento_blocos", "ceis_conhecimento_desafios",
                "ceis_conhecimento_plataformas", "ceis_conhecimento_produtos", "ceis_interesse_desafios",
                "ceis_interesse_produtos_emergencias", "ceis_interesse_produtos_agravos",
                "ceis_interesse_produtos_sugeridos", "tempo_percentual_buscas", "tempo_percentual_analises",
                "tempo_percentual_debates", "tempo_percentual_redacao", "tempo_percentual_reunioes",
                "tempo_percentual_comunicacao", "nivel_satisfacao", "nome_pesquisador"
            ]

            # Lê o arquivo Excel sem cabeçalho
            df = pd.read_excel(excel_file, header=None, dtype=str)

            # Define os nomes das colunas
            df.columns = colunas

            # Preenche os valores NaN com strings vazias
            df.fillna('', inplace=True)
            
            # Converte os dados para uma lista de dicionários
            interesses = []
            for index, row in df.iterrows():
                interesse = {
                    "id_levantamento": row['id_levantamento'],
                    "hora_inicio": row['hora_inicio'],
                    "hora_conclusao": row['hora_conclusao'],
                    "email": row['email'],
                    "nome_vazio": row['nome_vazio'],
                    "ultima_modificacao": row['ultima_modificacao'],
                    "consentimento_livre": row['consentimento_livre'],
                    "questoes_interesse": row['questoes_interesse'],
                    "palavras_chave": row['palavras_chave'].split('\n'),
                    "competencias_possuidas": row['competencias_possuidas'].split('\n'),
                    "competencias_desenvolver": row['competencias_desenvolver'].split('\n'),
                    "intencao_desenvolvimento": row['intencao_desenvolvimento'],
                    "trl_diagnosticos": row['trl_diagnosticos'],
                    "trl_pesquisa": row['trl_pesquisa'],
                    "trl_terapias": row['trl_terapias'],
                    "trl_servicos": row['trl_servicos'],
                    "trl_social": row['trl_social'],
                    "trl_digital": row['trl_digital'],
                    "ceis_conhecimento_blocos": row['ceis_conhecimento_blocos'],
                    "ceis_conhecimento_desafios": row['ceis_conhecimento_desafios'],
                    "ceis_conhecimento_plataformas": row['ceis_conhecimento_plataformas'],
                    "ceis_conhecimento_produtos": row['ceis_conhecimento_produtos'],
                    "ceis_interesse_desafios": row['ceis_interesse_desafios'],
                    "ceis_interesse_produtos_emergencias": row['ceis_interesse_produtos_emergencias'].split('\n'),
                    "ceis_interesse_produtos_agravos": row['ceis_interesse_produtos_agravos'].split('\n'),
                    "ceis_interesse_produtos_sugeridos": row['ceis_interesse_produtos_sugeridos'].split('\n'),
                    "tempo_percentual_buscas": row['tempo_percentual_buscas'],
                    "tempo_percentual_analises": row['tempo_percentual_analises'],
                    "tempo_percentual_debates": row['tempo_percentual_debates'],
                    "tempo_percentual_redacao": row['tempo_percentual_redacao'],
                    "tempo_percentual_reunioes": row['tempo_percentual_reunioes'],
                    "tempo_percentual_comunicacao": row['tempo_percentual_comunicacao'],
                    "nivel_satisfacao": row['nivel_satisfacao'],
                    "nome_pesquisador": row['nome_pesquisador']
                }
                interesses.append(interesse)

            # Salva os dados em um arquivo JSON
            with open(json_file, 'w') as f:
                json.dump(interesses, f, indent=4)

            print(f"Arquivo '{json_file}' gerado com sucesso!")

        except FileNotFoundError:
            print(f"Erro: Arquivo Excel '{excel_file}' não encontrado.")
            traceback.print_exc()  # Imprime o traceback
        except Exception as e:
            print(f"Erro ao processar o arquivo Excel: {e}")
            print(f"Erro na linha: {row}")
            traceback.print_exc()  # Imprime o traceback


    def generate_graph(self):
        nodes = []
        edges = []
        node_idx = 0
        node_id_map = {}

        # Adicionar nós dos produtos do CEIS
        with open('produtos_ceis.json', 'r') as f:
            data = json.load(f)
            for produto_data in data:
                features = self.extract_features_produtos(produto_data)
                nodes.append(features)
                node_id_map[produto_data['id']] = node_idx
                node_idx += 1

        # Criar arestas entre competências e produtos (exemplo)
        for i, node_comp in enumerate(nodes):
            if i < len(competencias_nodes):  # Verifica se é um nó de competência
                for j, node_prod in enumerate(nodes):
                    if j >= len(competencias_nodes):  # Verifica se é um nó de produto
                        similarity = self.calculate_similarity(node_comp, node_prod)  # Calcula a similaridade
                        if similarity > threshold:  # Se a similaridade for maior que um limiar
                            edges.append([i, j])

        for json_file in self.json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

                # Extrair nós e arestas do JSON
                for node_data in data['nodes']:
                    node_id = node_data['id']
                    features = self.extract_features(node_data, json_file)
                    nodes.append(features)
                    node_id_map[node_id] = node_idx
                    node_idx += 1

                for edge_data in data['edges']:
                    source = node_id_map[edge_data['from']]
                    target = node_id_map[edge_data['to']]
                    edges.append([source, target])

        # Criar o objeto Data do PyTorch Geometric
        x = torch.tensor(nodes, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.graph_data = Data(x=x, edge_index=edge_index)

        return self.graph_data

    def extract_features(self, node_data, json_file):
        """
            Extrai as features dos currículos: idiomas, formações, atuações profissionais, produções, projetos, bancas e orientações.
        """
        features = []
        if 'input_curriculos.json' in json_file:
            # Extrair features do nó de currículo
            features.append(len(node_data.get('Idiomas', [])))  # Número de idiomas
            features.append(len(node_data.get('Formação', {}).get('Acadêmica', [])))  # Número de formações acadêmicas
            features.append(len(node_data.get('Formação', {}).get('Complementar', [])))  # Número de formações complementares
            features.append(len(node_data.get('Atuação Profissional', [])))  # Número de atuações profissionais
            features.append(len(node_data.get('Linhas de Pesquisa', [])))  # Número de linhas de pesquisa
            features.append(len(node_data.get('Áreas', {})))  # Número de áreas de atuação
            features.append(len(node_data.get('Produções', {}).get('Artigos completos publicados em periódicos', [])))  # Número de artigos publicados
            features.append(len(node_data.get('Produções', {}).get('Resumos publicados em anais de congressos', {})))  # Número de resumos publicados
            features.append(len(node_data.get('Produções', {}).get('Apresentações de Trabalho', {})))  # Número de apresentações de trabalho
            features.append(len(node_data.get('Produções', {}).get('Outras produções bibliográficas', {})))  # Número de outras produções bibliográficas
            features.append(len(node_data.get('Produções', {}).get('Entrevistas, mesas redondas, programas e comentários na mídia', {})))  # Número de entrevistas e participações na mídia
            features.append(len(node_data.get('Produções', {}).get('Demais tipos de produção técnica', {})))  # Número de demais tipos de produção técnica
            features.append(len(node_data.get('ProjetosPesquisa', [])))  # Número de projetos de pesquisa
            features.append(len(node_data.get('ProjetosExtensão', [])))  # Número de projetos de extensão
            features.append(len(node_data.get('ProjetosDesenvolvimento', [])))  # Número de projetos de desenvolvimento
            features.append(len(node_data.get('ProjetosOutros', [])))  # Número de outros projetos
            features.append(len(node_data.get('Patentes e registros', {})))  # Número de patentes e registros
            features.append(len(node_data.get('Bancas', {}).get('Participação em bancas de trabalhos de conclusão', {})))  # Número de participações em bancas de trabalhos de conclusão
            features.append(len(node_data.get('Bancas', {}).get('Participação em bancas de comissões julgadoras', {})))  # Número de participações em bancas de comissões julgadoras
            features.append(len(node_data.get('Orientações', [])))  # Número de orientações
            features.append(len(node_data.get('JCR2', [])))  # Número de JCR2

        # Adicionar features de interesse dos pesquisadores
        elif 'interesses_pesquisadores.json' in json_file:
            with open('interesses_pesquisadores.json', 'r') as f:
                interesses_data = json.load(f)
                # ... extrair features dos interesses e adicionar à lista features ...

        elif 'input_process' in json_file:
            # Extrair features do nó de processo
            # ... features relevantes de processos ... (ex: número de etapas, tipo de processo, etc.)
            pass

        elif 'input_gestao.json' in json_file:
            # Extrair features do nó de gestão
            # ... features relevantes de gestão ...
            pass

        return features

    def extract_features_produtos(self, produto_data):
        features = []
        # ... extrair features relevantes dos produtos (bloco, produto, desafios) ...
        return features

    def calculate_similarity(self, node_comp, node_prod):
        # ... calcular a similaridade semântica entre as competências e os desafios dos produtos ...
        pass
