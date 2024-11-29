import os
import json
from tabnanny import verbose
import unicodedata
import networkx as nx
import matplotlib.pyplot as plt

from numpy import False_
from pyvis.network import Network
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

class Utils:
    @staticmethod
    def primeira_letra_maiuscula(texto):
        """
        Converte a primeira letra de uma string para maiúscula.
        """
        if not texto:
            return texto
        return texto[0].upper() + texto[1:]

class GrafoConhecimento:
    def __init__(self, dict_list, dados_demanda):
        self.dict_list = dict_list
        self.dados_demanda = dados_demanda
        self.grafo = nx.DiGraph()

    def info_subgrafo(self, subgrafo):
        # Adicionar aviso ao final da função
        num_nos = self.grafo.number_of_nodes()
        num_arestas = self.grafo.number_of_edges()

        # Contar nós de cada tipo
        tipos_nos = defaultdict(int)
        for _, dados in self.grafo.nodes(data=True):
            tipos_nos[dados['tipo']] += 1

        # Contar arestas de cada tipo
        tipos_arestas = defaultdict(int)
        for _, _, dados in self.grafo.edges(data=True):
            tipos_arestas[dados['relation']] += 1

        # Imprimir a mensagem com as informações adicionais
        print(f"\nSubgrafo de {subgrafo} criado com {num_nos} nós e {num_arestas} arestas.")
        print("  Nós por tipo:")
        for tipo, quantidade in tipos_nos.items():
            print(f"  - {tipo}: {quantidade}")
        print("  Arestas por tipo:")
        for tipo, quantidade in tipos_arestas.items():
            print(f"  - {tipo}: {quantidade}")

    def construir_grafo(self, verbose=False):
        """
        Constrói o grafo de conhecimento multicamadas.
        """
        # Criar índices
        self.indice_interesses = self.criar_indice_interesses()
        self.indice_produtos_desafios = self.criar_indice_produtos_desafios()

        # Criar subgrafos de oferta e demanda
        self.oferta = GrafoOferta(
            self.dict_list, 
            self.indice_interesses, 
            self.indice_produtos_desafios
        )
        self.demanda = GrafoDemanda(self.dados_demanda)

        # Construir o grafo de conhecimento por subgrafos em camadas
        self.demanda.subgrafo_demanda_pdi()
        self.oferta.subgrafo_oferta_pdi()
        self.oferta.adicionar_projetos()
        self.oferta.subgrafo_intencoes(verbose)

        # Integrar os subgrafos de oferta e demanda ao grafo principal
        self.integrar_subgrafos()

        # Adicionar nó CEIS e conectar aos blocos
        self.grafo.add_node('CEIS', tipo='instituicao')
        for no, dados in self.grafo.nodes(data=True):
            if dados.get('tipo') == 'bloco':
                self.grafo.add_edge('CEIS', no, relation='TEM_BLOCO')

        # Adicionar nó Fiocruz Ceará e conectar aos pesquisadores
        self.grafo.add_node('Fiocruz Ceará', tipo='instituicao')
        for no, dados in self.grafo.nodes(data=True):
            if dados.get('tipo') == 'pesquisador':
                self.grafo.add_edge('Fiocruz Ceará', no, relation='TEM_PESQUISADOR')

        # Gerar visualização do grafo com pyvis
        arquivo_html = self.visualizar()

        # Gerar visualização do grafo com matplotlib
        # arquivo_html = self.visualizar_grafo_matplotlib() 

    def integrar_subgrafos(self):
        """
        Integra os subgrafos de oferta e demanda ao grafo principal.
        """
        self.grafo.add_nodes_from(self.demanda.grafo.nodes(data=True))
        self.grafo.add_edges_from(self.demanda.grafo.edges(data=True))

        self.grafo.add_nodes_from(self.oferta.grafo.nodes(data=True))
        self.grafo.add_edges_from(self.oferta.grafo.edges(data=True))

        self.grafo.add_nodes_from(self.oferta.grafo.nodes(data=True))
        self.grafo.add_edges_from(self.oferta.grafo.edges(data=True))        

    def criar_indice_interesses(self):
        """
        Cria um índice para os interesses dos pesquisadores, 
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
        Cria um índice para os produtos e desafios do CEIS, 
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

    def agregar_camada_cnpq(self, camada):
        """
        Agrega a camada especificada no grafo.
        """
        if camada == 'area':
            # Agregar áreas e subáreas em um único nó "area_subarea"
            areas_subareas = defaultdict(list)
            for no, dados in self.grafo.nodes(data=True):
                if dados['tipo'] == 'subarea':
                    no = no.replace("Subárea: ", "")
                    for vizinho, attrs in self.grafo.adj[no].items():
                        if attrs['relation'] == 'CONTEM_SUBAREA':
                            areas_subareas[vizinho].append(no)

            for area, subareas in areas_subareas.items():
                novo_no = f"{area} - {', '.join(subareas)}"
                self.grafo.add_node(novo_no, tipo='area_subarea')
                for subarea in subareas:
                    for vizinho, attrs in self.grafo.adj[subarea].items():
                        if attrs['relation'] == 'ATUA_NA_AREA' and vizinho != area:
                            self.grafo.add_edge(novo_no, vizinho, relation='ATUA_NA_AREA_SUBAREA')
                    self.grafo.remove_node(subarea)
                self.grafo.remove_node(area)
        
        elif camada == 'outra_camada':
            # Acrescentar implementação para agregar outras camada, quando necessário no âmbito CNPQ
            # Para outras camadas de outras fontes usar outra função modular a ser chamada na integração
            pass

    def calcular_similaridade(self):
        """
        Calcula a similaridade entre as entidades do grafo.
        """
        caracteristicas = []
        nos = []
        for no, dados in self.grafo.nodes(data=True):
            if dados['tipo'] in ['pesquisador', 'area_subarea', 'produto', 'desafio', 'plataforma']:
                nos.append(no)
                caracteristicas_no = self.extrair_caracteristicas(no)
                caracteristicas.append(caracteristicas_no)

        similaridade = cosine_similarity(caracteristicas) # type: ignore

        for i in range(len(nos)):
            for j in range(i + 1, len(nos)):
                no1 = nos[i]
                no2 = nos[j]
                if similaridade[i, j] > 0:
                    self.grafo.add_edge(no1, no2, similaridade=similaridade[i, j])

    def extrair_caracteristicas(self, no):
        """
        Extrai as características relevantes de um nó.
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

    def identificar_lacunas(self):
        """
        Identifica lacunas de competências para a produção de produtos demandados.
        """
        lacunas = {}
        for produto in self.grafo.nodes(data=True):
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
        return lacunas

    def visualizar(self, nome_arquivo="grafo_conhecimento.html"):
        """
        Gera uma visualização interativa do grafo usando pyvis,
        ocupando toda a janela do browser, renderizando os nomes das 
        arestas e colorindo os nós de produtos em verde, plataformas 
        em cinza e desafios em laranja.
        """
        net = Network(notebook=True, 
                    directed=True, 
                    cdn_resources='in_line', 
                    height='1000')
        net.from_nx(self.grafo)

        # Definir cores e tamanho dos nós, e usar o nome como rótulo
        for node in net.nodes:
            try:
                if node['tipo'] == 'produto':
                    node['color'] = 'green'
                    node['size'] = node['size'] * 2  # Aumentar o tamanho do nó em 5 vezes
                elif node['tipo'] == 'pesquisador':
                    node['size'] = node['size'] * 2  # Aumentar o tamanho do nó em 5 vezes
                    node['color'] = 'orange'
                elif node['tipo'] == 'plataforma':
                    node['color'] = 'gray'
                elif node['tipo'] == 'desafio':
                    node['color'] = 'orange'
                elif node['tipo'] == 'bloco':
                    node['color'] = 'blue'
                    node['size'] = node['size'] * 5  # Aumentar o tamanho do nó em 10 vezes
                    node['label'] = node['nome']  # Usar o nome como rótulo
                elif node['tipo'] == 'area':
                    node['color'] = 'cyan'
                    node['size'] = node['size'] * 2  # Aumentar o tamanho do nó em 10 vezes
                elif node['tipo'] == 'instituicao':
                    node['color'] = 'yellow'
                    node['size'] = node['size'] * 10  # Aumentar o tamanho do nó em 10 vezes
            except KeyError:
                print(f"Erro: Nó '{node['id']}' não possui o atributo 'tipo'.")

        # Definir posições fixas para os nós CEIS e Fiocruz Ceará
        for node in net.nodes:
            if node['id'] == 'CEIS':
                node['x'] = 0
                node['y'] = 0
                node['fixed'] = True
            elif node['id'] == 'FiocruzCE':
                node['x'] = 300
                node['y'] = 0
                node['fixed'] = True

        # Adicionar title e estilo nas arestas
        for edge in net.edges:
            edge['title'] = edge['relation']
            if edge['relation'] == 'SIMILAR_A_PRODUTO_CEIS':
                edge['color'] = 'magenta'  # Cor magenta
                edge['width'] = 3  # Espessura da aresta

        # Personalizar a aparência (opcional)
        net.set_options("""
        const options = {
          "nodes": {
            "font": {
              "size": 20
            }
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true
              }
            },
            "color": {
              "inherit": true
            },
            "smooth": false
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "springLength": 100,
              "springConstant": 0.02
            },
            "minVelocity": 1.5
          }
        }
        """)

        print(f"\nGrafo de conhecimento gerado com sucesso:")
        return net.show(nome_arquivo) 

        print(f"\n HTML do Grafo de conhecimento gerado com sucesso:")
        return net.show(nome_arquivo)

    def visualizar_grafo_matplotlib(self):
        """
        Gera uma visualização do grafo usando networkx e matplotlib.
        """
        print(f"\n Gerando figura com matplotlib e netowrkx...")
        plt.figure(figsize=(80, 40))  # Aumentar o tamanho da figura (largura, altura) em polegadas

        # Definir layout e parâmetros de visualização
        pos = nx.spring_layout(self.grafo, k=0.3, iterations=50)  # Ajustar layout para melhor visualização
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

import os
import json
from pathlib import Path
from collections import defaultdict

import networkx as nx

class GrafoDemanda:
    def __init__(self, dados_demanda):
        self.grafo = nx.DiGraph()
        self.dados_demanda = dados_demanda
        self.base_repo_dir = self.find_repo_root()
        self.in_json = os.path.join(str(self.base_repo_dir), '_data', 'in_json')

    def find_repo_root(self, path='.', depth=10):
        ''' 
        Busca o arquivo .git e retorna string com a pasta raiz do repositório
        '''
        # Prevent infinite recursion by limiting depth
        if depth < 0:
            return None
        path = Path(path).absolute()
        if (path / '.git').is_dir():
            return path
        return self.find_repo_root(str(path.parent), depth-1)
    
    def info_subgrafo(self, subgrafo):
        """
        Imprime informações sobre o subgrafo, incluindo o número de nós, 
        arestas e a contagem de cada tipo de nó e aresta.

        Args:
            subgrafo (str): Nome do subgrafo.
        """
        # Adicionar aviso ao final da função
        num_nos = self.grafo.number_of_nodes()
        num_arestas = self.grafo.number_of_edges()

        # Contar nós de cada tipo
        tipos_nos = defaultdict(int)
        for _, dados in self.grafo.nodes(data=True):
            tipos_nos[dados['tipo']] += 1

        # Contar arestas de cada tipo
        tipos_arestas = defaultdict(int)
        for _, _, dados in self.grafo.edges(data=True):
            tipos_arestas[dados['relation']] += 1

        # Imprimir a mensagem com as informações adicionais
        print(f"\nSubgrafo de {subgrafo} criado com {num_nos} nós e {num_arestas} arestas.")
        print("  Nós por tipo:")
        for tipo, quantidade in tipos_nos.items():
            print(f"  - {tipo}: {quantidade}")
        print("  Arestas por tipo:")
        for tipo, quantidade in tipos_arestas.items():
            print(f"  - {tipo}: {quantidade}")

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
            self.grafo.add_node(bloco['id'], tipo='bloco', nome=bloco['nome'])
            # Conectar produtos ao bloco
            for produto in bloco['produtos']:  # Iterar sobre a lista de produtos
                self.grafo.add_node(produto['id'], tipo='produto', nome=produto['nome'])
                self.grafo.add_edge(bloco['id'], produto['id'], relation='CONTEM_PRODUTO')

            # Conectar desafios ao bloco e, em seguida, plataformas aos desafios
            for desafio in bloco['desafios']:  # Iterar sobre os desafios
                self.grafo.add_node(desafio['id'], tipo='desafio', nome=desafio['nome'])
                self.grafo.add_edge(bloco['id'], desafio['id'], relation='CONTEM_DESAFIO')

                for plataforma in desafio['plataformas']:  # Iterar sobre as plataformas
                    self.grafo.add_node(plataforma['id'], tipo='plataforma', nome=plataforma['nome'])
                    self.grafo.add_edge(desafio['id'], plataforma['id'], relation='REQUER_PLATAFORMA')
                    
        try:
            self.info_subgrafo('Demanda')
        except Exception as e:
            print(f"Não foi possível obter a informação do subgrafo")
            print(e)


class GrafoOferta:
    def __init__(self, dict_list, indice_interesses, indice_produtos_desafios):
        # self.grafo = nx.Graph()
        self.grafo = nx.DiGraph()  # Usar DiGraph para grafo direcionado
        self.dict_list = dict_list
        self.indice_interesses = indice_interesses
        self.indice_produtos_desafios = indice_produtos_desafios
        self.base_repo_dir = self.find_repo_root()
        self.in_json = os.path.join(str(self.base_repo_dir), '_data', 'in_json')
        self.qte_respostas_registradas = 0
        self.tipos_nos = defaultdict(int)
        self.tipos_arestas = defaultdict(int)
        self.model = SentenceTransformer('sentence-t5-base')

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
    

    def info_subgrafo(self, subgrafo):
        # Adicionar aviso ao final da função
        num_nos = self.grafo.number_of_nodes()
        num_arestas = self.grafo.number_of_edges()

        # Contar nós de cada tipo
        tipos_nos = defaultdict(int)
        for _, dados in self.grafo.nodes(data=True):
            tipos_nos[dados['tipo']] += 1

        # Contar arestas de cada tipo
        tipos_arestas = defaultdict(int)
        for _, _, dados in self.grafo.edges(data=True):
            tipos_arestas[dados['relation']] += 1

        # Imprimir a mensagem com as informações adicionais
        print(f"\nSubgrafo de {subgrafo} criado com {num_nos} nós e {num_arestas} arestas.")
        print("  Nós por tipo:")
        for tipo, quantidade in tipos_nos.items():
            print(f"  - {tipo}: {quantidade}")
        print("  Arestas por tipo:")
        for tipo, quantidade in tipos_arestas.items():
            print(f"  - {tipo}: {quantidade}")

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

    def calcular_similaridade_semantica(self, texto1, texto2):
        """
        Calcula a similaridade semântica entre dois textos.
        """
        embeddings1 = self.model.encode(texto1, convert_to_tensor=True)
        embeddings2 = self.model.encode(texto2, convert_to_tensor=True)
        cosine_sim = util.cos_sim(embeddings1, embeddings2)
        return cosine_sim.item()

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
            print(f"Aviso: Nome não informado. Usando ID Lattes genérico.")
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
        print(f"Aviso: Nome '{nome}' não encontrado no grafo.")
        return '9999999999999999'  # ID Lattes padrão se não encontrar


    def subgrafo_oferta_pdi(self):
        """
        Constrói o subgrafo de oferta com base nos dados dos pesquisadores.
        """
        if self.dict_list:
            for dicionario in self.dict_list:
                if isinstance(dicionario, dict):
                    id_lattes = dicionario.get('Identificação', {}).get('ID Lattes')
                    nome = dicionario.get('Identificação', {}).get('Nome')

                    # Dentro do método subgrafo_oferta_pdi
                    print(f"ID Lattes: {id_lattes}, Nome: {nome}")
                    self.grafo.add_node(id_lattes, tipo='pesquisador', nome=nome)
                    print(f"Nó criado: {id_lattes}, Atributos: {self.grafo.nodes[id_lattes]}")

                    if id_lattes and nome:
                        self.grafo.add_node(id_lattes, tipo='pesquisador', nome=nome)

                        areas = self.extrair_areas(dicionario.get('Áreas', {}))
                        for area in areas:
                            self.grafo.add_node(area, tipo='area')
                            self.grafo.add_edge(id_lattes, area, relation='ATUA_NA_AREA')

                            subareas = self.extrair_subareas(area)
                            for subarea in subareas:
                                self.grafo.add_node(subarea, tipo='subarea')
                                self.grafo.add_edge(area, subarea, relation='CONTEM_SUBAREA')
                else:
                    print(f"Erro com objeto dicionário: {type(dicionario)}")
            # Adicionar aviso ao final da função
            num_nos = self.grafo.number_of_nodes()
            num_arestas = self.grafo.number_of_edges()

            # Contar nós de cada tipo
            tipos_nos = defaultdict(int)
            for _, dados in self.grafo.nodes(data=True):
                tipos_nos[dados['tipo']] += 1

            # Contar arestas de cada tipo
            tipos_arestas = defaultdict(int)
            for _, _, dados in self.grafo.edges(data=True):
                tipos_arestas[dados['relation']] += 1

            # Imprimir a mensagem com as informações adicionais
            print(f"\nSubgrafo de oferta criado com {num_nos} nós e {num_arestas} arestas.")
            print("  Nós por tipo:")
            for tipo, quantidade in tipos_nos.items():
                print(f"  - {tipo}: {quantidade}")
            print("  Arestas por tipo:")
            for tipo, quantidade in tipos_arestas.items():
                print(f"  - {tipo}: {quantidade}")


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


    def subgrafo_intencoes(self, verbose=True):
        """
        Adiciona as intenções dos pesquisadores ao grafo,
        relacionando-as aos pesquisadores, às áreas de pesquisa
        e aos produtos e desafios do CEIS.
        """
        try:
            pathfilename = os.path.join(self.in_json, 'input_interesses_pesquisadores.json')
            with open(pathfilename, 'r') as f:  
                respostas_pesquisadores = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Erro ao ler o arquivo input_interesses_pesquisadores.json: {e}")
            return

        print(f"\nAssociando intenções dos pesquisadores da ICT com desafios do CEIS:")
        
        # Contadores
        num_arestas_interesse = 0
        
        respostas_nao_associadas = []

        for i, pesquisador in enumerate(respostas_pesquisadores):
            # Ignorar o primeiro dicionário (perguntas do questionário)
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
                            self.tipos_nos['competencia_declarada'] += 1

                            # Adicionar arestas COMPETENCIA_DECLARADA para id_lattes
                            if id_lattes:
                                self.grafo.add_edge(id_lattes, competencia_declarada, relation='COMPETENCIA_DECLARADA')
                                self.tipos_arestas['COMPETENCIA_DECLARADA'] += 1

                # --- Competências a Desenvolver ---
                competencias_desenvolver = pesquisador.get("competencias_desenvolver")
                if competencias_desenvolver and isinstance(competencias_desenvolver, list):
                    competencias_desenvolver = self.limpar_lista(competencias_desenvolver)
                    for competencia_desejada in competencias_desenvolver:
                        if isinstance(competencia_desejada, str):
                            # Criar nós de competencias_desenvolver
                            self.grafo.add_node(competencia_desejada, tipo='competencia_desejada')
                            self.tipos_nos['competencia_desejada'] += 1

                            # Adicionar arestas COMPETENCIA_DESEJADA para id_lattes
                            if id_lattes:
                                self.grafo.add_edge(id_lattes, competencia_desejada, relation='COMPETENCIA_DESEJADA')
                                self.tipos_arestas['COMPETENCIA_DESEJADA'] += 1

                # --- Intenções ---
                intencoes = []
                string_questoes = pesquisador.get("questoes_interesse")
                if string_questoes and isinstance(string_questoes, str):
                    lista_questoes = self.limpar_questoes(string_questoes)
                    intencoes.extend(lista_questoes) # adicionar elementos de um iterável (lista, tupla ou string) ao final da lista atual
                    for questao_pesquisa in lista_questoes:
                        if isinstance(questao_pesquisa, str):
                            # Adicionar cada questão de pesquisa como nó no grafo de conhecimento
                            self.grafo.add_node(questao_pesquisa, tipo='questao_pesquisa')
                        if self.grafo.has_node(id_lattes):
                            # Adicionar aresta ligando questão de pesquisa ao id_lattes no grafo de conhecimento
                            self.grafo.add_edge(id_lattes, questao_pesquisa, relation='TEM_INTERESSE_EM_DESAFIO')

                palavras_chave = pesquisador.get("palavras_chave")
                if palavras_chave and isinstance(palavras_chave, list):
                    palavras_chave = self.limpar_lista(palavras_chave)
                    for palavra in palavras_chave:
                        if isinstance(palavra, str):
                            intencoes.append(palavra.strip()) # adicionar um único elemento ao final da lista. 
                            # Obs.: Se usar append() com uma lista como argumento, adicionará a lista inteira como um único elemento, em vez de adicionar os elementos individuais.

                pretende_desenvolver = pesquisador.get("intencao_desenvolvimento")
                if pretende_desenvolver and isinstance(pretende_desenvolver, str):
                    intencoes.append(pretende_desenvolver.strip())

                ceis_interesse_desafios = pesquisador.get("ceis_interesse_desafios")
                if ceis_interesse_desafios and isinstance(ceis_interesse_desafios, str):
                    lista_desafios = [x.strip() for x in ceis_interesse_desafios.split(';')]
                    lista_desafios = self.limpar_lista(lista_desafios)
                    intencoes.extend(lista_desafios) # adicionar elementos de um iterável (lista, tupla ou string) ao final da lista atual

                    for desafio_ceis in lista_desafios:
                        if isinstance(desafio_ceis, str):
                            # Procurar o nó do desafio CEIS no grafo de conhecimento
                            if self.grafo.has_node(desafio_ceis):
                                # Adicionar aresta ligando desafio ao id_lattes no grafo de conhecimento
                                self.grafo.add_edge(id_lattes, desafio_ceis, relation='TEM_INTERESSE_EM_DESAFIO')

                ceis_interesse_produtos_emergencias = pesquisador.get("ceis_interesse_produtos_emergencias")
                if ceis_interesse_produtos_emergencias and isinstance(ceis_interesse_produtos_emergencias, list):
                    for produto_emergencial in ceis_interesse_produtos_emergencias:
                        if isinstance(produto_emergencial, str):
                            intencoes.append(produto_emergencial.strip())

                            # Procurar o nó do produto CEIS no grafo de conhecimento
                            for no, dados in self.grafo.nodes(data=True):
                                if dados.get('nome') == produto_emergencial:
                                    self.grafo.add_edge(id_lattes, no, relation='TEM_INTERESSE_EM_PRODUTO')
                                    break  # Interromper o loop após encontrar o nó

                ceis_interesse_produtos_agravos = pesquisador.get("ceis_interesse_produtos_agravos")
                if ceis_interesse_produtos_agravos and isinstance(ceis_interesse_produtos_agravos, list):
                    for produto in ceis_interesse_produtos_agravos:
                        if isinstance(produto, str):
                            intencoes.append(produto.strip())

                if verbose:
                    print(f"    questoes_pesquisa: {lista_questoes}")
                    print(f"       palavras_chave: {palavras_chave}")
                    print(f"      desenvolvimento: {pretende_desenvolver}")
                    print(f"      comp_declaradas: {competencias_presentes}")
                    print(f"       comp_desejadas: {competencias_desenvolver}")
                    print(f"        ceis_desafios: {lista_desafios}")
                    print(f"    ceis_prod_emergen: {ceis_interesse_produtos_emergencias}")
                    print(f"    ceis_prod_agravos: {ceis_interesse_produtos_agravos}")

                if verbose:
                    print(f"    Objeto de intenções tipo: {type(intencoes)} com {len(intencoes)} instancias")

                # --- Adicionar nós de intenções ao grafo de conhecimento ---
                if id_lattes and intencoes:
                    # Adicionar nó de intenção e contar
                    self.grafo.add_node(str(intencoes), tipo='intencao')  # Converter lista para string
                    self.tipos_nos['intencao'] += 1

                    # Adicionar aresta entre pesquisador e intenção e contar
                    self.grafo.add_edge(id_lattes, str(intencoes), relation='POSSUI_INTENCAO')
                    self.tipos_arestas['POSSUI_INTENCAO'] += 1

                    # Adicionar arestas para produtos e desafios do CEIS
                    self.adicionar_interesses_declarados_ceis(intencoes, 
                                                              pesquisador)

                    num_arestas_interesse += 1

                    # Adicionar nós de competências e contar
                    self.adicionar_competencias(id_lattes, nome, pesquisador, self.tipos_nos)


            except Exception as e:
                print(f"    Erro ao processar pesquisador {i}: {e}")

        try:
            self.info_subgrafo('Intenções em Pesquisa')
        except Exception as e:
            print(f"\nNão foi possível obter a informação do subgrafo 'Intenções em Pesquisa'")
            print(e)


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
            'As principais competências científicas e tecnológicas do grupo de pesquisa em que atuo, que podem contribuir para a implementação da Estratégia Nacional de Desenvolvimento do Complexo Econômico-Industrial da Saúde (CEIS), incluem:', 
            'Competências científicas: '
        ]
        return [item for item in lista if item not in ignorar]

    def adicionar_interesses_declarados_ceis(self, intencoes, pesquisador):
        """
        Adiciona arestas entre as intenções dos pesquisadores e os 
        produtos e desafios do CEIS, por escolha ou similaridade, 
        considerando a normalização dos nomes e a estrutura de listas.

        Args:
            intencoes (list): Lista de strings com as intenções do pesquisador.
            pesquisador (dict): Dicionário com os dados do pesquisador.
            respostas_nao_associadas (list): Lista para armazenar as respostas que não puderam ser associadas.
        """

        # 1. Obter os IDs dos produtos e desafios de interesse
        resposta_emergenciais = pesquisador.get('ceis_interesse_produtos_emergencias', [])
        produtos_emergenciais = [x.split(";") for x in resposta_emergenciais]
        resposta_agravos = pesquisador.get('ceis_interesse_produtos_agravos', [])
        produtos_agravos = [x.split(";") for x in resposta_agravos]

        # Achatar a lista produtos_interesse
        produtos_interesse = [produto for sublista in produtos_emergenciais + produtos_agravos for produto in sublista]
        
        # Limpar a lista de valores vazios e indevidos e aplicar a função primeira_letra_maiuscula
        produtos_interesse = self.limpar_lista(produtos_interesse)
        produtos_interesse = list(map(self.primeira_letra_maiuscula, produtos_interesse))
        if verbose:
            print(produtos_interesse)

        desafios_interesse = pesquisador.get('ceis_interesse_desafios', "").split(';')
        desafios_interesse = self.limpar_lista(desafios_interesse)
        desafios_interesse = list(map(self.primeira_letra_maiuscula, desafios_interesse))

        # 2. Criar arestas TEM_INTERESSE para os produtos e desafios de interesse
        for intencao in intencoes:
            for produto_id in produtos_interesse:
                # Normalizar o nome do produto para minúsculas e sem acentos
                produto_normalizado = unicodedata.normalize('NFKD', produto_id).encode('ASCII', 'ignore').decode('ASCII').lower()

                for no, dados in self.grafo.nodes(data=True):
                    # Normalizar o nome do nó para minúsculas e sem acentos
                    nome_no_normalizado = unicodedata.normalize('NFKD', dados.get('nome', '')).encode('ASCII', 'ignore').decode('ASCII').lower()

                    if nome_no_normalizado == produto_normalizado:
                        self.grafo.add_edge(no, intencao, relation='TEM_INTERESSE_EM_PRODUTO')  # Corrigido: ordem dos argumentos invertida
                        self.tipos_arestas['TEM_INTERESSE_EM_PRODUTO'] += 1
                        break

            for desafio_id in desafios_interesse:
                # Normalizar o nome do desafio para minúsculas e sem acentos
                desafio_normalizado = unicodedata.normalize('NFKD', desafio_id).encode('ASCII', 'ignore').decode('ASCII').lower()

                for no, dados in self.grafo.nodes(data=True):
                    # Normalizar o nome do nó para minúsculas e sem acentos
                    nome_no_normalizado = unicodedata.normalize('NFKD', dados.get('nome', '')).encode('ASCII', 'ignore').decode('ASCII').lower()

                    if nome_no_normalizado == desafio_normalizado:
                        self.grafo.add_edge(no, intencao, relation='TEM_INTERESSE_EM_DESAFIO')
                        self.tipos_arestas['TEM_INTERESSE_EM_DESAFIO'] += 1
                        break  # Interromper o loop após encontrar o nó

            # # 3. Criar arestas SIMILAR por aproximação semântica
            # self.adicionar_interesses_por_similaridade(intencao, pesquisador)

        # Imprimir a quantidade de relações por tipo criadas com sucesso
        print(f"\nRelações criadas com sucesso:")
        for relation, count in self.tipos_arestas.items():
            print(f"  - {relation}: {count}")

    def adicionar_interesses_por_similaridade(self, intencoes, pesquisador):
        """
        Adiciona arestas SIMILAR entre as intenções dos pesquisadores 
        e as áreas de pesquisa, produtos e desafios do CEIS, 
        por aproximação semântica.
        """
        threshold_similaridade = 0.7  # Defina o limiar de similaridade
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('sentence-t5-base')

        # 1. Obter o texto das intenções
        textos_intencoes = []
        for intencao in intencoes:
            if isinstance(intencao, str):
                textos_intencoes.append(intencao)

        # 2. Obter o texto das áreas de pesquisa, produtos e desafios
        textos_areas_produtos_desafios = []
        for no, dados in self.grafo.nodes(data=True):
            if dados.get('tipo') in ['area', 'produto', 'desafio']:
                textos_areas_produtos_desafios.append(dados.get('nome', ''))

        # 3. Calcular a similaridade de cosseno para cada intenção
        for intencao in textos_intencoes:
            for texto_area_produto_desafio in textos_areas_produtos_desafios:
                similaridade = self.calcular_similaridade_semantica(model, intencao, texto_area_produto_desafio)
                if similaridade >= threshold_similaridade:
                    # Encontrar o nó correspondente ao texto_area_produto_desafio
                    for no, dados in self.grafo.nodes(data=True):
                        if dados.get('nome') == texto_area_produto_desafio:
                            self.grafo.add_edge(no, intencao, relation='SIMILAR_A_PRODUTO_CEIS', similaridade=similaridade)
                            self.tipos_arestas['SIMILAR_A_PRODUTO_CEIS'] += 1
                            break  # Interromper o loop após encontrar o nó

    def adicionar_competencias(self, id_lattes, nome, pesquisador, tipos_nos):
        """
        Adiciona camada de nós de competências ao grafo, relacionando-as aos pesquisadores.
        Baseada em dados das respostas dos pesquisadores aos levantamentos e questionários
        """
        competencias_possuidas = pesquisador.get('competencias_possuidas', [])
        competencias_desenvolver = pesquisador.get('competencias_desenvolver', [])

        for competencias in competencias_possuidas:
            # Criar um nó para a competência, se ele ainda não existir
            if not self.grafo.has_node(competencias):
                self.grafo.add_node(competencias, tipo='competencia_possuida')

            # Criar uma aresta entre o pesquisador e a competência
            if id_lattes:
                self.grafo.add_edge(id_lattes, competencias, relation='POSSUI_COMPETENCIA')
                tipos_nos['competencia_possuida'] += 1

        for competencias in competencias_desenvolver:
            # Criar um nó para a competência, se ele ainda não existir
            if not self.grafo.has_node(competencias):
                self.grafo.add_node(competencias, tipo='competencia_desenvolver')

            # Criar uma aresta entre o pesquisador e a competência
            if id_lattes:
                self.grafo.add_edge(id_lattes, competencias, relation='DESEJA_DESENVOLVER_COMPETENCIA')
                tipos_nos['competencia_desenvolver'] += 1