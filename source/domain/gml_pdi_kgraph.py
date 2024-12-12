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
        Converter a primeira letra de uma string para maiúscula.
        """
        if not texto:
            return texto
        return texto[0].upper() + texto[1:]

class GrafoConhecimento:
    def __init__(self, dict_list, dados_demanda):
        self.dict_list = dict_list
        self.dados_demanda = dados_demanda
        self.grafo = nx.DiGraph()


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

    def ajustar_posicoes_nos_processo(self):
        """
        Ajusta as posições dos nós que começam com letras de A a H seguidas de underscore.
        Posiciona os nós em uma linha vertical com espaçamento de 1000 unidades,
        separando entre biológicos (x=1000) e pequenas moléculas (x=-1000).
        """
        # Dicionário para mapear letras às posições y
        posicoes_y = {
            'A': 4000,
            'B': 3000,
            'C': 2000,
            'D': 1000,
            'E': 0,
            'F': -1000,
            'G': -2000,
            'H': -3000
        }
        
        # Iterar sobre todos os nós do grafo
        for node_id, node_data in self.grafo.nodes(data=True):
            # Verificar se o ID do nó segue o padrão (letra maiúscula seguida de underscore)
            if isinstance(node_id, str) and len(node_id) >= 2:
                # Remover sufixos para verificar a letra inicial
                base_id = node_id.replace('_bio', '').replace('_sm', '')
                primeira_letra = base_id[0]
                
                if primeira_letra in posicoes_y and base_id[1] == '_':
                    # Determinar a posição x baseada no tipo do nó
                    if node_data.get('tipo') == 'processo_biologico':
                        # Definir posição fixa para o nó de tipo de produto
                        x_pos = 1000
                        
                        # Conectar nós das fases do processo ao nó do tipo de produto
                        self.grafo.add_edge('BIOLOGICOS', node_id, relation='TEM_FASE')

                    elif node_data.get('tipo') == 'processo_smallmolecule':
                        # Definir posição fixa para o nó de tipo de produto
                        x_pos = -1000

                        # Conectar nós das fases do processo ao nó do tipo de produto
                        self.grafo.add_edge('SMALLMOLECULE', node_id, relation='TEM_FASE')

                    else:
                        continue  # Pular nós que não são de processo
                    
                    # Atualizar atributos do nó
                    self.grafo.nodes[node_id].update({
                        'x': x_pos,
                        'y': posicoes_y[primeira_letra],
                        'physics': True, # Caso optar por False as fases ficam ordenadas na vertical
                        'fixed': False,
                        'size': 50
                    })




    ## PIPELINE DE CONSTRUÇÃO DO GRAFO DE CONHECIMENTO
    def construir_grafo(self, verbose=True):
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
                            label='ICT Competencies',)
        for no, dados in self.grafo.nodes(data=True):
            if dados.get('tipo') == 'pesquisador':
                self.grafo.add_edge('ICT', no, relation='TEM_PESQUISADOR')

        # Ajustar posições dos nós de processo
        self.ajustar_posicoes_nos_processo()

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
        """
        Calcular a similaridade entre as entidades do grafo.
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


    def identificar_lacunas(self):
        """
        Identificar lacunas de competências para a produção de produtos demandados.
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
        net = Network( 
            height='1000',
            bgcolor='#ffffff',
            font_color='#000000',
            directed=True
        )
        
        net.from_nx(self.grafo)
        
        # Configurar nós
        for node in net.nodes:
            big_size = 180
            try:
                if node['id'] == 'CEIS':
                    node['x'] = 4000
                    node['y'] = 0
                    node['physics'] = False
                    node['fixed'] = {'x': True, 'y': True}
                    node['font'] = {'size': big_size}
                elif node['id'] == 'ICT':
                    node['x'] = -4000  # Distância horizontal entre nós principais
                    node['y'] = 0
                    node['physics'] = False
                    node['fixed'] = {'x': True, 'y': True}
                    node['font'] = {'size': big_size}
                elif node['id'] == 'BIOLOGICOS':
                    node['font'] = {'size': big_size}
                elif node['id'] == 'SMALLMOLECULE':
                    node['font'] = {'size': big_size}

                if node['tipo'] == 'produto':
                    node['color'] = '#90EE90'  # verde claro
                    node['size'] = 20
                elif node['tipo'] == 'pesquisador':
                    node['color'] = '#FFA500'  # laranja
                    node['size'] = 20
                elif node['tipo'] == 'plataforma':
                    node['color'] = '#A9A9A9'  # cinza
                    node['size'] = 15
                elif node['tipo'] == 'desafio':
                    node['color'] = '#FFB6C1'  # rosa claro
                    node['size'] = 15
                elif node['tipo'] == 'bloco':
                    node['color'] = '#87CEEB'  # azul claro
                    node['size'] = 25
                elif node['tipo'] == 'instituicao':
                    node['color'] = '#FFD700'  # dourado
                    node['size'] = 30
                if node['id'] == 'CEIS':
                    node['size'] = 50
                    node['color'] = '#FFD500'                    
                elif node['id'] == 'ICT':
                    node['size'] = 50
                    node['color'] = '#FFD900'                    

            except KeyError:
                continue
        
        # Configurar arestas
        for edge in net.edges:
            edge['width'] = 1
            if 'relation' in edge:
                edge['title'] = edge['relation']

        # Configurações de física otimizadas com avoidOverlap
        net.set_options("""
        {
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -2000,
                    "centralGravity": 0.25,
                    "springLength": 200,
                    "springConstant": 0.15,
                    "damping": 0.09
                },
                "minVelocity": 0.75,
                "solver": "barnesHut",
                "avoidOverlap": 1
            }
        }
        """)

        # # Configurações de física mais leves
        # net.set_options("""
        # {
        #     "physics": {
        #         "barnesHut": {
        #             "gravitationalConstant": -2000,
        #             "centralGravity": 0.3,
        #             "springLength": 95,
        #             "springConstant": 0.04,
        #             "damping": 0.09
        #         },
        #         "minVelocity": 0.75
        #     }
        # }
        # """)
        
        try:
            net.write_html(nome_arquivo)
            print(f"\nArquivo HTML gerado com sucesso: {nome_arquivo}")
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
        
        # Adicionar nós principais dos processos
        self.grafo.add_node(
            'BIOLOGICOS',
            tipo='processo_principal',
            label='Bilogical Products Processes',
            color='#4169E1',
            size=75,
            x=0,
            y=-2000,
            physics=False,
            fixed=True
        )
        
        self.grafo.add_node(
            'SMALLMOLECULE',
            tipo='processo_principal',
            label='SmallMolecule Products Processes',
            color='#8B008B',
            size=50,
            x=0,
            y=2000,
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
                color='#87CEEB',
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
                color='#DDA0DD',
                size=20,
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
        self.dict_list = dict_list
        self.indice_interesses = indice_interesses
        self.indice_produtos_desafios = indice_produtos_desafios
        self.base_repo_dir = self.find_repo_root()
        self.in_json = os.path.join(str(self.base_repo_dir), '_data', 'in_json')
        self.qte_respostas_registradas = 0
        self.tipos_nos = defaultdict(int)
        self.tipos_arestas = defaultdict(int)
        self.model = SentenceTransformer('sentence-t5-base')
        self.grafo_conhecimento = grafo_conhecimento  # Referência para a classe pai

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

        print(f"\nPopulando subgrafo de oferta com as intenções dos pesquisadores da ICT_Competencies:")
        
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

                    # # Criar aresta do id_lattes para interesse em desafio do CEIS
                    # for desafio in lista_desafios:
                    #     if isinstance(desafio, str):
                    #         self.grafo.add_node(desafio, tipo='ceis_desafio')
                    #         if id_lattes:
                    #             self.grafo.add_edge(id_lattes, desafio, relation='INTERESSA_DESAFIO')

                ceis_interesse_produtos_emergencias = pesquisador.get("ceis_interesse_produtos_emergencias")
                if ceis_interesse_produtos_emergencias and isinstance(ceis_interesse_produtos_emergencias, list):
                    for produto_emergencial in ceis_interesse_produtos_emergencias:
                        if isinstance(produto_emergencial, str):
                            if ";" in produto_emergencial:
                                lista_produtos = produto_emergencial.split(';')
                                for produto in lista_produtos:
                                    if produto != '':
                                        intencoes["produtos_emergenciais"].append(produto.strip())
                                    
                                        # # Criar aresta do id_lattes para interesse em produto para emergências do CEIS
                                        # if isinstance(produto, str):
                                        #     self.grafo.add_node(produto, tipo='ceis_produto_emergencial')
                                        #     if id_lattes:
                                        #         self.grafo.add_edge(id_lattes, produto, relation='INTERESSA_PRODUTO')

                        elif isinstance(produto_emergencial, list):
                            for produto in produto_emergencial:
                                if produto != '':
                                    intencoes["produtos_emergenciais"].append(produto.strip())

                                    # # Criar aresta do id_lattes para interesse em produto para emergências do CEIS
                                    # if isinstance(produto, str):
                                    #     self.grafo.add_node(produto, tipo='ceis_produto_emergencial')
                                    #     if id_lattes:
                                    #         self.grafo.add_edge(id_lattes, produto, relation='INTERESSA_PRODUTO')


                ceis_interesse_produtos_agravos = pesquisador.get("ceis_interesse_produtos_agravos")
                if ceis_interesse_produtos_agravos and isinstance(ceis_interesse_produtos_agravos, list):
                    for produto_agravo in ceis_interesse_produtos_agravos:
                        if isinstance(produto_agravo, str):
                            if ";" in produto_agravo:
                                lista_produtos = produto_agravo.split(';')
                                for produto in lista_produtos:
                                    if produto != '':
                                        intencoes["produtos_agravos"].append(produto.strip())

                                        # # Criar aresta do id_lattes para interesse em produto para agravos do CEIS
                                        # if isinstance(produto, str):
                                        #     self.grafo.add_node(produto, tipo='ceis_produto_agravo')
                                        #     if id_lattes:
                                        #         self.grafo.add_edge(id_lattes, produto, relation='INTERESSA_PRODUTO')

                        if isinstance(produto_agravo, list):
                            for produto in produto_agravo:
                                if produto != '':
                                    intencoes["produtos_agravos"].append(produto.strip())

                                    # # Criar aresta do id_lattes para interesse em produto para agravos do CEIS
                                    # if isinstance(produto, str):
                                    #     self.grafo.add_node(produto, tipo='ceis_produto_agravo')
                                    #     if id_lattes:
                                    #         self.grafo.add_edge(id_lattes, produto, relation='INTERESSA_PRODUTO')

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
                    ## TO-FIX:objeto intencoes passado é só lista de tipos e não dicionário de inteções
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