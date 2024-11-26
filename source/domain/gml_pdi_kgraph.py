import os
import json
import networkx as nx
from pyvis.network import Network
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util



class GrafoDemanda:
    def __init__(self):
        self.grafo = nx.Graph()
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
    
    def subgrafo_demanda_pdi(self):
        """
        Constrói o subgrafo de demanda com base nos dados da matriz CEIS.
        """
        try:
            pathfilename = os.path.join(self.in_json, 'matriz_ceis.json')
            with open(pathfilename, 'r') as f:  
                dados_ceis = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Erro ao ler o arquivo matriz_ceis.json: {e}")
            return

        for bloco in dados_ceis['blocos']:
            for produto in bloco['produtos']:
                self.grafo.add_node(produto['id'], tipo='produto', nome=produto['nome'])
            for desafio in bloco['desafios']:
                self.grafo.add_node(desafio['id'], tipo='desafio', nome=desafio['nome'])
                for plataforma in desafio['plataformas']:
                    self.grafo.add_node(plataforma['id'], tipo='plataforma', nome=plataforma['nome'])
                    self.grafo.add_edge(desafio['id'], plataforma['id'], relation='REQUER_PLATAFORMA')
                    for produto in bloco['produtos']:
                        self.grafo.add_edge(plataforma['id'], produto['id'], relation='PRODUZ_PRODUTO')

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
        print(f"\nSubgrafo de demanda criado com {num_nos} nós e {num_arestas} arestas.")
        print("  Nós por tipo:")
        for tipo, quantidade in tipos_nos.items():
            print(f"  - {tipo}: {quantidade}")
        print("  Arestas por tipo:")
        for tipo, quantidade in tipos_arestas.items():
            print(f"  - {tipo}: {quantidade}")

class GrafoOferta:
    def __init__(self):
        self.grafo = nx.Graph()
        self.base_repo_dir = self.find_repo_root()
        self.in_json = os.path.join(str(self.base_repo_dir), '_data', 'in_json')
        
        # Ler o arquivo input_curriculos.json no construtor
        try:
            pathfilename = os.path.join(self.in_json, 'input_curriculos.json')
            with open(pathfilename, 'r') as f:  
                self.dict_list = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Erro ao ler o arquivo input_curriculos.json: {e}")
            self.dict_list = None

        self.tipos_arestas = defaultdict(int)

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

    def calcular_similaridade_semantica(self, model, texto1, texto2):
        """
        Calcula a similaridade semântica entre dois textos.

        Args:
            model: Modelo de sentence embedding.
            texto1 (str): Primeiro texto.
            texto2 (str): Segundo texto.

        Returns:
            float: Similaridade semântica entre os textos.
        """
        embeddings1 = model.encode(texto1, convert_to_tensor=True)
        embeddings2 = model.encode(texto2, convert_to_tensor=True)
        cosine_sim = util.cos_sim(embeddings1, embeddings2)
        return cosine_sim.item()

    def encontrar_id_lattes(self, nome):
        """
        Encontra o ID Lattes correspondente ao nome do pesquisador no grafo.

        Args:
            nome (str): O nome do pesquisador.

        Returns:
            str: O ID Lattes do pesquisador, ou '9999999999999999' se não for encontrado ou se o nome não for informado.
        """
        if nome.lower().split()[0] == 'não':  # Verificar se a primeira palavra é "não"
            print(f"Aviso: Nome não informado. Usando ID Lattes genérico.")
            return '9999999999999999'  # Retornar ID Lattes genérico

        for no, dados in self.grafo.nodes(data=True):
            if dados.get('nome'):
                nome_pesquisador = dados.get('nome').lower()
                # print(f"Buscando: {nome_pesquisador}")
                if dados.get('tipo') == 'pesquisador' and nome_pesquisador == nome.lower():
                    print(f"Pesquisador encontrado no grafo com nó: {no}")
                    return no

        # Se não encontrar o nome exato, imprimir aviso e sugestões
        print(f"Aviso: Nome '{nome}' não encontrado exatamente no grafo.")
        primeira_palavra = nome.split()[0]  # Obter a primeira palavra do nome
        print(f"Possíveis nomes correspondentes (contendo '{primeira_palavra}'):")
        for no, dados in self.grafo.nodes(data=True):
            if dados.get('tipo') == 'pesquisador' and primeira_palavra in dados.get('nome'):
                print(f"  - {dados.get('nome')}")

        return '9999999999999999'  # ID Lattes padrão se não encontrar

    def adicionar_intencoes(self, verbose=False):
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

        # Contadores
        num_arestas_interesse = 0
        tipos_nos = defaultdict(int)
        tipos_arestas = defaultdict(int)
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
                if competencias_presentes and isinstance(competencias_presentes, list): # Verificar se é uma lista
                    for competencia in competencias_presentes:
                        if isinstance(competencia, str):  # Verificar se é uma string
                            # Criar nós de competencias_possuidas
                            self.grafo.add_node(competencia, tipo='competencias_possuidas')
                            tipos_nos['competencias_possuidas'] += 1

                            # Adicionar arestas COMPETENCIA_DECLARADA para id_lattes
                            if id_lattes:
                                self.grafo.add_edge(id_lattes, competencia, relation='COMPETENCIA_DECLARADA')
                                tipos_arestas['COMPETENCIA_DECLARADA'] += 1

                # --- Competências a Desenvolver ---
                competencias_desenvolver = pesquisador.get("competencias_desenvolver")
                if competencias_desenvolver and isinstance(competencias_desenvolver, list):  # Verificar se é uma lista
                    for competencia in competencias_desenvolver:
                        if isinstance(competencia, str):  # Verificar se é uma string
                            # Criar nós de competencias_desenvolver
                            self.grafo.add_node(competencia, tipo='competencias_desenvolver')
                            tipos_nos['competencias_desenvolver'] += 1

                            # Adicionar arestas COMPETENCIA_DESEJADA para id_lattes
                            if id_lattes:
                                self.grafo.add_edge(id_lattes, competencia, relation='COMPETENCIA_DESEJADA')
                                tipos_arestas['COMPETENCIA_DESEJADA'] += 1

                # --- Intenções ---
                intencoes = []
                questoes = pesquisador.get("questoes_interesse")
                if questoes and isinstance(questoes, str):  # Verificar se é uma string
                    intencoes.append(questoes)
                palavras_chave = pesquisador.get("palavras_chave")
                if palavras_chave and isinstance(palavras_chave, list):  # Verificar se é uma lista
                    for palavra in palavras_chave:
                        if isinstance(palavra, str):  # Verificar se é uma string
                            intencoes.append(palavra)
                pretende_desenvolver = pesquisador.get("intencao_desenvolvimento")
                if pretende_desenvolver and isinstance(pretende_desenvolver, str):  # Verificar se é uma string
                    intencoes.append(pretende_desenvolver)
                ceis_interesse_desafios = pesquisador.get("ceis_interesse_desafios")
                if ceis_interesse_desafios and isinstance(ceis_interesse_desafios, str):  # Verificar se é uma string
                    intencoes.extend(ceis_interesse_desafios.split(';'))
                ceis_interesse_produtos_emergencias = pesquisador.get("ceis_interesse_produtos_emergencias")
                if ceis_interesse_produtos_emergencias and isinstance(ceis_interesse_produtos_emergencias, list):  # Verificar se é uma lista
                    for produto in ceis_interesse_produtos_emergencias:
                        if isinstance(produto, str):  # Verificar se é uma string
                            intencoes.append(produto)
                ceis_interesse_produtos_agravos = pesquisador.get("ceis_interesse_produtos_agravos")
                if ceis_interesse_produtos_agravos and isinstance(ceis_interesse_produtos_agravos, list):  # Verificar se é uma lista
                    for produto in ceis_interesse_produtos_agravos:
                        if isinstance(produto, str):  # Verificar se é uma string
                            intencoes.append(produto)

                if verbose:
                    print(f"       questoes_interesse: {questoes}")
                    print(f"           palavras_chave: {palavras_chave}")
                    print(f"     pretende_desenvolver: {pretende_desenvolver}")
                    print(f"   competencias_presentes: {competencias_presentes}")
                    print(f" competencias_desenvolver: {competencias_desenvolver}")
                    print(f"  ceis_interesse_desafios: {ceis_interesse_desafios}")
                    print(f"ceis_produtos_emergencias: {ceis_interesse_produtos_emergencias}")
                    print(f"    ceis_produtos_agravos: {ceis_interesse_produtos_agravos}")

                if id_lattes and intencoes:
                    # Adicionar nó de intenção e contar
                    self.grafo.add_node(str(intencoes), tipo='intencao')  # Converter a lista intencoes para string
                    tipos_nos['intencao'] += 1

                    # Adicionar aresta entre pesquisador e intenção e contar
                    self.grafo.add_edge(id_lattes, str(intencoes), relation='POSSUI_INTENCAO')
                    tipos_arestas['POSSUI_INTENCAO'] += 1

                    # Adicionar arestas para produtos e desafios do CEIS
                    self.adicionar_interesses_ceis(str(intencoes), pesquisador, respostas_nao_associadas)  # Converter a lista intencoes para string
                    num_arestas_interesse += 1

            except Exception as e:
                print(f"Erro ao processar pesquisador {i}: {e}")

        # Imprimir a mensagem com as informações sobre as arestas TEM_INTERESSE
        print(f"Foram criadas {num_arestas_interesse} arestas TEM_INTERESSE.")
        if respostas_nao_associadas:
            print("As seguintes respostas não puderam ser associadas a produtos da matriz_ceis:")
            for resposta in respostas_nao_associadas:
                print(f"  - {resposta}")

        # Imprimir a quantidade de nós por tipo
        print("Nós por tipo:")
        for tipo, quantidade in tipos_nos.items():
            print(f"  - {tipo}: {quantidade}")

        # Imprimir a quantidade de arestas por tipo
        print("Arestas por tipo:")
        for tipo, quantidade in tipos_arestas.items():
            print(f"  - {tipo}: {quantidade}")

    def adicionar_interesses_ceis(self, intencoes, pesquisador, respostas_nao_associadas):
        """
        Adiciona arestas entre as intenções dos pesquisadores e os 
        produtos e desafios do CEIS, por escolha ou similaridade.
        """
        # 1. Obter os IDs dos produtos e desafios de interesse
        produtos_interesse = pesquisador.get('ceis_interesse_produtos_emergencias', []) + \
                            pesquisador.get('ceis_interesse_produtos_agravos', [])
        desafios_interesse = pesquisador.get('ceis_interesse_desafios', "").split(';')

        # 2. Criar arestas TEM_INTERESSE para os produtos e desafios de interesse
        for produto_id in produtos_interesse:
            if self.grafo.has_node(produto_id):
                self.grafo.add_edge(intencoes, produto_id, relation='TEM_INTERESSE_EM_PRODUTO')
                self.tipos_arestas['TEM_INTERESSE_EM_PRODUTO'] += 1
            # else:
            #     try:
            #         # 3. Criar arestas SIMILAR por aproximação semântica
            #         self.adicionar_interesses_por_similaridade(intencoes, pesquisador)
            #     except:
            #         respostas_nao_associadas.append(produto_id)

        for desafio_id in desafios_interesse:
            if self.grafo.has_node(desafio_id):
                self.grafo.add_edge(intencoes, desafio_id, relation='TEM_INTERESSE_EM_DESAFIO')
                self.tipos_arestas['TEM_INTERESSE_EM_DESAFIO'] += 1
            else:
                respostas_nao_associadas.append(desafio_id)

    def adicionar_interesses_por_similaridade(self, intencoes, pesquisador):
        """
        Adiciona arestas SIMILAR entre as intenções dos pesquisadores e os 
        produtos e desafios do CEIS, por aproximação semântica.
        """
        threshold_similaridade = 0.7  # Defina o limiar de similaridade desejado

        # 1. Obter o texto das intenções e dos produtos/desafios
        texto_intencoes = pesquisador.get('intencao_desenvolvimento', "")
        textos_produtos_desafios = []
        for no in self.grafo.nodes(data=True):
            if 'tipo' in no[1] and no[1]['tipo'] in ['produto', 'desafio']:
                textos_produtos_desafios.append(no[1]['nome'])

        # 2. Calcular a similaridade de cosseno
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([texto_intencoes] + textos_produtos_desafios)

        # Verificar se há produtos/desafios para calcular a similaridade
        if tfidf_matrix.shape[0] > 1:
            similaridade = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        else:
            print("    Aviso: Não há produtos/desafios para calcular a similaridade.")
        # 3. Criar arestas SIMILAR para produtos/desafios com similaridade acima do limiar
        for i, no in enumerate(self.grafo.nodes(data=True)):
            if 'tipo' in no[1] and no[1]['tipo'] in ['produto', 'desafio'] and similaridade[0, i] >= threshold_similaridade:
                self.grafo.add_edge(intencoes, no[0], relation='SIMILAR', similaridade=similaridade[0, i])
                self.tipos_arestas['SIMILAR'] += 1

    def adicionar_competencias(self, id_lattes, nome, pesquisador, tipos_nos):
        """
        Adiciona nós de competências ao grafo, 
        relacionando-as aos pesquisadores.
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

    def extrair_areas_de_intencoes(self, intencoes):
        """
        Extrai as áreas de pesquisa das intenções dos pesquisadores.
        (Implemente a lógica de extração de acordo com o formato das intenções)
        """
        areas = []
        # ... (implementar lógica para extrair áreas de pesquisa das intenções) ...
        return areas

    def subgrafo_oferta_pdi(self):
        """
        Constrói o subgrafo de oferta com base nos dados dos pesquisadores.
        """
        if self.dict_list:
            for dicionario in self.dict_list:
                if isinstance(dicionario, dict):
                    id_lattes = dicionario.get('Identificação', {}).get('ID Lattes')
                    nome = dicionario.get('Identificação', {}).get('Nome')
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


class GrafoConhecimento:
    def __init__(self):
        self.grafo = nx.Graph()
    
    def visualizar(self, nome_arquivo="grafo_conhecimento.html"):
        """
        Gera uma visualização interativa do grafo usando pyvis.
        """
        net = Network(notebook=True, directed=True, cdn_resources='in_line')
        net.from_nx(self.grafo)

        # Definir cores dos nós
        for node in net.nodes:
            if node['tipo'] == 'produto':
                node['color'] = 'green'
            elif node['tipo'] == 'pesquisador':
                node['color'] = 'orange'

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
              "gravitationalConstant": -80000,
              "springLength": 250,
              "springConstant": 0.001
            },
            "minVelocity": 0.75
          }
        }
        """)
        print(f"\nGrafo de conhecimento gerado com sucesso:")
        return net.show(nome_arquivo)    
    
    def construir_grafo(self, verbose=False):
        """
        Constrói o grafo de conhecimento multicamadas.
        """
        # Criar subgrafos de oferta e demanda
        self.oferta = GrafoOferta()
        self.demanda = GrafoDemanda()

        # Construir o grafo de conhecimento por subgrafos em camadas
        self.demanda.subgrafo_demanda_pdi()
        if verbose:
            # Imprimir nós do subgrafo de demanda
            print("Nós do subgrafo de demanda:")
            for no, atributos in self.demanda.grafo.nodes(data=True):
                print(f"  - Nó: {no}, Atributos: {atributos}")

        self.oferta.subgrafo_oferta_pdi()
        if verbose:
            # Imprimir nós do subgrafo de oferta
            print("Nós do subgrafo de oferta:")
            for no, atributos in self.oferta.grafo.nodes(data=True):
                print(f"  - Nó: {no}, Atributos: {atributos}")

        self.oferta.adicionar_projetos()

        # Integrar os subgrafos de oferta e demanda ao grafo principal
        self.integrar_subgrafos()

        # Adicionar intenções dos pesquisadores
        self.oferta.adicionar_intencoes()  # Adicionar esta linha

        # Gerar visualização do grafo com pyvis
        caminho_html = self.visualizar()

    def integrar_subgrafos(self):
        """
        Integra os subgrafos de oferta e demanda ao grafo principal.
        """
        self.grafo.add_nodes_from(self.oferta.grafo.nodes(data=True))
        self.grafo.add_edges_from(self.oferta.grafo.edges(data=True))
        self.grafo.add_nodes_from(self.demanda.grafo.nodes(data=True))
        self.grafo.add_edges_from(self.demanda.grafo.edges(data=True))

    def agregar_camadas(self, camada):
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
                            self.grafo.add_edge(vizinho, novo_no, relation='ATUA_NA_AREA_SUBAREA')
                    self.grafo.remove_node(subarea)
                self.grafo.remove_node(area)
        
        elif camada == 'outra_camada':
            # Acrescentar implementação para agregar outras camada, quando necessário
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

        similaridade = cosine_similarity(caracteristicas)

        for i in range(len(nos)):
            for j in range(i + 1, len(nos)):
                no1 = nos[i]
                no2 = nos[j]
                if similaridade[i, j] > 0:
                    self.grafo.add_edge(no1, no2, similaridade=similaridade[i, j])


    def extrair_caracteristicas(self, no):
        """
        Extrai as características relevantes de um nó.

        Args:
            no (str): O nó do grafo.

        Returns:
            list: Uma lista de características numéricas do nó.
        """
        try:
            caracteristicas = []
            tipo_no = self.grafo.nodes[no]['tipo']

            # Extrair características pesquisador, como qte publicações, qte projetos, etc.
            if tipo_no == 'pesquisador':
                
                # Exemplo: usar o grau do nó como número de publicações
                num_publicacoes = self.grafo.degree(no)  
                
                caracteristicas.append(num_publicacoes)
                # ... (adicionar outras características do pesquisador) ...

            elif tipo_no == 'area_subarea':
                # Extrair características da área/subárea
                # Exemplo: número de pesquisadores, número de subáreas, etc.
                num_pesquisadores = len(list(self.grafo.predecessors(no)))  # Exemplo: contar quantos pesquisadores atuam na área/subárea
                caracteristicas.append(num_pesquisadores)
                # ... (adicionar outras características da área/subárea) ...

            elif tipo_no == 'produto':
                # Extrair características do produto, como tipo de produto, demanda, etc.
                demanda = self.grafo.nodes[no].get('demanda', 0)
                caracteristicas.append(demanda)
                # ... (adicionar outras características do produto) ...

            elif tipo_no == 'desafio':
                # Extrair características do desafio CEIS, plataformas, área de conhecimento, etc.
                num_plataformas = len(list(self.grafo.neighbors(no)))  # Exemplo: contar quantas plataformas o desafio requer
                caracteristicas.append(num_plataformas)
                # ... (adicionar outras características do desafio) ...

            elif tipo_no == 'plataforma':
                # Extrair características da plataforma
                # Exemplo: tipo de equipamento, número de produtos, etc.
                # ... (adicionar características da plataforma) ...
                pass

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
