import pandas as pd
import os, re, logging, json, neo4j
from pathlib import Path
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

class Neo4jPersister:
    def __init__(self, uri, user, password):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        self.configure_logging()

        # Devem ser persistidos como 
        self.tipos = [
            'Identificação',
            'Idiomas',
            'Formação',
            'Atuação Profissional',
            'Linhas de Pesquisa',
            'Áreas',
            'Produções',
            'ProjetosPesquisa',
            'ProjetosExtensão',
            'ProjetosDesenvolvimento',
            'ProjetosOutros',
            'Bancas',
            'Orientações',
            ]

        self.subtipos= [
            'Acadêmica',
            'Pos-Doc',
            'Complementar',
            'Artigos completos publicados em periódicos',
            'Resumos publicados em anais de congressos',
            'Apresentações de Trabalho',
            'Outras produções bibliográficas',
            'Entrevistas, mesas redondas, programas e comentários na mídia',
            'Concurso público',
            'Outras participações',
            'Livros publicados/organizados ou edições',
            'Capítulos de livros publicados',
            'Resumos expandidos publicados em anais de congressos',
            'Resumos publicados em anais de congressos (artigos)',
            'Trabalhos técnicos',
            'Demais trabalhos',
            'Mestrado',
            'Teses de doutorado',
            'Qualificações de Doutorado',
            'Qualificações de Mestrado',
            'Monografias de cursos de aperfeiçoamento/especialização',
            'Trabalhos de conclusão de curso de graduação',
            'Orientações e supervisões concluídas',
            'Orientações e supervisões em andamento',
            'Citações',
            'Trabalhos completos publicados em anais de congressos',
            'Produtos tecnológicos',
            'Artigos  aceitos para publicação',
            'Assessoria e consultoria',
            'Programas de computador sem registro',
            'Professor titular',
            'Avaliação de cursos',
            'Processos ou técnicas',
            'Outras produções artísticas/culturais',
            'Textos em jornais de notícias/revistas',
            'Redes sociais, websites e blogs',
            'Artes Visuais'            
            ]

        self.propriedades = [
            'Nome',
            'ID Lattes',
            'Última atualização',
            ]
        
    def close(self):
        self._driver.close()

    def configure_logging(self):
        logging.basicConfig(filename='logs/neo4j_persister.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

    @staticmethod
    def find_repo_root(path='.', depth=10):
        ''' 
        Busca o arquivo .git e retorna string com a pasta raiz do repositório.
        '''
        # Prevenir recursão infinita limitando a profundidade
        if depth < 0:
            return None
        path = Path(path).absolute()
        if (path / '.git').is_dir():
            return path
        # Corrigido para usar LattesScraper.find_repo_root para chamada recursiva
        return Neo4jPersister.find_repo_root(path.parent, depth-1)

    def persistir_revistas_da_planilha(self):
        """
        Persiste dados de revistas a partir da planilha 'classificações_publicadas_todas_as_areas_avaliacao1672761192111.xlsx' no Neo4j.

        Args:
            session: Objeto de sessão do Neo4j.
        """
        # Leitura da planilha
        dados_qualis = pd.read_excel(os.path.join(self.find_repo_root(),'_data','in_xls','classificações_publicadas_todas_as_areas_avaliacao1672761192111.xlsx'))

        # Extração e persistência de dados de revista
        with self._driver.session() as session:
            for index, row in dados_qualis.iterrows():
                issn = row['ISSN'].replace('-','')
                nome_revista = row['Título']
                area_avaliacao = row['Área de Avaliação']
                estrato = row['Estrato']

                # Verificação de existência da revista
                revista_node = session.run("""
                    MATCH (j:Revista {issn: $issn})
                    RETURN j
                """, issn=issn).single()

                if not revista_node:
                    # Criação da revista se não existir
                    session.run("""
                        CREATE (j:Revista {issn: $issn, nome_revista: $nome_revista, area_avaliacao: $area_avaliacao, estrato: $estrato})
                    """, nome_revista=nome_revista, issn=issn, area_avaliacao=area_avaliacao,  estrato=estrato)

    def persistir_areas_avaliacao_capes(self):
        """
        Persiste áreas de avaliação únicas como nós no Neo4j, relacionando-as às revistas.

        Args:
            session: Objeto de sessão do Neo4j (opcional). Se não fornecido, usa o driver da classe.
        """
        session = session or self._driver.session()

        # Consulta Cypher para criar nós de área de avaliação e relacionamentos
        query = """
        MATCH (r:Revista)
        UNWIND split(r.area_avaliacao, ', ') AS area
        MERGE (a:AreaAvaliacao {nome: area})
        MERGE (r)-[:AVALIADA_EM]->(a)
        """

        # Execução da consulta
        with session:
            session.run(query)

    def desenhar_grafo_revistas_capes(self):
        from pyvis.network import Network

        with self._driver.session() as session:
            result = session.run("""
                MATCH (a:AreaAvaliacao)<-[:AVALIADA_EM]-(revista:Revista)
                WITH a, collect(revista) AS revistas
                CALL apoc.path.subgraphAll(a, {relationshipFilter:'AVALIADA_EM'}) YIELD nodes, relationships
                RETURN nodes, relationships
            """)

            # Converter o resultado em um grafo Pyvis
            net = Network(notebook=True, cdn_resources='in_line')

            # Adicionar nós
            for record in result:
                for node in record["nodes"]:
                    if "Revista" in node.labels:
                        label = node.get("nome_revista")
                        node_id = label.replace(" ", "")  # Remover espaços em branco
                        net.add_node(node_id, label=label, shape="circle")
                    else:
                        net.add_node(node.id, label=node.get("nome"), shape="box")

            # Adicionar arestas
            for record in result:
                for rel in record["relationships"]:
                    start_node_id = rel.start_node.get("nome_revista").strip() if "Revista" in rel.start_node.labels else rel.start_node
                    end_node_id = rel.end_node.get("nome_revista").strip() if "Revista" in rel.end_node.labels else rel.end_node
                    net.add_edge(start_node_id, end_node_id)

            # Obter o caminho completo para o diretório 'templates' na raiz do projeto
            templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')

            # Criar o diretório 'templates' se ele não existir
            os.makedirs(templates_dir, exist_ok=True)

            # Salvar o HTML em 'templates/grafo_revistas.html'
            net.show(os.path.join(templates_dir, "grafo_revistas.html"))

    # Testes Ok! 
    def persist_docent_nodes(self, dict_list):
        query_pessoa = """
        MERGE (p:Docente {id_lattes: $id_lattes})
        ON CREATE SET p.nome = $nome, p.ultima_atualizacao = $ultima_atualizacao
        ON MATCH SET p.nome = $nome, p.ultima_atualizacao = $ultima_atualizacao
        """
        try:
            with self._driver.session() as session:
                for item in dict_list:
                    identificacao = item.get('Identificação')
                    nome = identificacao.get('Nome')
                    id_lattes = identificacao.get('ID Lattes')
                    ultima_atualizacao = identificacao.get('Última atualização')
                    if nome:
                        session.run(query_pessoa, id_lattes=id_lattes, nome=nome, ultima_atualizacao=ultima_atualizacao)
        except Exception as e:
            self.logger.error('Erro ao criar node "Pesquisador": {}'.format(e))

    def persist_discent_nodes(self, dict_list):
        query_pessoa = """
        MERGE (p:Discente {id_lattes: $id_lattes})
        ON CREATE SET p.nome = $nome, p.ultima_atualizacao = $ultima_atualizacao
        ON MATCH SET p.nome = $nome, p.ultima_atualizacao = $ultima_atualizacao
        """
        try:
            with self._driver.session() as session:
                for item in dict_list:
                    identificacao = item.get('Identificação')
                    nome = identificacao.get('Nome')
                    id_lattes = identificacao.get('ID Lattes')
                    ultima_atualizacao = identificacao.get('Última atualização')
                    if nome:
                        session.run(query_pessoa, id_lattes=id_lattes, nome=nome, ultima_atualizacao=ultima_atualizacao)
        except Exception as e:
            self.logger.error('Erro ao criar node "Discente": {}'.format(e))

    # Testes Ok!         
    def persist_pesquisador_grande_area_relationships(self, dict_list):
        query_rel_pessoa_grande_area = """
        MATCH (p:Pesquisador {id_lattes: $id_lattes})
        MATCH (ga:GrandeArea {nome: $grande_area_nome})
        MERGE (p)-[:ATUA_EM]->(ga)
        """

        with self._driver.session() as session:
            for item in dict_list:
                identificacao = item.get('Identificação')
                id_lattes = identificacao.get('ID Lattes')
                areas = item.get('Áreas').values()
                for area_string in areas:
                    grande_area_nome, _, _ = self.extract_area_info(area_string)
                    if grande_area_nome:
                        session.run(query_rel_pessoa_grande_area, id_lattes=id_lattes, grande_area_nome=grande_area_nome)

    # Testes Ok! 
    def persist_areas_nodes(self, dict_list):
        query_grande_area = """
        MERGE (ga:GrandeArea {nome: $nome})
        """
        query_area = """
        MATCH (ga:GrandeArea {nome: $grande_area_nome})
        MERGE (a:Area {nome: $nome}) ON CREATE SET a:Area
        MERGE (ga)-[:CONTEM]->(a)
        """
        query_subarea = """
        MATCH (a:Area {nome: $area_nome})
        MERGE (sa:Subarea {nome: $nome}) ON CREATE SET sa:Subarea
        MERGE (a)-[:CONTEM]->(sa)
        """
        query_rel_pessoa_grande_area = """
        MATCH (p:Pesquisador {id_lattes: $id_lattes})
        MATCH (ga:GrandeArea {nome: $grande_area_nome})
        MERGE (p)-[:ATUA_EM]->(ga)    
        """

        with self._driver.session() as session:
            for item in dict_list:
                areas = item.get('Áreas').values()
                for area_string in areas:
                    grande_area_nome, area_nome, subarea_nome = self.extract_area_info(area_string)
                    
                    # Verificar se o nome não está vazio
                    if grande_area_nome:
                        session.run(query_grande_area, nome=grande_area_nome)
                    if area_nome:
                        session.run(query_area, grande_area_nome=grande_area_nome, nome=area_nome)
                    if subarea_nome:
                        session.run(query_subarea, area_nome=area_nome, nome=subarea_nome)
                    
                    # Adicionar relacionamento Pesquisador - GrandeÁrea
                    id_lattes = item['Identificação']['ID Lattes']
                    if grande_area_nome:
                        session.run(query_rel_pessoa_grande_area, id_lattes=id_lattes, grande_area_nome=grande_area_nome)

    # Testes Ok! 
    @staticmethod
    def extract_area_info(area_string):
        # Extraindo os nomes de GrandeÁrea, Área e Subárea da string
        try:
            grande_area_nome = area_string.split('/')[0].strip().split(': ')[1]
        except:
            grande_area_nome = ''
        try:
            area_nome = area_string.split('/')[1].strip().split(': ')[1]
        except:
            area_nome = ''
        try:
            subarea_nome = area_string.split('/')[2].strip().split(': ')[1]
        except:
            subarea_nome = ''
        return grande_area_nome, area_nome, subarea_nome

    ## PRODUÇÕES
    def persist_producoes_pesquisador(self, dict_list):
        with self._driver.session() as session:
            for pesq in dict_list:
                identificacao = pesq.get('Identificação')
                id_lattes = identificacao.get('ID Lattes')
                producoes = pesq.get('Produções')

                if not isinstance(producoes, dict):
                    print(f"Erro!! Dicionário da seção 'Produções' não encontrado para {id_lattes}")
                    continue

                for chave_producao, valores_producao in producoes.items():
                    print(f'{chave_producao} | {valores_producao}')
                    if chave_producao == 'Artigos completos publicados em periódicos':
                        # self.persistir_artigos_completos(session, id_lattes, valores_producao)
                        self.persistir_artigos_revistas(session, id_lattes, valores_producao)

    def _get_or_create_node(self, session, label, properties):
        properties = [x.rstrip('.') for x in properties]
        node = session.run("MATCH (n: {label}) WHERE {properties} RETURN n", {"label": label, "properties": properties}).single()

        if not node:
            node = session.run("CREATE (n: {label} {properties}) RETURN n", {"label": label, "properties": properties}).single()["n"]
            self._node_created_count += 1

        return node

    def persist_tipo_producao(self, session, id_lattes, tipo_producao):
        query_create_node = """
        MERGE (p:Pesquisador {id_lattes: $id_lattes})
        MERGE (t:TipoProducao {nome: $tipo_producao})
        MERGE (p)-[:PRODUZ]->(t)
        """
        result = session.run(query_create_node, id_lattes=id_lattes, tipo_producao=tipo_producao)
        summary = result.consume()
        return summary.counters.nodes_created, summary.counters.nodes_deleted, summary.counters.relationships_created, summary.counters.relationships_deleted

    def persist_subtipo_producao(self, session, id_lattes, tipo_producao, subtipo_producao, dados_producao):
        def checar_e_serializar(dados):
            """ Verifica e serializa dicionários recursivamente """
            if isinstance(dados, dict):
                for chave, valor in dados.items():
                    if isinstance(valor, dict):
                        dados[chave] = json.dumps(valor)
                    # Checagem adicional para outros tipos inválidos, se necessário
            return dados
        # Serialização recursiva do dicionário
        dados_producao = checar_e_serializar(dados_producao)

        query_create_node = """
        MERGE (p:Pesquisador {id_lattes: $id_lattes})
        MERGE (t:TipoProducao {nome: $tipo_producao})
        MERGE (s:SubtipoProducao {nome: $subtipo_producao})
        MERGE (p)-[PRODUZ:]->(t)-[:DO_TIPO]->(s)
        """

        if subtipo_producao in ["ArtigoCompleto", "ResumoCongresso", "ApresentacaoTrabalho", "OutrasProducoesBibliograficas"]:
            query_create_node += """
            MERGE (o:Ocorrencia {tipo: $subtipo_producao, dados: $dados})
            MERGE (s)-[:OCORRENCIA]->(o)
            """

        result = session.run(query_create_node, id_lattes=id_lattes, tipo_producao=tipo_producao, 
                            subtipo_producao=subtipo_producao, dados=dados_producao)

        # Obtendo as informações de contadores
        summary = result.consume()

        return summary.counters.nodes_created, summary.counters.nodes_deleted, summary.counters.relationships_created, summary.counters.relationships_deleted

    def persistir_artigos_completos(self, session, id_lattes, dados):
        created_nodes = 0
        updated_nodes = 0
        created_relations = 0
        updated_relations = 0
        
        for dados_artigo in dados:
            dados_artigo['dados'] = json.dumps(dados_artigo)  # Conversão para JSON da estrutura completa
            query_create_node_artigo = """
                MERGE (p:Pesquisador {id_lattes: $id_lattes})
                CREATE (a:ArtigoPublicado {
                    ano: $ano,
                    fator_impacto_jcr: $fator_impacto_jcr,
                    ISSN: $ISSN,
                    titulo: $titulo,
                    revista: $revista,
                    autores: $autores,
                    Qualis: $Qualis,
                    DOI: $DOI,
                    dados: $dados_artigo
                })
                CREATE (p)-[:PRODUZ]->(a)

                MERGE (j:Revista {nome: $revista, issn: $ISSN})
                CREATE (a)-[:PUBLICADO_EM]->(j)
            """

            result_artigo = session.run(query_create_node_artigo, 
                                id_lattes=id_lattes, 
                                ano=dados_artigo['ano'],
                                fator_impacto_jcr=dados_artigo['fator_impacto_jcr'],
                                ISSN=dados_artigo['ISSN'],
                                titulo=dados_artigo['titulo'],
                                revista=dados_artigo['revista'],
                                autores=dados_artigo['autores'],
                                Qualis=dados_artigo['Qualis'],
                                DOI=dados_artigo['DOI'],
                                dados_artigo=dados_artigo
                                )
            summary_artigo = result_artigo.consume()
            created_nodes += summary_artigo.counters.nodes_created
            updated_nodes += summary_artigo.counters.nodes_deleted
            created_relations += summary_artigo.counters.relationships_created
            updated_relations += summary_artigo.counters.relationships_deleted

        return created_nodes, updated_nodes, created_relations, updated_relations

    def persistir_artigos_completos(self, session, id_lattes, dados):
        created_nodes = 0
        updated_nodes = 0
        created_relations = 0
        updated_relations = 0
                
        for dados_artigo in dados:
            ano = dados_artigo['ano']
            impact_jcr = dados_artigo['fator_impacto_jcr']
            issn = dados_artigo['ISSN']
            titulo = dados_artigo['titulo']
            revista = dados_artigo['revista']
            autores = dados_artigo['autores']
            qualis = dados_artigo['Qualis']
            doi = dados_artigo['DOI']

            query_create_node_artigo = """
                MERGE (p:Pesquisador {id_lattes: $id_lattes})
                CREATE (a:ArtigoPublicado {ano: $ano, impact_jcr: $impact_jcr, issn: $issn, titulo: $titulo, revista: $revista, autores: $autores, qualis: $qualis, doi: $doi})
                CREATE (p)-[:PRODUZ]->(a)
                MERGE (j:Revista {nome: $revista, issn: $issn})
                CREATE (a)-[:PUBLICADO_EM]->(j)
            """
            print(query_create_node_artigo)
            result_artigo = session.run(query_create_node_artigo, 
                                id_lattes=id_lattes, 
                                ano=ano,
                                impact_jcr=impact_jcr,
                                issn=issn,
                                titulo=titulo,
                                revista=revista,
                                autores=autores,
                                qualis=qualis,
                                doi=doi,
                                )
            summary_artigo = result_artigo.consume()
            created_nodes += summary_artigo.counters.nodes_created
            updated_nodes += summary_artigo.counters.nodes_deleted
            created_relations += summary_artigo.counters.relationships_created
            updated_relations += summary_artigo.counters.relationships_deleted

        return created_nodes, updated_nodes, created_relations, updated_relations

    def buscar_revista_por_issn(self, session, issn):
        """Função para buscar uma revista por ISSN no banco de dados Neo4j."""

        query = """
            MATCH (revista:Revista)
            WHERE revista.issn = "{issn}"
            RETURN revista
        """

        with session.begin_transaction() as tx:
            try:
                result = tx.run(query, issn=issn)
                return result.single()
            except Neo4jError as e:
                print(f"Erro Neo4j ao buscar a revista por ISSN: {e}")
                return None

    def persistir_artigos_revistas(self, session, id_lattes, dados):
        """
        Função para persistir os dados de artigos completos publicados em periódicos.

        Args:
            session (neo4j.Session): Sessão Neo4j.
            id_lattes (str): ID do Lattes do pesquisador.
            dados (dict): Dicionário contendo os dados dos artigos.

        Returns:
            None
        """

        for artigo in dados:
            # Extraindo informações do artigo
            revista_nome  = ''
            created_nodes = ''
            ano = artigo['ano']
            impact_jcr = artigo['fator_impacto_jcr']
            issn = artigo['ISSN']
            titulo = artigo['titulo']
            revista = artigo['revista']
            autores = artigo['autores']
            data_issn = artigo['data_issn']
            doi = artigo['DOI']
            qualis = artigo['Qualis']

            query_create_node_artigo = """
                MERGE (p:Pesquisador {id_lattes: $id_lattes})
                CREATE (a:ArtigoPublicado {ano: $ano, impact_jcr: $impact_jcr, issn: $issn, titulo: $titulo, revista: $revista, autores: $autores, qualis: $qualis, doi: $doi})
                CREATE (p)-[:PRODUZ]->(a)
                MERGE (j:Revista {nome: $revista, issn: $issn})
                CREATE (a)-[:PUBLICADO_EM]->(j)
            """
            
            # Buscando o nó da revista
            revista_node = self.buscar_revista_por_issn(session, issn)
            updated_nodes = 0
            created_relations = 0
            # Criando o nó do artigo
            with session.begin_transaction() as tx:
                tx.run(query_create_node_artigo,
                    id_lattes=id_lattes,
                    ano=ano,
                    impact_jcr=impact_jcr,
                    issn=issn,
                    titulo=titulo,
                    revista=revista,
                    autores=autores,
                    data_issn=data_issn,
                    doi=doi,
                    qualis=qualis
                    )

                # Criando o relacionamento PUBLICADO_EM
                if revista_node is not None:
                    node_revista = revista_node[0][1]
                    if node_revista is not None:
                        revista_nome = node_revista['nome_revista']
                        revista_issn = node_revista['issn']
                        revista_area_avaliacao = node_revista['area_avaliacao']
                        revista_estrato = node_revista['estrato']

                    if revista_nome:
                        tx.run("""
                            MATCH (a:ArtigoPublicado {doi: $doi}), (j:Revista {nome_revista: $revista_nome, issn: $revista_issn, area_avaliacao: $revista_area_avaliacao, estrato: $revista_estrato})
                            CREATE (a)-[:PUBLICADO_EM]->(j)
                        """, doi=doi, revista_nome=revista_nome, revista_issn=revista_issn, revista_area_avaliacao=revista_area_avaliacao, revista_estrato=revista_estrato)

                    else:
                        print("Erro: O nó da revista não foi encontrado para o ISSN", issn)
                        # Lógica de tratamento de erro (opcional)
                else:
                    print("Erro: O retorno para a revista com ISSN", issn, "é None.")

                tx.commit()

        # Atualização dos contadores
        with session.begin_transaction() as tx:
            created_nodes += str(tx.run("MATCH (n) WHERE n:ArtigoPublicado RETURN count(n)").single()[0])
            updated_nodes += tx.run("MATCH (n) WHERE n:ArtigoPublicado SET n.updated_at = datetime() RETURN count(n)").single()[0]
            created_relations += tx.run("MATCH (r) WHERE r:PUBLICADO_EM RETURN count(r)").single()[0]

    def persistir_resumos_congressos(self, session, id_lattes, dados):
        query_create_node = """
        MERGE (p:Pesquisador {id_lattes: $id_lattes})
        MERGE (r:ResumoCongresso {titulo: $titulo, ano: $ano, evento: $evento, autores: $autores, data_issn: $data_issn, doi: $doi})
        MERGE (p)-[:PRODUZ]->(r)
        """
        result = session.run(query_create_node, id_lattes=id_lattes, **dados)
        # Obtendo as informações de contadores
        summary = result.consume()
        created_nodes = summary.counters.nodes_created
        updated_nodes = summary.counters.nodes_deleted  
        created_relations = summary.counters.relationships_created
        updated_relations = summary.counters.relationships_deleted 

        return created_nodes, updated_nodes, created_relations, updated_relations

    def persistir_apresentacoes_trabalho(self, session, id_lattes, dados):
        query_create_node = """
        MERGE (p:Pesquisador {id_lattes: $id_lattes})
        MERGE (a:ApresentacaoTrabalho {
            titulo: $titulo,
            ano: $ano,
            evento: $evento,
            autores: $autores
        })
        MERGE (p)-[:PRODUZ]->(a)
        """
        result = session.run(query_create_node, id_lattes=id_lattes, **dados)

        # Obtendo as informações de contadores
        summary = result.consume()
        created_nodes = summary.counters.nodes_created
        updated_nodes = summary.counters.nodes_deleted  
        created_relations = summary.counters.relationships_created
        updated_relations = summary.counters.relationships_deleted 

        return created_nodes, updated_nodes, created_relations, updated_relations

    def persistir_outras_producoes_bibliograficas(self, session, id_lattes, dados):
        query_create_node = """
        MERGE (p:Pesquisador {id_lattes: $id_lattes})
        MERGE (o:OutrasProducoesBibliograficas {
            titulo: $titulo,
            ano: $ano,
            autores: $autores,
            doi: $doi
        })
        MERGE (p)-[:PRODUZ]->(o)
        """
        result = session.run(query_create_node, id_lattes=id_lattes, **dados)

        # Obtendo as informações de contadores
        summary = result.consume()
        created_nodes = summary.counters.nodes_created
        updated_nodes = summary.counters.nodes_deleted  
        created_relations = summary.counters.relationships_created
        updated_relations = summary.counters.relationships_deleted 

        return created_nodes, updated_nodes, created_relations, updated_relations

    ## ORIENTAÇÕES
    def extract_advices(self, docente, tipo_orientacao):
        linhas = docente.splitlines()
        nomes_discentes = []
        ano_inicio = None
        instituicao = None

        for linha in linhas:
            match_nome = re.search(r"^([A-Z]+(?: [A-Z]+)+)", linha)
            match_ano = re.search(r"Início: (\d{4})", linha)
            match_instituicao = re.search(r"\(([^)]+)\)\s*\.?$", linha)  # Captura a instituição entre parênteses no final da linha

            if match_nome:
                nomes_discentes.append(match_nome.group(1).strip())
            if match_ano:
                ano_inicio = int(match_ano.group(1))
            if match_instituicao:
                instituicao = match_instituicao.group(1).strip()

        return docente, nomes_discentes, tipo_orientacao, ano_inicio, instituicao

    def persist_advices_relationships(self, dict_docent_list):
        query_rel_advices = """
        MATCH (do:Docente {id_lattes: $id_lattes_docente})
        MERGE (di:Discente {nome: $nome_discente})
        MERGE (do)-[:ORIENTA {tipo: $tipo_orientacao, ano_inicio: $ano_inicio, instituicao: $instituicao}]->(di)
        """

        with self._driver.session() as session:
            for item in dict_docent_list:
                identificacao = item.get('Identificação')
                id_lattes_docente = identificacao.get('ID Lattes')
                
                # Processar as orientações em andamento
                orientacoes_andamento = item.get('Orientações e supervisões em andamento', {})
                for tipo_orientacao, trabalhos in orientacoes_andamento.items():
                    for trabalho in trabalhos.values():
                        _, nomes_discentes, tipo_orientacao_encontrado, ano_inicio, instituicao = self.extract_advices(trabalho, tipo_orientacao)
                        for nome_discente in nomes_discentes:
                            session.run(query_rel_advices, id_lattes_docente=id_lattes_docente, nome_discente=nome_discente, tipo_orientacao=tipo_orientacao_encontrado, ano_inicio=ano_inicio, instituicao=instituicao)
                
                # Processar as orientações concluídas
                orientacoes_concluidas = item.get('Orientações e supervisões concluídas', {})
                for tipo_orientacao, trabalhos in orientacoes_concluidas.items():
                    for trabalho in trabalhos.values():
                        _, nomes_discentes, tipo_orientacao_encontrado, _, _ = self.extract_advices(trabalho, tipo_orientacao)
                        for nome_discente in nomes_discentes:
                            session.run(query_rel_advices, id_lattes_docente=id_lattes_docente, nome_discente=nome_discente, tipo_orientacao=tipo_orientacao_encontrado, ano_inicio=None, instituicao=None) # Ano e instituição nulos para orientações concluídas

    def persistir_participacoes_bancas(self, session, id_lattes, dados):
        query_create_node = """
        MERGE (p:Docente {id_lattes: $id_lattes})
        MERGE (b:Banca {
            tipo: $tipo,
            titulo: $titulo,
            ano: $ano,
            instituicao: $instituicao
        })
        MERGE (p)-[:PARTICIPA_BANCA]->(b)
        """
        result = session.run(query_create_node, id_lattes=id_lattes, **dados)

        # Obtendo as informações de contadores
        summary = result.consume()
        created_nodes = summary.counters.nodes_created
        updated_nodes = summary.counters.nodes_deleted  
        created_relations = summary.counters.relationships_created
        updated_relations = summary.counters.relationships_deleted 

        return created_nodes, updated_nodes, created_relations, updated_relations

    def persistir_projetos_pesquisa(self, session, id_lattes, dados):
        query_create_node = """
        MERGE (p:Pesquisador {id_lattes: $id_lattes})
        MERGE (pr:ProjetoPesquisa {
            titulo: $titulo,
            ano_inicio: $ano_inicio,
            ano_fim: $ano_fim,
            agencia_financiadora: $agencia_financiadora,
            valor_financiamento: $valor_financiamento
        })
        MERGE (p)-[:COORDENA]->(pr)
        """
        result = session.run(query_create_node, id_lattes=id_lattes, dados=dados)

        # Obtendo as informações de contadores
        summary = result.consume()
        created_nodes = summary.counters.nodes_created
        updated_nodes = summary.counters.nodes_deleted  
        created_relations = summary.counters.relationships_created
        updated_relations = summary.counters.relationships_deleted 

        return created_nodes, updated_nodes, created_relations, updated_relations

    def persistir_premios_distincoes(self, session, id_lattes, dados):
        query_create_node = """
        MERGE (p:Pesquisador {id_lattes: $id_lattes})
        MERGE (pd:PremioDistincao {
            titulo: $titulo,
            ano: $ano,
            instituicao: $instituicao,
        })
        MERGE (p)-[:RECEBE]->(pd)
        """
        result = session.run(query_create_node, id_lattes=id_lattes, **dados)

        # Obtendo as informações de contadores
        summary = result.consume()
        created_nodes = summary.counters.nodes_created
        updated_nodes = summary.counters.nodes_deleted  
        created_relations = summary.counters.relationships_created
        updated_relations = summary.counters.relationships_deleted 

        return created_nodes, updated_nodes, created_relations, updated_relations


    @staticmethod
    def convert_to_primitives(input_data):
        if input_data is None:
            return None
        
        if isinstance(input_data, dict):
            return {key: Neo4jPersister.convert_to_primitives(value) for key, value in input_data.items()}
        
        elif isinstance(input_data, list):
            return [Neo4jPersister.convert_to_primitives(item) for item in input_data]
        
        elif isinstance(input_data, str):
            if 'http://' in input_data or 'https://' in input_data:
                parts = input_data.split(" ")
                new_parts = [urllib.parse.quote(part) if part.startswith(('http://', 'https://')) else part for part in parts]
                return " ".join(new_parts)
            return input_data
        
        elif isinstance(input_data, (int, float, bool)):
            return input_data
        
        else:
            return str(input_data)

    @staticmethod
    def debug_and_convert(input_data):
        try:
            return Neo4jPersister.convert_to_primitives(input_data)
        except:
            print("Conversion failed for:", input_data)
            raise

    def extract_lattes_id(self, infpes_list):
        """Extracts the Lattes ID from the InfPes list."""
        for entry in infpes_list:
            if 'ID Lattes:' in entry:
                # Extracting the numeric portion of the 'ID Lattes:' entry
                return entry.split(":")[1].strip()
        return None

    def persist_data(self, data_dict, label):
        data_dict_primitives = self.convert_to_primitives(data_dict)

        # Extracting the Lattes ID from the provided structure
        lattes_id = self.extract_lattes_id(data_dict.get("Identificação", []))
        
        if not lattes_id:
            print("Lattes ID not found or invalid.")
            return
        
        # Flatten the "Identificação" properties into the main dictionary
        if "Identificação" in data_dict_primitives:
            id_properties = data_dict_primitives.pop("Identificação")
            
            if isinstance(id_properties, dict):
                for key, value in id_properties.items():
                    # Adding a prefix to avoid potential property name conflicts
                    data_dict_primitives[f"Identificação_{key}"] = value
            else:
                # If it's not a dictionary, then perhaps store it as a single property (optional)
                data_dict_primitives["Identificação_value"] = id_properties

        with self._driver.session() as session:
            query = f"MERGE (node:{label} {{lattes_id: $lattes_id}}) SET node = $props"
            session.run(query, lattes_id=lattes_id, props=data_dict_primitives)

    def update_data(self, node_id, data_dict):
        data_dict_primitives = self.convert_to_primitives(data_dict)
        with self._driver.session() as session:
            query = f"MATCH (node) WHERE id(node) = {node_id} SET node += $props"
            session.run(query, props=data_dict_primitives)

    def parse_area(self, area_string):
        """Parses the area string and returns a dictionary with the parsed fields."""
        parts = area_string.split(" / ")
        area_data = {}
        for part in parts:
            # Separating key and value by the last colon found
            key_value = part.rsplit(':', 1)
            if len(key_value) == 2:
                key, value = key_value
                area_data[key.strip()] = value.strip()
        return area_data

    def process_all_person_nodes(self):
        """Iterates over all Person nodes and persists secondary nodes and relationships."""
        with self._driver.session() as session:
            result = session.run("MATCH (p:Person) RETURN p.name AS name, p.`Áreas de atuação` AS areas")

            for record in result:
                person_name = record["name"]
                
                # Check if name or areas is None
                if person_name is None or record["Áreas"] is None:
                    print(f"Skipping record for name {person_name} due to missing name or areas.")
                    continue

                # Check if the areas data is already in dict form
                if isinstance(record["Áreas"], dict):
                    areas = record["Áreas"]
                else:
                    # Attempt to convert from a string representation (e.g., JSON)
                    try:
                        areas = json.loads(record["Áreas"])
                    except Exception as e:
                        print(f"Failed to parse areas for name {person_name}. Error: {e}")
                        continue
                
                self.persist_secondary_nodes(person_name, areas)