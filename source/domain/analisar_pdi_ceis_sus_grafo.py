import json
import spacy
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import r2_score
import networkx as nx
import numpy as np
import cupy as cp  # Se disponível

# Carregar o modelo de linguagem português do spaCy
nlp = spacy.load("pt_core_news_lg")

# Carregar o modelo de similaridade semântica
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Conectar ao banco de dados Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

class ProcessamentoLinguagemNatural:
    """
    Classe para processamento de linguagem natural em questões de pesquisa.
    """

    @staticmethod
    def processar_questao_pesquisa(questao):
        """
        Processa a questão de pesquisa do usuário utilizando PLN e retorna entidades e relacionamentos relevantes.
        Utilizando de paradigma de orientação a objetos com compreensão de listas (list comprehensions).
        """

        # Analisar a questão com spaCy
        doc = nlp(questao)

        # Extrair entidades e relacionamentos (adapte as regras conforme necessário)
        entidades = [ent.text for ent in doc.ents]
        relacionamentos = [(child.text, child.dep_) for token in doc for child in token.children if child.dep_ in ["nsubj", "dobj"]]

        return entidades, relacionamentos

    def processar_questao_pesquisa_func(questao):
        """
        Processa a questão de pesquisa do usuário utilizando PLN e retorna entidades e  relacionamentos relevantes.
        Utilizando de paradigma de orientação a objetos com funções de alta ordem como map e filter.
        """

        doc = nlp(questao)

        entidades = list(map(lambda ent: ent.text, doc.ents))
        relacionamentos = list(filter(
            lambda child: child.dep_ in ["nsubj", "dobj"], 
            (child for token in doc for child in token.children)
        ))

        return entidades, relacionamentos

class InteracaoUsuario:
    """
    Classe para interagir com o pesquisador e refinar a consulta.
    """

    def __init__(self):
        self.entidades = []
        self.relacionamentos = []

    def interagir_com_pesquisador(self):
        """
        Realiza a interação com o pesquisador, permitindo que ele refine sua consulta.
        """

        entidades = []
        relacionamentos = []

        while True:
            questao = input("Digite sua questão de pesquisa ou 'sair' para encerrar: ")
            if questao.lower() == 'sair':
                break

            if not self.entidades and not self.relacionamentos:  # Primeira iteração
                entidades, relacionamentos = processar_questao_pesquisa(questao)
            else:
                novas_entidades, novos_relacionamentos = processar_questao_pesquisa(questao)
                entidades.extend(novas_entidades)
                relacionamentos.extend(novos_relacionamentos)

            print("\nEntidades identificadas:")
            for entidade in entidades:
                print(f"- {entidade}")

            print("\nRelacionamentos identificados:")
            for relacionamento in relacionamentos:
                print(f"- {relacionamento[0]} ({relacionamento[1]})")

            consulta = gerar_consulta_neo4j(entidades, relacionamentos)
            print("\nConsulta gerada:")
            print(consulta)

            # Executar a consulta no Neo4j e apresentar os resultados
            with driver.session() as session:
                resultado = session.run(consulta)

                # Verificar se há resultados
                if resultado.peek() is None:
                    print("\nNenhum resultado encontrado para a consulta.")
                else:
                    print("\nResultados da consulta:")
                    for registro in resultado:
                        # Apresentar os nós e relacionamentos encontrados
                        for node in registro.values():
                            if isinstance(node, Node):
                                print(f"Nó: {node['nome']} (tipo: {list(node.labels)[0]})")
                            elif isinstance(node, Relationship):
                                print(f"Relacionamento: {node.start_node['nome']} -[{node.type}]-> {node.end_node['nome']}")

            while True:
                resposta = input("\nDeseja explorar mais detalhes sobre alguma entidade ou relacionamento, ou realizar uma busca entre entidades? (s/n): ")
                if resposta.lower() != "s":
                    break

                opcao = input("Digite 'detalhar' para explorar uma entidade/relacionamento existente, ou 'buscar' para realizar uma busca entre entidades, ou 'adicionar' para sugerir novas entidades/relacionamentos: ")

                if opcao.lower() == 'detalhar':
                    print("Entidades e relacionamentos atuais:")
                    for i, entidade in enumerate(entidades):
                        print(f"{i+1}. Entidade: {entidade}")
                    for i, relacionamento in enumerate(relacionamentos):
                        print(f"{i+len(entidades)+1}. Relacionamento: {relacionamento[0]} ({relacionamento[1]})")

                    indice_elemento = int(input("Digite o número do elemento a ser explorado: ")) - 1

                    if indice_elemento < len(entidades):
                        entidade_selecionada = entidades[indice_elemento]
                        # Explosão do subgrafo
                        subgrafo = ExplosaoSubgrafos.recuperar_subgrafo(entidade_selecionada)
                        ExplosaoSubgrafos.apresentar_subgrafo(subgrafo)

                        # Recomendações (opcional, dependendo da sua implementação)
                        recomendacoes = RecomendacaoProjetos.recomendar_projetos(entidades, relacionamentos, subgrafo)
                        for recomendacao in recomendacoes:
                            print(f"- {recomendacao['projeto']}")
                            print(f"  Justificativa: {recomendacao['justificativa']}")
                            print(f"  Fontes: {recomendacao['fontes']}\n")

                    else:
                        relacionamento_selecionado = relacionamentos[indice_elemento - len(entidades)]
                        # ... (caso necessário tratar o relacionamento selecionado implementar detalhar relacionamento a partir daqui)
                        print(f"Relacionamento selecionado:\n {relacionamento_selecionado}")
                        print()

                elif opcao.lower() == 'buscar':
                    print("Entidades atuais:")
                    for i, entidade in enumerate(entidades):
                        print(f"{i+1}. {entidade}")

                    indice_inicio = int(input("Digite o número da entidade de início da busca: ")) - 1
                    indice_fim = int(input("Digite o número da entidade de fim da busca: ")) - 1

                    entidade_inicio = entidades[indice_inicio]
                    entidade_fim = entidades[indice_fim]

                    # Consulta para encontrar caminhos entre as entidades
                    consulta_caminhos = f"""
                        MATCH path = (inicio:Entidade {{nome: '{entidade_inicio}'}})-[*]-(fim:Entidade {{nome: '{entidade_fim}'}})
                        RETURN path
                    """

                    with driver.session() as session:
                        resultado_caminhos = session.run(consulta_caminhos)

                        # Apresentar os caminhos encontrados
                        if resultado_caminhos.peek() is None:
                            print("\nNenhum caminho encontrado entre as entidades.")
                        else:
                            print("\nCaminhos encontrados:")
                            for registro in resultado_caminhos:
                                caminho = registro["path"]
                                print(caminho)

                elif opcao.lower() == 'adicionar':
                    questao_adicional = input("Digite a questão de pesquisa que as novas entidades/relacionamentos devem abordar: ")
                    novas_entidades, novos_relacionamentos = ProcessamentoLinguagemNatural.processar_questao_pesquisa(questao_adicional)

                    # Apresentar as sugestões ao usuário para confirmação
                    print("\nSugestões de entidades:")
                    for entidade in novas_entidades:
                        print(f"- {entidade}")

                    print("\nSugestões de relacionamentos:")
                    for relacionamento in novos_relacionamentos:
                        print(f"- {relacionamento[0]} ({relacionamento[1]})")

                    confirmacao = input("\nDeseja adicionar estas sugestões ao modelo? (s/n): ")
                    if confirmacao.lower() == 's':
                        # Persistir as sugestões no Neo4j
                        with driver.session() as session:
                            for entidade in novas_entidades:
                                session.run("""
                                    MERGE (e:SugestaoEntidade {nome: $nome})
                                    ON CREATE SET e.data_criacao = datetime()
                                """, nome=entidade)

                            for relacionamento in novos_relacionamentos:
                                sujeito, tipo_relacionamento = relacionamento
                                session.run("""
                                    MERGE (s:SugestaoEntidade {nome: $sujeito})
                                    ON CREATE SET s.data_criacao = datetime()
                                    MERGE (r:SugestaoRelacionamento {tipo: $tipo})
                                    ON CREATE SET r.data_criacao = datetime()
                                    MERGE (s)-[:TEM_SUGESTAO]->(r)
                                """, sujeito=sujeito, tipo=tipo_relacionamento)

                        print("Sugestões de entidades e relacionamentos adicionadas ao banco de dados.")
                    else:
                        print("Sugestões descartadas.")

                    break 

                else:
                    print("Opção inválida. Digite 'detalhar', 'buscar' ou 'adicionar'.")

    def gerar_consulta_neo4j(self):
        """
        Gera uma consulta Cypher para o Neo4j com base nas entidades e relacionamentos extraídos.
        """
        # Construir a consulta (adaptar a lógica conforme a estrutura do modelo grafo em uso)
        match_clause = "MATCH "
        where_clause = "WHERE "
        for entidade in self.entidades:
            match_clause += f"(n{self.entidades.index(entidade)}:Entidade {{nome: '{entidade}'}})"
            if entidade != self.entidades[-1]:
                match_clause += ", "

        for relacionamento in self.relacionamentos:
            sujeito, tipo_relacionamento = relacionamento
            match_clause += f"-[r{self.relacionamentos.index(relacionamento)}:{tipo_relacionamento}]->"

        consulta = match_clause + " " + where_clause + " RETURN *"

        return consulta

    def gerar_consulta_neo4j_func(entidades, relacionamentos):
        """
        Gera uma consulta Cypher para o Neo4j com base nas entidades e relacionamentos extraídos.
        Versão com estilo mais funcional.
        """

        match_clause = "MATCH " + ", ".join(
            f"(n{i}:Entidade {{nome: '{entidade}'}})" for i, entidade in enumerate(entidades)
        )

        for i, relacionamento in enumerate(relacionamentos):
            sujeito, tipo_relacionamento = relacionamento
            match_clause += f"-[r{i}:{tipo_relacionamento}]->"

        consulta = match_clause + " WHERE " + " AND ".join(
            f"n{i}.nome = '{entidade}'" for i, entidade in enumerate(entidades)
        ) + " RETURN *"

        return consulta

class AnaliseGrafo:
    """
    Classe para analisar características do grafo Neo4j.
    """

    @staticmethod
    def calcular_densidade_grafo(grafo_neo4j):
        """
        Calcula a densidade do grafo Neo4j.
        """

        with driver.session() as session:
            resultado = session.run("""
                MATCH (n) 
                WITH count(n) AS num_nodes, 
                     sum(size((n)--()))/2 AS num_edges 
                RETURN toFloat(num_edges) / (toFloat(num_nodes) * (toFloat(num_nodes) - 1) / 2) AS density
            """)
            densidade = resultado.single()["density"]

        return densidade

    @staticmethod
    def analisar_distribuicao_graus(grafo_neo4j):
        """
        Analisa a distribuição de graus do grafo Neo4j e retorna "power_law" se a distribuição seguir
        aproximadamente uma lei de potência, ou "other" caso contrário.
        """

        with driver.session() as session:
            resultado = session.run("MATCH (n) RETURN size((n)--()) AS degree")
            graus = [registro["degree"] for registro in resultado]

        # Converter para NumPy array para cálculos mais rápidos
        graus_np = np.array(graus)

        # Opcional: usar CuPy para cálculos em GPU, se disponível
        if cp.cuda.is_available():
            graus_np = cp.asarray(graus_np)

        # Calcular histograma e analisar a distribuição
        histograma, _ = np.histogram(graus_np, bins='auto')

        # Filtrar zeros do histograma e calcular logaritmos
        graus_nao_zero = np.where(histograma > 0)[0]
        log_graus = np.log10(graus_nao_zero + 1)  # Adicionar 1 para evitar log(0)
        log_freqs = np.log10(histograma[graus_nao_zero])

        # Ajustar uma reta aos dados em escala log-log
        coeficientes = np.polyfit(log_graus, log_freqs, 1)

        # Calcular o coeficiente de determinação R²
        y_pred = np.polyval(coeficientes, log_graus)
        r2 = r2_score(log_freqs, y_pred)

        # Definir um limiar para o R² e retornar True ou False
        limiar_r2 = 0.9  # Ajuste conforme necessário
        if r2 >= limiar_r2:
            return "power_law"
        else:
            return "other"

    @staticmethod
    def visualizar_distribuicao_graus(grafo_neo4j):
        """
        Visualiza a distribuição de graus do grafo Neo4j em um histograma.
        """

        with driver.session() as session:
            resultado = session.run("MATCH (n) RETURN size((n)--()) AS degree")
            graus = [registro["degree"] for registro in resultado]

        # Criar um DataFrame para o Altair
        import pandas as pd
        df = pd.DataFrame({'Grau': graus})

        # Criar o histograma
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Grau:Q', bin=True, title='Grau do Nó'),
            y=alt.Y('count()', title='Frequência'),
            tooltip = ['Grau', 'count()']
        ).properties(
            title='Distribuição de Graus do Grafo'
        ).interactive()

        # Salvar o gráfico em um arquivo JSON
        chart.save('distribuicao_graus.json')        

import torch

class MapeamentoEntidades:
    """
    Classe responsável por mapear as entidades extraídas pela PLN para o modelo de grafo 
    e sugerir novas entidades quando necessário.
    """

    @staticmethod
    def mapear_entidades_gpu(entidades, grafo_neo4j):
        """
        Mapeia as entidades extraídas para o modelo de grafo e identifica entidades não encontradas.

        Args:
            entidades: Lista de entidades extraídas pela PLN.
            grafo_neo4j: Objeto representando o grafo no Neo4j.

        Returns:
            Uma tupla contendo duas listas:
            - entidades_mapeadas: Entidades encontradas no grafo.
            - entidades_nao_encontradas: Entidades não encontradas no grafo.
        """

        entidades_mapeadas = []
        entidades_nao_encontradas = []

        with driver.session() as session:
            resultados = session.run("UNWIND $entidades AS entidade MATCH (n:Entidade {nome: entidade}) RETURN n", entidades=entidades)
            entidades_encontradas = [registro["n"]["nome"] for registro in resultados if registro["n"] is not None]

        entidades_mapeadas = [entidade for entidade in entidades if entidade in entidades_encontradas]
        entidades_nao_encontradas = [entidade for entidade in entidades if entidade not in entidades_encontradas]

        return entidades_mapeadas, entidades_nao_encontradas

    @staticmethod
    def sugerir_novas_entidades_gpu(entidades_nao_encontradas):
        """
        Sugere novas entidades relacionadas às entidades não encontradas, com base em 
        similaridade semântica com entidades existentes no grafo.

        Args:
            entidades_nao_encontradas: Lista de entidades não encontradas no grafo.

        Returns:
            Uma lista de sugestões de novas entidades.
        """

        sugestoes = []

        # Obter todas as entidades existentes no grafo
        with driver.session() as session:
            resultado = session.run("MATCH (n:Entidade) RETURN n.nome AS nome")
            entidades_existentes = [registro["nome"] for registro in resultado]

        # Calcular embeddings das entidades não encontradas
        embeddings_nao_encontradas = model.encode(entidades_nao_encontradas, convert_to_tensor=True)

        # Calcular embeddings das entidades existentes
        embeddings_existentes = model.encode(entidades_existentes, convert_to_tensor=True)

        # Calcular similaridade de cosseno
        cosine_similarities = util.cos_sim(embeddings_nao_encontradas, embeddings_existentes)

        # Obter as 3 entidades mais similares para cada entidade não encontrada
        for i, entidade in enumerate(entidades_nao_encontradas):
            top_indices = torch.topk(cosine_similarities[i], k=3).indices
            entidades_similares = [entidades_existentes[j] for j in top_indices]
            sugestoes.append({
                "entidade_nao_encontrada": entidade,
                "sugestoes": entidades_similares
            })

        return sugestoes

    @staticmethod
    def mapear_entidades(entidades, grafo_neo4j):
        """
        Mapeia as entidades extraídas para o modelo de grafo e identifica entidades não encontradas.

        Args:
            entidades: Lista de entidades extraídas pela PLN.
            grafo_neo4j: Objeto representando o grafo no Neo4j.

        Returns:
            Uma tupla contendo duas listas:
            - entidades_mapeadas: Entidades encontradas no grafo.
            - entidades_nao_encontradas: Entidades não encontradas no grafo.
        """

        entidades_mapeadas = []
        entidades_nao_encontradas = []

        with driver.session() as session:
            for entidade in entidades:
                resultado = session.run("MATCH (n:Entidade {nome: $nome}) RETURN n", nome=entidade)
                if resultado.peek() is not None:
                    entidades_mapeadas.append(entidade)
                else:
                    entidades_nao_encontradas.append(entidade)

        return entidades_mapeadas, entidades_nao_encontradas

    @staticmethod
    def sugerir_novas_entidades(entidades_nao_encontradas):
        """
        Sugere novas entidades relacionadas às entidades não encontradas, com base em 
        similaridade semântica com entidades existentes no grafo.

        Args:
            entidades_nao_encontradas: Lista de entidades não encontradas no grafo.

        Returns:
            Uma lista de sugestões de novas entidades.
        """

        sugestoes = []

        # Obter todas as entidades existentes no grafo
        with driver.session() as session:
            resultado = session.run("MATCH (n:Entidade) RETURN n.nome AS nome")
            entidades_existentes = [registro["nome"] for registro in resultado]

        # Calcular embeddings das entidades não encontradas
        embeddings_nao_encontradas = model.encode(entidades_nao_encontradas, convert_to_tensor=True)

        # Calcular embeddings das entidades existentes
        embeddings_existentes = model.encode(entidades_existentes, convert_to_tensor=True)

        # Calcular similaridade de cosseno
        cosine_similarities = util.cos_sim(embeddings_nao_encontradas, embeddings_existentes)

        # Obter as 3 entidades mais similares para cada entidade não encontrada
        for i, entidade in enumerate(entidades_nao_encontradas):
            # Usando NumPy para obter os índices das 3 maiores similaridades
            top_indices = np.argpartition(cosine_similarities[i], -3)[-3:] 
            entidades_similares = [entidades_existentes[j] for j in top_indices]
            sugestoes.append({
                "entidade_nao_encontrada": entidade,
                "sugestoes": entidades_similares
            })

        return sugestoes

    @staticmethod
    def sugerir_novas_entidades(entidades_nao_encontradas):
        """
        Sugere novas entidades relacionadas às entidades não encontradas, com base em 
        similaridade semântica com entidades existentes no grafo.

        Args:
            entidades_nao_encontradas: Lista de entidades não encontradas no grafo.

        Returns:
            Uma lista de sugestões de novas entidades.
        """

        sugestoes = []

        # Obter todas as entidades existentes no grafo
        with driver.session() as session:
            resultado = session.run("MATCH (n:Entidade) RETURN n.nome AS nome")
            entidades_existentes = [registro["nome"] for registro in resultado]

        # Calcular embeddings das entidades não encontradas
        embeddings_nao_encontradas = model.encode(entidades_nao_encontradas, convert_to_tensor=True)

        # Calcular embeddings das entidades existentes
        embeddings_existentes = model.encode(entidades_existentes, convert_to_tensor=True)

        # Calcular similaridade de cosseno
        cosine_similarities = util.cos_sim(embeddings_nao_encontradas, embeddings_existentes)

        # Obter as 3 entidades mais similares para cada entidade não encontrada
        for i, entidade in enumerate(entidades_nao_encontradas):
            top_indices = torch.topk(cosine_similarities[i], k=3).indices
            entidades_similares = [entidades_existentes[j] for j in top_indices]
            sugestoes.append({
                "entidade_nao_encontrada": entidade,
                "sugestoes": entidades_similares
            })

        return sugestoes


import networkx as nx

class ExplosaoSubgrafos:
    """
    Classe responsável por realizar a consulta ao Neo4j para recuperar os subgrafos 
    filhos de uma entidade e apresentá-los ao usuário.
    """

    @staticmethod
    def recuperar_subgrafo(entidade_selecionada, profundidade=1):
        """
        Recupera o subgrafo da entidade selecionada até a profundidade especificada.

        Args:
            entidade_selecionada: Nome da entidade a ser expandida.
            profundidade: Profundidade da busca no grafo (padrão: 1).

        Returns:
            Um objeto DiGraph do NetworkX representando o subgrafo recuperado.
        """

        with driver.session() as session:
            resultado = session.run(f"""
                MATCH path = (n:Entidade {{nome: '{entidade_selecionada}'}})-[*..{profundidade}]-(m)
                RETURN path
            """)

            # Criar um grafo direcionado
            subgrafo = nx.DiGraph()

            # Iterar sobre os caminhos e adicionar nós e arestas ao grafo
            for registro in resultado:
                caminho = registro["path"]
                for i in range(len(caminho) - 1):
                    node1 = caminho[i]
                    node2 = caminho[i + 1]
                    edge = caminho[i + 1].relationships[0]

                    subgrafo.add_node(node1.id, nome=node1['nome'], tipo=list(node1.labels)[0])
                    subgrafo.add_node(node2.id, nome=node2['nome'], tipo=list(node2.labels)[0])
                    subgrafo.add_edge(node1.id, node2.id, tipo=edge.type)

        return subgrafo

    @staticmethod
    def apresentar_subgrafo(subgrafo):
        """
        Apresenta o subgrafo ao usuário de forma textual.

        Args:
            subgrafo: Objeto DiGraph representando o subgrafo a ser apresentado.
        """

        if subgrafo.number_of_nodes() > 0:
            print("\nSubgrafo:")
            for node_id in subgrafo.nodes:
                node_data = subgrafo.nodes[node_id]
                print(f"- Nó: {node_data['nome']} (tipo: {node_data['tipo']})")

            for edge in subgrafo.edges:
                edge_data = subgrafo.get_edge_data(*edge)
                print(f"- Relacionamento: {subgrafo.nodes[edge[0]]['nome']} -[{edge_data['tipo']}]-> {subgrafo.nodes[edge[1]]['nome']}")
        else:
            print("\nNenhum subgrafo encontrado para a entidade selecionada.")

    @staticmethod
    def apresentar_subgrafo_func(subgrafo):
        """
        Apresenta o subgrafo ao usuário de forma textual.
        """

        if subgrafo.number_of_nodes() > 0:
            print("\nSubgrafo:")
            print("\n".join(
                f"- Nó: {subgrafo.nodes[node_id]['nome']} (tipo: {subgrafo.nodes[node_id]['tipo']})" 
                for node_id in subgrafo.nodes
            ))

            print("\n".join(
                f"- Relacionamento: {subgrafo.nodes[edge[0]]['nome']} -[{subgrafo.get_edge_data(*edge)['tipo']}]-> {subgrafo.nodes[edge[1]]['nome']}"
                for edge in subgrafo.edges
            ))
        else:
            print("\nNenhum subgrafo encontrado para a entidade selecionada.")

class RecomendacaoProjetos:
    """
    Classe responsável por analisar o grafo e gerar recomendações explanáveis sobre 
    possíveis projetos de PDI, utilizando dados do grafo e fontes externas.
    """

    @staticmethod
    def recomendar_projetos(entidades, relacionamentos, subgrafo):
        """
        Gera recomendações de projetos de PDI com base nas entidades, relacionamentos e 
        subgrafo explorado pelo usuário.

        Args:
            entidades: Lista de entidades da consulta do usuário.
            relacionamentos: Lista de relacionamentos da consulta do usuário.
            subgrafo: Subgrafo explorado pelo usuário (objeto DiGraph do NetworkX).

        Returns:
            Uma lista de dicionários, cada um representando uma recomendação, 
            contendo "projeto", "justificativa" e "fontes".
        """

        recomendacoes = []

        # Lógica hipotética para gerar recomendações (adapte conforme suas necessidades)

        # Exemplo 1: Recomendação baseada em entidades chave
        entidades_chave = ["tecnologia X", "doença Y"]  # Adapte para suas entidades chave
        if any(entidade in entidades for entidade in entidades_chave):
            recomendacoes.append({
                "projeto": "Desenvolvimento de nova terapia para doença Y utilizando tecnologia X",
                "justificativa": "A consulta do usuário indica interesse em tecnologias para a doença Y, e a tecnologia X pode ser promissora nesse contexto.",
                "fontes": ["Artigo A sobre tecnologia X", "Banco de dados B sobre doença Y"]
            })

        # Exemplo 2: Recomendação baseada em relacionamentos específicos
        relacionamento_chave = ("entidade A", "colabora com")  # Adapte para seu relacionamento chave
        if relacionamento_chave in relacionamentos:
            recomendacoes.append({
                "projeto": "Fomentar colaboração entre entidade A e outras instituições para desenvolvimento de tecnologia Z",
                "justificativa": "A consulta do usuário destaca a importância da colaboração da entidade A, e o grafo indica potencial para parcerias em projetos de tecnologia Z.",
                "fontes": ["Relatório C sobre colaborações da entidade A", "Estudo D sobre oportunidades em tecnologia Z"]
            })

        # Exemplo 3: Recomendação baseada na análise do subgrafo (hipotético)
        if subgrafo is not None:
            # ... (analisar o subgrafo para identificar oportunidades de projetos)
            # ... (consultar fontes externas para enriquecer as recomendações)
            pass

        return recomendacoes


class FontesDados:
    """
    Classe ou módulo para gerenciar as fontes de dados externas e extrair 
    informações relevantes para as recomendações.
    """

    @staticmethod
    def consultar_fontes(termos_chave):
        """
        Consulta as fontes de dados externas relevantes com base nos termos-chave fornecidos.

        Args:
            termos_chave: Lista de termos-chave para a consulta.

        Returns:
            Uma lista de dicionários, cada um representando uma informação recuperada das fontes.
        """

        informacoes = []

        # Simulação de consulta a fontes externas (adapte para suas fontes reais)
        for termo in termos_chave:
            # Exemplo de consulta a uma base de dados hipotética
            if termo == "tecnologia X":
                informacoes.append({
                    "fonte": "Base de dados de patentes",
                    "informacao": "A tecnologia X tem potencial aplicação no tratamento da doença Y, conforme patente recente.",
                    "link": "https://www.exemplo.com/patente_tecnologia_x"
                })
            # Exemplo de consulta a uma API de notícias
            elif termo == "doença Y":
                informacoes.append({
                    "fonte": "API de notícias científicas",
                    "informacao": "Novo estudo identifica um biomarcador promissor para o diagnóstico precoce da doença Y."
                })

        return informacoes