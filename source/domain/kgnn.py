import torch
from py2neo import Graph
from sentence_transformers import SentenceTransformer


class KGNN(torch.nn.Module):

    def __init__(self, embedding_model_name, neo4j_uri, neo4j_user, neo4j_password):
        super().__init__()

        # Inicializa o modelo de embedding
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Inicializa a conexão com o Neo4j
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def criar_subgrafo_curriculo(self, curriculo_dict):
        """
        Cria um subgrafo para um currículo, incluindo suas informações e 
        relacionamentos com outras entidades.

        Args:
            curriculo_dict: Um dicionário contendo as informações do currículo.

        Returns:
            Um dicionário contendo as informações do subgrafo, com os nós e as arestas.
        """
        subgrafo = {"nos": [], "arestas": []}

        # --- Adicionar o nó do currículo ---
        curriculo_id = curriculo_dict['Identificação']['ID Lattes']
        subgrafo["nos"].append({"tipo": "Curriculo", "propriedades": curriculo_dict['Identificação']})

        # --- Adicionar nós e arestas para os artigos ---
        artigos = curriculo_dict.get('Produções', {}).get('Artigos completos publicados em periódicos', [])
        for artigo in artigos:
            artigo_id = artigo.get('DOI')  # Usando o DOI como ID do artigo
            if artigo_id:
                subgrafo["nos"].append({"tipo": "Artigo", "propriedades": artigo})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"DOI": artigo_id}, "tipo": "PUBLICOU_ARTIGO"})

        # --- Adicionar nós e arestas para as áreas de atuação ---
        areas = curriculo_dict.get('Áreas', {})
        for area_id, area_descricao in areas.items():
            subgrafo["nos"].append({"tipo": "Area", "propriedades": {"id": area_id, "descricao": area_descricao}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": area_id}, "tipo": "PESQUISA_AREA"})

        # --- Adicionar nós e arestas para a formação acadêmica ---
        formacao = curriculo_dict.get('Formação', {}).get('Acadêmica', [])
        for item in formacao:
            formacao_id = item.get('Descrição')  # Usando a Descrição como ID da formação
            if formacao_id:
                subgrafo["nos"].append({"tipo": "FormacaoAcademica", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"Descrição": formacao_id}, "tipo": "POSSUI_FORMACAO"})

        # --- Adicionar nós e arestas para o pós-doutorado ---
        posdoc = curriculo_dict.get('Formação', {}).get('Pos-Doc', [])
        for item in posdoc:
            posdoc_id = item.get('Descrição')  # Usando a Descrição como ID do pós-doutorado
            if posdoc_id:
                subgrafo["nos"].append({"tipo": "PosDoutorado", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"Descrição": posdoc_id}, "tipo": "POSSUI_POSDOC"})

        # --- Adicionar nós e arestas para a formação complementar ---
        formacao_complementar = curriculo_dict.get('Formação', {}).get('Complementar', [])
        for item in formacao_complementar:
            complementar_id = item.get('Descrição')  # Usando a Descrição como ID da formação complementar
            if complementar_id:
                subgrafo["nos"].append({"tipo": "FormacaoComplementar", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"Descrição": complementar_id}, "tipo": "POSSUI_COMPLEMENTAR"})

        # --- Adicionar nós e arestas para a atuação profissional ---
        atuacao_profissional = curriculo_dict.get('Atuação Profissional', [])
        for item in atuacao_profissional:
            atuacao_id = item.get('Instituição') + " - " + item.get('Ano')  # Usando a Instituição e o Ano como ID da atuação
            if atuacao_id:
                subgrafo["nos"].append({"tipo": "AtuacaoProfissional", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": atuacao_id}, "tipo": "POSSUI_VINCULO"})

        # --- Adicionar nós e arestas para as linhas de pesquisa ---
        linhas_de_pesquisa = curriculo_dict.get('Linhas de Pesquisa', [])
        for item in linhas_de_pesquisa:
            pesquisa_id = item.get('Descrição')  # Usando a Descrição como ID da linha de pesquisa
            if pesquisa_id:
                subgrafo["nos"].append({"tipo": "LinhaPesquisa", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"Descrição": pesquisa_id}, "tipo": "PESQUISA_LINHA"})

        # --- Adicionar nós e arestas para os idiomas ---
        idiomas = curriculo_dict.get('Idiomas', [])
        for item in idiomas:
            idioma_id = item.get('Idioma')  # Usando o Idioma como ID do idioma
            if idioma_id:
                subgrafo["nos"].append({"tipo": "Idioma", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"Idioma": idioma_id}, "tipo": "DOMINA_IDIOMA"})

        # Livros publicados/organizados ou edições
        livros = curriculo_dict.get('Produções', {}).get('Livros publicados/organizados ou edições', {})
        for livro_id, livro_descricao in livros.items():
            subgrafo["nos"].append({"tipo": "Livro", "propriedades": {"id": livro_id, "descricao": livro_descricao}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": livro_id}, "tipo": "PUBLICOU_LIVRO"})
        
        # Capítulos de livros publicados
        capitulos = curriculo_dict.get('Produções', {}).get('Capítulos de livros publicados', {})
        for capitulo_id, capitulo_descricao in capitulos.items():
            subgrafo["nos"].append({"tipo": "CapituloLivro", "propriedades": {"id": capitulo_id, "descricao": capitulo_descricao}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": capitulo_id}, "tipo": "PUBLICOU_CAPITULO"})

        # Resumos publicados em anais de congressos
        resumos_congressos = curriculo_dict.get('Produções', {}).get('Resumos publicados em anais de congressos', {})
        for resumo_id, resumo_descricao in resumos_congressos.items():
            subgrafo["nos"].append({"tipo": "ResumoCongresso", "propriedades": {"id": resumo_id, "descricao": resumo_descricao}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": resumo_id}, "tipo": "PUBLICOU_RESUMO_CONGRESSO"})

        # Apresentações de Trabalho
        apresentacoes = curriculo_dict.get('Produções', {}).get('Apresentações de Trabalho', {})
        for apresentacao_id, apresentacao_descricao in apresentacoes.items():
            subgrafo["nos"].append({"tipo": "ApresentacaoTrabalho", "propriedades": {"id": apresentacao_id, "descricao": apresentacao_descricao}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": apresentacao_id}, "tipo": "APRESENTOU_TRABALHO"})

        # Outras produções bibliográficas
        outras_producoes = curriculo_dict.get('Produções', {}).get('Outras produções bibliográficas', {})
        for producao_id, producao_descricao in outras_producoes.items():
            subgrafo["nos"].append({"tipo": "ProducaoBibliografica", "propriedades": {"id": producao_id, "descricao": producao_descricao}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": producao_id}, "tipo": "PUBLICOU_PRODUCAO"})

        # Entrevistas, mesas redondas, programas e comentários na mídia
        entrevistas = curriculo_dict.get('Produções', {}).get('Entrevistas, mesas redondas, programas e comentários na mídia', {})
        for entrevista_id, entrevista_descricao in entrevistas.items():
            subgrafo["nos"].append({"tipo": "Entrevista", "propriedades": {"id": entrevista_id, "descricao": entrevista_descricao}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": entrevista_id}, "tipo": "PARTICIPOU_ENTREVISTA"})

        # Demais tipos de produção técnica
        demais_producoes_tecnicas = curriculo_dict.get('Produções', {}).get('Demais tipos de produção técnica', {})
        for producao_tecnica_id, producao_tecnica_descricao in demais_producoes_tecnicas.items():
            subgrafo["nos"].append({"tipo": "ProducaoTecnica", "propriedades": {"id": producao_tecnica_id, "descricao": producao_tecnica_descricao}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": producao_tecnica_id}, "tipo": "PRODUZIU_TECNICA"})

        # --- Nós e Arestas para os projetos (pesquisa, extensão, etc.) ---
        # Projetos de Pesquisa
        projetos_pesquisa = curriculo_dict.get('ProjetosPesquisa', [])
        for projeto in projetos_pesquisa:
            projeto_id = projeto.get('chave')  # Usando a chave como ID do projeto
            if projeto_id:
                subgrafo["nos"].append({"tipo": "ProjetoPesquisa", "propriedades": projeto})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"chave": projeto_id}, "tipo": "PARTICIPOU_PROJETO_PESQUISA"})

        # Projetos de Extensão
        projetos_extensao = curriculo_dict.get('ProjetosExtensão', [])
        for projeto in projetos_extensao:
            projeto_id = projeto.get('chave')  # Usando a chave como ID do projeto
            if projeto_id:
                subgrafo["nos"].append({"tipo": "ProjetoExtensao", "propriedades": projeto})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"chave": projeto_id}, "tipo": "PARTICIPOU_PROJETO_EXTENSAO"})

        # Projetos de Desenvolvimento
        projetos_desenvolvimento = curriculo_dict.get('ProjetosDesenvolvimento', [])
        for projeto in projetos_desenvolvimento:
            projeto_id = projeto.get('chave')  # Usando a chave como ID do projeto
            if projeto_id:
                subgrafo["nos"].append({"tipo": "ProjetoDesenvolvimento", "propriedades": projeto})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"chave": projeto_id}, "tipo": "PARTICIPOU_PROJETO_DESENVOLVIMENTO"})

        # Projetos Outros
        projetos_outros = curriculo_dict.get('ProjetosOutros', [])
        for projeto in projetos_outros:
            projeto_id = projeto.get('chave')  # Usando a chave como ID do projeto
            if projeto_id:
                subgrafo["nos"].append({"tipo": "ProjetoOutro", "propriedades": projeto})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"chave": projeto_id}, "tipo": "PARTICIPOU_PROJETO_OUTRO"})

        # --- Nós e arestas para patentes e registros ---
        patentes = curriculo_dict.get('Patentes e registros', {})
        for patente_id, patente_info in patentes.items():
            # Considerando que cada patente_info é um dicionário com informações da patente
            subgrafo["nos"].append({"tipo": "Patente", "propriedades": patente_info})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": patente_id}, "tipo": "POSSUI_PATENTE"})

        # --- Nós e arestas para bancas e Orientações---
        bancas = curriculo_dict.get('Bancas', {})
        
        # Participação em bancas de trabalhos de conclusão
        bancas_trabalhos = bancas.get('Participação em bancas de trabalhos de conclusão', {})
        for banca_id, banca_info in bancas_trabalhos.items():
            subgrafo["nos"].append({"tipo": "BancaTrabalho", "propriedades": {"id": banca_id, "descricao": banca_info}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": banca_id}, "tipo": "PARTICIPOU_BANCA_TRABALHO"})

        # Participação em bancas de comissões julgadoras
        bancas_comissoes = bancas.get('Participação em bancas de comissões julgadoras', {})
        for banca_id, banca_info in bancas_comissoes.items():
            subgrafo["nos"].append({"tipo": "BancaComissao", "propriedades": {"id": banca_id, "descricao": banca_info}})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": banca_id}, "tipo": "PARTICIPOU_BANCA_COMISSAO"})

        # Orientações
        orientacoes = curriculo_dict.get('Orientações', [])
        for orientacao in orientacoes:
            orientacao_id = orientacao.get('nome')  # Usando o nome como ID da orientação
            if orientacao_id:
                subgrafo["nos"].append({"tipo": "Orientacao", "propriedades": orientacao})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"nome": orientacao_id}, "tipo": "ORIENTADOR"})

        # --- Nós e arestas para Fator de Impacto JCR---
        # JCR2
        jcr2 = curriculo_dict.get('JCR2', [])
        for item in jcr2:
            jcr2_id = item.get('doi')  # Usando o DOI como ID do JCR2
            if jcr2_id:
                subgrafo["nos"].append({"tipo": "JCR2", "propriedades": item})
                subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"doi": jcr2_id}, "tipo": "POSSUI_JCI"})

        return subgrafo


    def ingerir_subgrafo(self, subgrafo_dict):
        """
        Ingere um subgrafo no grafo de conhecimento do Neo4j.

        Args:
            subgrafo_dict: Um dicionário contendo as informações do subgrafo,
                           com os nós e as arestas.
        """
        # Extrair nós e arestas do subgrafo
        nos = subgrafo_dict.get("nos", [])
        arestas = subgrafo_dict.get("arestas", [])

        # Adicionar nós ao Neo4j
        for no in nos:
            propriedades = no.get("propriedades", {})
            tipo_no = no.get("tipo")
            query = f"""
                MERGE (n:{tipo_no} {{ {self._formatar_propriedades(propriedades)} }})
                RETURN n
            """
            self.graph.run(query)

        # Adicionar arestas ao Neo4j
        for aresta in arestas:
            no_origem = aresta.get("origem")
            no_destino = aresta.get("destino")
            tipo_aresta = aresta.get("tipo")
            propriedades = aresta.get("propriedades", {})
            query = f"""
                MATCH (n1 {{ {self._formatar_propriedades(no_origem)} }})
                MATCH (n2 {{ {self._formatar_propriedades(no_destino)} }})
                MERGE (n1)-[r:{tipo_aresta} {{ {self._formatar_propriedades(propriedades)} }}]->(n2)
                RETURN r
            """
            self.graph.run(query)


    def _formatar_propriedades(self, propriedades):
        """
        Formata as propriedades para a query Cypher.
        """
        propriedades_formatadas = []
        for chave, valor in propriedades.items():
            if isinstance(valor, str):
                valor = f'"{valor}"'
            propriedades_formatadas.append(f"{chave}: {valor}")
        return ", ".join(propriedades_formatadas)


    def gerar_embeddings(self, texto):
        """
        Gera embeddings para um texto usando o modelo SentenceTransformer.

        Args:
            texto: O texto a ser usado para gerar o embedding.

        Returns:
            Um tensor PyTorch com o embedding.
        """
        return self.embedding_model.encode(texto, convert_to_tensor=True)


    def forward(self, x):
        """
        Define o forward pass do KGNN.

        Args:
            x: Os dados de entrada.

        Returns:
            O resultado do forward pass.
        """
        # Implementar a lógica do forward pass aqui
        # ...
        return x