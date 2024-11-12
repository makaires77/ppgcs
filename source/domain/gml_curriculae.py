import re
import torch

from torch import nn
from git import Repo
from typing import Union
from collections import defaultdict

# Importar a classe KGNN abstrata
from kgnn import KGNN

# Importar a classe CompetenceExtractor
from competence_extraction import CompetenceExtractor


class CurriculoKGNN(KGNN):
    def __init__(self, embedding_model_name, neo4j_uri, neo4j_user, neo4j_password, curricula_file):
        super().__init__(embedding_model_name, neo4j_uri, neo4j_user, neo4j_password, curriculae_path="_data/out_json/input_curriculos.json")
        self.competence_extractor = CompetenceExtractor(curricula_file, embedding_model_name)
        # Mapear tipo de nó para a propriedade de identificação
        self.propriedades_identificacao = {
            'Curriculo': 'ID Lattes',
            'AreaAtuacao': 'descricao',
            'FormacaoAcademica': 'descricao',
            'Projeto': 'descricao',
            'Publicacao': 'descricao',
            'Orientacao': 'descricao',
            # ... adicionar outros tipos de nós conforme objetivo ...
        }
        self.relacionamentos_multiplex = {
            'ATUACAO_PROFISSIONAL': 'descricao',
            'POSSUI_FORMACAO': 'descricao',
            'PARTICIPOU_PROJETO': 'descricao',
            'PUBLICOU': 'descricao',
            'ORIENTADOR': 'descricao',
        }


    def criar_subgrafo(self, curriculo_dict):
        """
        Cria um subgrafo para um currículo, utilizando a classe CompetenceExtractor 
        para extrair as informações relevantes e construir os nós e arestas.

        Args:
            curriculo_dict: Um dicionário contendo as informações do currículo.

        Returns:
            Um dicionário contendo as informações do subgrafo, com os nós e as arestas.
        """

        subgrafo = {"nos": [], "arestas": []}

        # --- Adicionar o nó do currículo ---
        curriculo_id = curriculo_dict['Identificação']['ID Lattes']
        # Remove espaços e caracteres especiais do ID Lattes
        curriculo_id = re.sub(r"[^a-zA-Z0-9_]", "", curriculo_id)        
        subgrafo["nos"].append({"tipo": "Curriculo", "propriedades": curriculo_dict['Identificação']})

        # --- Extrair competências usando a CompetenceExtractor ---
        competencias = self.competence_extractor.extract_competences(curriculo_dict)

        # --- Adicionar nós e arestas para cada tipo de competência ---
        for competencia in competencias:
            # Definir tipo de nó e propriedades com base no tipo de competência
            if competencia.startswith('AtuaçãoPrf:'):
                tipo_no = 'AreaAtuacao'
                propriedades = {"descricao": competencia.split(': ')[1]}
                tipo_aresta = 'ATUACAO_PROFISSIONAL'
            elif competencia.startswith('FormaçãoAc:'):
                tipo_no = 'FormacaoAcademica'
                propriedades = {"descricao": competencia.split(': ')[1]}
                tipo_aresta = 'POSSUI_FORMACAO'
            elif competencia.startswith('Projeto'):
                tipo_no = 'Projeto'
                propriedades = {"descricao": competencia}
                tipo_aresta = 'PARTICIPOU_PROJETO'
            elif competencia.startswith('Publicação:'):
                tipo_no = 'Publicacao'
                propriedades = {"descricao": competencia}
                tipo_aresta = 'PUBLICOU'
            elif competencia.startswith('Ori'):
                tipo_no = 'Orientacao'
                propriedades = {"descricao": competencia}
                tipo_aresta = 'ORIENTADOR'
            else:
                # Tratar outros tipos de competência, se necessário
                continue

            #competencia_id = hashlib.sha256(competencia.encode()).hexdigest()  # Gera um ID único para a competência
            competencia_id = competencia # Usando a competência como ID 
            subgrafo["nos"].append({"tipo": tipo_no, "propriedades": propriedades})
            subgrafo["arestas"].append({"origem": {"ID Lattes": curriculo_id}, "destino": {"id": competencia_id}, "tipo": tipo_aresta})

        return subgrafo


    def extrair_texto_no(self, no):
        """
        Extrai o texto de um nó, concatenando suas propriedades.

        Args:
            no: Um dicionário contendo as informações do nó.

        Returns:
            Uma string com o texto extraído do nó.
        """
        texto = ""
        propriedades = no['propriedades']
        for chave, valor in propriedades.items():
            if isinstance(valor, str):
                texto += valor + ' '
            elif isinstance(valor, list):
                texto += ' '.join(valor) + ' '
        return texto

    def extrair_texto_no_vizinho(self, no_vizinho: Union[str, list, dict]) -> str:
        """
        Extrai texto das propriedades do nó vizinho, 
        manuseando strings, listas e dicionários.

        Args:
            no_vizinho: As propriedades do nó vizinho.

        Returns:
            Uma string com o texto extraído.
        """
        if isinstance(no_vizinho, str):
            return no_vizinho + ' '
        elif isinstance(no_vizinho, list):
            return ' '.join([
                item if isinstance(item, str) else ' '.join([
                    str(v) for k, v in item.items()
                ])
                for item in no_vizinho
            ]) + ' '
        elif isinstance(no_vizinho, dict):
            return ' '.join([
                str(valor) for chave, valor in no_vizinho.items()
            ]) + ' '
        else:
            return ''
    
    def obter_embeddings_vizinhos(self, no_embedding, tipo_no, tipo_relacionamento):
        """
        Obtém os embeddings dos vizinhos de um nó através de um tipo de 
        relacionamento, usando um dicionário para mapear o tipo de nó 
        para a propriedade de identificação.

        Args:
            no_embedding: O embedding do nó.
            tipo_no: O tipo do nó.
            tipo_relacionamento: O tipo de relacionamento.

        Returns:
            Uma lista com os embeddings dos vizinhos.
        """

        # Obter a propriedade de identificação do nó
        propriedade_id = self.propriedades_identificacao.get(tipo_no)
        if not propriedade_id:
            raise ValueError(f"Tipo de nó inválido: {tipo_no}")

        # Construir a consulta Cypher dinamicamente
        query = f"""
            MATCH (n:{tipo_no})-[r:{tipo_relacionamento}]-(m)
            WHERE n.{propriedade_id} = $id
            RETURN m
        """

        # Executar a consulta e obter os nós vizinhos
        resultados = self.graph.run(query, id=no_embedding).data()
        vizinhos = []
        for resultado in resultados:
            no_vizinho = resultado['m']

            # Verificar se no_vizinho é um dicionário
            if isinstance(no_vizinho, dict):
                # Extrair texto das propriedades do nó vizinho
                texto_vizinho = ' '.join([
                    self.extrair_texto_no_vizinho(valor) for chave, valor in no_vizinho.items()
                ])

            # Gerar embedding do nó vizinho
            embedding_vizinho = self.embedding_model.encode(texto_vizinho, convert_to_tensor=True)
            vizinhos.append(embedding_vizinho)

        return vizinhos


    def agregar_informacoes_vizinhos(self, embeddings, tipos_nos):
        """
        Agrega informações dos vizinhos de cada nó, considerando a nova estrutura 
        do subgrafo com base nas competências extraídas e grafos multiplex.

        Args:
            embeddings: Os embeddings dos nós.
            tipos_nos: Uma lista com os tipos dos nós.

        Returns:
            Um tensor com os embeddings agregados dos vizinhos.
        """

        embeddings_agregados = []
        for i, no in enumerate(embeddings):
            vizinhos_por_camada = defaultdict(list)  # Dicionário para armazenar vizinhos por camada
            tipo_no = tipos_nos[i]  # Obter o tipo do nó atual

            # Obter os embeddings dos vizinhos de acordo com os relacionamentos
            for tipo_relacionamento, propriedade_id in self.relacionamentos_multiplex.items():
                vizinhos = self.obter_embeddings_vizinhos(no, tipo_no, tipo_relacionamento)
                vizinhos_por_camada[tipo_relacionamento].extend(vizinhos)  # Adicionar vizinhos à camada correspondente

            # Agregar embeddings dos vizinhos por camada (usando a média)
            embeddings_camadas = []
            for tipo_relacionamento, vizinhos in vizinhos_por_camada.items():
                if vizinhos:
                    embeddings_camadas.append(torch.mean(torch.stack(vizinhos), dim=0))
                else:
                    # Se não houver vizinhos na camada, usar um tensor de zeros
                    embeddings_camadas.append(torch.zeros_like(no))

            # Concatenar os embeddings de todas as camadas
            embeddings_agregados.append(torch.cat(embeddings_camadas, dim=0))

        return torch.stack(embeddings_agregados)


    def combinar_embeddings(self, embeddings, embeddings_agregados):
        """
        Combina os embeddings dos nós com os embeddings agregados dos vizinhos usando concatenação.

        Args:
            embeddings: Os embeddings dos nós.
            embeddings_agregados: Os embeddings agregados dos vizinhos.

        Returns:
            Um tensor com os embeddings combinados.
        """
        return torch.cat([embeddings, embeddings_agregados], dim=1)


    def camadas_adicionais(self, embeddings_combinados):
        """
        Aplica camadas adicionais aos embeddings combinados (opcional).

        Args:
            embeddings_combinados: Os embeddings combinados.

        Returns:
            Um tensor com os embeddings após a aplicação das camadas adicionais.
        """
        # Definição das camadas adicionais (exemplo)
        self.linear1 = nn.Linear(embeddings_combinados.size(1), 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 64)

        # Aplicação das camadas
        saida = self.linear1(embeddings_combinados)
        saida = self.relu(saida)
        saida = self.linear2(saida)

        return saida