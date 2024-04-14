import os
import re
import csv
import json
import math
import torch
import hashlib
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm
import ipywidgets as widgets
from IPython.display import display, clear_output
from neo4j import GraphDatabase, Transaction, exceptions

from transformers import BertModel, BertTokenizer

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_people_without_teams(self):
        with self.driver.session() as session:
            query = "MATCH (p:Person) WHERE NOT (p)-[:MEMBER_OF]->(:Team) RETURN p.name, p.idLattes"
            result = session.run(query)
            return [{'name': record['p.name'], 'idLattes': record['p.idLattes']} for record in result]

    def get_available_teams(self):
        with self.driver.session() as session:
            query = "MATCH (t:Team) RETURN t.name"
            result = session.run(query)
            return [record['t.name'] for record in result]

    def get_existing_associations(self):
        with self.driver.session() as session:
            # Query para buscar associações existentes entre pessoas e equipes
            query = """
            MATCH (p:Person)-[r:MEMBER_OF]->(t:Team)
            RETURN p.idLattes AS idLattes, p.name AS name, t.name AS team
            """
            result = session.run(query)
            
            # Construir e retornar uma lista de associações
            associations = []
            for record in result:
                associations.append({
                    'idLattes': record['idLattes'],
                    'name': record['name'],
                    'team': record['team']
                })
            return associations

    def associate_person_to_team(self, person_id, team_name):
        with self.driver.session() as session:
            # Primeiro, remova a associação existente, se houver
            remove_query = (
                "MATCH (p:Person {idLattes: $person_id})-[r:MEMBER_OF]->(:Team) "
                "DELETE r"
            )
            session.run(remove_query, person_id=person_id)

            # Em seguida, crie a nova associação
            associate_query = (
                "MATCH (p:Person {idLattes: $person_id}), (t:Team {name: $team_name}) "
                "MERGE (p)-[:MEMBER_OF]->(t)"
            )
            session.run(associate_query, person_id=person_id, team_name=team_name)

    def extract_data_to_csv(self, csv_path):
        # Implementação para extrair dados do Neo4j e salvá-los em um arquivo CSV
        with self.driver.session() as session:
            # Query para extrair dados (ajuste conforme necessário)
            query = "MATCH (p:Person) RETURN p.idLattes AS idLattes, p.name AS name, p.equipe AS equipe"

            # Executar a query
            result = session.run(query)

            # Criar e escrever no arquivo CSV
            with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['idLattes', 'name', 'equipe'])
                writer.writeheader()
                for record in result:
                    writer.writerow({'idLattes': record['idLattes'], 'name': record['name'], 'equipe': ''})  # equipe inicialmente vazia

            print(f"Dados extraídos para o arquivo CSV: {csv_path}")

    def save_associations_to_csv(self, 
                                 input_folder = '/home/mak/gml_classifier-1/data/input/', 
                                 filename = 'relations_person_team.csv'):
        with self.driver.session() as session:
            query = """
            MATCH (p:Person)-[r:MEMBER_OF]->(t:Team)
            RETURN p.idLattes AS idLattes, p.name AS name, t.name AS team
            """
            result = session.run(query)

            input_folder 
            csv_file_path = input_folder+filename

            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['idLattes', 'name', 'team'])
                writer.writeheader()
                for record in result:
                    writer.writerow({'idLattes': record['idLattes'], 'name': record['name'], 'team': record['team']})

            print(f"Associações salvas em CSV: {csv_file_path}")

    def extract_article_data(self):
        with self.driver.session() as session:
            query = """
            MATCH (a:Article)
            WHERE a.embedding IS NULL AND a.title IS NOT NULL
            RETURN a.identifier AS identifier, a.title AS title, a.year AS year, a.resumo AS resumo
            """
            result = session.run(query)
            articles = []
            for record in result:
                article = {
                    'identifier': record['identifier'],
                    'title': record['title'],
                    'year': record['year']
                }
                # Adiciona 'resumo' apenas se estiver presente
                if 'resumo' in record:
                    article['resumo'] = record['resumo']
                articles.append(article)
            return articles


class BertEmbeddingsToNeo4j:
    def __init__(self, uri, user, password, model_name='bert-base-multilingual-cased', batch_size=100):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def close(self):
        self.driver.close()

    ## Processar os dicionários para persistir em Neo4j
    def load_data_from_json(self, json_file):
        with open(json_file, 'r') as file:
            return json.load(file)

    def extract_lattes_id(self, lattes_url):
        match = re.search(r'(\d{16})', lattes_url)
        return match.group(1) if match else None

    def serialize_dict(self, d):
        return json.dumps(d, ensure_ascii=False)

    ## Criar os nós de Pessoa
    def create_or_update_person_nodes(self, session, researcher):
        lattes_id = self.extract_lattes_id(researcher["Identificação"]["Lattes iD"])
        # Cria um dicionário com todas as informações do pesquisador
        researcher_properties = {
            'idLattes': lattes_id,
            'name': researcher.get("name", ""),
            'infPes': researcher.get("InfPes", []),
            'resumo': researcher.get("Resumo", ""),
            'formacao': researcher.get("Formação acadêmica/titulação", {}),
            'atuacao': researcher.get("Áreas de atuação", {}),
            # Continuar para outros subdicionários...
        }

        # Serializar dicionários aninhados
        formacao = self.serialize_dict(researcher.get("Formação acadêmica/titulação", {}))
        atuacao = self.serialize_dict(researcher.get("Áreas de atuação", {}))
        # ... serializar outros campos conforme necessário

        researcher_properties = {
            'idLattes': lattes_id,
            'name': researcher.get("name", ""),
            'infPes': researcher.get("InfPes", []),
            'resumo': researcher.get("Resumo", ""),
            'formacao': formacao,
            'atuacao': atuacao,
            # ... outros campos
        }

        query = (
            "MERGE (p:Person {idLattes: $idLattes}) "
            "SET p += $props"
        )
        session.run(query, idLattes=lattes_id, props=researcher_properties)

    ## Criar os nós de Artigo
    def create_article_nodes(self, session, researcher):
        lattes_id = self.extract_lattes_id(researcher["Identificação"]["Lattes iD"])

        count_articles = 0
        count_relationships = 0

        if "JCR2" in researcher:
            for article_info in researcher.get("JCR2", {}).values():
                doi = article_info.get("doi")
                title = article_info.get("titulo")
                year = article_info.get("jcr-ano", "").split(" ")[-1]
                
                # Trata None para strings vazias antes de usar strip()
                doi = (doi if doi is not None else "").strip()
                title = (title if title is not None else "").strip()

                # Verificação de unicidade do DOI
                if doi:
                    doi_exist_query = "MATCH (a:Article) WHERE a.doi = $doi RETURN count(a) > 0 AS exists"
                    doi_exists = session.run(doi_exist_query, doi=doi).single()[0]
                    if doi_exists:
                        print(f"Warning: Article with DOI {doi} already exists. Skipping creation.")
                        continue

                # Gerar um hash como identificador alternativo se não houver DOI
                if not doi:
                    identifier = hashlib.sha256(title.encode('utf-8')).hexdigest()
                    has_doi = False
                else:
                    identifier = doi
                    has_doi = True

                # Query Cypher para criar ou atualizar o nó do artigo
                article_query = (
                    "MERGE (a:Article {identifier: $identifier}) "
                    "ON CREATE SET a.doi = $doi, a.title = $title, a.year = $year, "
                    "a.hasDoi = $hasDoi, a.issn = $issn, a.volume = $volume, "
                    "a.initialPage = $initialPage, a.journalName = $journalName, a.impactFactor = $impactFactor "
                    "ON MATCH SET a.doi = $doi, a.title = $title, a.year = $year"
                )
                # Executar a query Cypher
                session.run(article_query, identifier=identifier, doi=doi, title=title, year=year,
                            hasDoi=has_doi, issn=article_info.get("issn", ""), 
                            volume=article_info.get("volume", ""),
                            initialPage=article_info.get("paginaInicial", ""), 
                            journalName=article_info.get("nomePeriodico", ""), 
                            impactFactor=article_info.get("impact-factor", 0),
                            originalTitle=article_info.get("original_title", ""))
                
                # Estrutura da query para criar relacionamento entre o pesquisador e o artigo
                relationship_query = (
                    "MATCH (p:Person {idLattes: $idLattes}), (a:Article {identifier: $identifier}) "
                    "MERGE (p)-[r:AUTHORED]->(a)"
                )
                
                # Executar a query Cypher para criar o relacionamento
                session.run(relationship_query, idLattes=lattes_id, identifier=identifier)
                # Atualiza as contagens
                
                print(f"Article node created/updated: {identifier}")
                print(f"Relationship created between researcher {lattes_id} and article {identifier}")

                count_articles += 1
                count_relationships += 1

        return count_articles, count_relationships

    ## Adicionar artigos direto do dataset normalizado em JSON
    def load_articles_from_json(self, json_file):
        with open(json_file, 'r') as file:
            return json.load(file)

    ## Gerar hash combinando título e ano do arquivo
    def generate_article_hash(self, title, year):
        return hashlib.sha256(f"{title}{year}".encode()).hexdigest()

    ## Criar as relações entre Artigos e Pesquisadores
    def create_or_update_article_nodes_and_relationships(self, session, lattes_id, articles):
        for article in articles:
            # Usando DOI como identificador, se disponível, caso contrário, gera um hash do título e do ano
            article_id = article['doi'] if article.get('doi') else self.generate_article_hash(article['title'], article['year'])
            impact_factor = float(article.get('impactFactor', '0.1'))

            # Cria ou atualiza o nó do artigo
            article_query = (
                "MERGE (a:Article {id: $article_id}) "
                "ON CREATE SET a.title = $title, a.year = $year, a.doi = $doi "
                "ON MATCH SET a.title = $title, a.year = $year, a.doi = $doi"
            )
            session.run(article_query, article_id=article_id, title=article['title'], year=article['year'], doi=article.get('doi', ''))

            # Cria relação entre o pesquisador e o artigo
            relation_query = (
                "MATCH (a:Article {id: $article_id}), (p:Person {idLattes: $lattes_id}) "
                "MERGE (p)-[r:AUTHORED]->(a) "
                "ON CREATE SET r.weight = $impact_factor"
            )

            try:
                session.run(relation_query, 
                            article_id=article_id, 
                            lattes_id=lattes_id, 
                            impact_factor=impact_factor)
                
                print(f"Relationship created/updated for article ID: {article_id} and researcher Lattes ID: {lattes_id}")
            except Exception as e:
                print(f"Error creating/updating relationship: {e}")

    ## Criar as relações entre Artigos e Pesquisadores
    def create_relationships(self, session, articles, researcher_name):
        for article in articles:
            article_id = article['doi'] if article['doi'] else self.generate_article_hash(article['title'], article['year'])
            query = (
                "MATCH (a:Article {id: $article_id}), (r:Researcher {name: $researcher_name}) "
                "MERGE (r)-[:AUTHORED]->(a)"
            )
            try:
                session.run(query, article_id=article_id, researcher_name=researcher_name)
            except Exception as e:
                print(f"Error creating/updating relationship: {e}")

    def normalize_string(self, s):
        # Remover acentuação
        try:
            s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
            # Converter para minúsculas
            return s.lower()
        except Exception as e:
            # print(f"Não foi possível normalizar: {s}")
            # print(e)
            pass

    def extract_area_names(self, area_str):
        area_parts = area_str.split('/')
        extracted_names = {}
        for part in area_parts:
            print(part)
            if 'Grande área' in part:
                print("GA destacada:", self.normalize_string(part.split(':')[1].strip()))
                extracted_names['grande_area'] = self.normalize_string(part.split(':')[1].strip())
            elif 'Área' in part:
                print(" A destacada:", self.normalize_string(part.split(':')[1].strip()))
                extracted_names['area'] = self.normalize_string(part.split(':')[1].strip())
            elif 'Subárea' in part:
                print("SA destacada:", self.normalize_string(part.split(':')[1].strip()))
                extracted_names['subarea'] = self.normalize_string(part.split(':')[1].strip())
        return extracted_names

    def create_area_relationships(self, session, lattes_id, areas_de_atuacao):
        for key, area_str in areas_de_atuacao.items():
            area_names = self.extract_area_names(area_str)

            if 'grande_area' in area_names:
                self.create_relationship_if_not_exists(session, lattes_id, area_names['grande_area'], 'GrandeArea')
            if 'area' in area_names:
                self.create_relationship_if_not_exists(session, lattes_id, area_names['area'], 'Area')
            if 'subarea' in area_names:
                self.create_relationship_if_not_exists(session, lattes_id, area_names['subarea'], 'Subarea')

    def process_researchers(self, researchers_list):
        with self.driver.session() as session:
            count_researchers = 0
            total_articles = 0
            total_relationships = 0

            for researcher in researchers_list:
                lattes_id = self.extract_lattes_id(researcher["Identificação"]["Lattes iD"])
                self.create_or_update_person_nodes(session, researcher)
                count_researchers += 1

                # Chama o método para criar nós de artigos e obter contagens
                count_articles, count_relationships = self.create_article_nodes(session, researcher)
                total_articles += count_articles
                total_relationships += count_relationships

            # Consulta as quantidades persistidas no Neo4j
            persisted_researchers = session.run("MATCH (p:Person) RETURN count(p) as count").single()["count"]
            persisted_articles = session.run("MATCH (a:Article) RETURN count(a) as count").single()["count"]
            persisted_relationships = session.run("MATCH (:Person)-[r:AUTHORED]->(:Article) RETURN count(r) as count").single()["count"]

            return {
                "researchers": {"processed": count_researchers, "persisted": persisted_researchers},
                "articles": {"processed": count_articles, "persisted": persisted_articles},
                "relationships": {"processed": count_relationships, "persisted": persisted_relationships}
            }

    ## Montar dataframe com dados do setor de pessoal
    def process_relationships_from_file(self, 
                                        pathdata='/home/mak/gml_classifier-1/data/xml_zip',
                                        filename='fioce_colaboradores-2023.xls'):
        # Construir o caminho completo do arquivo
        filepath = f"{pathdata}/{filename}"

        # Ler apenas os cabeçalhos do arquivo Excel
        headers = pd.read_excel(filepath, skiprows=3, header=0, nrows=0).columns

        # Definir função para filtrar colunas
        def cols_to_keep(col_name):
            return col_name not in ['QUANT', 'Unnamed: 3', 'Unnamed: 6', 'Unnamed: 9', 'ADICIONAL OCUPACIONAL',
                                    'EMPRESA/BOLSA/PROGRAMA', 'GESTOR', 'ADI', 'POSSE NA FIOCRUZ',
                                    'VIGÊNCIA BOLSA/ENCERRAMENTO DO CONTRATO', 'Unnamed: 17',
                                    'EMAIL INSTITUCIONAL', 'EMAIL PESSOAL', 'GENERO', 'DATA NASCIMENTO',
                                    'Unnamed: 22', 'FORMAÇÃO', 'ENDEREÇO RESIDENCIAL']

        # Filtrar colunas com base na função
        selected_columns = [col for col in headers if cols_to_keep(col)]

        # Ler dados do arquivo Excel
        df = pd.read_excel(filepath, skiprows=3, header=0, usecols=selected_columns)

        equipes=[]
        for i in df['ÁREA']:
            # print(type(i),i)
            try:
                i=i.lower()
                if 'Biotecnologia' in i:
                    equipes.append('Biotecnologia')
                elif 'família' in i:
                    equipes.append('Saúde da Família')
                elif 'ambiente' in i:
                    equipes.append('Saúde e Ambiente')
                elif 'digital' in i:
                    equipes.append('Saúde Digital')
                else:
                    equipes.append('administrativa')
            except:
                equipes.append('terceirizados')

        df['EQUIPE'] = equipes
        print(df['EQUIPE'].value_counts())
        print()

        # Chamar o método para criar relações a partir do DataFrame
        self.create_relationships_from_dataframe(df)

    ## Extrair nomes e criar arquivo local de associação à equipes
    def extract_data_to_csv(self, csv_file_path):
        with self.driver.session() as session, open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            # Escreve o cabeçalho do CSV
            writer.writerow(['id_lattes', 'name', 'team'])

            # Consulta para extrair os dados
            persons = session.run("MATCH (p:Person) RETURN p.idLattes AS idLattes, p.name AS name")

            # Escreve os dados no CSV
            for person in persons:
                writer.writerow([person['idLattes'], person['name'], ''])  # 'team' é deixado em branco

    ## Associar nomes à Equipes
    def update_teams_from_csv(self, csv_file_path):
        with self.driver.session() as session, open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                id_lattes = row['id_lattes']
                team = row['team']
                
                # Atualiza o nó da pessoa com a equipe associada
                if team:  # Verifica se a equipe não está vazia
                    session.run(
                        "MATCH (p:Person {idLattes: $idLattes}) "
                        "MERGE (t:Team {name: $teamName}) "
                        "MERGE (p)-[:MEMBER_OF]->(t)",
                        idLattes=id_lattes, teamName=team
                    )

    ## Atualizar nome para versão normalizada minúscula sem acentos
    def update_existing_names(self):
        with self.driver.session() as session:
            # Buscar todos os nós de Pessoa
            person_nodes = session.run("MATCH (p:Person) RETURN p.idLattes AS idLattes, p.name AS name")
            
            for person in person_nodes:
                normalized_name = self.normalize_string(person['name'])
                # Atualizar o nome da pessoa para a versão normalizada
                session.run(
                    "MATCH (p:Person {idLattes: $idLattes}) SET p.name = $normalized_name",
                    idLattes=person['idLattes'], normalized_name=normalized_name
                )

    ## Atualizar associação à uma equipe
    def update_existing_team_link(self):
        with self.driver.session() as session:
            # Buscar todos os nós de Pessoa que não tem relação MEMBER_OF com Team associada
            person_nodes = session.run("MATCH (p:Person) RETURN p.idLattes AS idLattes, p.name AS name")
            
            for person in person_nodes:
                normalized_name = self.normalize_string(person['name'])
                # Criar a relação com um dos nós Teams existentes
                session.run(
                    "MATCH (p:Person {idLattes: $idLattes}) SET p.name = $normalized_name",
                    idLattes=person['idLattes'], normalized_name=normalized_name
                )

    ## Disparar o processamento principal chamando outros métodos
    def create_relationships_from_dataframe(self, df):
        # Filtrar o DataFrame
        filtered_df = df[(df['VÍNCULO'] == 'SERVIDOR') & 
                         (df['STATUS'] == 'ATIVO') & 
                         (df['EQUIPE'].isin(['Biotecnologia', 
                                             'Saúde e Ambiente', 
                                             'Saúde da Família', 
                                             'Saúde Digital']))]
        match_count = 0
        total_relationships = 0
        relationship_counts = {'Biotecnologia': 0, 
                               'Saúde e Ambiente': 0, 
                               'Saúde da Família': 0, 
                               'Saúde Digital': 0}

        with self.driver.session() as session:
            for _, row in filtered_df.iterrows():
                # Normalizar o nome para correspondência
                normalized_name_df = self.normalize_string(row['NOME'])

                # Query para criar ou encontrar o nó Team
                team_query = (
                    "MERGE (t:Team {name: $teamName})"
                )
                session.run(team_query, teamName=row['EQUIPE'])

                # Query para criar relações no Neo4j
                relationship_query = (
                    "MATCH (p:Person), (t:Team) "
                    "WHERE toLower(p.name) = $name AND t.name = $teamName "
                    "MERGE (p)-[r:MEMBER_OF]->(t)"
                )
                result = session.run(relationship_query, name=normalized_name_df, teamName=row['EQUIPE'])

                # Verificar se a relação foi criada
                summary = result.consume()
                if summary.counters.relationships_created > 0:
                    match_count += 1
                    total_relationships += summary.counters.relationships_created
                    relationship_counts[row['EQUIPE']] += summary.counters.relationships_created

        print(f"Total matching researchers: {match_count}")
        print(f"Total relationships created: {total_relationships}")
        for team, count in relationship_counts.items():
            print(f"Relationships created for team '{team}': {count}")

    ## Criar um nó de Organização com base nas Equipes de trabalho
    def create_organization_and_connect_teams(self, organization_name, organization_sigla, team_names):
        with self.driver.session() as session:
            # Criar o nó da organização com os valores fornecidos
            organization_query = (
                "MERGE (o:Organization {name: $name, sigla: $sigla}) "
                "RETURN o"
            )
            organization = session.run(organization_query, name=organization_name, sigla=organization_sigla).single()
            print(f"Organization node created/updated: {organization['o']}")

            # Conectar a organização aos times correspondentes
            for team_name in team_names:
                connection_query = (
                    "MATCH (o:Organization {name: $name}), (t:Team {name: $teamName}) "
                    "MERGE (o)-[r:INCLUDES]->(t)"
                )
                session.run(connection_query, name=organization_name, teamName=team_name)
                print(f"Connected organization '{organization_name}' to team '{team_name}'")

        print("All specified teams have been connected to the organization.")

    def extract_area(self, areas, index):
        if len(areas) > index:
            parts = areas[index].split(':')
            if len(parts) > 1:
                return parts[1].strip()
        return None

    ## EMBEEDINGS
    def extract_and_process_data(self, extract_function):
        data = extract_function()
        for item in data:
            # Passa o item inteiro para o método prepare_text
            text = self.prepare_text(item['title'], item)
            if text:
                embedding = self.get_embedding(text)
                # Atualiza o embedding no banco de dados
                self.update_embeddings(item['identifier'], embedding)

    def prepare_text(self, title, item):
        title_text = title if title is not None else ""
        # Verifica se 'resumo' está presente no dicionário 'item'
        resumo_text = item.get('resumo', "")
        prepared_text = (title_text + " " + resumo_text).strip()
        return prepared_text

    def update_embeddings(self, identifier, embedding):
        """
        Atualiza um nó no Neo4j com o embedding fornecido.
        :param identifier: Identificador único do nó (e.g., id de um artigo).
        :param embedding: Embedding gerado pelo BERT.
        """
        try:
            query = "MATCH (n {identifier: $identifier}) SET n.embedding = $embedding"
            self.execute_query(query, {'identifier': identifier, 'embedding': embedding.tolist()})
            updated_node = self.execute_query(query, {'identifier': identifier, 'embedding': embedding.tolist()})
            if updated_node:
                print(f"Embedding successfully updated for article with ID: {identifier}")
            else:
                print(f"Failed to embed article: {identifier}")
        except Exception as e:
            print(f"Embedding error: {e}\n On article: {identifier}")

    def execute_all_embeddings_update(self, neo4j_connector):
        """
        Executa o processo completo de extração, processamento e atualização de embeddings.
        :param neo4j_connector: Instância do Neo4jConnector para extrair dados.
        """
        self.extract_and_process_data(neo4j_connector.extract_article_data)

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def get_total_articles_count(self):
        return self.execute_query("MATCH (a:Article) RETURN count(a)")[0][0]

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().detach().numpy()

    def get_articles_without_embedding(self):
        query = (
            "MATCH (a:Article) "
            "WHERE a.embedding IS NULL AND a.id IS NOT NULL "
            "RETURN a.id AS id, a.title AS title, a.resumo AS resumo "
            "LIMIT $batchSize"
        )
        parameters = {'batchSize': self.batch_size}
        return self.execute_query(query, parameters)


    def process_and_update_embeddings(self, neo4j_connector):
        # Extrair dados dos artigos
        articles = neo4j_connector.extract_article_data()

        # Processar cada artigo para gerar e atualizar embeddings
        for article in articles:
            identifier = article['identifier']
            title = article['title']

            # Gerar o embedding
            embedding = self.get_embedding(title)

            # Atualizar o artigo no Neo4j com o embedding
            update_query = """
            MATCH (a:Article {identifier: $identifier})
            SET a.embedding = $embedding
            """
            with self.driver.session() as session:
                session.run(update_query, identifier=identifier, embedding=embedding.tolist())
      
    def delete_all_nodes_of_type(self, node_type):
        with self.driver.session() as session:
            # Query Cypher para remover todos os nós do tipo especificado e suas relações
            query = (
                f"MATCH (n:{node_type}) "
                "DETACH DELETE n"
            )
            session.run(query)
            print(f"All nodes of type '{node_type}' and their relationships have been deleted.")


    def create_cnpq_relationships(self):
        with self.driver.session() as session:
            persons_query = "MATCH (p:Person) WHERE p.atuacao IS NOT NULL RETURN p.idLattes AS idLattes, p.atuacao AS atuacao"
            persons = session.run(persons_query).data()

            total_persons = 0
            relationship_counts = {'GrandeArea': 0, 'Area': 0, 'Subarea': 0}

            for person in persons:
                idLattes = person['idLattes']
                atuacoes = json.loads(person['atuacao'].replace("'", '"'))

                for atuacao_key, atuacao_value in atuacoes.items():
                    areas = atuacao_value.split('/')
                    grande_area = self.extract_area(areas, 0)
                    area = self.extract_area(areas, 1)
                    subarea = self.extract_area(areas, 2)

                    for area_type, area_name in [("GrandeArea", grande_area), ("Area", area), ("Subarea", subarea)]:
                        if area_name:
                            relationship_exists_query = (
                                f"MATCH (p:Person {{idLattes: $idLattes}})-[r:RELATED_TO]->(a:{area_type} {{name: $areaName}}) "
                                "RETURN count(r) as relationshipCount"
                            )
                            relationship_exists = session.run(relationship_exists_query, idLattes=idLattes, areaName=area_name).single()["relationshipCount"] > 0
                            
                            if relationship_exists:
                                print(f"Relationship between {idLattes} and {area_type} {area_name} already exists. Skipping.")
                                continue

                            relationship_query = (
                                f"MATCH (p:Person {{idLattes: $idLattes}}), (a:{area_type} {{name: $areaName}}) "
                                "MERGE (p)-[r:RELATED_TO]->(a) "
                                "ON CREATE SET r.weight = 1"
                            )
                            session.run(relationship_query, idLattes=idLattes, areaName=area_name)
                            relationship_counts[area_type] += 1

                    total_persons += 1

            print(f"Total persons found: {total_persons}")
            for label, count in relationship_counts.items():
                print(f"Relationships created for {label}: {count}")

            total_relationships = sum(relationship_counts.values())
            print(f"Total relationships created: {total_relationships}")

            if total_relationships == 0:
                print("Warning: No relationships were created. Check the queries and data.")

    def create_relationship_with_cnpq_node(self, session, idLattes, name, label, relationship_counts):
        query = (
            f"MATCH (p:Person {{idLattes: $idLattes}}), (c:{label} {{name: $name}}) "
            "MERGE (p)-[r:RELATED_TO]->(c)"
        )
        result = session.run(query, idLattes=idLattes, name=name)
        summary = result.consume()
        if summary.counters.relationships_created > 0:
            relationship_counts[label] += summary.counters.relationships_created

    def create_relationship_if_not_exists(self, session, idLattes, area_name, area_type, relationship_counts):
        if area_name:
            # Verificar se o relacionamento já existe
            relationship_exists_query = (
                f"MATCH (p:Person {{idLattes: $idLattes}})-[r:RELATED_TO]->(a:{area_type}) "
                "WHERE a.name = $areaName "
                "RETURN count(r) > 0 AS exists"
            )
            result = session.run(relationship_exists_query, idLattes=idLattes, areaName=area_name)
            relationship_exists = result.single()[0]

            if not relationship_exists:
                # Criar o relacionamento
                relationship_query = (
                    f"MATCH (p:Person {{idLattes: $idLattes}}), (a:{area_type} {{name: $areaName}}) "
                    "MERGE (p)-[r:RELATED_TO]->(a) "
                    "ON CREATE SET r.weight = 1"
                )
                print(f"Executing query: {relationship_query}")
                session.run(relationship_query, idLattes=idLattes, areaName=area_name)
                relationship_counts[area_type] += 1

                # Verificar se o relacionamento foi criado com sucesso
                verify_query = (
                    f"MATCH (p:Person {{idLattes: $idLattes}})-[r:RELATED_TO]->(a:{area_type} {{name: $areaName}}) "
                    "RETURN count(r) as relationshipCount"
                )
                verify_result = session.run(verify_query, idLattes=idLattes, areaName=area_name)
                for record in verify_result:
                    print(f"Verified relationships for {area_type} {area_name}: {record['relationshipCount']}")

class DataEntryInterface:
    def __init__(self, neo4j_connector):
        self.neo4j_connector = neo4j_connector

    def start_process(self):
        print("Iniciando o processo de associação de equipes.")
        self.display_existing_associations()
        self.display_unassociated_people_widget()

    def display_existing_associations(self):
        existing_associations = self.neo4j_connector.get_existing_associations()

        # Supondo que `get_existing_associations` retorne uma lista de associações
        for association in existing_associations:
            person_name = association['name']
            team_name = association['team']
            print(f"{person_name} ==> {team_name}.")
        
        # Aqui, você pode adicionar a opção para editar estas associações se necessário

    def display_unassociated_people_widget(self):
        # Obter pessoas não associadas em vez de associações existentes
        unassociated_people = self.neo4j_connector.get_people_without_teams()

        # Criar dropdown para seleção de pessoas
        people_dropdown = widgets.Dropdown(
            options=[(person['name'], person['idLattes']) for person in unassociated_people],
            description='Pessoas:',
            disabled=False,
        )

        teams_dropdown = widgets.Dropdown(
            options=self.neo4j_connector.get_available_teams(),
            description='Equipes:',
            disabled=False,
        )

        associate_button = widgets.Button(
            description='Associar',
            disabled=False,
            button_style='',
            tooltip='Clique para associar a pessoa à equipe selecionada',
        )

        def on_associate_button_clicked(b):
            selected_person = people_dropdown.value
            selected_team = teams_dropdown.value
            self.neo4j_connector.associate_person_to_team(selected_person, selected_team)
            print(f'Pessoa {people_dropdown.label} --> {selected_team}')
            clear_output()
            self.display_unassociated_people_widget()

        associate_button.on_click(on_associate_button_clicked)

        display(people_dropdown, teams_dropdown, associate_button)

    def display_and_edit_existing_associations(self):
        existing_associations = self.neo4j_connector.get_existing_associations()

        # Criar widgets para cada associação para permitir a edição
        for association in existing_associations:
            person_id = association['idLattes']
            person_name = association['name']
            current_team = association['team']

            print(f"{person_name} (ID: {person_id}) --> {current_team}")

            teams_dropdown = widgets.Dropdown(
                options=self.neo4j_connector.get_available_teams(),
                value=current_team,
                description='Nova Equipe:',
                disabled=False,
            )

            update_button = widgets.Button(
                description='Alterar Associação',
                disabled=False,
                button_style='',
                tooltip='Clique para atualizar a associação da equipe',
            )

            def on_update_button_clicked(b, person_id=person_id, dropdown=teams_dropdown):
                new_team = dropdown.value
                self.neo4j_connector.associate_person_to_team(person_id, new_team)
                print(f'Associação atualizada: {person_name} --> {new_team}')
                clear_output()
                self.display_and_edit_existing_associations()

            update_button.on_click(on_update_button_clicked)

            display(teams_dropdown, update_button)