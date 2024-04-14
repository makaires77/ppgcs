# Configuration for logging operations
import re
import json
import time
import torch
import psutil
import logging
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from scipy import optimize
from PyPDF2 import PdfReader
from neo4j import GraphDatabase

import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os, sys, pip, time, h5py, json
import requests, urllib, torch, sqlite3, asyncio
import glob, psutil, platform, subprocess, nltk
import warnings, logging, traceback, csv, string, re

from PIL import Image
from io import BytesIO
from pprint import pprint
from PyPDF2 import PdfReader
from string import Formatter
from neo4j import GraphDatabase
from nltk.corpus import stopwords
from urllib3.util.retry import Retry
from tqdm.notebook import trange, tqdm
from datetime import datetime, timedelta
from flask import render_template_string
from requests.adapters import HTTPAdapter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Any, List, Dict, Optional
from py2neo import Graph, Node, Relationship
from urllib.parse import urlparse, parse_qs
from typing import List, Optional, Dict, Union
from collections import deque, defaultdict, Counter
from bs4 import BeautifulSoup, Tag, NavigableString
from pyjarowinkler.distance import get_jaro_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython.display import clear_output, display, HTML

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common import exceptions
from selenium.common.exceptions import (
    NoSuchElementException, 
    StaleElementReferenceException,
    ElementNotInteractableException,
    TimeoutException,
    WebDriverException
)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='./logs/database_operations.log',
                    filemode='w')

class CNPqTree:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Inicialização da vetorização TF-IDF
        self.vectorizer = TfidfVectorizer()
        
        # Definição do modelo de classificação
        self.model = self.ClassifierModel()
    
    class ClassifierModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Camadas da rede neural
            self.fc1 = nn.Linear(5000, 1000)  
            self.fc2 = nn.Linear(1000, 100)
            self.fc3 = nn.Linear(100, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    def vectorize_titles(self, titles):
        X = self.vectorizer.fit_transform(titles)
        return torch.from_numpy(X.toarray())
    
    def train(self, train_data, train_labels, num_epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = optimize.Adam(self.model.parameters())

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.model(train_data)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    
    def classify(self, titles):
        tensor_titles = self.vectorize_titles(titles)
        with torch.no_grad():
            predicted_outputs = self.model(tensor_titles)
            _, predicted_labels = torch.max(predicted_outputs, 1)
        return predicted_labels

    def close(self):
        self._driver.close()

    # Additional methods for node creation
    def _merge_nodes(self, node_type, code, name):
        with self._driver.session() as session:
            session.run(f"MERGE (n:{node_type} {{code: $code, name: $name}})", code=code, name=name)

    def _create_relationship(self, parent_type, parent_code, child_type, child_code, relation_name):
        with self._driver.session() as session:
            session.run(f"""
                MATCH (p:`{parent_type}` {{code: $parent_code}})
                MATCH (c:`{child_type}` {{code: $child_code}})
                MERGE (p)-[:{relation_name}]->(c)
            """, parent_code=parent_code, child_code=child_code)

    def process(self, classifications):
        # First, create nodes for each classification
        for code, name in classifications:
            # Remove the verification digit
            code = code.rsplit('-', 1)[0]
            
            # Define regex patterns
            grande_area_pattern   = r"^\d\.00\.00\.00$"
            area_pattern          = r"^\d\.\d{2}\.00\.00$"  
            subarea_pattern       = r"^\d\.\d{2}\.\d{2}\.00$"  
            especialidade_pattern = r"^\d\.\d{2}\.\d{2}\.\d{2}$"

            if re.match(grande_area_pattern, code):
                self._merge_nodes('GrandeÁrea', code, name)
            elif re.match(area_pattern, code):
                self._merge_nodes('Área', code, name)
            elif re.match(subarea_pattern, code):
                self._merge_nodes('Subárea', code, name)
            elif re.match(especialidade_pattern, code):
                self._merge_nodes('Especialidade', code, name)
            else:
                logging.error(f"Invalid code format: {code}")
                raise ValueError(f"Invalid code format: {code}")

        # Next, establish relationships
        for code, _ in classifications:
            # Remove the verification digit
            code = code.rsplit('-', 1)[0]

            # For GrandeÁrea to Área
            if re.match(area_pattern, code):
                match = re.match(r"^(\d)\.\d{2}\.00\.00$", code)
                if match:
                    parent_code = match.group(1) + ".00.00.00"
                    self._create_relationship('GrandeÁrea', parent_code, 'Área', code, 'CONTÉM_ÁREA')
                else:
                    logging.error(f"Unexpected code format for Área: {code}")
                    raise ValueError(f"Unexpected code format for Área: {code}")

            # For Área to Subárea
            elif re.match(subarea_pattern, code):
                match = re.match(r"(\d\.\d{2})\.\d{2}\.00$", code)
                if match:
                    parent_code = match.group(1) + ".00.00"
                    self._create_relationship('Área', parent_code, 'Subárea', code, 'CONTÉM_SUBÁREA')
                else:
                    logging.error(f"Unexpected code format for Subárea: {code}")
                    raise ValueError(f"Unexpected code format for Subárea: {code}")

            # For Subárea to Especialidade
            elif re.match(especialidade_pattern, code):
                match = re.match(r"(\d\.\d{2}\.\d{2})\.\d{2}$", code)
                if match:
                    parent_code = match.group(1) + ".00"
                    self._create_relationship('Subárea', parent_code, 'Especialidade', code, 'CONTÉM_ESPECIALIDADE')
                else:
                    logging.error(f"Unexpected code format for Especialidade: {code}")
                    raise ValueError(f"Unexpected code format for Especialidade: {code}")

    def create_node_cnpq(self):
        with self._driver.session() as session:
            # Iniciar uma transação para executar a consulta Cypher
            result = session.execute_write(self._create_and_associate)

            return result

    @staticmethod
    def _create_and_associate(tx):
        # Definição da consulta Cypher usando MERGE
        query = (
            "MERGE (tree:ÁrvoreCNPq { name: 'Árvore do Conhecimento' }) "
            "WITH tree "
            "MATCH (ga:GrandeÁrea) "
            "MERGE (tree)-[:CONTÉM_GRANDEÁREA]->(ga)"
        )
        # Executar a consulta
        result = tx.run(query)
        return result.single()  # Retornar o resultado da consulta, se necessário


    pathdata = './../data/'
    file_cnpq = 'cnpq_tabela-areas-conhecimento.pdf'
    caminho = pathdata+file_cnpq

    def verifica_ponto_virgula(df):
        return df[df['Descricao'].str.contains(';', regex=False)]

    def verifica_virgula(df):
        return df[df['Descricao'].str.contains(',', regex=False)]

    def verifica_formato_descricao(self, descricao):
        excecoes = ["de", "do", "da", "dos", "das", "a", "o", "e", "em", "com", "para", "por", "sem"]
        palavras = descricao.split()
        
        for i, palavra in enumerate(palavras):
            if palavra.lower() in excecoes or palavra[0]=="(":
                continue
            if not palavra[0].isupper() or (palavra==palavras[-1] and palavra in excecoes):
                return False, i  # Retornar False e o índice da palavra problemática
        return True, None

        # for idx, word in enumerate(palavras):
        #     # Se a palavra inicia com letra minúscula e não é uma preposição ou artigo
        #     if word[0].islower() and word not in excecoes:
        #         # Aqui verificamos se a palavra anterior termina com uma letra e a palavra atual é uma preposição ou artigo
        #         if idx > 0 and palavras[idx - 1][-1].isalpha() and word in excecoes:
        #             return (False, idx)
        #         # Ou apenas a condição de começar com minúscula e não ser preposição ou artigo
        #         elif idx == 0 or (idx > 0 and not palavras[idx - 1][-1].isalpha()):
        #             return (False, idx)    

    def corrigir_descricao(self, descricao, word_index):
        excecoes = ["de", "do", "da", "dos", "das", "a", "o", "e", "em", "com", "para", "por", "sem"]
        palavras = descricao.split()

        # Se o índice anterior existir e a palavra atual começa com minúscula
        if word_index > 0 and palavras[word_index][0].islower():
            # Checar se a palavra é uma preposição ou artigo e se a anterior termina com uma letra
            if palavras[word_index] in excecoes:
                palavras[word_index - 1] += palavras[word_index]
                del palavras[word_index]
            else:
                # Juntar palavra atual com a palavra anterior
                palavras[word_index - 1] += palavras[word_index]
                del palavras[word_index]

        # Após as correções, juntamos as palavras de volta em uma única string
        nova_descricao = ' '.join(palavras)

        # Imprimindo para debug
        # print(f"Descrição ruim: {descricao}")
        # print(f"Correção feita: {palavra_anterior} + {palavra_incorreta} = {correcao}")
        # print(f"Nova descrição: {nova_descricao}\n")
        
        return nova_descricao

    def extrair_areas(self, caminho):
        texto_completo = ""

        reader = PdfReader(caminho)
        
        for npag, p in tqdm(enumerate(reader.pages), total=len(reader.pages), desc="Processando páginas do PDF das Áreas de pesquisa do CNPq.."):
            texto_completo += p.extract_text()

        texto_completo = texto_completo.replace('\n', ' ').replace(" -","-").replace(" ,",",").strip().replace("ã o","ão")
        texto_completo = re.sub(r'\s?(\d)\s?(\.)\s?(\d{2})\s?(\.)\s?(\d{2})\s?(\.)\s?(\d{2})\s?(-)\s?(\d)\s?', r'\1\2\3\4\5\6\7\8\9', texto_completo)

        pattern = r'(\d\.\d{2}\.\d{2}\.\d{2}-\d)([^0-9]+)'
        matches = re.findall(pattern, texto_completo)

        codigos = [match[0] for match in matches]
        descricoes = [match[1].strip() for match in matches]

        print(f'Total dos códigos   identificados: {len(codigos)}')
        print(f'Total de descrições identificadas: {len(descricoes)}')

        df_linhas = pd.DataFrame({'Codigo': codigos, 'Descricao': descricoes})

        # Verificação da divisão correta dos códigos/descrições
        descricoes_com_numeros = df_linhas[df_linhas['Descricao'].str.contains(r'\d')]
        if not descricoes_com_numeros.empty:
            print(f"Conferência: {len(descricoes_com_numeros)} descrições contêm números!")
        else:
            print(f"Nenhum erro de códigos em descrições!")

        # Identificando e printando a quantidade de possíveis erros
        erros = sum(1 for descricao in descricoes if not self.verifica_formato_descricao(descricao)[0])
        print(f"{erros} possíveis erros de descrição detectados.")

        # Barra de progresso para correção das descrições
        with tqdm(total=df_linhas.shape[0], desc="Corrigindo descrições...") as pbar:
            for index, row in df_linhas.iterrows():
                is_valid, word_index = self.verifica_formato_descricao(row['Descricao'])
                loop_count = 0
                while not is_valid and loop_count < 10:
                    row['Descricao'] = self.corrigir_descricao(row['Descricao'], word_index)
                    is_valid, word_index = self.verifica_formato_descricao(row['Descricao'])
                    loop_count += 1
                if loop_count == 10:
                    print(f"Problema corrigindo descrição: {row['Descricao']}")
                pbar.update(1)

        return df_linhas

    df_areas = extrair_areas(caminho)


class GraphExplore:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Fecha a conexão com o banco de dados
        self.driver.close()

    def explore_sons_nodes_bystring(self, string):
        with self.driver.session() as session:
            result = session.read_transaction(self._find_sons_nodes, string)
            return result

    @staticmethod
    def _find_sons_nodes(tx, string):
        query = (
            "MATCH (area) "
            "WHERE area.name CONTAINS $string "
            "MATCH path = (area)-[r*]->(descendant) "
            "WHERE ALL(rel IN r WHERE type(rel) STARTS WITH 'CONTÉM') "
            "RETURN path"
        )
        result = tx.run(query, string=string)
        return [record["path"] for record in result]

    @staticmethod
    def _create_subgraph(tx, area_name):
        create_subgraph_query = (
            "CALL gds.graph.project("
            "'subgraphComputacao', "
            "'MATCH (area:Área)-[r:CONTÉM*]->(descendant) WHERE area.name CONTAINS $area_name RETURN id(area) AS id UNION MATCH (area:Área)-[r:CONTÉM*]->(descendant) WHERE area.name CONTAINS $area_name RETURN id(descendant) AS id', "
            "'MATCH (area:Área)-[r:CONTÉM*]->(descendant) WHERE area.name CONTAINS $area_name RETURN id(area) AS source, id(descendant) AS target, type(r) AS type'"
            ")"
        )
        tx.run(create_subgraph_query, area_name=area_name)

    def apply_graph_algorithms(self, area_name):
        with self.driver.session() as session:
            # Criar um subgrafo em memória
            session.write_transaction(self._create_subgraph, area_name)
            # Aplicar um algoritmo ao subgrafo e retornar os resultados
            result = session.read_transaction(self._apply_pagerank)
            return result

    def paths_to_dataframe(self, paths):
        dados = []

        for path in paths:
            try:
                # Acessando nós e relações como atributos, não métodos
                nos = path.nodes
                relacoes = path.relationships

                # Assumindo que cada Path contém pelo menos 2 nós (início e fim)
                if len(nos) >= 2:
                    inicio = nos[0]
                    fim = nos[-1]

                    # Extrai as informações desejadas
                    id_inicio = inicio.id
                    rotulos_inicio = list(inicio.labels)
                    propriedades_inicio = dict(inicio)

                    id_fim = fim.id
                    rotulos_fim = list(fim.labels)
                    propriedades_fim = dict(fim)

                    # Adiciona os dados extraídos à lista
                    dados.append({
                        "ID Início": id_inicio,
                        "Rótulos Início": rotulos_inicio,
                        "Código Início": propriedades_inicio.get('code', ''),
                        "Nome Início": propriedades_inicio.get('name', ''),
                        "ID Fim": id_fim,
                        "Rótulos Fim": rotulos_fim,
                        "Código Fim": propriedades_fim.get('code', ''),
                        "Nome Fim": propriedades_fim.get('name', ''),
                        "Tamanho": len(relacoes)
                    })
            except Exception as e:
                print(f"Erro ao processar o Path: {e}")

        return pd.DataFrame(dados)


    def retrieve_titles(self, data, key="titulo"):
        """
        Recupera todos os títulos de publicações de uma estrutura JSON aninhada.
        
        Args:
        - data (dict): O dicionário contendo os dados.
        - key (str): A chave para buscar no dicionário (padrão é "titulo").
        
        Returns:
        - list: Uma lista contendo todos os títulos encontrados.
        """
        titles = []
        
        # Se o tipo de data for um dicionário e contiver a chave desejada
        if isinstance(data, dict):
            if key in data:
                titles.append(data[key])
            for value in data.values():
                titles.extend(self.retrieve_titles(value, key))
        
        # Se o tipo de data for uma lista, itere sobre ela
        elif isinstance(data, list):
            for item in data:
                titles.extend(self.retrieve_titles(item, key))
        
        return titles        
    

## Classe de extrair dados e gerar datasets para vetorizar
    # ExtractPublicationDataset
    # DatasetExtractor1 (usada quando operava tradução)

class ExtractPublicationDataset:
    def __init__(self):
        self.json_file_path = './../data/processed_data.json'
        self.data = self._load_data()

    def _load_data(self):
        with open(self.json_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _extract_id_lattes(self, inf_pes):
        id_lattes_info = inf_pes.get('2', '')
        id_lattes_match = re.search(r'ID Lattes: (\d+)', id_lattes_info)
        if id_lattes_match:
            id_lattes = id_lattes_match.group(1)
        else:
            id_lattes_match = re.search(r'Endereço para acessar este CV: http%3A//lattes.cnpq.br/(\d+)', id_lattes_info)
            id_lattes = id_lattes_match.group(1)
        return id_lattes if id_lattes else None

    def extract_productions_data(self, author_data):
        prod_data = author_data.get('Produções', {}).get('Produção bibliográfica', {}).get('Artigos completos publicados em periódicos', {})
        return prod_data
    
    def sep_define(self, prod_entry):
        if ' . ' in prod_entry:
            sep1 = ' . '
        elif '; ' in prod_entry:
            sep1 = '; '
        else:
            sep1 = '. '

        if 'v. ' in prod_entry:
            sep2 = '. '
        else:
            sep2 = ' '        
        return sep1,sep2

    def _extract_publication_info(self, jcr_data, jcr2_data, prod_data):
        publications = []

        for key, entry in jcr2_data.items():
            titulo_jcr2 = entry.get('titulo', '')

            publication = {
                'ano': '',
                'titulo': entry.get('titulo', ''),
                'issn': entry.get('issn', ''),
                'nomePeriodico': entry.get('journal', entry.get('original_title', '')),
                'impact_factor': entry.get('impact-factor', ''),
                'autores': '',
            }

            if not publication['impact_factor']:
                jcr_entry = jcr_data.get(key, {})
                publication['impact_factor'] = jcr_entry.get('impact-factor', '')
                    
            if not publication['issn']:
                jcr_entry = jcr_data.get(key, {})
                publication['issn'] = jcr_entry.get('data-issn', jcr_entry.get('issn'))

            if not publication['ano']:
                # Regular expression to extract year from production data
                for prod_key, prod_entry in prod_data.items():
                    if publication['titulo'] in prod_entry:
                        year_match = re.search(r'(\d{4})\.', prod_entry)
                        if year_match:
                            publication['ano'] = year_match.group(1)
                            break
                        else:
                            publication['ano'] = 2023
            # Usar dados do dicionário de Produções para complementar dados
            if publication['titulo'] != '':
                pass        
            else:
                ordem = str(int(key)+1)+'.'
                prod_entry = prod_data.get(ordem)
                sep1, sep2 = self.sep_define(prod_entry)
                publication['autores'] = prod_entry.split(sep1)[0] if prod_entry else ''
                publication['titulo']  = prod_entry.split(sep1)[1].split(sep2)[0] if prod_entry else ''                

            if publication['autores'] != '':
                pass        
            else:
                ordem = str(int(key)+1)+'.'
                prod_entry = prod_data.get(ordem)
                sep1, sep2 = self.sep_define(prod_entry)
                publication['autores'] = prod_entry.split(sep1)[0] if prod_entry else ''               

            publications.append(publication)

        return publications

    def extract_publications(self):
        all_publications = []

        for author_data in self.data:
            id_lattes = self._extract_id_lattes(author_data.get('InfPes', {}))
            jcr_data = author_data.get('JCR', {})
            jcr2_data = author_data.get('JCR2', {})
            # Extrai as ocorrências de artigos de cada currículo
            prod_data = self.extract_productions_data(author_data)

            publication_info = self._extract_publication_info(jcr_data, jcr2_data, prod_data)

            for info in publication_info:
                info['id_lattes'] = id_lattes
                all_publications.append(info)

        return all_publications

    def normalize_titles(self, input_data, file_path='./../data/clean_titles.json'):
        """
        Normaliza os títulos convertendo todos os caracteres em letras minúsculas e removendo palavras irrelevantes.
        Salva os títulos normalizados em um novo arquivo JSON com nome clean_titles.
        
        Parâmetros:
        - input_data (dict): O dicionário que contém os títulos em inglês.
        - file_path (str): O caminho onde o arquivo JSON será salvo.
        
        Retorna:
        None
        """
        # Initialize the stopwords
        nltk.download('stopwords')
        # stopwords for both English and Portuguese
        stop_words_en = set(stopwords.words('english'))
        stop_words_pt = set(stopwords.words('portuguese'))
        stop_words = stop_words_en.union(stop_words_pt)
        
        normalized_data = {}
        
        for path, entry in input_data.items():
            # Extract the translated title
            title = entry.get('titulo', '')
            
            # Convert to lowercase
            title = title.lower()
            
            # Remove stopwords
            title_words = title.split()
            filtered_words = [word for word in title_words if word not in stop_words]
            
            # Reassemble the title
            normalized_title = ' '.join(filtered_words)
            
            # Store in the normalized data dictionary
            normalized_data[path] = {'titulo': normalized_title, 'ano': entry.get('ano', 'Unknown')}
        
        # Save to disk
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(normalized_data, json_file, ensure_ascii=False, indent=4)    


class Neo4jService:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def _execute_query(self, transaction, query, parameters=None):
        result = transaction.run(query, parameters)
        return [record for record in result]

    def find_available_procedures(self):
        query = "SHOW PROCEDURES"
        return self.fetch_data(query)
    
    def find_available_functions(self):
        query = "SHOW FUNCTIONS"
        return self.fetch_data(query)

    def find_related_procedures(self, keyword):
        # Recuperamos todos os procedimentos disponíveis
        all_procedures = self.find_available_procedures()

        # Filtramos a lista de procedimentos com base na palavra-chave
        related_procedures = [proc for proc in all_procedures if keyword.lower() in proc['name'].lower()]

        return related_procedures

    def get_procedure_signatures(self, procedures_list):
        """
        Retrieves the signatures for a list of procedures.

        :param procedures_list: A list of strings representing the procedure names to fetch signatures for.
        :return: A dictionary with the procedure names as keys and their signatures as values.
        """
        # Start a string to build the Cypher query for multiple procedures
        if len(procedures_list) > 1:
            query_conditions = " OR ".join([f"name = '{procedure}'" for procedure in procedures_list])
        else:
            query_conditions = f"name = '{procedures_list[0]}'"

        get_signatures_query = f"""
        SHOW PROCEDURES
        YIELD name, signature
        WHERE {query_conditions}
        RETURN name, signature;
        """

        # print(get_signatures_query)

        # Execute the query and fetch the results
        try:
            results = self.execute_read(get_signatures_query)
            print(f"Quantidade de resultados encontrados: {len(results)}")
            # print(results)
            signatures = {record["name"]: record["signature"] for record in results}
            # for i,j in results:
            #     print(f"PROCEDIMENTO:{i}")
            #     print(f"  ASSINATURA:{j}")
            #     print()
            pprint(signatures,width=140)
            return signatures
        except Exception as e:
            print(f"An error occurred while fetching procedure signatures: {e}")
            return {}

    def list_projections(self):
        """
        Lists all the graph projections in the connected Neo4j database.

        Returns:
            A list of dictionaries, each representing a graph projection with its details.
        """
        list_projections_query = """
        CALL gds.graph.list()
        YIELD graphName, nodeCount, relationshipCount, schema
        RETURN graphName, nodeCount, relationshipCount, schema;
        """
        try:
            result = self.execute_read(list_projections_query)
            return [record for record in result]
        except Exception as e:
            print(f"An error occurred while listing projections: {e}")
            return []
        
    def fetch_data(self, query, parameters=None):
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def execute_read(self, query, parameters=None, db=None):
        with self._driver.session(database=db) as session:
            return session.execute_read(self._execute_query, query, parameters)

    def execute_write(self, query, parameters=None, db=None):
        with self._driver.session(database=db) as session:
            return session.execute_write(self._execute_query, query, parameters)

    def check_and_drop_projection(self, projection_name, db='neo4j'):
        # Check if the projection exists
        exists_query = """
        CALL gds.graph.exists($projection_name)
        YIELD exists
        RETURN exists
        """
        try:
            # Execute the query to check for existence
            exists_result = self.execute_read(exists_query, {'projection_name': projection_name}, db)
            exists = exists_result.single()['exists']
            
            # If the projection exists, drop it
            if exists:
                drop_query = "CALL gds.graph.drop($projection_name)"
                self.execute_write(drop_query, {'projection_name': projection_name}, db)
            else:
                print(f"Graph projection '{projection_name}' does not exist in database '{db}'.")
        except Exception as e:
            # Handle any exception that arises
            print(f"Exception occurred while dropping graph projection: {e}")

class CosineSimilarityRelationship(Neo4jService):
    def __init__(self, uri, user, password, model_name="default_model"):
        super().__init__(uri, user, password)
        self.model_name = model_name

    def check_connection(self):
        if not hasattr(self, '_driver') or not self._driver:
            logging.error("No database connection")
            raise Exception("No database connection")

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def normalize_similarity(self, similarity):
        return (similarity + 1) / 2

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        memory_MB = process.memory_info().rss / (1024 ** 2)
        return memory_MB

    def get_all_embeddings(self, label):
        with self._driver.session() as session:
            result = session.run(f"MATCH (n:{label}) RETURN id(n) AS id, n.embedding AS embedding")
            return [record for record in result]

    def process_similarity_for_nodes(self, source_embeddings, target_embeddings, source_label, target_label, threshold, batch_size):
        relationships_created_count = 0
        node_pairs_count = 0
        
        with self._driver.session() as session:
            tx = session.begin_transaction()
            try:
                for source in tqdm(source_embeddings, desc=f"Calculando similaridade semântica entre: {source_label}/{target_label}"):
                    for target in target_embeddings:
                        similarity = self.cosine_similarity(np.array(source["embedding"]), np.array(target["embedding"]))
                        normalized_similarity = self.normalize_similarity(similarity)
                        node_pairs_count += 1
                        if similarity > threshold:
                            query = f"""
                            MATCH (source:{source_label}) WHERE id(source) = $source_id
                            MATCH (target:{target_label}) WHERE id(target) = $target_id
                            MERGE (source)-[:SIMILAR {{score: $similarity, weight: $normalized_similarity}}]->(target)
                            """
                            tx.run(query, source_id=source["id"], target_id=target["id"], similarity=float(similarity), normalized_similarity=float(normalized_similarity))
                            relationships_created_count += 1
                            if relationships_created_count % batch_size == 0:
                                tx.commit()
                                # logging.info(f"Committed {batch_size} relationships for {source_label}/{target_label}.")
                                tx = session.begin_transaction()
                # Commit any remaining relationships after the loop
                tx.commit()
                # logging.info(f"Committed remaining relationships for {source_label}/{target_label}.")
            except Exception as e:
                logging.error(f"An error occurred during transaction: {e}")
                tx.rollback()
                raise
            return node_pairs_count, relationships_created_count

    def create_similarity_embeedings_relationships(self, threshold=0.7, batch_size=3000):
        start_time = time.time()
        self.check_connection()
        initial_memory = self.get_memory_usage()

        pub_embeddings = self.get_all_embeddings("Publicacao")
        esp_embeddings = self.get_all_embeddings("Especialidade")
        sub_embeddings = self.get_all_embeddings("Subárea")

        total_pub = len(pub_embeddings)
        total_esp = len(esp_embeddings)
        total_sub = len(sub_embeddings)

        logging.info(f"Nodes: Publicacao {total_pub}, Especialidade {total_esp}, Subárea {total_sub}")

        pub_sub_pairs, pub_sub_rels = self.process_similarity_for_nodes(pub_embeddings, sub_embeddings, "Publicacao", "Subárea", threshold, batch_size)
        pub_esp_pairs, pub_esp_rels = self.process_similarity_for_nodes(pub_embeddings, esp_embeddings, "Publicacao", "Especialidade", threshold, batch_size)

        final_memory = self.get_memory_usage()
        end_time = time.time()
        memory_difference = final_memory - initial_memory
        processing_time = end_time - start_time

        logging.info(f"RAM Consumption: {np.round(memory_difference,2)} MB")
        logging.info(f"Processing Time: {np.round(processing_time,2)} seconds")
        logging.info(f"Current Memory Usage: {np.round(final_memory,2)} MB")
        logging.info(f"Execution time for similarity calculations and relationship creation: {np.round(processing_time, 2)} seconds")
        logging.info(f"Similarity threshold: {threshold}")
        logging.info(f"Total node pairs analyzed: {pub_sub_pairs + pub_esp_pairs}")
        logging.info(f"Node pairs Publicacao/Subárea: {pub_sub_pairs}")
        logging.info(f"Node pairs Publicacao/Especialidade: {pub_esp_pairs}")
        logging.info(f"Total relationships created: {pub_sub_rels + pub_esp_rels}")
        logging.info(f"Relationships created Publicacao/Subárea: {pub_sub_rels}")
        logging.info(f"Relationships created Publicacao/Especialidade: {pub_esp_rels}")

    def run_similarity_operations(self, threshold=0.7):
        self.create_similarity_embeedings_relationships(threshold)


def plotar_publicacoes_por_id_seaborn(dados):
    # Contar as publicações por id_lattes
    contador = {}
    for item in dados:
        id_lattes = item['id_lattes']
        if id_lattes in contador:
            contador[id_lattes] += 1
        else:
            contador[id_lattes] = 1

    # Preparar os dados para o gráfico
    ids = list(contador.keys())
    publicacoes = list(contador.values())

    # Criar o gráfico de barras com Seaborn
    plt.figure(figsize=(19, 6))
    barplot = sns.barplot(x=ids, y=publicacoes, palette="viridis")
    plt.xlabel('ID Lattes')
    plt.ylabel('Número de Publicações')
    plt.title('Publicações por ID Lattes')
    plt.xticks(rotation=45, ha='right')

    # Adicionar rótulos de dados
    for p in barplot.patches:
        barplot.annotate(f'{int(p.get_height())}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha = 'center', va = 'center', 
                         xytext = (0, 9), 
                         textcoords = 'offset points')

    # Remover linhas de grade
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plotar_evolucao_publicacoes(dados, ano_inicio, ano_fim):
    # Preparar os dados
    lista_publicacoes = []
    for item in dados:
        ano_str = item['ano'].strip()
        if ano_str.isdigit():
            ano = int(ano_str)
            if ano_inicio <= ano <= ano_fim:
                lista_publicacoes.append((item['id_lattes'], ano))

    if not lista_publicacoes:
        print("Nenhum dado válido encontrado para o intervalo de anos especificado.")
        return

    df = pd.DataFrame(lista_publicacoes, columns=['id_lattes', 'ano'])
    df['count'] = 1
    df = df.groupby(['id_lattes', 'ano']).count().reset_index()

    # Criar o gráfico de linhas
    plt.figure(figsize=(19, 10))
    sns.lineplot(x='ano', y='count', hue='id_lattes', data=df, marker="o")
    plt.xlabel('Ano')
    plt.ylabel('Quantidade de Publicações')
    plt.title(f'Evolução do Número de Publicações por ID Lattes ({ano_inicio}-{ano_fim})')
    plt.legend(title='ID Lattes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(False)
    plt.tight_layout()
    plt.show()


    def plotar_boxplot_publicacoes(dados, ano_inicio=2000, ano_fim=2023):
        # Preparar os dados
        lista_publicacoes = []
        for item in dados:
            ano_str = item['ano'].strip()  # Remover espaços em branco extras

            # Verificar se o ano é um valor válido e está dentro do intervalo desejado
            if ano_str.isdigit() and ano_inicio <= int(ano_str) <= ano_fim:
                lista_publicacoes.append((item['id_lattes'], int(ano_str)))

        # Se não houver dados válidos, retornar
        if not lista_publicacoes:
            print("Nenhum dado válido encontrado para o intervalo de anos especificado.")
            return

        df = pd.DataFrame(lista_publicacoes, columns=['id_lattes', 'ano'])
        df['count'] = 1  # Adicionar uma coluna para contar as publicações
        df = df.groupby(['id_lattes', 'ano']).count().reset_index()  # Agrupar e contar

        # Ordenar os dados por ano
        df = df.sort_values('ano')

        # Criar o boxplot
        plt.figure(figsize=(19, 8))
        sns.boxplot(x='ano', y='count', data=df)
        plt.xlabel('Ano')
        plt.ylabel('Quantidade de Publicações')
        plt.title(f'Dispersão da Quantidade de Publicações por Ano para Cada ID Lattes ({ano_inicio}-{ano_fim})')
        plt.xticks(rotation=45, ha='right')
        plt.grid(False)
        plt.tight_layout()
        plt.show()                    