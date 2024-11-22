# import difflib, subprocess, string, sqlite3, asyncio, nltk, openpyxl, glob, stat, shutil, psutil
import warnings, platform, requests, urllib, logging, traceback, codecs, unicodedata
import os, re, bs4, time, json, h5py, pytz, pdfkit, sys, csv
import plotly.express.colors as px_colors
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np

import torch.optim as optim
import torch.nn as nn
import torch

from PIL import Image
from io import BytesIO
from pathlib import Path
from pprint import pprint
from zipfile import ZipFile
from string import Formatter
from PyPDF2 import PdfReader
from neo4j import GraphDatabase
from unidecode import unidecode
from Levenshtein import distance
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from Levenshtein import jaro_winkler
from urllib3.util.retry import Retry
from tqdm.notebook import trange, tqdm
from datetime import datetime, timedelta
from flask import render_template_string
from requests.adapters import HTTPAdapter
from urllib.parse import urlparse, parse_qs
from py2neo import Graph, Node, Relationship
from sklearn.metrics import silhouette_score
from typing import List, Dict, Any, Optional, Union
from collections import deque, defaultdict, Counter
from bs4 import BeautifulSoup, Tag, NavigableString
from pyjarowinkler.distance import get_jaro_distance
from IPython.display import clear_output, display, HTML
from typing import Tuple, List, Dict, Any, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common import exceptions
from selenium.common.exceptions import (
    StaleElementReferenceException,
    NoSuchElementException, 
    TimeoutException,
    ElementNotInteractableException,
    WebDriverException
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class PlotProduction:
    def __init__(self, df):
        """
        Inicializa a classe com o DataFrame de entrada.

        Args:
            df (pd.DataFrame): DataFrame com a estrutura especificada na pergunta.
                               Os índices são os nomes dos pesquisadores e as colunas são os anos.
        """
        self.df = df
        self.df = self.df.reset_index()
        self.df = self.df.rename(columns={'index': 'Autor'})
        self.df['Autor'] = self.df['Autor'].astype(str)

    def preparar_dados(self):
        """
        Prepara os dados para plotagem, anonimizando os pesquisadores e reestruturando o DataFrame.
        """

        # Cria uma nova coluna 'ID' com IDs anônimos para os pesquisadores
        self.df['ID'] = 'p' + (self.df.index + 1).astype(str)

        try:
            # Remove a coluna 'Pesquisador' original
            self.df.drop('Autor', axis=1, inplace=True)

            # Remove a coluna 'Pesquisador' original
            self.df.drop('Contagem', axis=1, inplace=True)        
        except:
            pass
        
        # Reestrutura o DataFrame para o formato longo
        self.df = self.df.melt(id_vars='ID', value_vars=self.df.columns[:-1],
                                 var_name='Ano', value_name='Quantidade')

        return self.df

    def plotar_linhas(self):
        """
        Plota o gráfico de linhas com Plotly Express.
        """

        df_longo = self.preparar_dados()

        fig = px.line(df_longo, x='Ano', y='Quantidade', color='ID',
              title='Produção de artigos por ano, segmentada por autor',
              width=800, height=900)

        # Ajustando o eixo x para mostrar todos os anos
        fig.update_xaxes(type='category')

        # Adicionando marcadores manualmente
        fig.update_traces(mode='lines+markers')

        fig.show()
        
    def ajustar_posicao_rotulos(self, df_longo, df_anotacoes):
        """
        Calcula a posição correta dos rótulos no eixo y para o modo 'overlay'.
        """

        # Agrupa df_longo por 'Ano' e 'ID' e calcula a soma acumulada de 'Quantidade'
        df_longo['Soma Acumulada'] = df_longo.groupby(['Ano', 'ID'])['Quantidade'].cumsum()

        # Calcula a posição y dos rótulos adicionando um pequeno deslocamento à soma acumulada
        df_longo['Posicao Rotulo'] = df_longo['Soma Acumulada'] - df_longo['Quantidade'] / 2

        # Retorna as posições y dos rótulos
        return df_longo['Posicao Rotulo']

    def plotar_barras_empilhadas(self, barmode='group'):
        """
        Plota o gráfico de barras empilhadas com Plotly Express, incluindo anotações com a
        quantidade total de artigos e a contagem de pesquisadores por ano.
        """

        df_longo = self.preparar_dados()

        # Converte a coluna 'Ano' para str
        df_longo['Ano'] = df_longo['Ano'].astype(str)

        # # Normaliza os valores de 'Quantidade' para o intervalo [0, 1]
        # df_longo['Quantidade_normalizada'] = (df_longo['Quantidade'] - df_longo['Quantidade'].min()) / (df_longo['Quantidade'].max() - df_longo['Quantidade'].min())

        # Calcula a quantidade total de artigos por ano
        total_artigos_ano = df_longo.groupby('Ano')['Quantidade'].sum().reset_index(name='Total Artigos')

        # Calcula a contagem de pesquisadores que publicaram por ano
        contagem_pesquisadores_ano = df_longo.groupby('Ano')['ID'].nunique().reset_index(name='Pesquisadores')

        # Junta as informações em um novo DataFrame
        df_anotacoes = pd.merge(contagem_pesquisadores_ano, total_artigos_ano, on='Ano')

        # Apura quantidades totais e formata as anotações
        df_anotacoes['Anotacao'] = df_anotacoes.apply(lambda row: f"{row['Total Artigos']}", axis=1)

        fig = px.bar(df_longo, 
                     x='Ano', 
                     y='Quantidade',
                     color='Quantidade',
                     title='Publicações de Artigos por Ano',
                     color_continuous_scale=px.colors.sequential.Greens,
                     barmode=barmode,
                     width=800, height=600)

        # Ajustando o eixo x para mostrar todos os anos
        fig.update_xaxes(type='category')
        
        # Remove a legenda
        fig.update_layout(showlegend=False)

        # Adiciona as anotações
        anos = df_longo['Ano'].unique()
        y = df_anotacoes['Total Artigos']
        text = df_anotacoes['Anotacao']

        for i, ano in enumerate(anos):
            fig.add_annotation(x=i, y=y[i], text=text[i], showarrow=False, font=dict(size=14), yshift=10)

        fig.show()

    def plotar_barras_periodo(self, ano_inicio=None, ano_fim=None, barmode='stack'):
        """
        Plota o gráfico de barras com Plotly Express, incluindo anotações com a
        quantidade total de artigos e a contagem de pesquisadores por ano.
        """
        df_longo = self.preparar_dados()

        # Converte a coluna 'Ano' para string
        df_longo['Ano'] = df_longo['Ano'].astype(str)

        # Remove caracteres não numéricos da coluna 'Ano'
        df_longo['Ano'] = df_longo['Ano'].str.replace('[^0-9]', '', regex=True)

        # Converte a coluna 'Ano' para inteiro
        df_longo['Ano'] = pd.to_numeric(df_longo['Ano'])

        # Filtrando os dados por ano
        if ano_inicio:
            df_longo = df_longo[df_longo['Ano'].astype(int) >= int(ano_inicio)] # Correção aqui
        if ano_fim:
            df_longo = df_longo[df_longo['Ano'].astype(int) <= int(ano_fim)]   # Correção aqui

        fig = px.bar(df_longo, x='Ano', 
                     y='Quantidade', 
                     color='Quantidade',
                     title='Produção de artigos por ano, segmentada por autor',
                     color_continuous_scale=px.colors.sequential.Greens, 
                     barmode='group',
                     width=800, height=600)

        # Ajustando o eixo x para mostrar todos os anos
        fig.update_xaxes(type='category')

        fig.show()

    def plotar_barras(self, barmode='stack'): # Corrigido: indentação alinhada com outras funções
        """
        Plota o gráfico de barras com Plotly Express, incluindo anotações com a
        quantidade total de artigos e a contagem de pesquisadores por ano.
        """

        df_longo = self.preparar_dados()

        # Converte a coluna 'Ano' para str
        df_longo['Ano'] = df_longo['Ano'].astype(str)

        # Calcula a quantidade total de artigos por ano
        total_artigos_ano = df_longo.groupby('Ano')['Quantidade'].sum().reset_index(name='Total Artigos')

        # Calcula a contagem de pesquisadores que publicaram por ano
        contagem_pesquisadores_ano = df_longo.groupby('Ano')['ID'].nunique().reset_index(name='Pesquisadores')

        # Junta as informações em um novo DataFrame
        df_anotacoes = pd.merge(contagem_pesquisadores_ano, total_artigos_ano, on='Ano')

        # Apura quantidades totais e formata as anotações
        df_anotacoes['Anotacao'] = df_anotacoes.apply(lambda row: f"{row['Total Artigos']}", axis=1)

        fig = px.bar(df_longo, 
                     x='Ano', 
                     y='Quantidade', 
                     color='Quantidade',
                     title='Produção de artigos por ano, segmentada por autor',
                     color_continuous_scale=px.colors.sequential.Greens, 
                     width=800, height=600,
                     barmode=barmode)

        # Ajustando o eixo x para mostrar todos os anos
        fig.update_xaxes(type='category')
        
        # Remove a legenda
        fig.update_layout(showlegend=False)
        
        # Adiciona as anotações
        anos = df_longo['Ano'].unique()
        y_max = df_longo.groupby('Ano')['Quantidade'].max().tolist()  # Calcula o valor máximo de Quantidade para cada ano
        text = df_anotacoes['Anotacao']

        for i, ano in enumerate(anos):
            fig.add_annotation(x=i, y=y_max[i], text=text[i], showarrow=False, font=dict(size=14), yshift=10)

        fig.show()

    def plotar_boxplot(self):
        """
        Plota o gráfico de boxplot com Plotly Express para cada ano.
        """

        df_longo = self.preparar_dados()

        # Converte a coluna 'Ano' para str
        df_longo['Ano'] = df_longo['Ano'].astype(str)

        fig = px.box(df_longo, 
                     x='Ano', 
                     y='Quantidade',
                     title='Distribuição da produção de artigos por ano',
                     width=800, height=600)

        # Ajustando o eixo x para mostrar todos os anos
        fig.update_xaxes(type='category')
        
        # Remove a legenda
        fig.update_layout(showlegend=False)

        fig.show()


class JSONFileManager:
    def list_json(self, folder):
        # Criar uma lista para armazenar os nomes dos arquivos JSON
        json_files = []

        for i in os.listdir(folder):
            try:
                ext = i.split('.')[1]
                if ext == 'json':
                    json_files.append(i)
            except IndexError:
                # Ignora arquivos sem extensão
                pass

        # Ordenar a lista de arquivos JSON em ordem alfabética
        json_files.sort()

        # Imprimir os arquivos JSON em ordem
        print('Arquivos disponíveis na pasta para dados de entrada:')
        for file in json_files:
            print(f'  {file}')

        return json_files

    # Carregar arquivo 'dict_list.json' para a variável com dados de criação e modificação
    def load_from_json(self, file_path):
        """
        Carrega um arquivo JSON e retorna seu conteúdo e data de criação.
        Parâmetros:
            file_path (str): O caminho para o arquivo JSON.
        Retorna:
            dict, str: O conteúdo do arquivo JSON e sua data de criação.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Obter datas de criação e modificação do arquivo
        creation_date = os.path.getctime(file_path)
        modification_date = os.path.getmtime(file_path)

        # Converter timestamps para datas no fuso horário de Brasília
        brasilia_tz = pytz.timezone('America/Sao_Paulo')
        creation_date_brasilia = datetime.fromtimestamp(creation_date).astimezone(brasilia_tz)
        modification_date_brasilia = datetime.fromtimestamp(modification_date).astimezone(brasilia_tz)

        # Formatar datas com dd/mm/aaaa hh:mm:ss
        formatted_creation_date = creation_date_brasilia.strftime("%d/%m/%Y %H:%M:%S")
        formatted_modification_date = modification_date_brasilia.strftime("%d/%m/%Y %H:%M:%S")

        # Calcular contagem de horas até a data atual
        now = datetime.now(brasilia_tz)
        time_delta = now - modification_date_brasilia

        # Determinar unidade de tempo (minutos, horas ou dias)
        if time_delta.total_seconds() < (60*60):
            unit = "minutos"
            time_count = round(time_delta.total_seconds() / 60, 1)
        elif time_delta.total_seconds() < (60*60*24):
            unit = "horas"
            time_count = round(time_delta.total_seconds() / 3600, 1)
        else:
            unit = "dias"
            time_count = round(time_delta.total_seconds() / 86400, 1)

        return data, formatted_creation_date, formatted_modification_date, time_count, unit

    def verifify_json(self, folder, filename):
        pathfilename = os.path.join(folder,filename)
        dict_list, formatted_creation_date, formatted_modification_date, time_count, unit = self.load_from_json(os.path.join(pathfilename))
        print(f"\n{len(dict_list)} currículos carregados na lista de dicionários '{filename}'")
        # print(f"Arquivo criado inicialmente em {formatted_creation_date} carregado com sucesso")
        print(f"Extração realizada em {formatted_modification_date} a {time_count} {unit}")    

    # Função para salvar a lista de dicionários em um arquivo .json
    def save_to_json(self, data, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


class attribute_to_be_non_empty:
    """
    An expectation for checking that an attribute of a specific element is not empty,
    and return the value of the attribute.
    """
    def __init__(self, locator: Tuple[By, str], attribute: str):
        self.locator = locator
        self.attribute = attribute

    def __call__(self, driver):
        element = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(self.locator))
        attribute_value = element.get_attribute(self.attribute)
        if attribute_value:  # Se o valor do atributo não for vazio, retorne o valor.
            return attribute_value
        else:
            return False  # Retorne False caso contrário.

class SoupParser:
    def __init__(self, driver):
        self.configure_logging()
        self.verbose = False
        self.base_url = 'http://buscatextual.cnpq.br'
        self.driver = driver
        self.delay = 10
        self.soup = None

    def configure_logging(self):
        logging.basicConfig(filename='lattes_scraper.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.driver.quit()
        self.driver = None

    def serialize_to_file(self, data: Dict, filename: str, format: str = "json"):
        try:
            if format == "json":
                with open(filename, 'w') as f:
                    json.dump(data, f)
            elif format == "hdf5":
                with h5py.File(filename, 'w') as hdf5_file:
                    for key, value in data.items():
                        hdf5_file.create_dataset(key, data=json.dumps(value))
            else:
                logging.error("Unsupported format specified for serialization.")
        except Exception as e:
            logging.error(f"An error occurred while saving data: {e}")

    def extract_section_data(self, soup, section_titles: List[str]) -> Dict:
        data_dict = {}
        for title in section_titles:
            sections = soup.find_all('div', text=title)
            for section in sections:
                data_dict[title] = self.extract_data_from_section(section)
        return data_dict

    def extract_data_from_section(self, section) -> Dict:
        # Placeholder for actual logic to extract data from given section
        data = {}
        # Example of extracting data
        if self.verbose:
            print(f"Extracting data from section: {section.text}")
        return data

    def extract_all_data(self, soup: BeautifulSoup) -> Dict:
        # Define titles or markers of sections to extract
        section_titles = ['Identificação', 'Formação acadêmica/titulação', 'Eventos']
        all_data = {}
        for title in section_titles:
            all_data.update(self.extract_section_data(soup, [title]))
        return all_data

    def parse_soup(self, soup: BeautifulSoup) -> Dict:
        """
        Parses the BeautifulSoup object and extracts relevant data.
        """
        self.soup = soup
        return self.extract_all_data(soup)

    def extract_and_save_data(self, soup: BeautifulSoup, json_filename: str, hdf5_filename: str):
        data = self.parse_soup(soup)
        self.serialize_to_file(data, json_filename, format="json")
        self.serialize_to_file(data, hdf5_filename, format="hdf5")

class DictToHDF5:
    def __init__(self, data_list):
        self.data_list = data_list

    def create_dataset(self, filename, directory=None):
        with h5py.File(f"{directory or ''}{filename}", "w") as f:
            null_group = f.create_group("0000")
            for person_dict in self.data_list:  # Corrigido de self.data para self.data_list
                if 'curriculo' not in person_dict:
                    name = person_dict.get('name', 'Unknown')  # Uso de get() para evitar KeyError
                    null_group.attrs[name] = "No curriculum"  # Adicionando como atributos ao grupo '0000'
                    continue
                person_group = f.create_group(person_dict['id'])
                for key, value in person_dict.items():
                    if value is None:
                        continue
                    if isinstance(value, list):
                        if not value:  # Skip empty lists
                            continue

                        dtype = type(value[0])
                        if dtype == str:
                            dt = h5py.string_dtype(encoding='utf-8')
                            person_group.create_dataset(key, (len(value),), dtype=dt, data=value)
                        else:
                            value = np.array(value, dtype=dtype)
                            person_group.create_dataset(key, data=value)
                    elif isinstance(value, str):
                        dt = h5py.string_dtype(encoding='utf-8')
                        person_group.create_dataset(key, (1,), dtype=dt, data=value)
                    elif isinstance(value, dict) or isinstance(value, list):
                        json_str = json.dumps(value)
                        dt = h5py.string_dtype(encoding='utf-8')
                        person_group.create_dataset(key, (1,), dtype=dt, data=json_str)
                    else:
                        person_group.create_dataset(key, data=value)

    def extract_id_lattes(self, data_dict):
        inf_pes = data_dict.get('InfPes', [])
        for item in inf_pes:
            if 'ID Lattes:' in item:
                return item.split('ID Lattes: ')[-1]
        return "0000" + str(data_dict.get("name"))

    # Para visualização
    @staticmethod
    def print_hdf5_structure(hdf5_file_path: str, indent=0):
        """
        Imprime a estrutura do arquivo HDF5 com informações adicionais.
        
        Parâmetros:
        - hdf5_file_path: str, caminho para o arquivo HDF5.
        - indent: int, nível de recuo para representação hierárquica (default=0).
        """
        with h5py.File(hdf5_file_path, 'r') as file:
            DictToHDF5._print_hdf5_group_structure(file, indent)
            
    @staticmethod
    def _print_hdf5_group_structure(group, indent=0):
        """
        Auxilia na impressão da estrutura hierárquica do grupo HDF5.
        
        Parâmetros:
        - group: h5py.Group, grupo HDF5 para imprimir.
        - indent: int, nível de recuo para representação hierárquica.
        """
        for key, value in group.items():
            print(" " * indent + f"{key} : {str(value)}")

            if isinstance(value, h5py.Group):
                DictToHDF5._print_hdf5_group_structure(value, indent + 4)
                
            elif isinstance(value, h5py.Dataset):
                # Informações adicionais sobre o conjunto de dados
                print(" " * (indent + 4) + f"Shape: {value.shape}, Dtype: {value.dtype}")
                
                if value.size:
                    # Visualização do conteúdo do conjunto de dados
                    if value.size < 10:  # imprimir todo o conjunto de dados se ele for pequeno
                        print(" " * (indent + 4) + f"Data: {value[...]}")
                    else:  # imprimir uma amostra dos dados se o conjunto de dados for grande
                        sample = value[...]
                        if value.ndim > 1:
                            sample = sample[:min(3, value.shape[0]), :min(3, value.shape[1])]
                        else:
                            sample = sample[:min(3, value.size)]
                        print(" " * (indent + 4) + f"Sample Data: {sample}")

    @staticmethod
    def print_json_structure(json_file_path: str, indent: int = 0, max_sample_size: int = 10):
        """
        Imprime a estrutura e os dados do arquivo JSON.

        Parâmetros:
        - json_file_path: str, caminho para o arquivo JSON.
        - indent: int, nível de recuo para representação hierárquica (default=0).
        - max_sample_size: int, tamanho máximo da amostra para visualização (default=10).
        """
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
            DictToHDF5._print_json_data_structure(json_data, indent, max_sample_size)

    @staticmethod
    def _print_json_data_structure(json_data, indent: int, max_sample_size: int):
        """
        Auxilia na impressão da estrutura e dos dados de um objeto JSON.

        Parâmetros:
        - json_data: qualquer tipo de dado serializável em JSON.
        - indent: int, nível de recuo para representação hierárquica.
        - max_sample_size: int, tamanho máximo da amostra para visualização.
        """
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                print(" " * indent + f"{key}: {type(value).__name__}")
                DictToHDF5._print_json_data_structure(value, indent + 4, max_sample_size)

        elif isinstance(json_data, list):
            print(" " * indent + f"List of length {len(json_data)}")
            sample = json_data[:min(len(json_data), max_sample_size)]
            for index, value in enumerate(sample):
                print(" " * (indent + 4) + f"[{index}]: {type(value).__name__}")
                DictToHDF5._print_json_data_structure(value, indent + 8, max_sample_size)

        else:
            print(" " * indent + f"{json_data}")

    # Para persisistir em N4j
    def persist_to_neo4j(self, filepath, neo4j_url, username, password):
        graph = Graph(neo4j_url, auth=(username, password))
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                group = f[key]
                properties = {}
                if key == '0000':  # Tratar grupo "0000" diferentemente
                    for attr_name, attr_value in group.attrs.items():
                        properties[attr_name] = attr_value
                    node = Node("NoCurriculumGroup", **properties)  # Criação de um nó específico para o grupo
                else:
                    for ds_key in group.keys():
                        dataset = group[ds_key]
                        properties[ds_key] = dataset[()]
                    node = Node("Person", **properties)  # Assumindo que o nó seja do tipo "Person"
                graph.create(node)

class LattesScraper:
    def __init__(self, search_terms, neo4j_uri, neo4j_user, neo4j_password, only_doctors=False):
        self.verbose = False
        self.configure_logging()
        self.driver = self.connect_driver(only_doctors)
        self.only_doctors = only_doctors
        self.search_terms = search_terms
        self.session = requests.Session()
        self.delay = 30
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

    def configure_logging(self):
        logging.basicConfig(filename='logs/lattes_scraper.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

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
        return LattesScraper.find_repo_root(path.parent, depth-1)

    @staticmethod
    def connect_driver(only_doctors):
        '''
        Conecta ao servidor do CNPq para busca de currículo
        '''
        # print(f'Conectando com o servidor do CNPq...')
        # print(f'Iniciada extração de {len(lista_nomes)} currículos')
        ## https://www.selenium.dev/documentation/pt-br/webdriver/browser_manipulation/
        # options   = Options()
        # options.add_argument("--headless")
        # driver   = webdriver.Chrome(options=options)

        driver_path = None
        try:
            # Caminho para o chromedriver no sistema local
            if platform.system() == "Windows":
                driver_path=LattesScraper.find_repo_root(os.getcwd())/'chromedriver'/'chromedriver.exe'
            else:
                driver_path=LattesScraper.find_repo_root(os.getcwd())/'chromedriver'/'chromedriver'
        except Exception as e:
            print("Não foi possível estabelecer uma conexão, verifique o chromedriver")
            print(e)
        
        # print(driver_path)
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service)
        driver.set_window_position(-20, -10)
        driver.set_window_size(170, 1896)
        # only_doctors = True
        if only_doctors:
            print('Buscando currículos apenas entre nível de doutorado')
            url_docts = 'http://buscatextual.cnpq.br/buscatextual/busca.do?buscarDoutores=true&buscarDemais=false&textoBusca='
            driver.get(url_docts) # acessa a url de busca somente de doutores 
        else:
            print('Buscando currículos com qualquer nível de formação')
            url_busca = 'http://buscatextual.cnpq.br/buscatextual/busca.do?buscarDoutores=true&buscarDemais=true&textoBusca='
            driver.get(url_busca) # acessa a url de busca do CNPQ
            # Localize o elemento do checkbox
            checkbox = driver.find_element(By.ID, "buscarDemais")
            # try:
            #     checkbox = WebDriverWait(driver, 10).until(
            #         EC.element_to_be_clickable((By.CSS_SELECTOR, "div.input-checkbox input[type='checkbox']"))
            #     )

            #     action_chains = ActionChains(driver)
            #     action_chains.move_to_element(checkbox).perform()

            #     if not checkbox.is_selected():
            #         driver.execute_script("arguments[0].click();", checkbox)

            # except Exception as e:
            #     print(f"Erro ao localizar checkbox: {e}")
            #     driver.save_screenshot("erro_screenshot.png") 
            # Verifique se o checkbox está marcado
            if not checkbox.is_selected():
                # Se o checkbox não estiver marcado, mova o mouse até ele e clique
                actions = ActionChains(driver)
                actions.move_to_element(checkbox).click().perform()
            # driver.get(url_busca) # acessa a url de busca do CNPQ
        driver.mouse = webdriver.ActionChains(driver)
        return driver

    def strfdelta(self, tdelta, fmt='{H:02}h {M:02}m {S:02}s', inputtype='timedelta'):
        if inputtype == 'timedelta':
            remainder = int(tdelta.total_seconds())
        else:
            conversion_factors = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400, 'w': 604800}
            remainder = int(tdelta) * conversion_factors[inputtype]
        f = Formatter()
        desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
        possible_fields = ('W', 'D', 'H', 'M', 'S')
        constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
        values = {}
        for field in possible_fields:
            if field in desired_fields and field in constants:
                values[field], remainder = divmod(remainder, constants[field])
        return f.format(fmt, **values)
    
    def tempo(self, start, end):
        t = end - start
        tempo = timedelta(
            weeks=t // (3600 * 24 * 7),
            days=t // (3600 * 24) % 7,
            hours=t // 3600 % 24,
            minutes=t // 60 % 60,
            seconds=t % 60
        )
        fmt = '{H:02}:{M:02}:{S:02}'
        return self.strfdelta(tempo, fmt=fmt, inputtype='timedelta')

    # Salvar a lista de dicionários em um arquivo .json
    def save_to_json(self, data, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    # Normalizar os nomes de autores
    def normalizar_nome(self, nome):
        """Normaliza um nome, removendo acentos e caracteres especiais, convertendo para minúsculas e padronizando espaços."""

        # Remover acentos e caracteres especiais
        nome_sem_acentos = unidecode(nome)

        # Converter para minúsculas e remove espaços extras
        nome_normalizado = nome_sem_acentos.lower().strip().replace("  ", " ")

        return nome_normalizado

    # def scrape_and_persist(self, data):
    #     self._scrape(data)
    #     self._persist(data)

    def _persist(self, data):
        # Conectar ao banco de dados Neo4j
        graph = Graph(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))

        # Percorrer os dados e persistir no Neo4j
        for pessoa in data:
            # Criar nó para a pessoa
            pessoa_node = Node("Pessoa", nome=pessoa["Identificação"]["Nome"])
            graph.create(pessoa_node)

            # Criar nós para idiomas e relacionamentos com a pessoa
            for idioma in pessoa["Idiomas"]:
                idioma_node = Node("Idioma", nome=idioma["Idioma"])
                graph.create(idioma_node)
                rel = Relationship(pessoa_node, "FALA", idioma_node)
                graph.create(rel)

            # Criar nós para formações e relacionamentos com a pessoa
            for formacao in pessoa["Formação"]["Acadêmica"]:
                formacao_node = Node("Formacao", ano=formacao["Ano"], descricao=formacao["Descrição"])
                graph.create(formacao_node)
                rel = Relationship(pessoa_node, "FORMOU", formacao_node)
                graph.create(rel)

            # Criar nós para projetos de pesquisa e relacionamentos com a pessoa
            for projeto in pessoa["ProjetosPesquisa"]:
                projeto_node = Node("ProjetoPesquisa", titulo=projeto["titulo_projeto"], descricao=projeto["descricao"])
                graph.create(projeto_node)
                rel = Relationship(pessoa_node, "PARTICIPA", projeto_node)
                graph.create(rel)

    def retry(self, func, expected_ex_type=Exception, limit=0, wait_ms=500,
              wait_increase_ratio=2, on_exhaust="throw"):
        attempt = 1
        while True:
            try:
                return func()
            except Exception as ex:
                if not isinstance(ex, expected_ex_type):
                    raise ex
                if 0 < limit <= attempt:
                    if on_exhaust == "throw":
                        raise ex
                    return on_exhaust
                attempt += 1
                time.sleep(wait_ms / 1000)
                wait_ms *= wait_increase_ratio

    def wait_for_element(self, css_selector: str, ignored_exceptions=None):
        WebDriverWait(self.driver, self.delay, ignored_exceptions=ignored_exceptions).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)))

    def handle_stale_file_error(self, max_retries=5, retry_interval=10):
        for attempt in range(max_retries):
            try:
                error_div = self.driver.find_element(By.CSS_SELECTOR, 'resultado')
                linha1 = error_div.find_element('li')
                if 'Stale file handle' in linha1.text:
                    time.sleep(retry_interval)
                else:
                    return True
            except NoSuchElementException:
                return True
        return False

    def extract_data_from_cvuri(self, element) -> dict:
        """
        Extracts data from the cvuri attribute of the given element.
        :param element: WebElement object
        :return: Dictionary of extracted data
        """
        cvuri = element.get_attribute('cvuri')
        parsed_url = urlparse(cvuri)
        params = parse_qs(parsed_url.query)
        data_dict = {k: v[0] for k, v in params.items()}
        return data_dict

    # Usar só em caso onde a busca for somente na base de doutores, pois carrega página de busca com check 'demais pesquisadores' desabilitado
    def new_search(self):
        try:
            WebDriverWait(self.driver, self.delay).until(
                EC.element_to_be_clickable((By.ID, "botaoBuscaFiltros"))).click()
            if self.verbose:
                print("Efetuando nova busca...")
        except Exception as e:
            print(f"Erro ao clicar em Nova consulta: {e}")

    # Usar para retonar para página de buscas por todos os níveis de formação e nacionalidades        
    def return_search_page(self):
        flag = str(not self.only_doctors).lower()
        url_busca = f'http://buscatextual.cnpq.br/buscatextual/busca.do?buscarDoutores=true&buscarDemais={flag}&textoBusca='
        self.driver.get(url_busca) # acessa a url de busca do CNPQ

    def switch_to_new_window(self):
        # Esperar até que uma nova janela seja aberta
        WebDriverWait(self.driver, self.delay).until(EC.number_of_windows_to_be(2))
        original_window = self.driver.current_window_handle
        new_window = [window for window in self.driver.window_handles if window != original_window][0]
        self.driver.switch_to.window(new_window)
        return original_window

    def switch_back_to_original_window(self):
        # Armazenar o handle da janela original (primeira aba aberta)
        original_window = self.driver.window_handles[0]

        # Verificar se existem mais de uma aba aberta
        if len(self.driver.window_handles) > 1:
            # Muda o foco para a última aba aberta
            self.driver.switch_to.window(self.driver.window_handles[-1])
            # Fecha a última aba
            self.driver.close()

        # Voltar o foco para a janela original
        self.driver.switch_to.window(original_window)

    def close_all_other_tabs(self, wait_time=1):
        original_window = self.driver.window_handles[0]
        for window_handle in self.driver.window_handles:
            if window_handle != original_window:
                self.driver.switch_to.window(window_handle)
                self.driver.close()
                time.sleep(wait_time) 
        self.driver.switch_to.window(original_window)

    def close_current_tab(self):
        # Mudar para a aba que deseja fechar
        self.driver.switch_to.window(self.driver.window_handles[-1])
        # Enviar a combinação de teclas para fechar a aba
        try:
            print("Tentando fechar aba corrente com Ctrl+w...")
            self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.CONTROL + 'w')
        except:
            print("Tentando fechar aba corrente com Ctrl+F4...")
            self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.CONTROL + Keys.F4)

        # Aguardar um momento para o fechamento da aba
        time.sleep(1)

        # Opcional: voltar para a aba principal se necessário
        if len(self.driver.window_handles) > 1:
            self.driver.switch_to.window(self.driver.window_handles[0])
        else:
            print("A aba principal foi fechada inesperadamente.")

    def close_popup(self):
        try:
            WebDriverWait(self.driver, self.delay).until(
                EC.element_to_be_clickable((By.ID, "idbtnfechar"))
                ).click()
            if self.verbose:
                print("       Pop-up fechado com sucesso.")
        except Exception as e:
            print(f"Erro ao fechar o pop-up: {e}")

    def extrair_informacoes_infpessoa(self, soup):
        # Dicionário para armazenar os dados extraídos
        dados_extraidos = {
            'nome': None,
            'id_lattes': None,
            'data_atualizacao': None
        }
        # Extrair o nome
        nome = soup.find('h2', class_='nome')
        if nome:
            dados_extraidos['nome'] = nome.text.strip()
        # Extrai o ID Lattes e a data de atualização
        ul = soup.find('ul', class_='informacoes-autor')
        if ul:
            for li in ul.find_all('li'):
                texto_li = li.text.strip()
                if texto_li.startswith('Endereço para acessar este CV:'):
                    id_lattes = texto_li.split('/')[-1]
                    dados_extraidos['id_lattes'] = id_lattes
                elif 'Última atualização do currículo em' in texto_li:
                    data_atualizacao = texto_li.split('em')[-1].strip()
                    dados_extraidos['data_atualizacao'] = data_atualizacao
        return dados_extraidos

    ## Extrações para cada seção do currículo
    def get_general_data(self, soup):
        """Extrai dados gerais do currículo."""      
        div = soup.find('div', class_='infpessoa')
        ul = soup.find('ul', class_="informacoes-autor")
        general_data = {}
        nome_pesquisador = div.find('h2', class_='nome')
        if nome_pesquisador:
            nome_pesquisador=nome_pesquisador.get_text(strip=True)
            general_data['nome'] = nome_pesquisador
        idlattes = ul.find('span', style=lambda value: value and 'color: #326C99;' in value)
        if idlattes:
            idlattes=idlattes.get_text(strip=True)
            general_data['lattes_id'] = idlattes
        linha_atualizado = ul.find('li', string=lambda text: 'Última atualização do currículo em' in text)
        if linha_atualizado:
            try:
                data_atualizado=linha_atualizado.get_text.split('em')[-1].strip()
            except:
                pass
        info_list = soup.find('ul', class_='informacoes-autor')
        if info_list:
            # Busca todos os <li> dentro da lista
            list_items = info_list.find_all('li')
            for item in list_items:
                # Procura pela frase que indica a última atualização
                if 'Última atualização do currículo em' in item.text:
                    update_date = item.text.split('em')[-1].strip()
                    print(update_date)
                    break            
        if update_date:
            general_data['ultima_atualizacao'] = update_date
        if self.verbose:
            if nome_pesquisador:
                print(f'            Nome: {nome_pesquisador}')
            if idlattes:
                print(f'       ID Lattes: {idlattes}')
            if data_atualizado:
                print(f'     Atualização: {data_atualizado}')
        return general_data

    def get_abstract(self,soup):
        """Extrai parágrafo de resumo do currículo."""
        abs_par = soup.find('p', class_='resumo')
        if abs_par:
            print(type(abs_par))
            abs_text = abs_par.text.strip()
            return abs_text
        else:
            p_tag = soup.find('p')
            print(type(p_tag))
            abs_text = abs_par.text.strip()
            return abs_text

    def get_formation(self, soup):
        """Extrai informações de formação acadêmica."""
        formation = []
        for item in soup.find_all('div', class_='layout-cell layout-cell-12 data-cell'):
            info = {}
            title = item.find_previous('h1').get_text(strip=True) if item.find_previous('h1') else None
            if title and 'Formação acadêmica/titulação' in title:
                info['titulo'] = item.find('b').get_text(strip=True) if item.find('b') else None
                info['descricao'] = item.find('div', class_='layout-cell-pad-5').get_text(strip=True) if item.find('div', class_='layout-cell-pad-5') else None
                formation.append(info)
        return formation

    def get_professional_experience(self, soup):
        """Extrai informações da atuação profissional."""
        experiences = []
        atuacao_profissional_section = soup.find('a', name='AtuacaoProfissional')
        if atuacao_profissional_section:
            all_experiences = atuacao_profissional_section.find_next_siblings('div', limit=1)[0].find_all('div', class_='inst_back')
            for exp in all_experiences:
                experience_info = {}
                experience_info['instituicao'] = exp.get_text(strip=True)
                experiences.append(experience_info)
        return experiences

    def get_section_productions(self, soup, section_name):
        section = soup.find('a', name=section_name)
        if not section:
            return []
        productions_list = []
        current_element = section.find_next_sibling()
        while current_element and current_element.name != 'a':
            text = current_element.get_text(strip=True, separator=" ")
            if text:  # Filter out empty texts
                productions_list.append(text)
            current_element = current_element.find_next_sibling()
        return productions_list
    
    def get_productions(self, soup):
        logging.debug("Extraindo produções do currículo.")
        try:
            productions = {
                "artigos_completos": self.get_section_productions(soup, "ArtigosCompletos"),
                "livros_publicados": self.get_section_productions(soup, "LivrosCapitulos"),
                "capitulos_de_livros_publicados": self.get_section_productions(soup, "LivrosCapitulos"),
                "trabalhos_completos_em_anais": self.get_section_productions(soup, "TrabalhosPublicadosAnaisCongresso"),
                "apresentacoes_de_trabalho": self.get_section_productions(soup, "ApresentacoesTrabalho"),
                "outras_producoes_bibliograficas": self.get_section_productions(soup, "OutrasProducoesBibliograficas"),
                "assessoria_e_consultoria": self.get_section_productions(soup, "AssessoriaConsultoria"),
                "produtos_tecnologicos": self.get_section_productions(soup, "ProdutosTecnologicos"),
                "trabalhos_tecnicos": self.get_section_productions(soup, "TrabalhosTecnicos"),
                "demais_tipos_de_producao_tecnica": self.get_section_productions(soup, "DemaisProducaoTecnica"),
                "demais_trabalhos": self.get_section_productions(soup, "DemaisTrabalhos")
            }
            return productions
        except Exception as e:
            logging.error("Erro ao extrair produções: %s", e)


    def get_events(self, soup):
        """Extrai informações de eventos participados."""
        events = []
        eventos_section = soup.find('a', name='Eventos')
        if eventos_section:
            all_events = eventos_section.find_next_siblings('div', class_='layout-cell layout-cell-12 data-cell')[0].find_all('div', class_='transform')
            for event in all_events:
                event_info = {}
                event_info['nome'] = event.get_text(strip=True)
                events.append(event_info)
        return events

    def get_orientations(self, soup):
        """Extrai informações de orientações concluídas."""
        orientations = []
        orientacoes_section = soup.find('a', name='Orientacoesconcluidas')
        if orientacoes_section:
            all_orientations = orientacoes_section.find_next_siblings('div', class_='layout-cell layout-cell-12 data-cell')[0].find_all('div', class_='transform')
            for orientation in all_orientations:
                orientation_info = {}
                orientation_info['titulo'] = orientation.get_text(strip=True)
                orientations.append(orientation_info)
        return orientations

    def medir_tempo_resposta(self, url_busca):
        try:
            response = requests.get(url_busca)
            tempo_resposta = response.elapsed.total_seconds()
            print(f"Tempo de resposta do servidor: {tempo_resposta:.2f} segundos")
        except requests.exceptions.RequestException as e:
            print(f"Erro ao fazer solicitação HTTP: {e}")

    def extract_dom_element(self):
        """
        Extrair elemento com vínculo detectado a partir da página de resultados.
        """
        try:
            # Esperar o elemento que indica a quantidade de resultados
            css_qteresultados = ".tit_form > b:nth-child(1)"
            WebDriverWait(self.driver, self.delay).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_qteresultados)))

            # Extrair a quantidade de resultados
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            div_element = soup.find('div', {'class': 'tit_form'})
            match = re.search(r'<b>(\d+)</b>', str(div_element))
            if match:
                qte_res = int(match.group(1))
            else:
                print(f'Resultados não encontrados na busca Lattes')
                return None

            # Loop para lidar com Stale File Handler
            for _ in range(5):  # self.max_tentativas define o número máximo de tentativas
                try:
                    # Esperar carregar a lista de resultados
                    css_resultados = ".resultado"
                    WebDriverWait(self.driver, self.delay).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, css_resultados)))

                    # Encontrar os elementos da lista de resultados
                    resultados = self.driver.find_elements(By.CSS_SELECTOR, css_resultados)

                    # Verificar se há erro de Stale File Handler
                    if self.is_stale_file_handler_present():
                        raise StaleElementReferenceException

                    return resultados

                except StaleElementReferenceException:
                    print(f"Erro Stale File Handler detectado. Tentando novamente...")
                    time.sleep(2)  # Aguarda um tempo antes de tentar novamente

            print(f"Erro Stale File Handler persistiu.")
            return None

        except Exception as e:
            print(f"Erro ao extrair elemento DOM: {e}")
            print(f"Conteúdo HTML: {self.driver.page_source}")
            return None

    def get_element_without_pagination(self, NOME, resultados, termos_busca):
        """
        Extrair o elemento com vínculo de acordo com termos de busca, em páginas sem paginação
        """
        limite = 5
        duvidas = []
        qte_res = len(resultados)
        force_break_loop = False
        elm_vinculo = None
        for n,i in enumerate(resultados):
            linhas = i.text.split('\n\n')
            if self.verbose:
                print(f'       Quantidade de homônimos: {len(linhas):02}')
            if self.is_stale_file_handler_present():
                raise StaleElementReferenceException
                # return np.NaN, NOME, np.NaN, 'Stale file handle', self.driver
            for m,linha_multipla in enumerate(linhas):
                nome_achado = linhas[m].split('\n')[0]
                linha = linha_multipla.replace("\n", " ")
                if self.verbose:
                    print(f'       Currículo: {nome_achado}')
                    print(f'        Conteúdo: {linha}')
                if self.verbose:
                    print(f'       Currículo {m+1:02}/{len(linhas):02}: {linha}')
                for termo in termos_busca:
                    replaces = [('\xa0',' ')]
                    print(f"       {termo:>25} | {termo in linha} | {[linha.replace(repl[0],repl[1]) for repl in replaces]}")
                    if self.verbose:
                        print(f'       Resultado: {m+1}, de total de linhas {len(linhas)}')
                        print(f'        Conteúdo: {linha}')
                    if termo.lower() in linha.lower():
                        count=m
                        while get_jaro_distance(nome_achado.lower(), str(NOME).lower()) < 0.85 and count>0:
                            count-=1
                            print(f'       Contador: {count}')
                        found = m+1
                        # nome_vinculo = linhas[count].replace('\n','\n       ').strip()
                        # print(f'       Achado: {nome_vinculo}')
                        css_vinculo = f".resultado > ol:nth-child(1) > li:nth-child({m+1}) > b:nth-child(1) > a:nth-child(1)"
                        # print('\nCSS_SELECTOR usado:', css_vinculo)
                        WebDriverWait(self.driver, self.delay).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, css_vinculo)))            
                        elm_vinculo  = self.driver.find_element(By.CSS_SELECTOR, css_vinculo)
                        nome_vinculo = elm_vinculo.text
                        if self.verbose:
                            print(f'       Tipo extraído: {type(elm_vinculo)}')
                            print(f'       Nome extraído: {nome_vinculo}')
                        ## Tentar repetidamente clicar no elemento encontrado
                        self.retry(ActionChains(self.driver).click(elm_vinculo).perform(),
                            wait_ms=500,
                            limit=limite,
                            on_exhaust=(f'       Problema ao clicar no link do nome. {limite} tentativas sem sucesso.'))
                        force_break_loop = True
                        # Parar loop de termos quando um termo já tiver sido achado
                        break
                # Parar loop de currículos quando um termo já tiver sido achado
                if force_break_loop:
                    break
                ## Caso percorra toda lista e não encontre vínculo adiciona à dúvidas quanto ao nome
                if m==(qte_res):
                    print(f'       Nenhuma vínculo encontrado para {NOME}')
                    duvidas.append(NOME)
                    # clear_output(wait=True)
                    # driver.quit()
            # Parar loop da div de resultados quando um termo já tiver sido achado
            if force_break_loop:
                break
        return elm_vinculo

    def get_pages_numbers(self):
        '''
        Helper function to page results on the search page
        '''
        pagenumbers = []
        css_paginacao = "div.paginacao:nth-child(2)"
        try:
            WebDriverWait(self.driver, self.delay).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_paginacao)))
            pages = self.driver.find_element(By.CSS_SELECTOR, css_paginacao)
            pagenumbers_parts = pages.text.split(' ')
            remove = ['', 'anterior', '...']
            pagenumbers = [x for x in pagenumbers_parts if x not in remove]
        except Exception as e:
            print('  ERRO!! Ao rodar função get_pages_numbers():', e)
        return pagenumbers

    def load_page_results(self, page_link):
        '''
        Helper function to load pages results on each search
        '''
        # Extrair o link para acessar a página
        page_href = page_link.get_attribute('href')

        # Acessar a página usando o driver Selenium
        self.driver.get(page_href)

        # Esperar o carregamento da página
        wait = WebDriverWait(self.driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'resultado')))

        # Encontrar os links de resultado na página
        result_links = self.driver.find_elements(By.CSS_SELECTOR, '.resultado a[href^="javascript:abreDetalhe"]')
        # if self.verbose:
        print(result_links)

        return result_links

    def obter_resultados_pagina(numero_pagina, intLRegPagina, strLQuery):
        """
        Função para obter resultados de uma página específica.

        Args:
            numero_pagina (int): Número da página a ser acessada.

        Returns:
            BeautifulSoup object: Conteúdo HTML da página solicitada.
        """
        url_pagina = f"https://exemplo.com/pagina-resultados?numeroPagina={numero_pagina}&registros=0;{intLRegPagina}&{strLQuery}"
        response = requests.get(url_pagina)
        return BeautifulSoup(response.content, 'html.parser')

    ## Refazendo todos os passos
    def get_html(self, url):
        """Obtém o conteúdo HTML da URL especificada."""
        response = requests.get(url)
        response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
        return response.text
            
    def get_pagination_links(self, html):
        """Extrai os links para as páginas da paginação."""
        soup = BeautifulSoup(html, 'lxml')
        pagination_div = soup.find('div', class_='paginacao')
        links = []
        for a in pagination_div.find_all('a', data_role='paginacao'):
            links.append(a['href'])
        return links

    def get_results(self, html):
        """Extrai os resultados da pesquisa (currículos)."""
        soup = BeautifulSoup(html, 'lxml')
        results_div = soup.find('div', class_='resultado')
        results = []
        for li in results_div.find_all('li'):
            result = {
                'nome': li.find('a').text,  # Nome da pessoa
                'link': li.find('a')['href'],  # Link para o currículo
                'detalhes': li.text.strip()  # Detalhes do currículo (texto completo da li)
            }
            results.append(result)
        return results

    def search_lattes(self, query, termos_busca=None, max_pages=100):
        """Realiza a pesquisa no Lattes e retorna os resultados."""
        base_url = 'https://buscatextual.cnpq.br/buscatextual/busca.do?metodo=busca'
        start_url = base_url + '&query=' + query
        results = []
        # Extrai os resultados da primeira página
        page_html = self.get_html(start_url)
        results += self.get_results(page_html)
        # Extrai os resultados das demais páginas (até o limite especificado)
        for page_url in self.get_pagination_links(page_html)[:max_pages - 1]:
            page_html = self.get_html(page_url)
            results += self.get_results(page_html)
        # Filtrar resultados por homônimos (se termos_busca for fornecido)
        if termos_busca:
            filtered_results = []
            for result in results:
                for termo in termos_busca:
                    if termo.lower() in result['detalhes'].lower():
                        filtered_results.append(result)
                        break  # Encontra um termo de busca, interrompe a iteração
            results = filtered_results
        return results

    def process_names(self, names, termos_busca):
        """Processa uma lista de nomes e retorna os resultados da busca no Lattes."""
        results = []
        for name in names:
            # Gerar query de busca
            query = '+idx_nme_pessoa:' + name.replace(' ', '+')
            print(f'       {query}')
            # Realizar a busca e filtrar por homônimos
            lattes_results = self.search_lattes(query, termos_busca)
            # Selecionar o resultado mais relevante (se houver)
            if lattes_results:
                result = self.select_most_relevant_result(lattes_results)
                results.append(result)
        return results

    def select_most_relevant_result(results):
        """Seleciona o resultado mais relevante com base na frequência dos termos de busca."""
        # TODO: Implementar lógica para selecionar o resultado mais relevante
        return results[0]

    def extract_results_page(driver, url):
        """ Extrai os resultados de uma determinada página de resultados. """

        driver.get(url)

        # Encontra a div de resultados
        soup = BeautifulSoup(driver.page_source, 'lxml')
        resultados_div = soup.find('div', class_='resultado')

        resultados = []
        for li in resultados_div.find_all('li'):
            titulo = li.text.strip()
            link = li.find('a')['href']

            if any(char.isdigit() for char in titulo):  # Verifica números
                resultados.append({
                    'titulo': titulo,
                    'link': link 
                })   
        return resultados

    def extrair_dados_pagina(driver):
        # Encontrar os elementos que contêm os dados (ex: resultado da pesquisa)
        elementos_dados = driver.find_elements(By.CSS_SELECTOR, ".resultado li")

        # Extrair os dados de cada elemento
        dados_pagina = []
        for elemento in elementos_dados:
            # Extrair informações relevantes de cada elemento (nome, nacionalidade, etc.)
            nome = elemento.find_element(By.CSS_SELECTOR, "a").text
            dados = elemento.find_element(By.CSS_SELECTOR, "li").text

            # Armazenar os dados extraídos em um dicionário
            dados_pagina.append({
                "nome": nome,
                "dados": dados,
            })

        return dados_pagina

    def tentar_paginar(self, driver, elemento):
        """Tenta diferentes estratégias de paginação e trata os possíveis erros.

        Args:
            driver: O WebDriver do Selenium.
            elemento: O elemento HTML a ser clicado para paginação.
        """

        href_elemento = elemento.get_attribute("href")  # Obtém o href antes das tentativas

        estrategias = [
            # Estratégia 1: Clique direto no href (considerando comportamento ideal)
            lambda: ActionChains(driver).move_to_element(elemento).click().perform(),
            # Estratégia 2: CTRL + Clique
            lambda: ActionChains(driver).key_down(Keys.CONTROL).click(elemento).perform(),            
            # Estratégia 3: Simular navegação com driver.get()
            # lambda: driver.get(href_elemento)
            # Estratégia 4: Clique direto no href (considerando comportamento ideal)
            # lambda: elemento.click(),
        ]
        count=0
        for estrategia in estrategias:
            count+=1
            try:
                estrategia()
                print(f"        Paginação bem-sucedida usando a estratégia: {count}")
                return  # Termina a função se uma estratégia funcionar
            except Exception as e:
                print(f"        Erro ao paginar com a estratégia {count}: {e}")

        # Se todas as estratégias falharam:
        print(f"        Falha ao paginar com todas as estratégias para o link: {href_elemento}")


    def extrair_resultados_paginados(self, url_base, pagina_inicial=0, pagina_final=None):
        """
        Extrai os dados dos resultados de pesquisa de todas as páginas, utilizando paginação.

        Argumentos:
            url_base (str): URL base da consulta de pesquisa (ex: 'https://busca.lattes.cnpq.br/busca').
            pagina_inicial (int): Número da página inicial para extração (padrão: 1).
            pagina_final (int): Número da página final para extração (opcional).

        Retorna:
            list: Lista de dicionários com os dados de cada resultado (nome, link) de todas as páginas.
        """
        if self.verbose:
            print(f'       pagina_atual: {pagina_inicial}')

        dados_resultados = []
        pagina_atual = pagina_inicial

        while pagina_atual <= pagina_final or pagina_final is None:
            if self.verbose:
                print(f'       pagina_atual: {pagina_atual}')            
            # Formato a ser construído

            # Constrói a URL da página atual
            url_base='https://buscatextual.cnpq.br'
            url_pagina = url_base + f"/buscatextual/busca.do?metodo=forwardPaginaResultados&amp;registros={pagina_atual*10};10&amp;query=%28%2Bidx_nme_pessoa%3A%28aline%29+%2Bidx_nme_pessoa%3A%28silva%29+%2Bidx_nme_pessoa%3A%28soares%29++%2Bidx_nacionalidade%3Ae%29+or+%28%2Bidx_nme_pessoa%3A%28aline%29+%2Bidx_nme_pessoa%3A%28silva%29+%2Bidx_nme_pessoa%3A%28soares%29++%2Bidx_nacionalidade%3Ab%29&amp;analise=cv&amp;tipoOrdenacao=null&amp;paginaOrigem=index.do&amp;mostrarScore=false&amp;mostrarBandeira=true&amp;modoIndAdhoc=null"
            if self.verbose:            
                print(f'       url_pagina: {url_pagina}')
            response = requests.get(url_pagina)
            if self.verbose:
                print(response.status_code)

            # Verifica se a requisição foi bem-sucedida
            if response.status_code == 200:
                # Extrai os dados dos resultados da página atual
                pagina_html = response.content
                try:
                    dados_pagina = self.extrair_resultados(pagina_html)
                    if self.verbose:
                        print(f'       dados_pagina: {dados_pagina}')
                except Exception as e:
                    print(f'       Erro com {e}')

                # Adiciona os dados da página atual à lista final
                dados_resultados.extend(dados_pagina)
                if self.verbose:
                    print(f'       Qte de resultado: {len(dados_resultados)}')
                    print(f'       Dados Resultados: {dados_resultados}')

                # Extrai os parâmetros de paginação da página atual
                try:
                    parametros_paginacao = self.extrair_parametros_paginacao(pagina_html)
                except Exception as e:
                    print(f'       Erro com {e}')                

                # Atualiza a página atual para a próxima página (se existir)
                pagina_atual = parametros_paginacao["pagina_seguinte"]
            else:
                # Erro na requisição: interrompe a extração e registra o erro
                print(f'Erro na requisição para página {pagina_atual}: {response.status_code}')
            break

        return dados_resultados

    def extrair_parametros_paginacao(self, pagina_atual):
        """
        Extrai os parâmetros de paginação a partir da página HTML atual.

        Argumentos:
            pagina_atual (str): Conteúdo HTML da página atual.

        Retorna:
            dict: Dicionário com os parâmetros de paginação (página_inicial, pagina_final, pagina_anterior, pagina_seguinte).
        """
        ## TO-DO: incluir clique em 'próximo' a cada dezena para paginar mais de 20 resultados
        # Extrai parâmetros da página atual
        parametros_pagina_atual = re.findall(r'paginaAtual": (\d+),', pagina_atual)
        pagina_atual = int(parametros_pagina_atual[0]) if parametros_pagina_atual else None

        # Extrai parâmetros da primeira página
        parametros_primeira_pagina = re.findall(r'primeiraPagina": (\d+),', pagina_atual)
        pagina_inicial = int(parametros_primeira_pagina[0]) if parametros_primeira_pagina else None

        # Extrai parâmetros da última página
        parametros_ultima_pagina = re.findall(r'ultimaPagina": (\d+),', pagina_atual)
        pagina_final = int(parametros_ultima_pagina[0]) if parametros_ultima_pagina else None

        # Extrai parâmetros da página anterior (se existir)
        pagina_anterior = pagina_atual - 1 if pagina_atual > 1 else None

        # Extrai parâmetros da próxima página (se existir)
        pagina_seguinte = pagina_atual + 1 if pagina_atual < pagina_final else None

        # Retorna dicionário com os parâmetros extraídos
        return {
            "pagina_inicial": pagina_inicial,
            "pagina_final": pagina_final,
            "pagina_anterior": pagina_anterior,
            "pagina_seguinte": pagina_seguinte,
        }

    def extrair_resultados(self, pagina_html):
        """
        Extrai os dados dos resultados de pesquisa presentes na página HTML.

        Argumentos:
            pagina_html (str): Conteúdo HTML da página de resultados.

        Retorna:
            list: Lista de dicionários com os dados de cada resultado (nome, link).
        """

        soup = BeautifulSoup(pagina_html, 'lxml')

        # Localiza os elementos de resultado na página
        resultados = soup.find_all('li', class_='resultado')

        # Extrai dados de cada resultado
        dados_resultados = []
        for resultado in resultados:
            nome = resultado.find('a', class_='nome-autor').text.strip()
            dado = resultado.find('b', class_='nome-autor').text.strip()
            link = resultado.find('a', class_='nome-autor')['href']

            dados_resultados.append({
            "nome": nome,
            "dado": dado,
            "link": link,
            })

        return dados_resultados

    ###### Experiências com tratamento de homônimos
                            # css_elemento = self.driver.find_element(By.CSS_SELECTOR, elemento.text)
                            # css_elemento = self.driver.find_element(By.XPATH, f"//a[contains(text(), {elemento.text})]")
                            
                            ## TO-FIX: PRECISA DISPARAR JAVASCRIPT PARA CARREGAR OS RESULTADOS DE CADA PÁGINA DA PAGINAÇÃO
                            # clicar no elemento encontrado para carregar próxima página na paginação
                            
                            # WebDriverWait(soup, self.delay).until(
                            #     EC.presence_of_element_located((By.CSS_SELECTOR, ".tit_form"))
                            # )                            
                            # Carregar os resultados da nova página a verificar presença de termos_busca
                            
                            # resultados = soup.findChildren('li')
                            # count+=len(resultados)
                            # print(f'       Carregada página {count_pages:02}/{len(numpaginas):02} de resultados')
                            # text_results = [x.text.replace('\xa0',' ') for x in resultados]
                            # if self.verbose:
                            #     print(f'       Elem text: {len(text_results)} | {text_results}')
                            
                            # # iterar em cada resultado
                            # for n,i in enumerate(resultados):
                            #     try:
                            #         # Ler dados prévios dos currículos e buscar termos
                            #         elm_vinculo = self.get_element_without_pagination(NOME, resultados, termos_busca)
                            #     except Exception as e:
                            #         print(f'       Não foi possível extrair currículo, erro em get_element_without_pagination')
                            #         print(f'       ERRO: {e}')
                            #         return None                            

                        # Ler dados prévios dos currículos e buscar termos
                        # resultados = soup.findChildren('li')
                        # if self.verbose:
                        #     print(f'       {len(resultados):2} Resultados: {type(resultados)} | {resultados}')
                        # if resultados:
                        #     try:
                        #         # procurar nos resultados termos de busca
                        #         elm_vinculo = self.get_element_without_pagination(NOME, resultados, termos_busca)
                        #         if elm_vinculo:
                        #             break
                        #     except Exception as e:
                        #         print(f'       Não foi possível extrair currículo, erro em get_element_without_pagination')
                        #         print(f'       ERRO: {e}')
                        #         traceback_str = ''.join(traceback.format_tb(e.__traceback__))
                        #         print(f'       Ocorrido em: {traceback_str}')
                        #         print(f"       {'-'*120}")
                        #         return None
                        # else:
                        #     print('       Não foi possível obter resultados para a busca')

    # def handle_pagination_and_collect_profiles(self):
    #     def extrair_dados_cv():
    #         resultados = self.driver.find_elements(By.CSS_SELECTOR, "div.resultado")
    #         dados_pessoas = []

    #         for resultado in resultados:
    #             nome = resultado.find_element(By.TAG_NAME, "a").text
    #             link_detalhes = [y.get_attribute("href") for y in x for x in resultado.find_elements(By.TAG_NAME, "a")]
    #             preview = resultado.find_element(By.CSS_SELECTOR, "br + br").text

    #             dados_pessoa = {
    #                 "nome": nome,
    #                 "link_detalhes": link_detalhes,
    #                 "dados_curriculo": preview,
    #             }
    #             dados_pessoas.append(dados_pessoa)
    #         return dados_pessoas

    #     preview = {}
    #     profiles = []
    #     pages_links=[]
    #     while True:
    #         pagination_div = self.driver.find_element(By.CLASS_NAME, "paginacao")
    #         for x in pagination_div.find_elements(By.TAG_NAME, "a"):
    #             pages_links.append(x.get_attribute("href"))
    #             for n,page in enumerate(pages_links):
    #                 try:
    #                     print(f'{n}/{len(pages_links)} sendo lido...')
    #                     page.click(page)
    #                     WebDriverWait(self.driver, self.delay).until(
    #                         EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".resultado")))
    #                     results = self.driver.find_elements(By.CSS_SELECTOR, ".resultado > ol > li")
    #                     dados_pessoas = extrair_dados_cv()
    #                     profiles.append(dados_pessoas)
    #                 except:
    #                     pass
        
    #     preview['Curriculos'] = profiles
    #             # next_button = self.driver.find_elements(By.CSS_SELECTOR, "próximo")
    #             # if next_button:
    #             #     next_button[0].click()
    #             # else:
    #             #     break
    #     return preview

    def obter_dados_homonimo_paginacao(self, nome, termos_busca, qte_res):
        """
        Obtém os dados do homônimo mais compatível em páginas com homônimos e paginação.

        Argumentos:
            nome (str): Nome do pesquisador.
            termos_busca (list): Lista de termos de busca.

        Retorna:
            tuple: Tupla contendo os dados do homônimo (nome_vinculo, nome_completo, id_lattes, resumo, driver).
        """

        # Variáveis de controle
        resultado_encontrado = False
        iteracoes = 0
        links_relevantes = []
        pagin = qte_res // 10 + 1  # Número de páginas (aproximado)
        count = None
        found = None
        result_counter = 1

        # Acessar todas as páginas e avaliar cada homônimo
        for pagina in range(1, pagin + 1):
            url_pagina = f"https://buscatextual.cnpq.br/busca?pesquisador={nome}&pagina={pagina}"
            self.driver.get(url_pagina)

            # Acessar os links de cada página
            melhor, pontuacao = self.acessar_links_paginados(termos_busca)

            # Avaliar cada homônimo
            for linha in self.resultado_pagina:
                iteracoes += 1
                nome_completo = linha[0].strip()

                # Verificar se o nome completo e os termos de busca coincidem
                if self.avaliar_coincidencia(resumo, termos_busca):
                    count = iteracoes
                    found = [nome_completo, linha[1], linha[2]]
                    links_relevantes.append(linha)
                    resultado_encontrado = True
                    break

            # Se o homônimo foi encontrado, parar a iteração
            if resultado_encontrado:
                break

        # Se o homônimo foi encontrado, retornar seus dados
        if found:
            nome_vinculo = linhas[count].replace('\n', '\n     ').strip()
            print(f'    Escolhido o homônimo {found[0]}: {nome_vinculo}')
            resumo = self.pegar_resumo(self.driver)
            return nome_vinculo, found[0], found[1], resumo, self.driver

        # Se não foi encontrado em nenhuma página, retornar None
        else:
            print('    Homônimo não encontrado em nenhuma página.')
            return None, NOME, np.NaN, 'Homônimo não encontrado', self.driver

    def pegar_resumo(self):
        """
        Extrai o resumo do texto da página de detalhes.

        Argumentos:
            driver: Instância do Selenium WebDriver.

        Retorno:
            O texto do resumo.
        """
        # Localize o iframe pai
        # iframe_pai = self.driver.find_element(By.CSS_SELECTOR, 'div.moldal div.conteudo iframe')
        # iframe_pai = self.driver.find_element(By.CLASS_NAME, "iframe-modal")
        # try:
        #     iframe_pai = self.driver.find_element(By.CSS_SELECTOR, 'iframe.iframe-modal')
        #     self.driver.switch_to.frame(iframe_pai)
        # except Exception as e:
        #     print(e)
        #     traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        #     print(traceback_str)

        # Localize o iframe do resumo
        try:
            iframe = self.driver.find_element(By.ID, "id_form_previw")
            self.driver.switch_to.frame(iframe)  
        except Exception as e:
            print(e)
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            print(traceback_str)

        # # Extrair o resumo
        # soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        # resumo_element = soup.find('p', class_='resumo')
        # resumo = resumo_element.text

        # Correção da sintaxe, supondo que "resumo" é o nome correto da classe
        resumo_element = self.driver.find_element(By.CLASS_NAME, "resumo")
        resumo = resumo_element.text

        # Sair do iframe
        self.driver.switch_to_default_content()

        return resumo

    def acessar_links_paginados(self, termos_busca):
        # Encontrar a div de paginação
        pagination_div = self.driver.find_element(By.CLASS_NAME, "paginacao")

        # Extrair todos os links das páginas
        page_links = pagination_div.find_elements(By.TAG_NAME, 'a')
        maximo = -1
        for page_link in page_links:
            # Extrair o link para acessar a página
            page_url = page_link.get_attribute('href')

            # Acessar a página usando o driver Selenium
            self.driver.get(page_url)

            # Esperar o carregamento da página
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'resultado')))

            # Encontrar os links de resultado na página
            result_links = self.driver.find_elements(By.CSS_SELECTOR, '.resultado a[href^="javascript:abreDetalhe"]')

            for result_link in result_links:
                # Obter o JavaScript a ser executado
                javascript_code = result_link.get_attribute('href')

                # Executar o JavaScript via Selenium 
                self.driver.execute_script(javascript_code)

                # Esperar carregamento da página de detalhes
                wait.until(EC.presence_of_element_located((By.ID, 'idbtnabrircurriculo')))

                # Processar o conteúdo da página de detalhes
                resumo = ''
                informacoes_adicionais = ''
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                abs_par = soup.find('p', class_='resumo')
                if abs_par:
                    print(type(abs_par.strip()))
                    abs_text = abs_par.text.strip()               
                    resumo = abs_text
                else:
                    resumo = self.get_abstract(soup)
                print(f'    Resumo: {resumo}')

                # Avaliar pontuação
                pontuacao = self.calcular_pontuacao_compatibilidade(termos_busca, resumo, informacoes_adicionais)

                # Retornar para a página de resultados
                self.driver.back()
            
            if pontuacao > maximo:
                maximo = pontuacao
                melhor = result_link
        
        return melhor, pontuacao


    def extrair_dados_pessoa(self, soup):
        """Extrai o nome da pessoa e a descrição a partir do elemento 'resultado'.

        Args:
            soup (BeautifulSoup): Objeto BeautifulSoup do elemento 'resultado'.

        Returns:
            tuple: (nome, descricao)
        """
        
        nome_element = soup.find('a', href=re.compile(r'javascript:abreDetalhe'))  # Selecionar o link do nome
        if nome_element:
            nome = nome_element.text.strip()
        else:
            nome = None

        # Encontrar o elemento que contém a descrição (ajuste o seletor conforme a estrutura)
        descricao_element = soup.find('div', class_='content_result') 
        if descricao_element:
            descricao = descricao_element.text.strip()
        else:
            descricao = None

        return nome, descricao

    def extrair_preview(self, li):
        """Extrai e limpa as informações adicionais do elemento <li>."""
        nome_link = self.extrair_dados_pessoa(li.find('a'))  # Obtem o nome da pessoa (já extraído)
        texto_completo = li.get_text().strip()  # Obtém todo o texto do elemento <li>
        preview = texto_completo.replace(nome_link, '', 1).strip()  # Remove o nome da pessoa
        return preview

    def extrair_numero_pagina_atual(self, url):
        """Extrai o número da página atual da URL."""
        match = re.search(r"page=(\d+)", url)
        if match:
            return int(match.group(1))
        else:
            return 1 

    def gerar_url_proxima_pagina(self, url, numero_pagina_atual):
        """Gera a URL da próxima página."""
        base_url = re.sub(r"page=\d+", "", url)  
        proxima_pagina = base_url + f"page={numero_pagina_atual + 1}"
        return proxima_pagina

    def navegar_paginas(self, url_inicial, termos_busca):
        """
        Navega pelas páginas de resultados e coleta links relevantes.

        Args:
            url_inicial (str): URL da página inicial de resultados.
            termos_busca (list): Lista de termos para buscar nos links.

        Returns:
            list: Lista de todos os links relevantes encontrados.
        """

        links_coletados = []
        url_atual = url_inicial

        while True:
            response = requests.get(url_atual)
            response.raise_for_status()  # Verifica se a requisição foi bem-sucedida

            links_coletados += self.encontrar_links_relevantes(response.text, termos_busca)

            # Extrai o número da página atual da URL
            numero_pagina_atual = self.extrair_numero_pagina_atual(url_atual)

            # Gera a URL da próxima página
            proxima_pagina = self.gerar_url_proxima_pagina(url_atual, numero_pagina_atual)

            # Verifica se a próxima página existe
            response_proxima_pagina = requests.get(proxima_pagina)
            if response_proxima_pagina.status_code == 200:
                url_atual = proxima_pagina
            else:
                break

        return links_coletados

    def encontrar_links_relevantes(self, html, termos_busca):
        """
        Encontra links relevantes dentro do HTML fornecido, com base nos termos de busca.

        Args:
            html (str): Conteúdo HTML da página.
            termos_busca (list): Lista de termos para filtrar os links.

        Returns:
            list: Lista de links (href strings) que correspondem aos termos de busca.
        """

        soup = BeautifulSoup(html, 'html.parser')
        resultados = soup.find_all('div', class_='resultado')
        links_relevantes = []

        for resultado in resultados:
            link_element = resultado.find('a')
            if link_element:
                texto_link = link_element.text.lower()
                if any(termo.lower() in texto_link for termo in termos_busca):
                    links_relevantes.append(link_element['href'])

        return links_relevantes

    def escolher_homonimo(self, html_content):
        """
        Retorna o link do homônimo mais compatível com os termos de busca.

        Args:
            html_content (str): Conteúdo HTML da página de resultados.

        Returns:
            str: Link do homônimo mais compatível, ou None se nenhum for encontrado.
        """

        soup = BeautifulSoup(html_content, 'html.parser')
        resultados = soup.find_all('div', class_='resultado')
        homonimo_mais_compativel = None
        pontuacao_maxima = 0

        for resultado in resultados:
            nome_pessoa = self.extrair_dados_pessoa(resultado)
            informacoes_adicionais = self.extrair_preview(resultado)
            link_pessoa = resultado.find('a')['href']

            pontuacao_compatibilidade = self.calcular_pontuacao_compatibilidade(
                self.termos_busca, nome_pessoa, informacoes_adicionais
            )

            if pontuacao_compatibilidade > pontuacao_maxima:
                homonimo_mais_compativel = link_pessoa
                pontuacao_maxima = pontuacao_compatibilidade

        return homonimo_mais_compativel

    def calcular_pontuacao_compatibilidade(self, termos_busca, resumo, informacoes_adicionais):
        """
        Calcula a pontuação de compatibilidade entre os termos de busca e as informações do homônimo.

        Args:
            termos_busca (list): Lista de termos de busca.
            nome_pessoa (str): Nome da pessoa.
            informacoes_adicionais (str): Informações adicionais sobre a pessoa.

        Returns:
            int: Pontuação de compatibilidade.
        """

        pontuacao = 0
        texto_comparacao = resumo.lower() + " " + informacoes_adicionais.lower()

        for termo in termos_busca:
            if termo.lower() in texto_comparacao:
                pontuacao += 1

        return pontuacao

    def fill_name(self, NOME, retry_count=3):
        '''
        Move cursor to the search field and fill in the specified name.
        '''
        if self.driver is None:
            logging.error("O driver não foi inicializado corretamente.")
            return
        try:
            search_input_selector = "#textoBusca"
            WebDriverWait(self.driver, self.delay).until(EC.visibility_of_element_located((By.CSS_SELECTOR, search_input_selector)))
            nome = lambda: self.driver.find_element(By.CSS_SELECTOR, search_input_selector)
            nome().send_keys(Keys.CONTROL + "a")
            nome().send_keys(NOME)
            search_button_selector = "#botaoBuscaFiltros"  # usar # seguido do id do elemento desejado
            WebDriverWait(self.driver, self.delay).until(EC.element_to_be_clickable((By.CSS_SELECTOR, search_button_selector)))
            search_button = self.driver.find_element(By.CSS_SELECTOR, search_button_selector)
            search_button.click()
        except Exception as e:
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            if self.verbose:
                print(f'  {e}')
                print(f'  {traceback_str}')
                # traceback.print_exc()
            if retry_count > 0:
                print(f"       Erro ao inserir o nome com função fill_name(), tentando novamente...")
                self.return_search_page()
                self.fill_name(NOME, retry_count - 1)
            else:
                print("       Tentativas esgotadas. Abortando ação.")

    def extract_tooltip_data(self, retries=3, delay=2):
        """
        Extracts tooltip data from articles section using Selenium with retry logic.
        :param retries: Number of retries if element is not interactable.
        :param delay: Wait time before retrying.
        :return: List of dictionaries containing the extracted tooltip data.
        """
        tooltip_data_list = []
        try:
            WebDriverWait(self.driver, self.delay).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, "#artigos-completos img.ajaxJCR")))
            layout_cells = self.driver.find_elements(By.CSS_SELECTOR, '#artigos-completos .layout-cell-11')
            if self.verbose:
                print(f'       {len(layout_cells):>003} células de layout encontradas')
            for cell in layout_cells:
                tooltip_data = {}
                # Tentativas para encontrar os dados do DOI e de citações, caso existam.
                try:
                    doi_elem = cell.find_element(By.CSS_SELECTOR, "a.icone-producao.icone-doi")
                    tooltip_data["doi"] = doi_elem.get_attribute("href")
                except NoSuchElementException:
                    tooltip_data["doi"] = None
                tooltip_elems = cell.find_elements(By.CSS_SELECTOR, "img.ajaxJCR")
                for tooltip_elem in tooltip_elems:
                    ActionChains(self.driver).move_to_element(tooltip_elem).perform()
                    # time.sleep(delay)  # Dando tempo para o tooltip ser carregado
                    # Espera até que o atributo 'original-title' do tooltip não esteja vazio
                    original_title = WebDriverWait(self.driver, 10).until(
                        attribute_to_be_non_empty((By.CSS_SELECTOR, "img.ajaxJCR"), "original-title")
                    )
                    if original_title:
                        match = re.search(r"Fator de impacto \(JCR \d{4}\): (\d+\.\d+)", original_title)
                        if match:
                            tooltip_data["impact-factor"] = match.group(1)
                            tooltip_data["original_title"] = original_title.split('<br />')[0].strip()
                            # Adicionar mais dados ao tooltip_data como valores de citações WoS e Scopus
                            break  # Sair do loop após sucesso na captura dos dados

                tooltip_data_list.append(tooltip_data)
            print(f'       {len(tooltip_data_list):>003} artigos extraídos')
        
        # ## TO-DO implementar o retry aqui
        # except StaleElementReferenceException:
        #     print(f"Erro Stale File Handler detectado. Tentando novamente...")
        #     time.sleep(2)  # Aguarda um tempo antes de tentar novamente

        except TimeoutException:
            print(f"       Requisição ao CNPq sem resposta, tooltips das publicações indisponíveis.")
        except Exception as e:
            print(f"       Erro inesperado ao extrair tooltips: {e}")
        return tooltip_data_list

    def buscar_qualis(self, lista_dados_autor):
        for dados_autor in lista_dados_autor:
            for categoria, artigos in dados_autor['Produções'].items():
                if categoria == 'Artigos completos publicados em periódicos':
                    for artigo in artigos:
                        issn_artigo = artigo['ISSN'].replace('-','')
                        qualis = self.encontrar_qualis_por_issn(issn_artigo)
                        print(f'{issn_artigo:8} | {qualis}')
                        if qualis:
                            artigo['Qualis'] = qualis
                        else:
                            artigo['Qualis'] = 'NA'

    def buscar_qualis_e_atualizar_arquivo(self, lista_dados_autor, pathfilename):
        """
        Busca o Qualis de cada artigo completo publicado em periódicos e atualiza o arquivo original com os dados encontrados.

        Args:
            lista_dados_autor (list): Lista de dicionários com os dados dos autores.
            pathfilename (str): Caminho completo e Nome do arquivo JSON a ser atualizado.
        """

        try:
            # Verifique se o arquivo existe
            if not os.path.exists(pathfilename):
                # O arquivo não existe, então vamos criá-lo
                with open(pathfilename, 'w', encoding='utf-8') as arquivo:
                    json.dump({}, arquivo, indent=4)

            # Abra o arquivo no modo r+ para leitura e escrita
            with codecs.open(pathfilename, 'r+', encoding='utf-8') as arquivo:
                # Carregue os dados do arquivo
                dados = json.load(arquivo)

                # Percorra os dados de cada autor
                for m, dados_autor in enumerate(lista_dados_autor):
                    for categoria, artigos in dados_autor['Produções'].items():
                        if categoria == 'Artigos completos publicados em periódicos':
                            for n, artigo in enumerate(artigos):
                                print(f'{n+1:3}/{len(artigos):3} artigos do autor {m+1:3}/{len(lista_dados_autor):3}')
                                clear_output(wait=True)

                                # Recupere o ISSN do artigo
                                issn_artigo = artigo['ISSN'].replace('-','')

                                # Busque o Qualis do artigo
                                qualis = self.encontrar_qualis_por_issn(issn_artigo)

                                # Atualize o campo 'Qualis' do artigo
                                if qualis:
                                    artigo['Qualis'] = qualis
                                else:
                                    artigo['Qualis'] = 'NA'

                # Reposicione o cursor no início do arquivo
                arquivo.seek(0)

                # Reescreva os dados formatados no arquivo
                json.dump(dados, arquivo, indent=4)

                return dados

        except Exception as e:
            print(f"Erro ao atualizar o arquivo: {e}")
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            print(f"{traceback_str}")

    def encontrar_qualis_por_issn(self, issn):
        """
        Busca o Qualis do artigo pelo ISSN.

        Args:
            issn (str): ISSN do artigo.

        Returns:
            str: Estrato do Qualis do artigo ou 'NA' caso não seja encontrado.
        """

        qualis = self.dados_planilha[self.dados_planilha['ISSN'].str.replace('-','') == issn]['Estrato'].tolist()
        if qualis:
            return qualis[0]
        else:
            return None

    # Conectar ao banco de dados Neo4j
    def persistir_dados(self, dados):
        for pessoa in dados:
            # Criar nó para a pessoa
            pessoa_node = Node("Pessoa", nome=pessoa["Identificação"]["Nome"])
            graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
            graph.create(pessoa_node)
            # Criar nós para idiomas e relacionamentos com a pessoa
            for idioma in pessoa["Idiomas"]:
                idioma_node = Node("Idioma", nome=idioma["Idioma"])
                graph.create(idioma_node)
                rel = Relationship(pessoa_node, "FALA", idioma_node)
                graph.create(rel)
            # Criar nós para formações e relacionamentos com a pessoa
            for formacao in pessoa["Formação"]["Acadêmica"]:
                formacao_node = Node("Formacao", ano=formacao["Ano"], descricao=formacao["Descrição"])
                graph.create(formacao_node)
                rel = Relationship(pessoa_node, "FORMOU", formacao_node)
                graph.create(rel)
            # Criar nós para projetos de pesquisa e relacionamentos com a pessoa
            for projeto in pessoa["ProjetosPesquisa"]:
                projeto_node = Node("ProjetoPesquisa", titulo=projeto["titulo_projeto"], descricao=projeto["descricao"])
                graph.create(projeto_node)
                rel = Relationship(pessoa_node, "PARTICIPA", projeto_node)
                graph.create(rel)

    def is_stale_file_handler_present(self):
        try:
            resultado_div = self.driver.find_element(By.CLASS_NAME, "resultado")
            if "Stale file handle" in resultado_div.text:
                return True
            else:
                return False
        except NoSuchElementException:
            return False

    def retry_click_vinculo(self, elm_vinculo, retry_count=3):
        if elm_vinculo is None:
            logging.info("       Nenhum dos vínculos esperados encontrado no currículo...")
            self.return_search_page()
            return

        for _ in range(retry_count):
            try:
                # Tentar clicar no vínculo
                self.click_vinculo(elm_vinculo)
                break  # Se o clique for bem-sucedido, saia do loop de retry
            except TimeoutException as se:
                logging.error(f"Erro ao clicar no vínculo: {se}. Tentando novamente...")
                # Lidar com a situação em que o pop-up ainda está aberto após uma tentativa de clique falha
                if self.is_popup_open():
                    self.close_popup()  # Fechar o pop-up
                time.sleep(1)  # Aguardar um segundo antes de tentar novamente
        else:
            logging.error("Todas as tentativas de clicar no vínculo falharam.")

    def click_vinculo(self, elm_vinculo):
        # Aguardar até que o botão esteja presente na página
        botao_abrir_curriculo = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '#idbtnabrircurriculo'))
        )
        actions = ActionChains(self.driver)
        actions.move_to_element(botao_abrir_curriculo).perform()
        botao_abrir_curriculo.click()
        # Checar se a página foi carregada
        # WebDriverWait(self.driver, self.delay).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, "#artigos-completos img.ajaxJCR")))
        layout_cells = self.driver.find_elements(By.CSS_SELECTOR, '#artigos-completos .layout-cell-11')

    def click_pagination_element(self, page_text):
        """
        Função para localizar um link com base no texto e clicar nele.

        Args:
            page_text (str): Texto a ser pesquisado no elemento.
        """
        # Aguardar até a página estar completamente carregada
        WebDriverWait(self.driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '*')))

        # Localizar o elemento `<a>` do resultado desejado
        resultado_element = soup.find('li', class_='resultado').find('a', text=page_text)

        # Mover o mouse para o link e clicar nele
        actions = ActionChains(self.driver)
        actions.move_to_element(resultado_element).perform()
        resultado_element.click()

        # Verificar se a nova página foi carregada
        # (Opcional, implemente a verificação de acordo com seus critérios)


    ## TO-FIX: Não está detectando o stale file handler, avisa erro inesperado e avança para próximo
    def find_terms(self, NOME, termos_busca, delay, limite=5):
        """
        Função para manipular o HTML até abir a página HTML de cada currículo, resolvendo homônimos
        Parâmetros:
            - NOME: É o nome completo de cada pesquisador
            - termos_busca: Lista de strings a buscar no currículo para escolher homônimo
            - driver (webdriver object): The Selenium webdriver object.
            - limite (int): Número máximo de tentativas em casos de erro.
            - delay (int): tempo em milisegundos a esperar nas operações de espera.
        Retorna:
            elm_vinculo, np.NaN, np.NaN, np.NaN, driver.
        Em caso de erro retorna:
            None, NOME, np.NaN, e, driver
        """
        
        # Inicializar variáveis para evitar UnboundLocalError
        count = 0
        qte_res = 0
        elm_vinculo = None
        try:
            # Esperar carregar a lista de resultados na página
            css_resultados = ".resultado"
            # ignored_exceptions=(NoSuchElementException)
            WebDriverWait(self.driver, self.delay).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_resultados)))
            resultados = self.driver.find_elements(By.CSS_SELECTOR, css_resultados)
            if resultados:
                count+=len(resultados*10)
            if self.is_stale_file_handler_present():
                raise StaleElementReferenceException
            # Ler quantidade de resultados apresentados pela busca de nome
            css_qteresultados = ".tit_form > b:nth-child(1)"
            WebDriverWait(self.driver, self.delay).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_qteresultados)))
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            div_element = soup.find('div', {'class': 'tit_form'})
            match = re.search(r'<b>(\d+)</b>', str(div_element))
            if match:
                qte_res = int(match.group(1))
                # print(f'{qte_res} resultados para {NOME}')
            else:
                return None, NOME, np.NaN, 'Currículo não encontrado', self.driver

            # Escolher função extração dada quantidade de resultados da lista apresentada na busca
            numpaginas = self.get_pages_numbers()
            if qte_res==1:
                # Para resultado único, capturar link para o primeiro nome resultado da busca
                css_linknome = ".resultado > ol:nth-child(1) > li:nth-child(1) > b:nth-child(1) > a:nth-child(1)"
                WebDriverWait(self.driver, self.delay).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, css_linknome)))
                elm_vinculo = self.driver.find_element(By.CSS_SELECTOR, css_linknome)
                nome_vinculo = elm_vinculo.text
                # print('Clicar no nome único:', nome_vinculo)
                try:
                    # Ao achar um dos termos clicar no elemento elm_vinculo com link do nome
                    self.retry(ActionChains(self.driver).click(elm_vinculo).perform(),
                                wait_ms=200,
                                limit=limite,
                                on_exhaust=(f'       Problema ao clicar no link do nome. {limite} tentativas sem sucesso.'))
                except:
                    print('       Erro ao clicar no único nome encontrado anteriormente')
                    return None, NOME, np.NaN, None, self.driver

            elif numpaginas == []:
                # Páginas até 10 resultados, sem paginar buscar termos e retornar elemento do resultado
                print(f'       {qte_res:>2} currículos homônimos: {NOME}')
                if self.is_stale_file_handler_present():
                    raise StaleElementReferenceException

                # iterar em cada resultado
                for n,i in enumerate(resultados):
                    try:
                        elm_vinculo = self.get_element_without_pagination(NOME, resultados, termos_busca)
                    except Exception as e:
                        print(f'       Não foi possível extrair currículo, erro em get_element_without_pagination')
                        print(f'       ERRO: {e}')
                        return None

            elif numpaginas != []:
                # Páginas onde precisa paginar escolher resultado, antes de buscar termos
                print(f'       {qte_res:>2} currículos homônimos: {NOME} (com paginação)')
                ## Paginar com clique no hiperlink já em cada link dentro do elemento de paginação
                try:
                    # Obter variáveis do JavaScript
                    script_tag = soup.find('script', language='JavaScript')
                    javascript_code = script_tag.text
                    intLTotReg = int(re.findall(r"intLTotReg = (\d+)", javascript_code)[0])
                    intLRegPagina = int(re.findall(r"intLRegPagina = (\d+)", javascript_code)[0])
                    strLQuery = re.findall(r"strLQuery = (.*)", javascript_code)
                    parametros_paginacao = {
                        'registros': intLRegPagina,
                        'intLTotReg': intLTotReg,
                        'strLQuery': strLQuery
                    }
                    if self.verbose:
                        print(f'       Extraído do Javascript')
                        print(f'            total_registros: {intLTotReg}')
                        print(f'       registros_por_pagina: {intLRegPagina}')
                        print(f'                  strLQuery: {strLQuery}')
                        print(f'       parametros_paginacao: {parametros_paginacao}')
                    # Localizar a primeira lista de paginação
                    elementos_paginacao = self.driver.find_elements(By.CSS_SELECTOR, ".paginacao a[data-role='paginacao']")
                    elemento = elementos_paginacao[0]
                    href_elemento = elemento.get_attribute('href')
                    if self.verbose:
                        print(f'           elemento: {type(elemento)}')
                        print(f'           elem_txt: {elemento.text}')
                        print(f"               href: {href_elemento}")
                    try:
                        # Acessar a nova página a partir do texto contido no elemento
                        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                        elm_vinculo = self.get_element_without_pagination(NOME, resultados, termos_busca)
                        count_pages=0
                        while not elm_vinculo and count_pages<=(intLTotReg//intLRegPagina):
                            count_pages+=1
                            elemento = elementos_paginacao[count_pages]
                            print(f"       Página a ser carregada: {elemento.text}")
                            try:
                                # elemento.click()
                                # self.driver.get(href_elemento)
                                ActionChains(self.driver).move_to_element(elemento).click().perform()
                                # ActionChains(self.driver).key_down(Keys.CONTROL).click(elemento).perform()
                                
                                ## TO-IMPROVE como carregar os dados da nova página dinamicamente
                                resultados = self.driver.find_elements(By.CSS_SELECTOR, css_resultados)
                                if resultados:
                                    count+=len(resultados*10)
                                elm_vinculo = self.get_element_without_pagination(NOME, resultados, termos_busca)

                            except Exception as e:
                                erro_regex = r"(?:\w+): (.*)"
                                desc = re.findall(erro_regex, str(e))[0] if re.findall(erro_regex, str(e)) else None
                                if desc:
                                    print(f'       Erro com: {desc}')
                                else:
                                    print(f'       Erro com: {e}')
                    except Exception as e:
                        print(f'       Erro ao paginar {elemento.text}')
                        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
                        print(f'       Na linha: {traceback_str}')
                except Exception as e:
                    print(f'       Erro na paginação dos elementos de resultados homônimos')
                    erro_regex = r"(?:\w+): (.*)"
                    print(f'       Erro com: {re.findall(erro_regex, str(e))[0]}')
                    # traceback_str = ''.join(traceback.format_tb(e.__traceback__))
                    # print(f'       Na linha: {traceback_str}')
                if self.is_stale_file_handler_present():
                    raise StaleElementReferenceException      
        
        ## Retornar None ao detectar erro de StaleFileHanlder
        except StaleElementReferenceException as e:
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            if self.verbose:
                print(traceback_str)
            return None, np.NaN, np.NaN, np.NaN, self.driver
       
        ## Verificar antes de retornar para garantir que elm_vinculo foi definido
        if elm_vinculo is None:
            if count > 1:
                des='s'
            else:
                des=''
            print(f"       Nenhum termo de vínculo achado em {count:02}/{intLTotReg:02} resultado{des} verificados. Verificada até página {count_pages:02}/{len(numpaginas):02}")
            return None, NOME, np.NaN, 'Termos não encontrados', self.driver
        ## Retorna a saída de sucesso
        return elm_vinculo, np.NaN, np.NaN, np.NaN, self.driver


    ## Função para iterar busca por cada nome, lidando internamente na find terms caso achar
    def search_profile(self, name, termos_busca):
        '''
        Usa a função de achar os termos-chave para assegurar escolha do homônimo correto
        '''
        try:
            # Interagir internamente com a busca realizando várias tentativas
            profile_element, _, _, _, _ = self.find_terms(
                name,
                termos_busca,
                10,  # delay extração tooltips (10 funciona sem erros em dia normal)
                3 # limite de tentativas
            )
            # print('Elemento encontrado:', profile_element)
            if profile_element:
                return profile_element
            else:
                logging.info(f'Currículo não encontrado: {name}')
                self.return_search_page()

            if profile_element is None:
                logging.error(f"Erro StaleElementReferenceException ao buscar currículo")
                raise StaleElementReferenceException

        except requests.HTTPError as e1:
            logging.error(f"HTTPError occurred: {str(e1)}")
            return None
        except Exception as e2:
            logging.error(f"Erro inesperado ao buscar: {str(e2)}")
            return None

    def scrape_single(self, name, termos_busca):
        dict_list = []  # Inicialize a lista de dicionários vazia
        try:
            self.fill_name(name)
            try:
                elm_vinculo = self.search_profile(name, termos_busca)
            ## TO-FIX: implementar tentativas de refresh e buscar novamente aqui, fora do find terms
            except StaleElementReferenceException as e:
                traceback_str = ''.join(traceback.format_tb(e.__traceback__))
                if self.verbose:
                    print(traceback_str)
                base = 2  # Fator de multiplicação exponencial
                max_wait_time = 120  # Tempo máximo de espera em segundos
                for i in range(1, 6):  # Tentativas máximas com espera exponencial
                    wait_time = min(base ** i, max_wait_time)  # Limita o tempo máximo de espera
                    print(f"       {'-'*120}")
                    print(f"       Erro no servidor CNPq ao buscar nome, tentar novamente em {wait_time} segundos...")
                    time.sleep(wait_time)
                    try:
                        self.search_profile(name, termos_busca)
                        break  # Se o clique for bem-sucedido, saia do loop de retry
                    except TimeoutException as se:
                        traceback_str = ''.join(traceback.format_tb(se.__traceback__))
                        print(se)
                        print(traceback_str)
                        logging.error(f"Tentativa {i} falhou: {traceback_str}.")
                        limite+=1
                if limite <= 0:
                    print("       Tentativas esgotadas. Abortando ação.")                
            if elm_vinculo:
                if self.verbose:
                    print(f"       {name}: vínculo encontrado no currículo, tentando abrir...")
                self.retry_click_vinculo(elm_vinculo) # Tentativas recorrentes para clicar no abrirCurrículo
                if self.verbose:
                    print(f"       {name}:  mudar para nova janela após clique para abrir currículo...")
                window_before = self.switch_to_new_window()
                if self.verbose:
                    print(f"       {name}: mudado para nova janela com sucesso após abrir currículo...")                
                if self.is_stale_file_handler_present():
                    fib = [0, 1]
                    print(f"       {name}: Erro 'Stale File Handler' detectado na página. Tentando novamente em {fib[-1]} segundos...")
                    for i in range(2, 12):  # Tentativas máximas com espera para contornar erro de Stale File Handler
                        fib.append(fib[i-1] + fib[i-2])
                    for i, wait_time in enumerate(fib):
                        logging.info(f"       Tentativa {i+1} com tempo de espera de {wait_time} segundos...")
                        time.sleep(wait_time)
                        try:
                            self.retry_click_vinculo(elm_vinculo)
                            break  # Se o clique for bem-sucedido, saia do loop de retry
                        except TimeoutException as se:
                            logging.error(f"Tentativa {i+1} falhou: {se}.")
                tooltip_data_list = self.extract_tooltip_data() # Extraindo dados ocultos nos tooltips
                if self.verbose:
                    print(f'       {len(tooltip_data_list):003} tooltips extraídos com sucesso...')
                page_source = self.driver.page_source
                if page_source is not None:
                    if self.verbose:
                        print(f'       Encontrada page_source do driver...')
                    try:
                        parser = HTMLParser(page_source)
                        data = parser.to_json()
                        if self.verbose:
                            print(f'       HTMLParser rodado com sucesso...')
                        data['JCR2'] = tooltip_data_list
                    except Exception as e:
                        print(f'       Erro ao fazer o parser do HTML:')
                        print(f'       {e}')
                    print(f'       Extração bem-sucedida')
                    # Adicione o dicionário de dados à lista
                    dict_list.append(data)
                    if self.verbose:
                        print('       Tentar voltar a página de busca...')
                    self.switch_back_to_original_window()
                    if self.verbose:
                        print('       Disparado switch_back_to_original_window()...')
                    self.close_popup()
                    self.return_search_page()
                    if self.verbose:
                        print('       Disparado retorno para página de busca...')
                else:
                    print(f"{name}: página de resultado vazia.")
        
        ## Captura corretamente o erro de StaleFileHanlder
        except StaleElementReferenceException as e:
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            if self.verbose:
                print(traceback_str)
            base = 2  # Fator de multiplicação exponencial (pode ser ajustado)
            max_wait_time = 120  # Tempo máximo de espera em segundos
            for i in range(1, 12):  # Tentativas máximas com espera exponencial
                wait_time = min(base ** i, max_wait_time)  # Limita o tempo máximo de espera
                print(f"       {'-'*120}")
                print(f"        Servidor do CNPq com StaleFileError ao procurar termos, tentando novamente em {wait_time} segundos...")
                time.sleep(wait_time)
                
                ## Refresh e reinserir o mesmo nome e tentar clicar novamente no botão buscar

                ## Tentar clicar novamente no botão de abrir currículo achado
                try:
                    self.scrape_single(name, termos_busca)
                    break  # Se o clique for bem-sucedido, saia do loop de retry
                except TimeoutException as se:
                    traceback_str = ''.join(traceback.format_tb(se.__traceback__))
                    print(se)
                    print(traceback_str)
                    logging.error(f"Tentativa {i} falhou: {traceback_str}.")
                    limite+=1
            if limite <= 0:
                print("       Tentativas esgotadas. Abortando ação.")

        except Exception as e:
            raise TimeoutException(f"       Erro ao realizar a extração para {name}: {e}")
        return dict_list

    def scrape_retry(self, name, termos_busca, retry_count):
        dict_list = []
        try:
            for _ in range(retry_count):
                try:
                    dict_list.extend(self.scrape_single(name, termos_busca))
                    break  # Se o a extração for bem-sucedido, saia do loop de retry
                except TimeoutException as se:
                    logging.error(f"Tentativa de extrair currículo de {name} em scrape_retry() falhou: {se}. Tentando novamente...")
                    time.sleep(1)  # Aguarda um segundo antes de tentar novamente
            else:
                logging.error(f"Todas as tentativas em em scrape_retry() falharam para {name}.")
        except Exception as e:
            raise TimeoutException(f"Erro inesperado em scrape_retry() para {name}")
        return dict_list

    # Realizar chamada recursiva para processar cada nome da lista
    def scrape(self, name_list, termos_busca, retry_count=5):
        dict_list = []
        for k, name in enumerate(name_list):
            print(f'{k+1:>2}/{len(name_list)}: {name}')
            try:
                dict_list.extend(self.scrape_retry(name, termos_busca, retry_count))
            except TimeoutException:
                logging.error(f"Erro de Timeout ao extrair {name}")
                if retry_count > 0:
                    logging.info(f"Tentando novamente para {name}...")
                    # Realiza novar tentativa para o mesmo nome passando número de tentativas decrementado de 1
                    dict_list.extend(self.scrape([name], termos_busca, retry_count-1))
                else:
                    logging.error(f"Todas as tentativas falharam para {name}")
            except Exception as e:
                logging.error(f"Erro inesperado ao extrair {name}: {e}")
                traceback_str = ''.join(traceback.format_tb(e.__traceback__))
                logging.error(traceback_str)
        self.driver.quit()
        try:
            filepath = os.path.join(str(LattesScraper.find_repo_root()),'_data','in_csv','temp_dict_list.json')
            print(f'Arquivo salvo em {filepath}')
        except:
            print('Não foi possível salvar extração em arquivo')
        self.save_to_json(dict_list, filepath)

        return dict_list

    def verificar_remanescentes(self, lista_busca, dom_dict_list):
        lista_restante = lista_busca[:]
        print(f'{len(lista_busca)} currículos buscados')
        print(f'{len(dom_dict_list)} currículos extraídos com sucesso')
        total_extraidos = 0
        total_nao_extraidos = 0

        # Função para normalizar os nomes
        def normalizar_nome(nome):
            # Normaliza o nome para comparar de forma mais flexível
            nome_normalizado = nome.lower().replace(' ', '').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('â', 'a').replace('ê', 'e').replace('ô', 'o').replace('ã', 'a').replace('õ', 'o').replace('ç', 'c').replace('ü', 'u')
            return nome_normalizado

        # Lista para armazenar nomes extraídos com sucesso
        nomes_extraidos = []

        for i in dom_dict_list:
            nome = i.get('Identificação').get('Nome')
            nome_normalizado = normalizar_nome(nome)
            encontrado = False

            # Verifica se o nome ou uma forma similar já foi extraído
            for nome_original in lista_restante:
                nome_original_normalizado = normalizar_nome(nome_original)
                if nome_original_normalizado == nome_normalizado:
                    lista_restante.remove(nome_original)
                    encontrado = True
                    break

            # Incrementa o contador correspondente
            if encontrado:
                total_extraidos += 1
            else:
                total_nao_extraidos += 1

            # Imprime o status da extração
            if encontrado:
                print(f'({total_extraidos:>02}) {nome}')
            else:
                print(f'({total_nao_extraidos}) Não extraído: {nome}')
            # Adiciona à lista de nomes extraídos apenas se não for exatamente igual ao buscado
            if encontrado and nome_original_normalizado != nome_normalizado:
                nomes_extraidos.append(nome)

        print(f'\n{len(lista_restante)} currículos não extraídos')

        for i in lista_restante:
            print(f'   {i}')
        
        return lista_restante

    def extract_remanescents(self, lista_restante, dict_list_actual, search_terms):
        print(f'Resta extrair {len(lista_restante)} currículos:')
        print(lista_restante)
        print('-'*110)
        
        if lista_restante:
            # Iniciar a extração de currículos remanescentes
            t0 = time.time()
            scraper = LattesScraper(search_terms, 'bolt://localhost:7687', 'neo4j', 'password', only_doctors=False)
            dict_list_1 = scraper.scrape(lista_restante, search_terms)
            print(f'\n{self.tempo(t0,time.time())} para busca de {len(lista_restante)} nomes com extração de dados de {len(dict_list_1)} dicionários')

            # Dicionário vazio para armazenar elementos combinados
            dicionario_combinado = {}

            # Percorrer a lista de dicionários existentes (dict_list_docents)
            for dicionario in dict_list_actual:
                # Obter a chave única do dicionário
                chave_unica = dicionario.get("Identificação").get("Nome")

                # Verificar se a chave é hashável
                if isinstance(chave_unica, (str, int, float, tuple)):
                    # Se a chave for hashável, adicionar o dicionário ao dicionário combinado
                    dicionario_combinado[chave_unica] = dicionario
                else:
                    # Se a chave não for hashável, registrar um erro
                    print(f"Erro: Chave 'Identificação' não hashável em dicionário: {dicionario}")

            # Percorrer a lista de dicionários recém-extraídos (dict_list_1)
            for dicionario in dict_list_1:
                # Obter a chave única do dicionário
                chave_unica = dicionario.get("Identificação").get("Nome")

                # Verificar se a chave é hashável
                if isinstance(chave_unica, (str, int, float, tuple)):
                    # Se a chave for hashável, adicionar o dicionário ao dicionário combinado
                    # Verifique se a chave já existe no dicionário combinado
                    if chave_unica not in dicionario_combinado:
                        dicionario_combinado[chave_unica] = dicionario
                    # else:
                    #     # Se a chave já existe, combinar os valores dos dicionários
                    #     dicionario_combinado = dict_list_actual.copy()
                    #     dicionario_combinado.append(dicionario)                
                else:
                    # Se a chave não for hashável, registrar um erro
                    print(f"Erro: Chave 'Identificação' não hashável em dicionário: {dicionario}")

            # Converter o dicionário em lista
            lista_dict_combinado = list(dicionario_combinado.values())
            print(f'Total de dicionários na lista completa: {len(lista_dict_combinado)}')
            
            # Obter o caminho do arquivo JSON
            pathfilename = os.path.join(os.getcwd(), '_data', 'in_csv', 'combined_dict_list.json')
            self.save_to_json(lista_dict_combinado, pathfilename)

            print(f"Arquivo JSON salvo em: {pathfilename}")

            return lista_dict_combinado
        else:
            print(f'{len(dict_list_actual)-len(lista_restante)} Currículos já extraídos com sucesso.')

    def avaliar_remanescentes(self, lista_busca, dict_list_docents, filename='temp_dict_list.json'):
        print(f'{len(lista_busca)} currículos a buscar no total')
        print(f'{len(dict_list_docents)} currículos já extraídos')
        
        total_extraidos = 0
        total_nao_extraidos = 0
        self.driver.quit()
        
        ## Lista para armazenar nomes extraídos com sucesso
        nomes_extraidos = []

        jfm = JSONFileManager()
        ## Carregar arquivo dict_list_temp.json
        # pathfilename = os.path.join(self.find_repo_root(), '_data','in_csv', filename)
        # dict_list_docents, formatted_creation_date, formatted_modification_date, time_count, unit = jfm.load_from_json(pathfilename)
        lista_restante = lista_busca[:]

        for i in dict_list_docents:
            nome = i.get('Identificação').get('Nome')
            nome_normalizado = self.normalizar_nome(nome)
            encontrado = False

            ## Verificar se o nome ou uma forma similar já foi extraído
            for nome_original in lista_restante:
                nome_original_normalizado = self.normalizar_nome(nome_original)
                if nome_original_normalizado == nome_normalizado or nome_original_normalizado in nome_normalizado:
                    lista_restante.remove(nome_original)
                    encontrado = True
                    break

            ## Incrementar o contador correspondente
            if encontrado:
                total_extraidos += 1
            else:
                total_nao_extraidos += 1

            ## Imprimir o status da extração
            if encontrado:
                print(f'({total_extraidos:>02}) {nome}')
            else:
                print(f'({total_nao_extraidos}) Não extraído: {nome}')
            ## Adicionar à lista de nomes extraídos apenas se não for exatamente igual ao buscado
            if encontrado and nome_original_normalizado != nome_normalizado:
                nomes_extraidos.append(nome)

        print(f'\n{len(lista_restante)} currículos não extraídos')
        for i in lista_restante:
            print(f'   {i}')

        return lista_restante


class HTMLParser:
    def __init__(self, html):
        self.soup = BeautifulSoup(html, 'html.parser')
        self.dados_planilha = pd.read_excel(os.path.join(LattesScraper.find_repo_root(),'_data','in_xls','classificações_publicadas_todas_as_areas_avaliacao1672761192111.xlsx'))
        self.ignore_classes = ["header", "sub_tit_form", "footer", "rodape-cv", "menucent", "menu-header", "menuPrincipal", "to-top-bar", "header-content max-width", "control-bar-wrapper", "megamenu","layout-cell-6"]
        self.ignore_elements = ['script', 'style', 'head']
        self.visited_structures = defaultdict(int)
        self.estrutura = {}
        self.verbose = False

    def should_ignore(self, node):
        if isinstance(node, bs4.element.Tag):
            if node.name in self.ignore_elements:
                return True
            if any(cls in self.ignore_classes for cls in node.get('class', [])):
                return True
        return False

    # Revelar a estrutura hierárquica do HTML
    def explore_structure(self):
        self.print_node_hierarchy(self.soup)

    def print_node_hierarchy(self, node, level=0, parent_counts=defaultdict(int)):
        if self.should_ignore(node):
            return

        prefix = "  " * level
        if isinstance(node, bs4.element.Tag):
            # Verificar se o nó é parte de uma lista repetitiva
            if node.name == 'b' and node.text.isdigit():
                num = int(node.text)
                if num == 1 or (num - 1) == parent_counts[node.parent.name]:
                    print(f"{prefix}[Estrutura repetida começa aqui]")
                parent_counts[node.parent.name] = num
                
                if num > 1:
                    return

            print(f"{prefix}<{node.name} class='{node.get('class', '')}'>")
            for child in node.children:
                self.print_node_hierarchy(child, level + 1, parent_counts)
        elif isinstance(node, bs4.element.NavigableString) and node.strip():
            print(f"{prefix}{node.strip()}")

    def find_path_to_text(self, node, text, path=None):
        if path is None:
            path = []

        if self.should_ignore(node):
            return None

        if node.string and text.lower() in node.string.lower():
            return path

        if isinstance(node, bs4.element.Tag):
            for i, child in enumerate(node.children):
                child_path = path + [(node.name, node.get('class'), i)]
                found_path = self.find_path_to_text(child, text, child_path)
                if found_path:
                    return found_path

        return None

    def find_element_by_path(self, path):
        """
        Encontra o elemento no BeautifulSoup object baseado no caminho fornecido.
        O caminho é uma lista de tuplas (tag, classe, índice).
        """
        current_element = self.soup
        for tag, classe, index in path:
            try:
                if classe:  # Se uma classe foi especificada
                    current_element = current_element.find_all(tag, class_=classe)[index]
                else:  # Se nenhuma classe foi especificada
                    current_element = current_element.find_all(tag)[index]
            except IndexError:
                return None  # Retorna None se o caminho não levar a um elemento válido
        return current_element

    def extract_data_from_path(self, path):
        """
        Extrai dados de um caminho especificado.
        O caminho é uma lista de direções para navegar na árvore HTML.
        """
        # Inicia no elemento raiz (soup)
        current_element = self.soup

        # Navega pelo caminho
        for tag, classe, index in path:
            # Tenta encontrar o próximo elemento no caminho
            if classe:  # Se classe for especificada
                current_element = current_element.find_all(tag, class_=classe)
            else:  # Se não, apenas pela tag
                current_element = current_element.find_all(tag)

            # Tenta acessar o elemento pelo índice, se falhar retorna None
            try:
                current_element = current_element[index]
            except IndexError:
                return None

        # Retorna o elemento encontrado
        return current_element

    def explore_structure_for_text(self, target_text):
        """
        Encontra o caminho hierárquico até o texto desejado, considerando tags, classes e índices.
        """
        def find_path(element, path=[]):
            if element.name in self.ignore_elements:
                return False
            if isinstance(element, Tag):
                for class_ in self.ignore_classes:
                    if class_ in element.get('class', []):
                        return False
                for child in element.children:
                    new_path = path + [(element.name, ' '.join(element.get('class', [])), element.index(child))]
                    if child.string and target_text in child.string:
                        return new_path
                    else:
                        found_path = find_path(child, new_path)
                        if found_path:
                            return found_path
            return False

        found_path = find_path(self.soup)
        if found_path:
            # Transforma o caminho em uma lista de dicionários.
            path_dicts = [{'Tag': tag, 'Classes': classes, 'Index': index} for tag, classes, index in found_path]
            return path_dicts
        else:
            return f"Caminho para '{target_text}' não encontrado."

    ## Processamentos da extração de dados
    # Identificação OK!!            
    def process_identification(self):
        nome = self.soup.find(class_="nome").text.strip()
        id_lattes = self.soup.find("span", style=lambda value: value and "color: #326C99" in value).text.strip()
        
        ultima_atualizacao_element = self.soup.find(lambda tag: tag.name == "li" and "Última atualização do currículo em" in tag.text)
        if ultima_atualizacao_element:
            ultima_atualizacao = ultima_atualizacao_element.text.split("em")[1].strip()
        else:
            ultima_atualizacao = "Não encontrado"

        self.estrutura["Identificação"] = {
            "Nome": nome,
            "ID Lattes": id_lattes,
            "Última atualização": ultima_atualizacao
        }

    # Idiomas OK!!!
    def process_idiomas(self):
        idiomas = []
        idiomas_header = self.soup.find("h1", text=lambda text: text and "Idiomas" in text)
        
        if idiomas_header:
            idiomas_container = idiomas_header.find_next("div", class_="data-cell")
            
            if not idiomas_container:
                # Se não encontrou usando a classe "data-cell", tenta buscar o próximo container de maneira mais genérica
                idiomas_container = idiomas_header.find_next_sibling()

            if idiomas_container:
                idioma_divs = idiomas_container.find_all("div", recursive=False)
                for idioma_div in idioma_divs:
                    idioma = idioma_div.find("div", class_="layout-cell-pad-5 text-align-right")
                    proficiencia = idioma_div.find_next("div", class_="layout-cell layout-cell-9")
                    
                    if idioma and proficiencia:
                        idioma_text = idioma.text.strip()
                        proficiencia_text = proficiencia.text.strip()
                        idiomas.append({"Idioma": idioma_text, "Proficiência": proficiencia_text})
                    else:
                        continue
            else:
                print("       Container de idiomas não encontrado")
        else:
            print("       Seção de idiomas não encontrada")

        self.estrutura["Idiomas"] = idiomas

    # Formação acadêmica e complementar OK!
    def process_formacao(self):
        formacao_academica = []
        formacao_posdoc = []
        formacao_complementar = []
        
        # Encontrar todas as seções 'title-wrapper' que contêm os títulos 'Formação acadêmica/titulação' e 'Formação Complementar'
        secoes = self.soup.find_all('div', class_='title-wrapper')
        
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1:
                titulo = titulo_h1.get_text(strip=True)
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                if data_cell:
                    # Processar cada item de formação dentro da data_cell
                    # A estrutura é sempre a mesma: ano à direita, descrição à esquerda
                    anos_divs = data_cell.find_all('div', class_='layout-cell layout-cell-3 text-align-right')
                    descricoes_divs = data_cell.find_all('div', class_='layout-cell layout-cell-9')
                    
                    for ano_div, descricao_div in zip(anos_divs, descricoes_divs):
                        ano = ano_div.get_text(strip=True)
                        descricao = descricao_div.get_text(separator=' ', strip=True).replace(' .', '.')
                        formacao = {"Ano": ano, "Descrição": descricao}
                        
                        if 'Formação acadêmica/titulação' in titulo:
                            formacao_academica.append(formacao)
                        elif 'Pós-doutorado' in titulo:
                            formacao_posdoc.append(formacao)
                        elif 'Formação Complementar' in titulo:
                            formacao_complementar.append(formacao)

        # Armazenar ou retornar os dados de formação
        self.estrutura["Formação"] = {
            "Acadêmica": formacao_academica,
            "Pos-Doc": formacao_posdoc,
            "Complementar": formacao_complementar
        }
        
        # Retorna o dicionário de formação se necessário
        return self.estrutura["Formação"]

    ## Linhas de Pesquisa OK!
    def process_linhas_pesquisa(self):
        linhas_pesquisa = []
        # Encontrar a seção específica de linhas de pesquisa
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Linhas de pesquisa' in titulo_h1.get_text(strip=True):
                # Encontrar todos os blocos de dados dentro da seção
                data_cell = secao.find('div', class_='layout-cell-12')
                if data_cell:
                    # Encontrar todos os elementos de título e descrição dentro da data_cell
                    elements = data_cell.find_all(recursive=False)
                    detalhes = ""
                    for i, element in enumerate(elements):
                        if element.name == 'div' and 'layout-cell-9' in element.get('class', []):
                            descricao = element.get_text(strip=True)
                            if 'Objetivo:' in descricao and i+1 < len(elements):
                                descricao = element.find_previous_sibling('div', class_='layout-cell-9').get_text(strip=True)
                                detalhes += element.get_text(separator=' ', strip=True) + ' '
                            elif element.name == 'div' and 'text-align-right' in element.get('class', []):
                                continue

                            linhas_pesquisa.append({
                                "Descrição": descricao,
                                "Detalhes": detalhes.strip()
                            })
        
        self.estrutura["Linhas de Pesquisa"] = linhas_pesquisa
        return linhas_pesquisa

    ## Atuação Profissional OK!
    def process_atuacao_profissional(self):
        atuacoes_profissionais = []
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Atuação Profissional' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                if data_cell:
                    # Iniciamos a coleta de dados do primeiro bloco após inst_back
                    elements = data_cell.find_all(recursive=False)
                    current_instituicao = None
                    current_block = []

                    # Iteramos sobre os elementos para capturar informações até a próxima inst_back
                    for element in elements:
                        if element.name == 'div' and 'inst_back' in element.get('class', []):
                            if current_instituicao:  # Se já havia uma instituição, processamos o bloco acumulado
                                self.extract_atuacao_from_block(current_block, atuacoes_profissionais, current_instituicao)
                                current_block = []  # Reiniciamos o bloco para a próxima instituição
                            current_instituicao = element.get_text(strip=True)
                        elif current_instituicao:  # Estamos dentro do bloco de uma instituição
                            current_block.append(element)

                    # Não esqueça de processar o último bloco
                    if current_instituicao and current_block:
                        self.extract_atuacao_from_block(current_block, atuacoes_profissionais, current_instituicao)

        self.estrutura["Atuação Profissional"] = atuacoes_profissionais
        return self.estrutura["Atuação Profissional"]

    def extract_atuacao_from_block(self, block, atuacoes_profissionais, instituicao_nome):
        ano_pattern = re.compile(r'(\d{2}/)?\d{4}\s*-\s*(\d{2}/)?(?:\d{4}|Atual)')
        # Removemos os padrões que não serão usados diretamente na identificação de elementos

        ano = None
        descricao = None
        outras_informacoes = []

        for element in block:
            # Captura o ano e descrição
            if element.name == 'div' and 'text-align-right' in element.get('class', []):
                if ano_pattern.search(element.get_text(strip=True)):
                    if ano:  # Se um ano já foi capturado, então terminamos de processar o bloco anterior
                        atuacao = {
                            "Instituição": instituicao_nome,
                            "Ano": ano,
                            "Descrição": descricao,
                            "Outras informações": ' '.join(outras_informacoes)
                        }
                        atuacoes_profissionais.append(atuacao)
                        outras_informacoes = []  # Reiniciamos a lista para o próximo bloco
                    ano = element.get_text(strip=True)
                    descricao = element.find_next('div', class_='layout-cell-9').get_text(separator=' ', strip=True) if descricao else ""
            elif element.name == 'div' and 'layout-cell-9' in element.get('class', []):
                # Acumula todas as informações das divs 'layout-cell-9' dentro do mesmo bloco
                outras_infos = element.get_text(separator=' ', strip=True)
                if outras_infos:  # Verifica se há texto dentro do elemento
                    outras_informacoes.append(outras_infos)

        # Verifica se ainda existe um bloco a ser adicionado após o loop
        if ano:
            atuacao = {
                "Instituição": instituicao_nome,
                "Ano": ano,
                "Descrição": descricao,
                "Outras informações": ' '.join(outras_informacoes)
            }
            atuacoes_profissionais.append(atuacao)

        return atuacoes_profissionais

    def process_producao_bibliografica(self):
        # Inicializa a lista de produções bibliográficas
        self.estrutura["ProducaoBibliografica"] = {
            "Artigos completos publicados em periódicos": [],
            "Livros e capítulos": [],
            "Trabalhos completos publicados em anais de congressos": [],
            # Adicione mais categorias conforme necessário
        }

        # Mapeia os identificadores das seções para as categorias de produção bibliográfica
        secoes = {
            "ArtigosCompletos": "Artigos completos publicados em periódicos",
            "LivrosCapitulos": "Livros e capítulos",
            "TrabalhosPublicadosAnaisCongresso": "Trabalhos completos publicados em anais de congressos",
        }

        # Percorre cada seção de interesse no documento HTML
        for secao_id, categoria in secoes.items():
            secao_inicio = self.soup.find("a", {"name": secao_id})
            if not secao_inicio:
                continue

            # Encontra todos os itens dentro da seção até a próxima seção
            proxima_secao = secao_inicio.find_next_sibling("a", href=True)
            itens_secao = []
            atual = secao_inicio.find_next_sibling("div", class_="layout-cell layout-cell-11")
            while atual and atual != proxima_secao:
                if atual.text.strip():
                    itens_secao.append(atual.text.strip())
                atual = atual.find_next_sibling("div", class_="layout-cell layout-cell-11")

            # Adiciona os itens encontrados à categoria correspondente
            self.estrutura["ProducaoBibliografica"][categoria].extend(itens_secao)

    def process_producao_bibliografica(self):
        producoes = []
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Produções' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                div_artigos = data_cell.find_all("div", id="artigos-completos")
                for div in div_artigos:
                    if div:
                        # Iniciamos a coleta de dados do primeiro bloco após inst_back
                        articles = div.find_all("div", class_="artigo-completo", recursive=False)
                        # print(f'{len(articles)} divs de artigos')
                        current_instituicao = None
                        current_block = []

                        # Iteramos sobre os elementos para capturar informações até a próxima inst_back
                        for element in articles:
                            if element.name == 'div' and 'inst_back' in element.get('class', []):
                                if current_instituicao:  # Se já havia uma produção, processamos o bloco acumulado
                                    self.extract_producao_from_block(current_block, producoes, current_instituicao)
                                    current_block = []  # Reiniciamos o bloco para a próxima instituição
                                current_instituicao = element.get_text(strip=True)
                            elif current_instituicao:  # Estamos dentro do bloco de uma instituição
                                current_block.append(element)

                        # Processar o último bloco
                        if current_instituicao and current_block:
                            self.extract_producao_from_block(current_block, producoes, current_instituicao)

        self.estrutura["ProducaoBibliografica"] = producoes
        return self.estrutura["ProducaoBibliografica"]

    def extract_producao_from_block(self, block, producoes, instituicao_nome):
        ano_pattern = re.compile(r'(\d{2}/)?\d{4}\s*-\s*(\d{2}/)?(?:\d{4}|Atual)')
        # Removemos os padrões que não serão usados diretamente na identificação de elementos

        ano = None
        descricao = None
        outras_informacoes = []

        for element in block:
            # Captura o ano e descrição
            if element.name == 'div' and 'text-align-right' in element.get('class', []):
                if ano_pattern.search(element.get_text(strip=True)):
                    if ano:  # Se um ano já foi capturado, então terminamos de processar o bloco anterior
                        atuacao = {
                            "Instituição": instituicao_nome,
                            "Ano": ano,
                            "Descrição": descricao,
                            "Outras informações": ' '.join(outras_informacoes)
                        }
                        producoes.append(atuacao)
                        outras_informacoes = []  # Reiniciamos a lista para o próximo bloco
                    ano = element.get_text(strip=True)
                    descricao = element.find_next('div', class_='layout-cell-9').get_text(separator=' ', strip=True) if descricao else ""
            elif element.name == 'div' and 'layout-cell-9' in element.get('class', []):
                # Acumula todas as informações das divs 'layout-cell-9' dentro do mesmo bloco
                outras_infos = element.get_text(separator=' ', strip=True)
                if outras_infos:  # Verifica se há texto dentro do elemento
                    outras_informacoes.append(outras_infos)

        # Verifica se ainda existe um bloco a ser adicionado após o loop
        if ano:
            atuacao = {
                "Instituição": instituicao_nome,
                "Ano": ano,
                "Descrição": descricao,
                "Outras informações": ' '.join(outras_informacoes)
            }
            producoes.append(atuacao)

        return producoes

    def extrair_texto_sup(element):
        # Busca por todos os elementos <sup> dentro do elemento fornecido
        sup_elements = element.find_all('sup')
        # Lista para armazenar os textos extraídos
        textos_extras = []

        for sup in sup_elements:
            # Verifica se o elemento <sup> contém um elemento <img> com a classe 'ajaxJCR'
            if sup.find('img', class_='ajaxJCR'):
                # Extrai o valor do atributo 'original-title', se disponível
                texto = sup.find('img')['original-title'] if sup.find('img').has_attr('original-title') else None
                if texto:
                    textos_extras.append(texto)
        
        return textos_extras

    def extrair_dados_jcr(texto):
        # Regex para capturar o nome do periódico e o fator de impacto
        regex = r"(.+?)\s*\((\d{4}-\d{4})\)<br />\s*Fator de impacto \(JCR (\d{4})\):\s*(\d+\.\d+)"
        match = re.search(regex, texto)

        if match:
            periódico = f"{match.group(1)} ({match.group(2)})"
            fator_de_impacto = f"Fator de impacto (JCR {match.group(3)}): {match.group(4)}"
            return periódico, fator_de_impacto
        else:
            return None, None

    def extrair_dados_jcr(self, html_element):
        sup_tag = html_element.find('sup')
        img_tag = sup_tag.find('img')
        attributes_dict = {}
        # Extraia os atributos básicos
        # not_extract=['class','id','src']
        # attributes_dict = {key: value for key, value in img_tag.attrs.items() if key != 'original-title' and key not in not_extract}
        issn = sup_tag.find('img', class_='data-issn')
        # print(f'ISSN: {issn}')
        attributes_dict['data-issn'] = issn
        # original_title = sup_tag['original-title'].replace('&lt;br /&gt;', '')
        original_title = sup_tag.find('img', class_='original-title')
        # print(f'ISSN: {original_title}')
        parts = original_title.split(': ')
        periodico_info = parts[0].split('(')
        fator_impacto = parts[1]

        # Atualiza o dicionário com as informações processadas
        attributes_dict['periodico'] = f"{periodico_info[0].strip()} ({periodico_info[1].split('<br />')[0].strip(')')})"
        attributes_dict['fator_impacto'] = float(fator_impacto.split(' ')[0])
        attributes_dict['JCR'] = parts[0].split('(')[-1].split(')')[0]
        return attributes_dict
    
    def extract_year(self, soup):
        # Encontre o elemento <span> com a classe 'informacao-artigo' e data-tipo-ordenacao='ano'
        year_span = soup.findChild('span', {'class': 'informacao-artigo', 'data-tipo-ordenacao': 'ano'})
        
        # Recupera o texto do elemento, que deve ser o ano
        year = year_span.text if year_span else 'Ano não encontrado'

        return year

    def extract_first_author(self, soup):
        # Encontre o elemento <span> com a classe 'informacao-artigo' e data-tipo-ordenacao='autor'
        author_span = soup.findChild('span', {'class': 'informacao-artigo', 'data-tipo-ordenacao': 'autor'})
        
        # Recupera o texto do elemento, que deve ser o nome do primeiro autor
        first_author = author_span.text if author_span else 'Ano não encontrado'

        return first_author

    def extract_periodico(self, soup):
        # Encontre o elemento <span> com a classe 'informacao-artigo' e data-tipo-ordenacao='autor'
        img_tag = soup.findChild('sup')
        dados_periodico = img_tag.findChild('img', class_='original-title')
        if dados_periodico:
            parts = dados_periodico.split('(')
            print(parts)
        else:
            print(f"       Não foi possível extrair dados do periódico de {soup}")
        # Recupera o texto do elemento, que deve ser o nome do primeiro autor
        periodico = dados_periodico.text if dados_periodico else None

        return periodico
    
    def extract_qualis(self, soup):
        # Extração de informações do Qualis a partir do elemento 'p'
        p_tag = soup.find('p')
        qualis_text = p_tag.get_text(strip=True) if p_tag else ''
        qualis_match = re.search(r'[ABC]\d', qualis_text)
        qualis = qualis_match.group(0) if qualis_match else 'Indisponível'

        # Extração de informações JCR a partir do elemento 'sup'
        sup_tag = soup.find('sup')
        jcr_info = sup_tag.find('img')['original-title'] if sup_tag and sup_tag.find('img') else ''
        jcr_parts = jcr_info.split('<br />') if jcr_info else []
        jcr = jcr_parts[-1].split(': ')[-1].strip() if len(jcr_parts) > 1 else 'Indisponível'

        # Compilando resultados
        results = {
            'Qualis': qualis,
            'JCR': jcr
        }
        return results

    ## Não considerava os casos onde lista de autores vem oculta por et al. com javscript
    # def extract_info(self):
    #     soup = BeautifulSoup(self.html_element, 'html.parser')
    #     qualis_info = self.extract_qualis(soup)

    #     # Extrai o primeiro autor
    #     autores = soup.find_all('span', class_='informacao-artigo', data_tipo_ordenacao='autor')
    #     primeiro_autor = autores[0].text if autores else None
    #     # Considera todos os textos após o autor como parte da lista de autores até um elemento estrutural significativo (<a>, <b>, <sup>, etc.)
    #     autores_texto = self.html_element.split('autor">')[-1].split('</span>')[0] if autores else ''

    #     ano_tag = soup.find('span', {'class': 'informacao-artigo', 'data-tipo-ordenacao': 'ano'})
    #     ano = int(ano_tag.text) if ano_tag else 'Ano não disponível'

    #     # Extrai o título, periódico, e outras informações diretamente do texto
    #     texto_completo = soup.get_text(separator=' ', strip=True)
        
    #     # Assume que o título vem após os autores e termina antes de uma indicação de periódico ou volume
    #     titulo_match = re.search(r'; ([^;]+?)\.', texto_completo)
    #     titulo = titulo_match.group(1) if titulo_match else None

    #     # Periódico e detalhes como volume, página, etc., 
    #     periodico_match = re.search(r'(\. )([^.]+?),( v\. \d+, p\. \d+, \d+)', texto_completo)
    #     periodico = periodico_match.group(2) if periodico_match else None
    #     detalhes_periodico = periodico_match.group(3) if periodico_match else None

    #     # Extrai citações se disponível
    #     citacoes = soup.find('span', class_='numero-citacao')
    #     citacoes = int(citacoes.text) if citacoes else 0

    #     # Extrai ISSN
    #     issn = soup.find('img', class_='ajaxJCR')
    #     issn = issn['data-issn'] if issn else None

    #     # Qualis/CAPES pode ser extraído se existir um padrão identificável
    #     qualis_capes = "quadriênio 2017-2020"  # Hardcoded, mas pode ser ajustado futuramente

    #     # Monta o dicionário de resultados
    #     resultado = {
    #         "dados_gerais": texto_completo,
    #         "primeiro_autor": primeiro_autor,
    #         "ano": ano,
    #         "autores": autores_texto,
    #         "titulo": titulo,
    #         "periodico": f"{periodico}{detalhes_periodico}",
    #         "data-issn": issn,
    #         "impacto": qualis_info.get('JCR'),
    #         "Qualis/CAPES": qualis_capes,
    #         "qualis": qualis_info.get('Qualis'),
    #         "citacoes": citacoes,
    #     }

    #     return resultado, json.dumps(resultado, ensure_ascii=False)

    def extract_info(self):
        soup = BeautifulSoup(self.html_element, 'html.parser')
        qualis_info = self.extract_qualis(soup)

        # **Author List Extraction**
        inicial_autores = []
        for autor in soup.find_all('span', class_='informacao-artigo', data_tipo_ordenacao='autor'):
            inicial_autores.append(autor.text.strip())

        et_al_link = soup.find('a', class_='tooltip autores-et-al-plus')
        if et_al_link:
            et_al_link.click()
            autores_expandidos = soup.find('span', class_='autores-et-al').text.strip()
        else:
            autores_expandidos = ''

        todos_autores = ', '.join(inicial_autores + autores_expandidos.split(','))
        texto_completo = soup.get_text(separator=' ', strip=True)
        autores = soup.find_all('span', class_='informacao-artigo', data_tipo_ordenacao='autor')
        primeiro_autor = autores[0].text if autores else None
        ano_tag = soup.find('span', {'class': 'informacao-artigo', 'data-tipo-ordenacao': 'ano'})
        ano = int(ano_tag.text) if ano_tag else 'Ano não disponível'
        titulo_match = re.search(r'; ([^;]+?)\.', texto_completo)
        titulo = titulo_match.group(1) if titulo_match else None
        periodico_match = re.search(r'(\. )([^.]+?),( v\. \d+, p\. \d+, \d+)', texto_completo)
        periodico = periodico_match.group(2) if periodico_match else None
        detalhes_periodico = periodico_match.group(3) if periodico_match else None
        citacoes = soup.find('span', class_='numero-citacao')
        citacoes = int(citacoes.text) if citacoes else 0
        issn = soup.find('img', class_='ajaxJCR')
        issn = issn['data-issn'] if issn else None
        qualis_capes = "quadriênio 2017-2020"
        resultado = {
            "dados_gerais": texto_completo,
            "primeiro_autor": primeiro_autor,
            "ano": ano,
            "autores": todos_autores,
            "titulo": titulo,
            "periodico": f"{periodico}{detalhes_periodico}",
            "data-issn": issn,
            "impacto": qualis_info.get('JCR'),
            "Qualis/CAPES": qualis_capes,
            "qualis": qualis_info.get('Qualis'),
            "citacoes": citacoes,
        }

        return resultado, json.dumps(resultado, ensure_ascii=False)

    def process_areas(self):
        self.estrutura["Áreas"]={}
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Áreas de atuação' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                ## Extrair cada área de atuação
                ocorrencias = {}
                # Encontrar todos os elementos irmãos seguintes de div_cita_artigos
                next_siblings = data_cell.findChildren("div")
                # print(len(next_siblings))
                # Listas para armazenar os divs encontrados
                divs_indices = []
                divs_ocorrencias = []

                # Iterar sobre os elementos irmãos
                for sibling in next_siblings:
                    # Verificar se o irmão tem a classe "cita-artigos"
                    if 'title-wrapper' in sibling.get('class', []):
                        # Encontramos o marcador para parar, sair do loop
                        break
                    # Verificar as outras classes e adicionar aos arrays correspondentes
                    if 'layout-cell layout-cell-3 text-align-right' in " ".join(sibling.get('class', [])):
                        divs_indices.append(sibling)
                    elif 'layout-cell layout-cell-9' in " ".join(sibling.get('class', [])):
                        divs_ocorrencias.append(sibling)
                
                if len(divs_indices) == len(divs_ocorrencias):
                    # Itera sobre o intervalo do comprimento de uma das listas
                    for i in range(len(divs_indices)):
                        # Usa o texto ou outro identificador único dos elementos como chave e valor
                        chave = divs_indices[i].get_text(strip=True)
                        valor = divs_ocorrencias[i].get_text(strip=True)

                        # Adiciona o par chave-valor ao dicionário
                        ocorrencias[chave] = valor

                self.estrutura["Áreas"] = ocorrencias

    def process_projetos_pesquisa(self):
        self.estrutura["ProjetosPesquisa"] = []
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Projetos de pesquisa' in titulo_h1.get_text(strip=True):
                data_cell = secao.find_next('div', class_='layout-cell layout-cell-12 data-cell')
                next_siblings = data_cell.find_all("div", recursive=False)

                chave = None
                titulo_projeto = None
                descricao = None
                estado = 0  # Estado inicial

                for sibling in next_siblings:
                    if 'title-wrapper' in sibling.get('class', []):
                        break  # Encontrou nova seção, finaliza o loop

                    classes = " ".join(sibling.get('class', []))
                    if estado == 0 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        chave = sibling.get_text(strip=True)
                        estado = 1
                    elif estado == 1 and 'layout-cell layout-cell-9' in classes:
                        titulo_projeto = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        estado = 2  # Ignora a próxima "layout-cell-3"
                    elif estado == 2 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        estado = 3  # Próximo estado para buscar a descrição
                    elif estado == 3 and 'layout-cell layout-cell-9' in classes:
                        descricao = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        # Prepara para adicionar ao dicionário e reinicia o ciclo
                        if chave and titulo_projeto:
                            projeto_pesquisa = {
                                "chave": chave,
                                "titulo_projeto": titulo_projeto,
                                "descricao": descricao
                            }
                            self.estrutura["ProjetosPesquisa"].append(projeto_pesquisa)
                            chave = titulo_projeto = descricao = None  # Reinicia para o próximo ciclo
                            estado = 0  # Volta ao estado inicial

    def process_projetos_extensao(self):
        self.estrutura["ProjetosExtensão"] = []
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Projetos de extensão' in titulo_h1.get_text(strip=True):
                data_cell = secao.find_next('div', class_='layout-cell layout-cell-12 data-cell')
                next_siblings = data_cell.find_all("div", recursive=False)

                chave = None
                titulo_projeto = None
                descricao = None
                estado = 0  # Estado inicial

                for sibling in next_siblings:
                    if 'title-wrapper' in sibling.get('class', []):
                        break  # Encontrou nova seção, finaliza o loop

                    classes = " ".join(sibling.get('class', []))
                    if estado == 0 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        chave = sibling.get_text(strip=True)
                        estado = 1
                    elif estado == 1 and 'layout-cell layout-cell-9' in classes:
                        titulo_projeto = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        estado = 2  # Ignora a próxima "layout-cell-3"
                    elif estado == 2 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        estado = 3  # Próximo estado para buscar a descrição
                    elif estado == 3 and 'layout-cell layout-cell-9' in classes:
                        descricao = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        # Prepara para adicionar ao dicionário e reinicia o ciclo
                        if chave and titulo_projeto:
                            projeto_pesquisa = {
                                "chave": chave,
                                "titulo_projeto": titulo_projeto,
                                "descricao": descricao
                            }
                            self.estrutura["ProjetosExtensão"].append(projeto_pesquisa)
                            chave = titulo_projeto = descricao = None  # Reinicia para o próximo ciclo
                            estado = 0  # Volta ao estado inicial

    def process_projetos_desenvolvimento(self):
        self.estrutura["ProjetosDesenvolvimento"] = []
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Projetos de desenvolvimento' in titulo_h1.get_text(strip=True):
                data_cell = secao.find_next('div', class_='layout-cell layout-cell-12 data-cell')
                next_siblings = data_cell.find_all("div", recursive=False)

                chave = None
                titulo_projeto = None
                descricao = None
                estado = 0  # Estado inicial

                for sibling in next_siblings:
                    if 'title-wrapper' in sibling.get('class', []):
                        break  # Encontrou nova seção, finaliza o loop

                    classes = " ".join(sibling.get('class', []))
                    if estado == 0 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        chave = sibling.get_text(strip=True)
                        estado = 1
                    elif estado == 1 and 'layout-cell layout-cell-9' in classes:
                        titulo_projeto = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        estado = 2  # Ignora a próxima "layout-cell-3"
                    elif estado == 2 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        estado = 3  # Próximo estado para buscar a descrição
                    elif estado == 3 and 'layout-cell layout-cell-9' in classes:
                        descricao = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        # Prepara para adicionar ao dicionário e reinicia o ciclo
                        if chave and titulo_projeto:
                            projeto_pesquisa = {
                                "chave": chave,
                                "titulo_projeto": titulo_projeto,
                                "descricao": descricao
                            }
                            self.estrutura["ProjetosDesenvolvimento"].append(projeto_pesquisa)
                            chave = titulo_projeto = descricao = None  # Reinicia para o próximo ciclo
                            estado = 0  # Volta ao estado inicial

    def process_projetos_outros(self):
        self.estrutura["ProjetosOutros"] = []
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Outros Projetos' in titulo_h1.get_text(strip=True):
                data_cell = secao.find_next('div', class_='layout-cell layout-cell-12 data-cell')
                next_siblings = data_cell.find_all("div", recursive=False)

                chave = None
                titulo_projeto = None
                descricao = None
                estado = 0  # Estado inicial

                for sibling in next_siblings:
                    if 'title-wrapper' in sibling.get('class', []):
                        break  # Encontrou nova seção, finaliza o loop

                    classes = " ".join(sibling.get('class', []))
                    if estado == 0 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        chave = sibling.get_text(strip=True)
                        estado = 1
                    elif estado == 1 and 'layout-cell layout-cell-9' in classes:
                        titulo_projeto = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        estado = 2  # Ignora a próxima "layout-cell-3"
                    elif estado == 2 and 'layout-cell layout-cell-3 text-align-right' in classes:
                        estado = 3  # Próximo estado para buscar a descrição
                    elif estado == 3 and 'layout-cell layout-cell-9' in classes:
                        descricao = sibling.get_text(strip=True).replace('\t','').replace('\n',' ')
                        # Prepara para adicionar ao dicionário e reinicia o ciclo
                        if chave and titulo_projeto:
                            projeto_pesquisa = {
                                "chave": chave,
                                "titulo_projeto": titulo_projeto,
                                "descricao": descricao
                            }
                            self.estrutura["ProjetosOutros"].append(projeto_pesquisa)
                            chave = titulo_projeto = descricao = None  # Reinicia para o próximo ciclo
                            estado = 0  # Volta ao estado inicial
                            
    def add_qualis(self):
        file_name = 'classificações_publicadas_todas_as_areas_avaliacao1672761192111.xls'
        planilha_excel = os.path.join(self.find_repo_root(), '_data', file_name)
        planilha = pd.read_excel(planilha_excel)
        for artigo in self.json_data['Produções']['Artigos completos publicados em periódicos']:
            try:
                issn = artigo['ISSN']
                estrato = planilha.loc[planilha['ISSN'].strip('-') == issn, 'Estrato'].values[0]
                artigo['Qualis'] = estrato
            except:
                artigo['Qualis'] = ''

    def process_producoes(self):
        self.estrutura["Produções"]={}
        dados_artigos = []
        ano=''
        issn=''
        titulo=''
        sem_doi=0
        autores=''
        doi_link = ''
        data_issn = ''
        jcr_impact = ''
        subsec_name = ''
        primeiro_autor=''
        ano_publicacao=''
        fator_impacto = ''
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Produções' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                ## Extrair dados dos artigos em periódicos
                div_artigos = data_cell.find_all("div", id="artigos-completos", recursive=False)
                for div_artigo in div_artigos:
                    sem_doi = 0
                    subsecao = div_artigo.find('b')
                    if subsecao:
                        subsec_name = subsecao.get_text(strip=True)
                        # print(subsec_name)
                    artigos_completos = div_artigo.find_all("div", class_="artigo-completo", recursive=False)
                    for artigo_completo in artigos_completos:
                        dados_qualis = artigo_completo.find('p')
                        try:
                            layout_cell = artigo_completo.find("div", class_="layout-cell layout-cell-11")
                            # print(f'\nlayout_cell: {layout_cell}')

                            # Extrai especificamente o ano da publicação
                            # ano_tag = layout_cell.find('span', {'class': 'informacao-artigo', 'data-tipo-ordenacao': 'ano'})
                            # ano = int(ano_tag.text) if ano_tag else 'Ano não disponível'                        
                            # ano = self.extract_year(layout_cell)
                            # print(f'Primeira extração de ano: {ano}')

                            # # Extrair estrato qualis e impacto JCR
                            # qualis_info = self.extract_qualis(layout_cell)

                            # Extrai o título, periódico, e outras informações diretamente do texto
                            texto_completo = layout_cell.get_text(separator=' ', strip=True)
                            
                            # Assume que título vem após autores e termina antes de periódico ou volume
                            titulo_match = re.search(r'; ([^;]+?)\.', texto_completo)
                            autores = titulo_match.groups() if titulo_match else None

                            # Expressão regular para capturar as partes especificadas
                            pattern = re.compile(
                                r'(?P<primeiro_autor>.*?) ' # até o primeiro espaço antes do ano, não gananciosa
                                r'(?P<ano>\d{4}) ' # Captura o ano como uma sequência de 4 dígitos
                                r'(?P<autores>.+?) ' # após o ano até ponto fim dos autores, não gananciosa
                                r'\. ' # Capta o ponto e o espaço que indica o término da seção de autores
                                r'(?P<titulo_revista>.+?) ' # Capta o título até encontrar "v. ", não gananciosa
                                r'v\. ' # Identifica o início dos detalhes da publicação, marcando o fim do título
                            )

                            # Busca na string pelos padrões
                            match = pattern.search(texto_completo)

                            # Verifica se houve correspondência e extrai os grupos
                            if match:
                                ano_publicacao = match.group('ano')
                                # print(f' Segunda extração de ano: {ano_publicacao}')
                                primeiro_autor = match.group('primeiro_autor')
                                autores = match.group('autores')
                            else:
                                sem_doi +=1

                            span_citado = layout_cell.find("span", class_="citado")
                            if span_citado:
                                # Extrair os atributos do elemento
                                attrs = span_citado.attrs
                                
                                # Converter os atributos em um dicionário
                                attributes_dict = dict(attrs)

                                # Extrair os parâmetros de cvuri
                                if 'cvuri' in attributes_dict:
                                    cvuri_params = urllib.parse.parse_qs(attributes_dict['cvuri'])
                                    attributes_dict['cvuri_params'] = cvuri_params

                            img_original_title = layout_cell.find("img")
                            if img_original_title:
                                attrs = img_original_title.attrs # Extrair os atributos do elemento
                                img_attributes_dict = dict(attrs) # Converter os atributos em um dicionário
                                complete_text = img_attributes_dict.get('original-title')
                                parts = complete_text.split('(') if complete_text else ""
                                issn = parts[1].split(')')[0].strip() if parts else ""
                                jcr_impact = parts[2].split(':')[-1].strip() if parts else ""
                                # print("ISSN:", issn)
                                # print("Impacto JCR:", jcr_impact)

                            # Encontrar o elemento <a> com as classes "icone-producao" e "icone-doi"
                            a_icone_doi = layout_cell.find("a", class_="icone-producao icone-doi")
                            if a_icone_doi:
                                # Extrair o atributo href
                                doi_link = a_icone_doi.get('href')
                                # print("DOI link:", doi_link)
                            # else:
                            #     print("       Link DOI não encontrado.")

                            # Encontrar o elemento <span> com a classe "citado"
                            span_citado = layout_cell.find("span", class_="citado")
                            if span_citado:
                                # Extrair os atributos do elemento
                                attrs = span_citado.attrs
                                
                                # Converter os atributos em um dicionário
                                attributes_dict = dict(attrs)
                                # Extrair os parâmetros de cvuri, se existirem
                                if 'cvuri' in attributes_dict:
                                    cvuri_params = urllib.parse.parse_qs(attributes_dict['cvuri'])
                                    attributes_dict['cvuri_params'] = cvuri_params
                                    if attributes_dict.get('cvuri_params').get('titulo'):
                                        titulo = attributes_dict.get('cvuri_params').get('titulo')[0]
                                    else:
                                        titulo = ''
                                    # print("Título:", titulo)
                                    if attributes_dict.get('cvuri_params').get('nomePeriodico'):
                                        periodico = attributes_dict.get('cvuri_params').get('nomePeriodico')[0]
                                    else:
                                        periodico = ''
                                    # print("Revista:", periodico)
                                    all_texts = layout_cell.get_text(strip=True).lstrip(ano_publicacao)
                                    autores = all_texts.split(titulo)[0].replace('. .','.')
                                    # print("Autores:", autores)
                                    if attributes_dict.get('cvuri_params').get('issn'):
                                        data_issn = attributes_dict.get('cvuri_params').get('issn')[0]
                                    else:
                                        data_issn = ''
                                    # print("data_issn:", data_issn)

                            else:
                                print("       Elemento <span> com a classe 'citado' não encontrado.")
                            segmentos_dict = {
                                "ano": ano_publicacao if ano_publicacao else "",
                                "fator_impacto_jcr": jcr_impact if jcr_impact else "",
                                "ISSN": attributes_dict.get('cvuri_params').get('issn')[0] if attributes_dict.get('cvuri_params').get('issn') else "",
                                "titulo": attributes_dict.get('cvuri_params').get('titulo')[0] if attributes_dict.get('cvuri_params').get('titulo') else "",
                                "revista": attributes_dict.get('cvuri_params').get('nomePeriodico')[0] if attributes_dict.get('cvuri_params').get('nomePeriodico') else "",
                                "autores": autores if autores else "",
                                "data_issn": data_issn if data_issn else "",
                                "DOI": doi_link if doi_link else "",
                                # "qualis": self.get_qualis(data_issn),
                                # "fonte": " ".join(segmentos[2:]).replace("fonteQualis/","").strip(),                            
                            }
                            dados_artigos.append(segmentos_dict)
                        except Exception as e:
                            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
                            print(f'       Um erro impediu de extrair a produção:')
                            print(f'       {e} | {traceback_str}')
                if sem_doi == 1:
                    print(f"       DOI indisponível para {sem_doi:02} artigo extraído")
                elif sem_doi > 1:
                    print(f"       DOI indisponível em {sem_doi:02} artigos extraídos")
                
                self.estrutura["Produções"][subsec_name] = dados_artigos

                ## Extrair demais produções
                divs_cita_artigos = data_cell.find_all("div", class_="cita-artigos", recursive=False)
                for div_cita_artigos in divs_cita_artigos:
                    ocorrencias = {}
                    subsecao = div_cita_artigos.find('b')
                    if subsecao:
                        subsec_name = subsecao.get_text(strip=True)
                        # print(subsec_name)
                    # Encontrar todos os elementos irmãos seguintes de div_cita_artigos
                    next_siblings = div_cita_artigos.find_next_siblings("div")

                    # Listas para armazenar os divs encontrados
                    divs_indices = []
                    divs_ocorrencias = []

                    # Iterar sobre os elementos irmãos
                    for sibling in next_siblings:
                        # Verificar se o irmão tem a classe "cita-artigos"
                        if 'cita-artigos' in sibling.get('class', []):
                            # Encontramos o marcador para parar, sair do loop
                            break
                        # Verificar as outras classes e adicionar aos arrays correspondentes
                        if 'layout-cell layout-cell-1 text-align-right' in " ".join(sibling.get('class', [])):
                            divs_indices.append(sibling)
                        elif 'layout-cell layout-cell-11' in " ".join(sibling.get('class', [])):
                            divs_ocorrencias.append(sibling)
                    
                    if len(divs_indices) == len(divs_ocorrencias):
                        # Itera sobre o intervalo do comprimento de uma das listas
                        for i in range(len(divs_indices)):
                            # Usa o texto ou outro identificador único dos elementos como chave e valor
                            chave = divs_indices[i].get_text(strip=True)
                            valor = divs_ocorrencias[i].get_text(strip=True)

                            # Adiciona o par chave-valor ao dicionário
                            ocorrencias[chave] = valor

                    self.estrutura["Produções"][subsec_name] = ocorrencias

        return self.estrutura["Produções"]

    def process_bancas(self):
        self.estrutura["Bancas"]={}
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Bancas' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                subsecoes = data_cell.find_all('div', class_='inst_back')
                geral={}
                for subsecao in subsecoes:
                    if subsecao:
                        subsec_name = subsecao.get_text(strip=True)
                        # print(subsec_name)
                    ## Extrair cada banca
                    divs_cita_artigos = data_cell.find_all("div", class_="cita-artigos", recursive=False)
                    for div_cita_artigos in divs_cita_artigos:
                        ocorrencias = {}
                        subsubsecao = div_cita_artigos.find('b')
                        if subsubsecao:
                            subsubsecao_name = subsubsecao.get_text(strip=True)
                            # print(f'      Subseção: {subsubsecao_name}')
                        # Encontrar todos os elementos irmãos seguintes de div_cita_artigos
                        next_siblings = div_cita_artigos.find_next_siblings("div")

                        # Listas para armazenar os divs encontrados
                        divs_indices = []
                        divs_ocorrencias = []

                        # Iterar sobre os elementos irmãos
                        for sibling in next_siblings:
                            # Verificar se o irmão tem a classe "cita-artigos"
                            if 'cita-artigos' in sibling.get('class', []):
                                # Encontramos o marcador para parar, sair do loop
                                break
                            # Verificar as outras classes e adicionar aos arrays correspondentes
                            if 'layout-cell layout-cell-1 text-align-right' in " ".join(sibling.get('class', [])):
                                divs_indices.append(sibling)
                            elif 'layout-cell layout-cell-11' in " ".join(sibling.get('class', [])):
                                divs_ocorrencias.append(sibling)
                        
                        if len(divs_indices) == len(divs_ocorrencias):
                            # Itera sobre o intervalo do comprimento de uma das listas
                            for i in range(len(divs_indices)):
                                # Usa o texto ou outro identificador único dos elementos como chave e valor
                                chave = divs_indices[i].get_text(strip=True).replace('\t','').replace('\n',' ')
                                valor = divs_ocorrencias[i].get_text(strip=True).replace('\t','').replace('\n',' ')

                                # Adiciona o par chave-valor ao dicionário
                                ocorrencias[chave] = valor
                        # geral.append(ocorrencias)
                        self.estrutura["Bancas"][subsec_name] = ocorrencias

    def process_patentes(self):
        self.estrutura["Patentes e registros"] = {}
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Patentes e registros' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')

                # Encontrar todas as subseções (Patente e Marca registrada)
                subsecoes = data_cell.find_all('div', class_='inst_back')

                for subsecao in subsecoes:
                    if subsecao:
                        nome_subsecao = subsecao.get_text(strip=True)
                        self.estrutura["Patentes e registros"][nome_subsecao] = {}

                        # Encontrar os itens de cada subseção
                        itens_subsecao = subsecao.find_next_siblings('div', class_='layout-cell layout-cell-11')
                        for i, item in enumerate(itens_subsecao, start=1):
                            nome_ocorrencia = str(i)

                            # Extrair informações do item (ano, título, autores, etc.)
                            texto_item = item.get_text(strip=True).replace('\t', '').replace('\n', ' ')
                            ano_match = re.search(r'\b(\d{4})\b', texto_item)
                            ano = int(ano_match.group(1)) if ano_match else None

                            # Criar um dicionário para armazenar as informações da patente
                            patente_info = {
                                'ano': ano,
                                'texto': texto_item
                            }

                            # Extrair autores
                            autores_elemento = item.find_all('a', class_='tooltip')
                            autores = [autor.get_text(strip=True) for autor in autores_elemento]
                            patente_info['autores'] = autores

                            # Adicionar as informações da patente diretamente ao dicionário da subseção
                            self.estrutura["Patentes e registros"][nome_subsecao][nome_ocorrencia] = patente_info
    
    def process_orientacoes(self):
        self.estrutura["Orientações"] = [] 

        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Orientações' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')

                secao_atual = {"nome": titulo_h1.get_text(strip=True), "subsecoes": []}
                subsecao_atual = None
                tipo_orientacao = None

                for elemento in data_cell.children:
                    if isinstance(elemento, str):  # Ignora elementos de texto como quebras de linha
                        continue

                    if elemento.name == 'div' and 'inst_back' in elemento.get('class', []):
                        # Início de uma nova subseção
                        if subsecao_atual:  # Se já existe uma subseção em andamento, adiciona à seção atual
                            secao_atual["subsecoes"].append(subsecao_atual)
                        subsecao_atual = {"nome": elemento.get_text(strip=True), "orientacoes": []}
                        tipo_orientacao = None  # Reinicia o tipo de orientação para a nova subseção
                    elif elemento.name == 'div' and 'cita-artigos' in elemento.get('class', []):
                        # Tipo de orientação
                        tipo_orientacao = elemento.find('b').get_text(strip=True)
                    elif elemento.name == 'div' and 'layout-cell layout-cell-11' in " ".join(elemento.get('class', [])):
                        # Dados da orientação
                        if subsecao_atual and tipo_orientacao:
                            orientacao = elemento.get_text(strip=True).replace('\t', '').replace('\n', ' ')
                            subsecao_atual["orientacoes"].append({"tipo": tipo_orientacao, "detalhes": orientacao})

                # Adiciona a última subseção à seção atual
                if subsecao_atual:
                    secao_atual["subsecoes"].append(subsecao_atual)

                self.estrutura["Orientações"].append(secao_atual)

    def process_all(self):
        ## IDENTIFICAÇÃO/FORMAÇÃO
        self.process_identification()           # Ok!
        self.process_idiomas()                  # Ok!
        self.process_formacao()                 # Ok!
        ## ATUAÇÃO
        self.process_atuacao_profissional()     # Ok!
        self.process_linhas_pesquisa()          # Ok!
        self.process_areas()                    # Ok!
        ## PRODUÇÕES TODOS OS TIPOS
        self.process_producoes()                # Ok!
        ## PROJETOS
        self.process_projetos_pesquisa()        # Ok!
        self.process_projetos_extensao()        # Ok!
        self.process_projetos_desenvolvimento() # Ok!
        self.process_projetos_outros()          # Ok!      
        ## PRODUTOS TECNOLÓGICOS
        self.process_patentes()                 # FALTA TESTAR
        ## EDUCAÇÃO
        self.process_bancas()                   # Ok!
        self.process_orientacoes()              # Ok!

        # TO-DO-LATER em Produções extraindo vazio
        # "Inovação"
        # "Citações": {},                           
        # "Resumos publicados em anais de congressos (artigos)": {} 

    def to_json(self):
        self.process_all()
        json_string = json.dumps(self.estrutura, ensure_ascii=False, indent=4)
        return json.loads(json_string)

class GetQualis:
    def __init__(self):
        self.dados_planilha = pd.read_excel(os.path.join(LattesScraper.find_repo_root(),'_data','in_xls','classificações_publicadas_todas_as_areas_avaliacao1672761192111.xlsx'))

    def buscar_qualis_e_atualizar_arquivo(self, lista_dados_autor, nome_arquivo):
        """
        Busca o Qualis de cada artigo completo publicado em periódicos e atualiza o arquivo original com os dados encontrados.

        Args:
            lista_dados_autor (list): Lista de dicionários com os dados dos autores.
            nome_arquivo (str): Nome do arquivo JSON a ser atualizado.
        """
        try:
            # with open(nome_arquivo, 'r+') as arquivo:
            #     # Carregue os dados do arquivo
            #     dados = json.load(arquivo)
            with codecs.open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
                 # Carregue os dados do arquivo
                 dados = json.load(arquivo)

            # Percorra os dados de cada autor
            for m,dados_autor in enumerate(lista_dados_autor):
                for categoria, artigos in dados_autor['Produções'].items():
                    if categoria == 'Artigos completos publicados em periódicos':
                        for n,artigo in enumerate(artigos):
                            print(f'{n+1:3}/{len(artigos):3} artigos do autor {m+1:3}/{len(lista_dados_autor):3}')
                            clear_output(wait=True)
                            # Recupere o ISSN do artigo
                            issn_artigo = artigo['ISSN'].replace('-','')

                            # Busque o Qualis do artigo
                            qualis = self.encontrar_qualis_por_issn(issn_artigo)

                            # Atualize o campo 'Qualis' do artigo
                            if qualis:
                                artigo['Qualis'] = qualis
                            else:
                                artigo['Qualis'] = 'NA'

            # Reescreva os dados formatados no arquivo
            # json.dump(dados, arquivo, indent=4)
            with codecs.open(nome_arquivo, 'w', encoding='utf-8') as arquivo:
                arquivo.seek(0)
                json.dump(dados, arquivo, indent=4)
        
        except Exception as e:
            print(f"Erro ao atualizar o arquivo: {e}")
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            print(f"{traceback_str}")

    def encontrar_qualis_por_issn(self, issn):
        qualis = self.dados_planilha[self.dados_planilha['ISSN'].str.replace('-','') == issn]['Estrato'].tolist()
        if qualis:
            return qualis[0]
        else:
            return None

class DiscentCollaborationCounter:
    def __init__(self, dict_list):
        self.data_list = dict_list
        self.verbose = False

    def get_coauthors(self, string):
        """
        Extrai a lista de coautores de uma string que contém todos os dados de cada artigo

        Argumentos:
            string (str): A string que contém todos dados de cada artigo no formato:
            * prim_autor|ano|todos_autores|nome_revista|titulo_artigo|vol|ano|citações.

        Retorna:
            lista_coautores:
            * tupla com a lista de autores individualizados

            padroes: Um dicionário contendo as posições de cada padrão encontrado para separar dados:
            * 'quatrodigitos': Posição dos quatro dígitos (ou None se não encontrado)
            * 'pontos_duplos': Posição dos dois pontos (ou None se não encontrado)
            * 'ponto_esp_pnt': Posição do espaço seguido de ponto seguido de espaço (ou None se não encontrado)
            * 'ponto_inicial': Posições onde ponto como símbolo de abreviatura de nome (ou None se não encontrado)
            * 'ponto_virgula': Posições onde ocorrem o símbolo do ponto e vírgula (ou None se não encontrado)
        """

        padroes = {
            'quatrodigitos': None,
            'pontos_duplos': None,
            'ponto_esp_pnt': None,
            'ponto_inicial': None,
            'ponto_virgula': None,
        }

        # Expressões regulares para cada padrão
        regex_quatrodigitos = r"\d{4}?.*$"
        regex_pontos_duplos = r"\.\."
        regex_ponto_esp_pnt = r"\s+\.\s+"
        regex_ponto_virgula = r"\;"
        regex_pnto_iniciais = r"\."
        regex_ponto_fim_esp = r"\.\s+(\w+)"
        # regex_restante_strg = r"\.\s+(\w+)+.*$"

        # Encontrar as primeiras ocorrências de cada padrão
        string = str(string)
        match_quatrodigitos = re.search(regex_quatrodigitos, string)
        match_pontos_duplos = re.search(regex_pontos_duplos, string)
        match_ponto_esp_pnt = re.search(regex_ponto_esp_pnt, string)
        # match_ponto_fim_esp = re.search(regex_pntofinal_esp, string)
        # Encontrar todas ocorrências de padrões de abreviação/separação 
        match_ponto_virgula = re.finditer(regex_ponto_virgula, string)
        match_pnto_iniciais = re.finditer(regex_pnto_iniciais, string)
        match_ponto_fim_esp = re.finditer(regex_ponto_fim_esp, string)

        # Atualizar o dicionário dos padrões com as posições encontradas
        if match_quatrodigitos:
            if isinstance(match_quatrodigitos, str):
                padroes['quatrodigitos'] = match_quatrodigitos.end()
            elif isinstance(match_quatrodigitos, list):
                padroes['quatrodigitos'] = match_quatrodigitos[0].end()
        else:
            padroes['quatrodigitos'] = 0
        if match_pontos_duplos == None:
            padroes['pontos_duplos'] = len(string)
        else:
            padroes['pontos_duplos'] = match_pontos_duplos.start()
        if match_ponto_esp_pnt == None:
            padroes['ponto_esp_pnt'] = len(string)
        else:
            padroes['ponto_esp_pnt'] = match_ponto_esp_pnt.start()
        if match_pnto_iniciais:
            padroes['ponto_inicial'] = [x.end() for x in match_pnto_iniciais]
        else:
            padroes['ponto_inicial'] = []    
        if match_ponto_virgula:
            padroes['ponto_virgula'] = [x.end() for x in match_ponto_virgula]
        else:
            padroes['ponto_virgula'] = []
        if match_ponto_fim_esp:
            padroes['ponto_final'] = [x.end() for x in match_ponto_fim_esp]
        else:
            padroes['ponto_final'] = []            

        beg = padroes['quatrodigitos']
        end = min(padroes['pontos_duplos'], padroes['ponto_esp_pnt'])
        string_coautores = string[beg:end]
        lista_coautores = [x.strip() for x in string_coautores.split(';')]
            
        if self.verbose:
            print(f"quatrodigitos: {padroes['quatrodigitos']:003} | {type(padroes['quatrodigitos'])}")
            print(f"pontos_duplos: {padroes['pontos_duplos']:003} | {type(padroes['pontos_duplos'])}")
            print(f"ponto_esp_pnt: {padroes['ponto_esp_pnt']:003} | {type(padroes['ponto_esp_pnt'])}")
            print(f"ponto_inicial: {padroes['ponto_inicial']} | {type(padroes['ponto_inicial'])}")
            print(f"ponto_virgula: {padroes['ponto_virgula']} | {type(padroes['ponto_virgula'])}")
            print(f"pnto_espfinal: {padroes['ponto_final']} | {type(padroes['ponto_final'])}")
            print(f"       inicio: {beg:003} | final: {end:003}")

        return lista_coautores, padroes

    def get_articles_coauthorings(self, dict_list, ano_inicio=2017, ano_final=2024, limite_similaridade_sobrenome=0.88, limite_similaridade_iniciais=0.8):
        """
        Extrair cada artigo de cada dicionário de currículo

        Argumentos:
            dict_list (dict): Lista de dicionários  contendo todos os dados extraídos dos currículos
        
        Retorna:
            colaboracoes (list): Uma lista de dicionários com chave nome do dono do currículo e valores lista de todos autores
        """
        lista_normalizada_discentes = self.normalize_discents()
        colaboracoes = []
        percentuais = {}
        print(f'{len(dict_list)} currículos a analisar colaborações')
        for dic in dict_list:
            autor = dic.get('Identificação',{}).get('Nome',{})
            artigos = dic.get('Produções', {}).get('Artigos completos publicados em periódicos', {})
            lista_coautores_artigo = []
            total_artigos_periodo = 0            
            colaboracoes_com_discentes = 0
            if artigos:    
                try:
                    for i in artigos:
                        colaborou = False
                        ano = i.get('ano',{})
                        qualis = i.get('Qualis',{})
                        doi = i.get('DOI',{})
                        string_autores = i.get('autores')
                        coautores, padroes = self.get_coauthors(string_autores)
                        regex_quatrodigitos = r"\d{4}?.*$"
                        match_quatrodigitos = re.search(regex_quatrodigitos, coautores[0])
                        if match_quatrodigitos:
                            nome_prim_autor = re.sub(regex_quatrodigitos, "", coautores[0])
                            coautores[0] = nome_prim_autor
                        regex_restantedados = r"\.\s+(\w+)+.*$"
                        match_restantedados = re.search(regex_quatrodigitos, coautores[-1])
                        if match_restantedados:
                            nome_ultimo_autor = re.sub(regex_restantedados, "", coautores[-1])
                            coautores[-1] = nome_ultimo_autor
                        coautores = [self.iniciais_nome(x) for x in coautores]
                        lista_coautores_artigo.append(coautores)
                        colaboracoes.append({autor: coautores})
                        if ano and int(ano) >= ano_inicio:
                            total_artigos_periodo +=1
                            for nome_coautor in coautores:
                                nome_discente=''
                                for nome_discente in lista_normalizada_discentes:
                                    similaridade_sobrenome, similaridade_iniciais = self.similar_index(nome_discente, nome_coautor)
                                    if similaridade_sobrenome > limite_similaridade_sobrenome and similaridade_iniciais > limite_similaridade_iniciais:
                                        colaborou = True
                                        # print(f'{autor:40} {nome_discente:20} {nome_coautor:25} | {similaridade_sobrenome:.6f} | {similaridade_iniciais:.6f}')
                            if colaborou == True:
                                colaboracoes_com_discentes += 1
                                print(f"ANO {ano} DOI {doi:51} com colaboração com discente '{nome_discente}'")
                            else:
                                print(f'ANO {ano} DOI {doi:51} nome de discente em coautorias não encontrado')
                            if len(artigos) == 0:
                                porcentagem_colab_discentes = 0
                            else:
                                porcentagem_colab_discentes = np.round((colaboracoes_com_discentes / total_artigos_periodo) * 100, 2)
                                percentuais[autor] = porcentagem_colab_discentes
                        if ano == '':
                            print(f'Ano não extraído para {i}')
                except Exception as e:
                    print('Erro ao contar colaborações:')
                    print(e)
            else:
                porcentagem_colab_discentes = 0
                percentuais[autor] = porcentagem_colab_discentes
            print('-'*120)
            print(f'Nome de discente detectado em {colaboracoes_com_discentes} dos {total_artigos_periodo} artigos de {autor}, perfazendo {porcentagem_colab_discentes}% de colaborações com discente')
            print('='*120)

        return colaboracoes, percentuais

    def discent_direct_counter(self, coauthors_list, discent_list):
        count=0
        for name in coauthors_list:
            if name in discent_list:
                count+=1
        return count

    def normalize(self, texto):
        """
        Remove acentuação gráfica e converte para minúsculas.

        Argumentos:
            texto (str): String a ser normalizada.

        Retorna:
            str: String normalizada.
        """

        # Remover acentuação gráfica
        texto_sem_acento = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')

        # Converter para minúsculas
        texto_minusculo = texto_sem_acento.lower()

        return texto_minusculo

    def similar_index(self, name_discent, name_coauthor):
        """
        Calcula a similaridade de Levenshtein entre duas strings normalizadas.

        Argumentos:
            name_discent (str): Nome do discente.
            name_coauthor (str): Nome do coautor.

        Retorna:
            float: Índice de similaridade de Levenshtein (entre 0.0 e 1.0).
        """

        # Normalizar as strings
        name_discent_norm = self.normalize(name_discent).lower()
        name_coauthor_norm = self.normalize(name_coauthor).lower()

        # Cálcular da distância de Levenshtein
        # distancia_levenshtein = distance(name_discent_norm, name_coauthor_norm)

        # Cálcular índice de similaridade
        # max_length = max(len(name_discent_norm), len(name_coauthor_norm))
        # similarity_index = 1 - (distancia_levenshtein / max_length)
        try:
            surname_discent      = name_discent_norm.split(',')[0].strip()
            surname_coauthor     = name_coauthor_norm.split(',')[0].strip()
            surname_levenshtein  = distance(surname_discent, surname_coauthor)
            surname_similarity   = 1 - (surname_levenshtein / (len(surname_discent) + len(surname_coauthor)))

            initials_discent     = name_discent_norm.split(',')[1].strip()
            initials_coauthor    = name_coauthor_norm.split(',')[1].strip()        
            initials_levenshtein = distance(initials_discent, initials_coauthor)
            initials_similarity  = 1 - (initials_levenshtein / (len(initials_discent) + len(initials_coauthor)))
            
            return surname_similarity, initials_similarity
        except:
            return 0,0

    def normalize_discents(self, fonte_planilha = 'ppgcs_estudantes_2021-2024.xlsx'):
        """
        Calcula o percentual de colaboração entre um pesquisador e seus discentes.

        Args:
            pesquisador: O nome do pesquisador.
            lista_normalizada_discentes: A lista de nomes de discentes normalizados.
            colabs: A lista de colaborações (dicionários com informações sobre artigos).
            discent_collab_counter: Objeto que contém métodos para calcular a similaridade entre nomes.
            limite_similaridade_sobrenome: Limite mínimo para a similaridade do sobrenome ser considerada (padrão: 0.87).
            limite_similaridade_iniciais: Limite mínimo para a similaridade das iniciais ser considerada (padrão: 0.8).

        Returns:
            Dicionário com as seguintes chaves:
            - colaboracao_com_discentes: Número de colaborações com discentes.
            - total_colaboracoes: Número total de colaborações do pesquisador.
            - porcentagem_colab_discentes: Percentual de colaboração com discentes.
        """

        dados_discentes = pd.read_excel(os.path.join(LattesScraper.find_repo_root(),'_data','in_xls',fonte_planilha), header=1)
        lista_discentes = list(dados_discentes['Discente'].unique())
        print(f'{len(lista_discentes)} discentes no período 2021-2024 informados pelos programa')
        print('='*120)

        lista_normalizada_discentes=[]
        for i in lista_discentes:
            lista_normalizada_discentes.append(self.iniciais_nome(i))

        return lista_normalizada_discentes

    ## PADRONIZAÇÃO DE NOMES DE AUTOR E ANÁLISE DE SIMILARIDADES
    def padronizar_nome(self, linha_texto):
        '''Procura sobrenomes e abreviaturas e monta nome completo
        Recebe: String com todos os sobrenomes e nomes, abreviados ou não
        Retorna: Nome completo no formato padronizado em SOBRENOME AGNOME, Prenomes
        Autor: Marcos Aires (Mar.2022)
        '''
        # print('               Analisando:',linha_texto)
        string = ''.join(ch for ch in unicodedata.normalize('NFKD', linha_texto) if not unicodedata.combining(ch))
        string = string.replace('(Org)','').replace('(Org.)','').replace('(Org).','').replace('.','').replace('\'','')
        string = string.replace(',,,',',').replace(',,',',').replace('Ãº','ú').replace('Ã¡','á').replace('Ã³','ó').replace('Ã´','ô').replace('a¡U','áu')
        string = re.sub(r'[0-9]+', '', string)
            
        # Expressões regulares para encontrar padrões de divisão de nomes de autores
        sobrenome_inicio   = re.compile(r'^[A-ZÀ-ú-a-z]+,')                  # Sequência de letras maiúsculas no início da string
        sobrenome_composto = re.compile(r'^[A-ZÀ-ú-a-z]+[ ][A-ZÀ-ú-a-z]+,')  # Duas sequências de letras no início da string, separadas por espaço, seguidas por vírgula
        letra_abrevponto   = re.compile(r'^[A-Z][.]')                        # Uma letra maiúscula no início da string, seguida por ponto
        letra_abrevespaco  = re.compile(r'^[A-Z][ ]')                        # Uma letra maiúscula no início da string, seguida por espaço
        letras_dobradas    = re.compile(r'[A-Z]{2}')                         # Duas letras maiúsculas juntas no início da string, seguida por espaço
        letras_dobradasini = re.compile(r'[A-Z]{2}[ ]')                      # Duas letras maiúsculas juntas no início da string, seguida por espaço
        letras_dobradasfim = re.compile(r'[ ][A-Z]{2}')                      # Duas letras maiúsculas juntas no final da string, precedida por espaço
        letras_duasconsnts = re.compile(r'[B-DF-HJ-NP-TV-XZ]{2}')            # Duas Letras maiúsculas e consoantes juntas
        letras_tresconsnts = re.compile(r'[B-DF-HJ-NP-TV-XZ]{3}')            # Três Letras maiúsculas e consoantes juntas
        
        # Agnomes e preprosições a tratar, agnomes vão maiúsculas para sobrenome e preposições vão para minúsculas nos nomes
        nomes=[]
        agnomes       = ['NETO','JUNIOR','FILHO','SEGUNDO','TERCEIRO']
        preposicoes   = ['da','de','do','das','dos']
        nome_completo = ''
        
        # Ajustar lista de termos, identificar sobrenomes compostos e ajustar sobrenome com ou sem presença de vírgula
        div_sobrenome      = sobrenome_inicio.findall(string)
        div_sbrcomposto    = sobrenome_composto.findall(string)
        
        # print('-'*100)
        # print('                 Recebido:',string)
        
        # Caso haja vírgulas na string, tratar sobrenomes e sobrenomes compostos
        if div_sobrenome != [] or div_sbrcomposto != []:
            # print('CASO_01: Há víruglas na string')
            div = string.split(', ')
            sobrenome     = div[0].strip().upper()
            try:
                div_espaco    = div[1].split(' ')
            except:
                div_espaco    = ['']
            primeiro      = div_espaco[0].strip('.')
            
            # print('     Dividir por vírgulas:',div)
            # print('      Primeira DivVirgula:',sobrenome)
            # print('Segunda DivVrg/DivEspaços:',div_espaco)
            # print('      Primeira DivEspaços:',primeiro)
                
            # Caso primeiro nome sejam somente duas letras maiúsculas juntas, trata-se de duas iniciais
            if len(primeiro)==2 or letras_tresconsnts.findall(primeiro):
                # print('CASO_01.a: Há duas letras ou três letras consoantes juntas, são iniciais')
                primeiro_nome=primeiro[0].strip()
                # print('          C01.a1_PrimNome:',primeiro_nome)
                nomes.append(primeiro[1].strip().upper())
                try:
                    nomes.append(primeiro[2].strip().upper())
                except:
                    pass
            else:
                # print('CASO_01.b: Primeiro nome maior que 2 caracteres')
                primeiro_nome = div_espaco[0].strip().title()
                # print('          C01.a2_PrimNome:',primeiro_nome)
            
            # Montagem da lista de nomes do meio
            for nome in div_espaco:
                # print('CASO_01.c: Para cada nome da divisão por espaços após divisão por vírgula')
                if nome not in nomes and nome.lower()!=primeiro_nome.lower() and nome.lower() not in primeiro_nome.lower() and nome!=sobrenome:   
                    # print('CASO_01.c1: Se o nome não está nem como primeiro nome, nem sobrenomes')
                    # print(nome, len(nome))
                    
                    # Avaliar se é abreviatura seguida de ponto e remover o ponto
                    if len(nome)<=2 and nome.lower() not in preposicoes:
                        # print('    C01.c1.1_Nome<=02:',nome)
                        for inicial in nome:
                            # print(inicial)
                            if inicial not in nomes and inicial not in primeiro_nome:
                                nomes.append(inicial.replace('.','').strip().title())
                    elif len(nome)==3 and nome.lower() not in preposicoes:
                            # print('    C01.c1.2_Nome==03:',nome)
                            for inicial in nome:
                                if inicial not in nomes and inicial not in primeiro_nome:
                                    nomes.append(inicial.replace('.','').strip().title())
                    else:
                        if nome not in nomes and nome!=primeiro_nome and nome!=sobrenome and nome!='':
                            if nome.lower() in preposicoes:
                                nomes.append(nome.replace('.','').strip().lower())
                            else:
                                nomes.append(nome.replace('.','').strip().title())
                            # print(nome,'|',primeiro_nome)
                            
            #caso haja sobrenome composto que não esteja nos agnomes considerar somente primeiro como sobrenome
            if div_sbrcomposto !=[] and sobrenome.split(' ')[1] not in agnomes and sobrenome.split(' ')[0].lower() not in preposicoes:
                # print('CASO_01.d: Sobrenome composto sem agnomes')
                # print(div_sbrcomposto)
                # print('Sobrenome composto:',sobrenome)
                nomes.append(sobrenome.split(' ')[1].title())
                sobrenome = sobrenome.split(' ')[0].upper()
                # print('Sobrenome:',sobrenome)
                for i in nomes:
                    if i.lower() in sobrenome.lower():
                        nomes.remove(i)
                # print('    Nomes:',nomes)
            
            #caso haja preposição como agnome desconsiderar e passar para final dos nomes
            if div_sbrcomposto !=[] and sobrenome.split(' ')[0].lower() in preposicoes:
                # print('CASO_01.e: Preposição no Sobrenome passar para o final dos nomes')
                # print('   div_sbrcomposto:', div_sbrcomposto)
                # print('Sobrenome composto:',div_sbrcomposto)
                nomes.append(div_sbrcomposto[0].split(' ')[0].lower())
                # print('    Nomes:',nomes)
                sobrenome = div_sbrcomposto[0].split(' ')[1].upper().strip(',')
                # print('Sobrenome:',sobrenome)
                for i in nomes:
                    # print('CASO_01.e1: Para cada nome avaliar se o sobrenome está na lista')
                    if i.lower() in sobrenome.lower():
                        nomes.remove(i)
                # print('  Nomes:',nomes)
            # print('Ao final do Caso 01')
            # print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
            # print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
            # print('           Lista de nomes:',nomes, len(nomes),'nomes')
            
        # Caso não haja vírgulas na string considera sobrenome o último nome da string dividida com espaço vazio
        else:
            # print('CASO_02: Não há víruglas na string')
            try:
                div = string.split(' ')
                # print('      Divisões por espaço:',div)
                
                if div[-1] in agnomes: # nome final é um agnome
                    sobrenome     = div[-2].upper().strip()+' '+div[-1].upper().strip()
                    for i in div[1:-2]:
                        if i not in sobrenome and i not in preposicoes:
                            nomes.append(i.title().strip())
                        if i in preposicoes:
                            nomes.append(i.lower().strip())
                else:
                    if len(div[-1]) > 2:
                        sobrenome     = div[-1].upper().strip()
                        primeiro_nome = div[1].title().strip()
                        for i in div[1:-1]:
                            if i != sobrenome and i not in preposicoes:
                                nomes.append(i.title().strip())
                            if i in preposicoes:
                                nomes.append(i.lower().strip())
                    else:
                        sobrenome     = div[-2].upper().strip()
                        for i in div[-1]:
                            nomes.append(i.title())
                        primeiro_nome = nomes[0].title().strip()
                        for i in div[1:-1]:
                            if i != sobrenome and i not in preposicoes:
                                nomes.append(i.title().strip())
                            if i in preposicoes:
                                nomes.append(i.lower().strip())
            except:
                sobrenome = div[-1].upper().strip()
                for i in div[1:-1]:
                        if i != sobrenome and i not in preposicoes:
                            nomes.append(i.title().strip())
                        if i in preposicoes:
                            nomes.append(i.lower().strip())
            if sobrenome.lower() != div[0].lower().strip():
                primeiro_nome=div[0].title().strip()
            else:
                primeiro_nome=''
            
            # print('Ao final do Caso 02')
            # print('    Sobrenome sem vírgula:',sobrenome, len(sobrenome),'letras')
            # print('Primeiro nome sem vírgula:',primeiro_nome, len(primeiro_nome),'letras')
            # print('Nomes do meio sem vírgula:',nomes, len(nomes),'nomes')
        
        # Encontrar e tratar como abreviaturas termos com apenas uma ou duas letras iniciais juntas, com ou sem ponto
        for j in nomes:
            # print('CASO_03: Avaliar cada nome armazenado na variável nomes')
            # Procura padrões com expressões regulares na string
            div_sobrenome      = sobrenome_inicio.findall(j)
            div_sbrcomposto    = sobrenome_composto.findall(j)
            div_abrevponto     = letra_abrevponto.findall(j)
            div_abrevespaco    = letra_abrevespaco.findall(j)
            div_ltrdobradasini = letras_dobradasini.findall(j)
            div_ltrdobradasfim = letras_dobradasfim.findall(j)
            div_ltrdobradas    = letras_dobradas.findall(j)
            tamanho=len(j)
            # print('\n', div_ltrdobradasini, div_ltrdobradasfim, tamanho, 'em:',j,len(j))
            
            #caso houver abreviatura com uma letra em maiúscula nos nomes
            if div_abrevponto !=[] or tamanho==1:
                # print('CASO_03.1: Há abreviaturas uma letra maiúscula nos nomes')
                nome = j.replace('.','').strip()
                if nome not in nomes and nome != sobrenome and nome != primeiro_nome:
                    # print('CASO_03.1a: Há abreviaturas uma letra maiúscula nos nomes')
                    nomes.append(nome.upper())
            
            #caso houver duas inicias juntas em maiúsculas
            elif div_ltrdobradasini !=[] or div_ltrdobradasfim !=[] or div_ltrdobradas !=[] :
                # print('CASO_03.2: Há abreviaturas uma letra maiúscula nos nomes')
                for letra in j:
                    # print('CASO_03.2a: Avaliar cada inicial do nome')
                    if letra not in nomes and letra != sobrenome and letra != primeiro_nome:
                        # print('CASO_03.2a.1: Se não estiver adicionar inicial aos nomes')
                        nomes.append(letra.upper())
            
            #caso haja agnomes ao sobrenome
            elif sobrenome in agnomes:
                # print('CASO_03.3: Há agnomes nos sobrenomes')
                sobrenome = nomes[-1].upper()+' '+sobrenome
                # print(sobrenome.split(' '))
                # print('Sobrenome composto:',sobrenome)
                for i in nomes:
                    if i.lower() in sobrenome.lower():
                        nomes.remove(i)
                # print('Nomes do meio:',nomes)
                
            else:
                # print('CASO_03.4: Não há agnomes nos sobrenomes')
                if j not in nomes and j not in sobrenome and j != primeiro_nome:
                    if len(nomes) == 1:
                        # adicionar sobrenome com todas letas em maiúsculas
                        nomes.append(j.upper())
                    elif 1 < len(nomes) <= 3:
                        # adicionar preposições em minúsculas
                        nomes.append(j.lower())
                    else:
                        # adicionar nomes com somente a primeira em maúscula
                        nomes.append(j.title())
            
            # print('Ao final do Caso 03')
            # print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
            # print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
            # print('Nomes do meio com vírgula:',nomes, len(nomes),'nomes')
            
        nomes_meio=' '.join([str for str in nomes]).strip()
        # print('        Qte nomes do meio:',nomes,len(nomes))
        
        if primeiro_nome.lower() == sobrenome.lower():
            # print('CASO_04: Primeiro nome é igual ao sobrenome')
            try:
                primeiro_nome=nomes_meio.split(' ')[0]
            except:
                pass
            try:
                nomes_meio.remove(sobrenome)
            except:
                pass
        
            # print('Ao final do caso 04')
            # print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
            # print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
            # print('Nomes do meio com vírgula:',nomes, len(nomes),'nomes')
        
        # Caso sobrenome seja só de 1 letra passá-lo para nomes e considerar o próximo nome como sobrenome
        for i in range(len(div)):
            if len(sobrenome)==1 or sobrenome.lower() in preposicoes:
                # print('CASO_05: Mudar sobrenomes até o adequado')
                div    = string.split(', ')
                # print('Divisão por vírgulas:',div)
                avaliar0       = div[0].split(' ')[0].strip()
                if 1< len(avaliar0) < 3:
                    # print('CASO_05.1: 1 < Sobrenome < 3 fica em minúsculas')
                    sbrn0          = avaliar0.lower()
                else:
                    # print('CASO_05.2: Sobrenome de tamanho 1 ou maior que 3 fica em maiúsculas')
                    sbrn0          = avaliar0.title()
                # print('sbrn0:',sbrn0, len(sbrn0))
                
                try:
                    avaliar1=div[0].split(' ')[1].strip()
                    # print('avaliar0',avaliar0)
                    # print('avaliar1',avaliar1)
                    if 1 < len(avaliar1) <=3:
                        sbrn1     = avaliar1.lower()
                    else:
                        sbrn1     = avaliar1.title()
                    # print('sbrn1:',sbrn1, len(sbrn1))

                except:
                    pass

                if div != []:
                    # print('CASO_05.3: Caso haja divisão por vírgulas na string')
                    try:
                        div_espaco     = div[1].split(' ')
                    except:
                        div_espaco     = div[0].split(' ')
                    sobrenome      = div_espaco[0].strip().upper()
                    try:
                        primeiro_nome  = div_espaco[1].title().strip()
                    except:
                        primeiro_nome  = div_espaco[0].title().strip()
                    if len(sbrn0) == 1:
                        # print('CASO_05.3a: Avalia primeiro sobrenome de tamanho 1')
                        # print('Vai pros nomes:',str(sbrn0).title())
                        nomes_meio = nomes_meio+str(' '+sbrn0.title())
                        # print('   NomesMeio:',nomes_meio)

                    elif 1 < len(sbrn0) <= 3:
                        # print('CASO_05.3b: Avalia primeiro sobrenome 1< tamanho <=3')
                        # print('Vão pros nomes sbrn0:',sbrn0, 'e sbrn1:',sbrn1)

                        div_tresconsoantes = letras_tresconsnts.findall(sobrenome)
                        if div_tresconsoantes != []:
                            # print('CASO_05.4: Três consoantes como sobrenome')
                            for letra in sobrenome:
                                nomes.append(letra)

                            if len(sobrenome) >2:
                                sobrenome=nomes[0]
                            else:
                                sobrenome=nomes[1]
                            nomes.remove(sobrenome)
                            primeiro_nome=nomes[0]
                            nomes_meio=' '.join([str for str in nomes[1:]]).strip()
                            nome_completo=sobrenome.upper()+', '+nomes_meio                
                        
                        try:                       
                            # print(' 05.3b    Lista de Nomes:',nomes_meio)
                            nomes_meio=nomes_meio.replace(sbrn0,'')
                            # print(' 05.3b ReplaceSobrenome0:',nomes_meio)
                            nomes_meio=nomes_meio.replace(sbrn1,'')
                            # print(' 05.3b ReplaceSobrenome1:',nomes_meio)
                        except Exception as e:
                            # print('   Erro ReplaceSobrenome:',e)
                            pass
                        try:
                            nomes_meio.replace(primeiro_nome.title(),'')
                            nomes_meio.replace(primeiro_nome.lower(),'')
                            nomes_meio.replace(primeiro_nome,'')
                            # print(' 05.3b Replace PrimNome:',nomes_meio)
                        except Exception as e:
                            print('Erro no try PrimeiroNome:',e)
                            pass
                        nomes_meio = nomes_meio.replace(sobrenome,'')
                        try:
                            for n,i in enumerate(avaliar1):
                                nomes.append(i.upper())
                                sbrn1     = avaliar1[0]
                            else:
                                sbrn1     = avaliar1.title()
                            # print('sbrn1:',sbrn1, len(sbrn1))
                            nomes_meio = nomes_meio+str(' '+sbrn0)+str(' '+sbrn1)
                        except:
                            nomes_meio = nomes_meio+str(' '+sbrn0)
                        nomes      = nomes_meio.strip().strip(',').split(' ')
                        # print(' 05.3b NomesMeio:',nomes_meio)
                        # print(' 05.3b     Nomes:',nome)

                    else:
                        # print('CASO_05.3c: Avalia primeiro sobrenome >3')
                        nomes_meio = nomes_meio+str(' '+div[0].strip().title())
                        nomes      = nomes_meio.strip().split(' ')
                        # print(' 05.3c NomesMeio:',nomes_meio)
                        # print(' 05.3c     Nomes:',nomes)

                    nomes_meio=nomes_meio.replace(sobrenome,'').replace(',','').strip()
                    nomes_meio=nomes_meio.replace(primeiro_nome,'').strip()

                # print('Ao final do caso 05')
                # print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
                # print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
                # print('Nomes do meio com vírgula:',nomes, len(nomes),'nomes')
        
        if sobrenome != '' and primeiro_nome !='':
            nome_completo=sobrenome.upper().replace(',','')+', '+primeiro_nome.replace(',','')+' '+nomes_meio.replace(sobrenome,'').replace(',','')
        elif sobrenome != '':
            nome_completo=sobrenome.upper().replace(',','')+', '+nomes_meio.replace(sobrenome,'').replace(',','')
        else:
            nome_completo=sobrenome.upper()
        
    #     print('Após ajustes finais')
    #     print('     Sobrenome:',sobrenome)
    #     print(' Primeiro Nome:',primeiro_nome)
    #     print('         Nomes:',nomes)
    #     print('     NomesMeio:',nomes_meio)        
            
    #     print('                Resultado:',nome_completo)
        
        return nome_completo.strip()

    def iniciais_nome(self, linha_texto):
        '''Função para retornar sobrenome+iniciais dos nomes, na forma: SOBRENOME, X Y Z
        Recebe: String com nome
        Retorna: Nome e sua versão padronizada em sobrenome+agnomes em maiúsculas, seguida de vírgula e iniciais dos nomes 
        Autor: Marcos Aires (Mar.2022)
        '''
        sobrenome_iniciais = ''
        # print('               Analisando:',linha_texto)
        string = ''.join(ch for ch in unicodedata.normalize('NFKD', linha_texto) if not unicodedata.combining(ch))
        string = string.replace('(Org)','').replace('(Org.)','').replace('(Org).','').replace('.','').replace('Ãº','ú').replace('Ã¡','á').replace('Ã³','ó').replace('Ã´','ô').replace('a¡U','áu')
            
        # Expressões regulares para encontrar padrões de divisão de nomes de autores
        sobrenome_inicio   = re.compile(r'^[A-ZÀ-ú-a-z]+,')                 # Sequência de letras maiúsculas no início da string
        sobrenome_composto = re.compile(r'^[A-ZÀ-ú-a-z]+[ ][A-ZÀ-ú-a-z]+,') # Duas sequências de letras no início da string, separadas por espaço, seguidas por vírgula
        letra_abrevespaco  = re.compile(r'^[A-Z][ ]')                       # Uma letra maiúscula no início da string, seguida por espaço
        letra_abrevponto   = re.compile(r'[A-Z][.]')                        # Uma letra maiúscula em qualquer lugar da string, seguida por ponto
        letra_abrevisolada = re.compile(r'[A-Z]{1}')                        # Uma letra maiúscula isolada em qualquer lugar da string
        letras_dobradas    = re.compile(r'[A-Z]{2}')                        # Duas letras maiúsculas juntas no início da string, seguida por espaço
        letras_dobradasini = re.compile(r'[A-Z]{2}[ ]')                     # Duas letras maiúsculas juntas no início da string, seguida por espaço
        letras_dobradasfim = re.compile(r'[ ][A-Z]{2}')                     # Duas letras maiúsculas juntas no final da string, precedida por espaço
            
        nomes         = []
        agnomes       = ['NETO','JUNIOR','FILHO','SEGUNDO','TERCEIRO']
        preposicoes   = ['da','de','do','das','dos','DA','DE','DOS','DAS','DOS','De']
        nome_completo = ''
        
        # Ajustar lista de termos, identificar sobrenomes compostos e ajustar sobrenome com ou sem presença de vírgula
        div_sobrenome      = sobrenome_inicio.findall(string)
        div_sbrcomposto    = sobrenome_composto.findall(string)
        
        # Caso haja vírgulas na string, tratar sobrenomes e sobrenomes compostos
        if div_sobrenome != [] or div_sbrcomposto != []:
            div   = string.split(', ')
            sobrenome     = div[0].strip().upper()
            try:
                div_espaco    = div[1].split(' ')
            except:
                div_espaco  = ['']
            primeiro      = div_espaco[0].strip('.')
            
            # Caso primeiro nome sejam somente duas letras maiúsculas juntas, trata-se de duas iniciais
            if len(primeiro)==2:
                primeiro_nome=primeiro[0].strip()
                nomes.append(primeiro[1].strip())
            else:
                primeiro_nome = div_espaco[0].strip().title()
            
            # Montagem da lista de nomes do meio
            for nome in div_espaco:
                if nome not in nomes and nome.lower()!=primeiro_nome.lower() and nome!=sobrenome:   
                    # print(nome, len(nome))
                    
                    # Avaliar se é abreviatura seguida de ponto e remover o ponto
                    if len(nome)<=2 and nome.lower() not in preposicoes:
                        for inicial in nome:
                            nomes.append(inicial.replace('.','').strip().title())
                    else:
                        if nome not in nomes and nome!=primeiro_nome and nome!=sobrenome and nome!='':
                            if nome.lower() in preposicoes:
                                nomes.append(nome.replace('.','').strip().lower())
                            else:
                                nomes.append(nome.replace('.','').strip().title())
                            # print(nome,'|',primeiro_nome)
                            
            #caso haja sobrenome composto que não esteja nos agnomes considerar somente primeiro como sobrenome
            if div_sbrcomposto !=[] and sobrenome.split(' ')[1] not in agnomes:
                # print(div_sbrcomposto)
                # print('Sobrenome composto:',sobrenome)
                nomes.append(sobrenome.split(' ')[1].title())
                sobrenome = sobrenome.split(' ')[0].upper()
                # print('Sobrenome:',sobrenome.split(' '))
                for i in nomes:
                    if i.lower() in sobrenome.lower():
                        nomes.remove(i)
                # print('Nomes do meio:',nomes)
            
            # print('    Sobrenome com vírgula:',sobrenome, len(sobrenome),'letras')
            # print('Primeiro nome com vírgula:',primeiro_nome, len(primeiro_nome),'letras')
            # print('Nomes do meio com vírgula:',nomes, len(nomes),'nomes')
            
        # Caso não haja vírgulas na string considera sobrenome o último nome da string dividida com espaço vazio
        else:
            try:
                div       = string.split(' ')
                if div[-2] in agnomes:
                    sobrenome = div[-2].upper()+' '+div[-1].strip().upper()
                    for i in nomes[1:-2]:
                        if i not in sobrenome and i not in preposicoes:
                            nomes.append(i.strip().title())
                        if i in preposicoes:
                            nomes.append(i.strip().lower())
                else:
                    sobrenome = div[-1].strip().upper()
                    for i in div[1:-1]:
                        if i not in sobrenome and i not in preposicoes:
                            nomes.append(i.strip().title())
                        if i in preposicoes:
                            nomes.append(i.strip().lower())
            except:
                sobrenome = div[-1].strip().upper()
                for i in div[1:-1]:
                        if i not in sobrenome and i not in preposicoes:
                            nomes.append(i.strip().title())
                        if i in preposicoes:
                            nomes.append(i.strip().lower())
                
            if sobrenome.lower() != div[0].strip().lower():
                primeiro_nome=div[0].strip().title()
            else:
                primeiro_nome=''
            
            # print('    Sobrenome sem vírgula:',sobrenome)
            # print('Primeiro nome sem vírgula:',primeiro_nome)
            # print('Nomes do meio sem vírgula:',nomes)
        
        # Encontrar e tratar como abreviaturas termos com apenas uma ou duas letras iniciais juntas, com ou sem ponto
        for j in nomes:
            # Procura padrões com expressões regulares na string
            div_sobrenome      = sobrenome_inicio.findall(j)
            div_sbrcomposto    = sobrenome_composto.findall(j)
            div_abrevponto     = letra_abrevponto.findall(j)
            div_abrevespaco    = letra_abrevespaco.findall(j)
            div_abrevisolada   = letra_abrevisolada.findall(j)
            div_ltrdobradasini = letras_dobradasini.findall(j)
            div_ltrdobradasfim = letras_dobradasfim.findall(j)
            div_ltrdobradas    = letras_dobradas.findall(j)
            tamanho=len(j)
            # print('\n', div_ltrdobradasini, div_ltrdobradasfim, tamanho, 'em:',j,len(j))
            
            #caso houver abreviatura com uma letra em maiúscula nos nomes
            if div_abrevponto !=[] or tamanho==1:
                cada_nome = j.replace('.','').strip()
                if cada_nome not in nomes and cada_nome != sobrenome and nome != primeiro_nome:
                    nomes.append(cada_nome)

            #caso houver abreviatura com uma letra em maiúscula nos nomes
            if div_abrevespaco !=[] or tamanho==1:
                cada_nome = j.replace('.','').strip()
                if cada_nome not in nomes and cada_nome != sobrenome and nome != primeiro_nome:
                    nomes.append(cada_nome)

            #caso houver abreviatura com uma letra em maiúscula isolada nos nomes
            if div_abrevisolada !=[] or tamanho==1:
                cada_nome = j.replace('.','').strip()
                if cada_nome not in nomes and cada_nome != sobrenome and nome != primeiro_nome:
                    nomes.append(cada_nome)

            #caso houver duas inicias juntas em maiúsculas
            elif div_ltrdobradasini !=[] or div_ltrdobradasfim !=[] or div_ltrdobradas !=[] :
                for letra in j:
                    if letra not in nomes and letra != sobrenome and letra != primeiro_nome:
                        nomes.append(letra)
            
            #caso haja agnomes ao sobrenome
            elif sobrenome in agnomes:
                sobrenome = nomes[-1].upper()+' '+sobrenome
                # print(sobrenome.split(' '))
                # print('Sobrenome composto:',sobrenome)
                for i in nomes:
                    if i.lower() in sobrenome.lower():
                        nomes.remove(i)
                # print('Nomes do meio:',nomes)
                
            else:
                if j not in nomes and j not in sobrenome and j != primeiro_nome:
                    nomes.append(j)
        
        nomes_meio=' '.join([str[0] for str in nomes]).strip()
        # print('Qte nomes do meio',len(nomes),nomes)
        if sobrenome != '' and primeiro_nome !='':
            sobrenome_iniciais = sobrenome+', '+primeiro_nome[0]+' '+nomes_meio
        elif sobrenome != '':
            sobrenome_iniciais = sobrenome

        return sobrenome_iniciais.strip()

    def similares(self, lista_autores, lista_grupo, limite_jarowinkler, distancia_levenshtein):
        """Função para aplicar padronização no nome de autor da lista de pesquisadores e buscar similaridade na lista de coautores
        Recebe: Lista de pesquisadores do grupo em análise gerada pela lista de nomes dos coautores das publicações em análise
        Utiliza: get_jaro_distance(), editdistance()
        Retorna: Lista de autores com fusão de nomes cuja similaridade esteja dentro dos limites definidos nesta função
        Autor: Marcos Aires (Fev.2022)
        
        Refazer: Inserir crítica de, mantendo sequência ordem alfabética, retornar no final nome mais extenso em caso de similaridade;
        """
        from pyjarowinkler.distance import get_jaro_distance
        from IPython.display import clear_output
        import editdistance
        import numpy as np
        import time
        
        t0=time.time()
        
        # limite_jarowinkler=0.85
        # distancia_levenshtein=6
        similares_jwl=[]
        similares_regras=[]
        similares=[]
        tempos=[]
        
        count=0
        t1=time.time()
        for i in lista_autores:
            count+=1
            if count > 0:
                tp=time.time()-t1
                tmed=tp/count*2
                tempos.append(tp)
        #     print("Analisar similaridades com: ", nome_padronizado)
            
            count1=0
            for nome in lista_autores:
                if count1 > 0:
                    resta=len(lista_autores)-count
                    print(f'Analisando {count1:3}/{len(lista_autores)} resta analisar {resta:3} nomes. Previsão de término em {np.round(tmed*resta/60,1)} minutos')
                else:
                    print(f'Analisando {count1:3}/{len(lista_autores)} resta analisar {len(lista_autores)-count1} nomes.')
                
                t2=time.time()
                count1+=1            

                try:
                    similaridade_jarowinkler = get_jaro_distance(i, nome)
                    print(f'{i:40} | {nome:40} | Jaro-Winkler: {np.round(similaridade_jarowinkler,2):4} Levenshtein: {editdistance.eval(i, nome)}')
                    similaridade_levenshtein = editdistance.eval(i, nome)

                    # inferir similaridade para nomes que estejam acima do limite ponderado definido, mas não idênticos e não muito distantes em edição
                    if  similaridade_jarowinkler > limite_jarowinkler and similaridade_jarowinkler!=1 and similaridade_levenshtein < distancia_levenshtein:
                        # Crítica no nome mais extenso como destino no par (origem, destino)
                        
                        similares_jwl.append((i,nome))

                except:
                    pass

                clear_output(wait=True)
        
        # Conjunto de regras de validação de similaridade
        # Monta uma lista de nomes a serem retirados antes de montar a lista de troca
        trocar=[]
        retirar=[]
        for i in similares_jwl:
            sobrenome_i = i[0].split(',')[0]
            sobrenome_j = i[1].split(',')[0]

            try:
                iniciais_i  = iniciais_nome(i[0]).split(',')[1].strip()
            except:
                iniciais_i  = ''

            try:
                iniciais_j  = iniciais_nome(i[1]).split(',')[1].strip()
            except:
                iniciais_j  = ''

            try:
                primnome_i = i[0].split(',')[1].strip().split(' ')[0].strip()
            except:
                primnome_i = ''

            try:
                primnome_j = i[1].split(',')[1].strip().split(' ')[0].strip()
            except:
                primnome_j = ''    

            try:
                inicial_i = i[0].split(',')[1].strip()[0]
            except:
                inicial_i = ''

            try:
                resto_i   = i[0].split(',')[1].strip().split(' ')[0][1:]
            except:
                resto_i   = ''

            try:
                inicial_j = i[1].split(',')[1].strip()[0]
            except:
                inicial_j = ''

            try:
                resto_j   = i[1].split(',')[1].strip().split(' ')[0][1:]
            except:
                resto_j = ''

            # Se a distância de edição entre os sobrenomes
            if editdistance.eval(sobrenome_i, sobrenome_j) > 2 or inicial_i!=inicial_j:
                retirar.append(i)
            else:
                if primnome_i!=primnome_j and len(primnome_i)>1:
                    retirar.append(i)
                if primnome_i!=primnome_j and len(primnome_i)>1 and len(primnome_j)>1:
                    retirar.append(i)
                if resto_i!=resto_j and resto_i!='':
                    retirar.append(i)
                if len(i[1]) < len(i[0]):
                    retirar.append(i)
                if len(iniciais_i) != len(iniciais_j):
                    retirar.append(i)

        for i in similares_jwl:
            if i not in retirar:
                trocar.append(i)

            if iniciais_nome(i[0]) in iniciais_nome(i[1]) and len(i[0]) < len(i[1]):
                trocar.append(i)

            if iniciais_nome(i[0]) == iniciais_nome(i[1]) and len(i[0]) < len(i[1]):
                trocar.append(i)
        
        lista_extra = [
                        # ('ALBUQUERQUE, Adriano B', 'ALBUQUERQUE, Adriano Bessa'),
                        # ('ALBUQUERQUE, Adriano', 'ALBUQUERQUE, Adriano Bessa'),
                        # ('COELHO, Andre L V', 'COELHO, Andre Luis Vasconcelos'),
                        # ('DUARTE, Joao B F', 'DUARTE, Joao Batista Furlan'),
                        # ('FILHO, Raimir H','HOLANDA FILHO, Raimir'),
                        # ('FILHO, Raimir','HOLANDA FILHO, Raimir'),
                        # ('FORMIGO, A','FORMICO, Maria Andreia Rodrigues'),
                        # ('FORMICO, A','FORMICO, Maria Andreia Rodrigues'),
                        # ('FURLAN, J B D', 'FURLAN, Joao Batista Duarte'),
                        # ('FURTADO, Elizabeth', 'FURTADO, Maria Elizabeth Sucupira'),
                        # ('FURTADO, Elizabeth S', 'FURTADO, Maria Elizabeth Sucupira'),
                        # ('FURTADO, Elizabeth Sucupira','FURTADO, Maria Elizabeth Sucupira'),
                        # ('FURTADO, M E S', 'FURTADO, Maria Elizabeth Sucupira'),
                        # ('FURTADO, Vasco', 'FURTADO, Joao Jose Vasco Peixoto'),
                        # ('FURTADO, J P', 'FURTADO, Joao Jose Vasco Peixoto'),
                        # ('FURTADO, J V P', 'FURTADO, Joao Jose Vasco Peixoto'),
                        # ('FURTADO, Vasco', 'FURTADO, Joao Jose Vasco Peixoto'),
                        # ('FURTADO, Elizabeth','FURTADO, Maria Elizabeth Sucupira'),
                        # ('HOLANDA, Raimir', 'HOLANDA FILHO, Raimir'),
                        # ('LEITE, G S', 'LEITE, Gleidson Sobreira'),
                        # ('PEQUENO, T H C', 'PEQUENO, Tarcisio Haroldo Cavalcante'),
                        # ('PEQUENO, Tarcisio','PEQUENO, Tarcisio Haroldo Cavalcante'),
                        # ('PEQUENO, Tarcisio Cavalcante', 'PEQUENO, Tarcisio Haroldo Cavalcante'),
                        # ('PINHEIRO, Placido R', 'PINHEIRO, Placido Rogerio'),
                        # ('PINHEIRO, Vladia', 'PINHEIRO, Vladia Celia Monteiro'),
                        # ('RODRIGUES, M A F', 'RODRIGUES, Maria Andreia Formico'),
                        # ('RODRIGUES, Andreia', 'RODRIGUES, Maria Andreia Formico'),
                        # ('JOAO, Batista F Duarte,', 'FURLAN, Joao Batista Duarte'),
                        # ('MACEDO, Antonio Roberto M de', 'MACEDO, Antonio Roberto Menescal de'),
                        # ('MACEDO, D V', 'MACEDO, Daniel Valente'),
                        # ('MENDONCA, Nabor C', 'MENDONCA, Nabor das Chagas'),
                        # ('PEQUENO, Tarcisio', 'PEQUENO, Tarcisio Haroldo Cavalcante'),
                        # ('PEQUENO, Tarcisio H', 'PEQUENO, Tarcisio Haroldo Cavalcante'),
                        # ('PINHEIRO, Mirian C D', 'PINHEIRO, Miriam Caliope Dantas'),
                        # ('PINHEIRO, Mirian Caliope Dantas', 'PINHEIRO, Miriam Caliope Dantas'),
                        # ('PINHEIRO, P G C D', 'PINHEIRO, Pedro Gabriel Caliope Dantas'),
                        # ('PINHEIRO, Pedro G C', 'PINHEIRO, Pedro Gabriel Caliope Dantas'),
                        # ('PINHEIRO, Placido R', 'PINHEIRO, Placido Rogerio'),
                        # ('PINHEIRO, Vladia', 'PINHEIRO, Vladia Celia Monteiro'),
                        # ('ROGERIO, Placido Pinheiro', 'PINHEIRO, Placido Rogerio'),
                        # ('REBOUCRAS FILHO, Pedro', 'REBOUCAS FILHO, Pedro Pedrosa'),
                        # ('SAMPAIO, A', 'SAMPAIO, Americo Tadeu Falcone'),
                        # ('SAMPAIO, Americo', 'SAMPAIO, Americo Tadeu Falcone'),
                        # ('SAMPAIO, Americo Falcone', 'SAMPAIO, Americo Tadeu Falcone'),
                        # ('SUCUPIRA, Elizabeth Furtado','FURTADO, Maria Elizabeth Sucupira'),
                    ]
        
        trocar=trocar+lista_extra
        trocar.sort()
        
        return trocar

    def extrair_variantes(self, df_dadosgrupo):
        ''' Utiliza campo de Nome em Citações do currículo como filtro para obter variantes do nome de cada membro
        Recebe: Dataframe com os dados brutos do grupo de pesquisa agrupados; lista de nomes de pesquisadores de interesse
        Retorna: Lista de tuplas com pares a serem trocados da variante pelo nome padronizado na forma (origem, destino)
        '''
        filtro1   = 'Nome'
        lista_nomes = df_dadosgrupo[(df_dadosgrupo.ROTULOS == filtro1)]['CONTEUDOS'].values

        variantes=[]
        filtro='Nome em citações bibliográficas'
        variantes=df_dadosgrupo[(df_dadosgrupo.ROTULOS == filtro)]['CONTEUDOS'].to_list()

        trocar=[]
        for j in range(len(variantes)):
            padrao_destino = padronizar_nome(lista_nomes[j])
            trocar.append((lista_nomes[j], padrao_destino))
            for k in variantes[j]:
                padrao_origem = padronizar_nome(k)
                trocar.append((k, padrao_destino))
                trocar.append((padrao_origem, padrao_destino))
        
        return trocar

    def inferir_variantes(self, nome):
        ''' Quebra um nome inicialmente por vírgula para achar sobrenomes, e depois por ' ' para achar nomes
        Recebe: Par de nomes a comparar, nome1 é nome padronizado na função padronizar_nome(), nome2 é o que será analisado
        Utiliza: Função padronizar_nome(nome)
        Retorna: Lista de tuplas, no formato (origem, destino), com variantes de nome a serem trocadas pela forma padronizada
        Autor: Marco Aires (Fev.2022)
        '''
        trocar = []
        nomes  = []
        try:
            div0  = nome.split(',').strip()
            sobrenome=div0[0]
            try:
                div1 = div0[1].split(' ').strip()
                for i in div1:
                    nomes.append(i)
            except:
                pass
            
        except:
            pass
            
        trocar.append(nome, iniciais_nome(nome))
        
        return trocar

    def comparar_nomes(self, nome1,nome2):
        ''' Compara dois nomes por seus sobrenomes e iniciais do primeiro nome
        Recebe: Par de nomes a comparar, nome1 é nome padronizado na função padronizar_nome(), nome2 é o que será analisado
        Utiliza: Função padronizar_nome(nome)
        Retorna: Lista de tuplas, no formato (origem, destino), com variantes de nome a serem trocadas pela forma padronizada
        Autor: Marco Aires (Fev.2022)
        '''
        trocar=[]
        qte_nomes1=0
        nome_padronizado1 = padronizar_nome(nome1)
        sobrenome1        = nome_padronizado1.split(',')[0]
        if sobrenome1!='':
            qte_nomes1+=1
        primeiro_nome1    = nome_padronizado1.split(',')[1].split(' ')[0]
        if primeiro_nome1!='':
            qte_nomes1+=1
        inicial_primnome1 = primeiro_nome1[0]
        demais_nomes1     = nome_padronizado1.split(',')[1].split(' ')[1:]
        qte_nomes1=qte_nomes1+len(demais_nomes1)
        
        qte_nomes2=0
        nome_padronizado2 = padronizar_nome(nome2)
        sobrenome2        = nome_padronizado2.split(',')[0]
        if sobrenome2!='':
            qte_nomes2+=1    
        primeiro_nome2    = nome_padronizado2.split(',')[1].split(' ')[0]
        if primeiro_nome2!='':
            qte_nomes2+=1
        inicial_primnome2 = primeiro_nome2[0]
        demais_nomes2     = nome_padronizado2.split(',')[1].split(' ')[1:]
        qte_nomes2=qte_nomes2+len(demais_nomes2)
        
        if sobrenome1==sobrenome2 and primeiro_nome1==primeiro_nome2:
            trocar.append((nome1,nome_padronizado2))

        if sobrenome1==sobrenome2 and primeiro_nome1==primeiro_nome2:
            trocar.append((nome1,nome_padronizado2))
            
        return trocar


class ArticlesCounter:
    def __init__(self, dict_list):
        self.data_list = dict_list

    def contar_artigos(self, dict_list):
        ## Contagem de artigos para simples confererência
        print(f'{len(dict_list)} dicionários montados')
        qte_artigos=0
        qte_titulos=0
        for k,i in enumerate(dict_list):
            try:
                qte_jcr = len(i.get('Produções').get('Artigos completos publicados em periódicos'))
            except:
                qte_jcr = 0
            try:
                qte_jcr2 = len(i['JCR2'])
            except:
                qte_jcr2 = 0
            qte_artigos+=qte_jcr
            qte_titulos+=qte_jcr2
            status=qte_jcr2-qte_jcr
            print(f"{k:>2}C {qte_jcr:>03}A {qte_jcr2:>03}T Dif:{status:>03} {i.get('Identificação').get('name')} ")
        print(f'\nTotal de artigos em todos períodos: {qte_artigos}')
        print(f'Total de títulos em todos períodos: {qte_titulos}')    
        return qte_artigos, qte_titulos


    ## Funções auxiliares da extração de artigos e orientações
    def buscar_qualis(self, lista_dados_autor):
        for dados_autor in lista_dados_autor:
            for categoria, artigos in dados_autor['Produções'].items():
                if categoria == 'Artigos completos publicados em periódicos':
                    for artigo in artigos:
                        issn_artigo = artigo['ISSN'].replace('-','')
                        qualis = self.encontrar_qualis_por_issn(issn_artigo)
                        print(f'{issn_artigo:8} | {qualis}')
                        if qualis:
                            artigo['Qualis'] = qualis
                        else:
                            artigo['Qualis'] = 'NA'

    def encontrar_qualis_por_issn(self, issn):
        qualis = self.dados_planilha[self.dados_planilha['ISSN'].str.replace('-','') == issn]['Estrato'].tolist()
        if qualis:
            return qualis[0]
        else:
            return None

    # Usar unicodedata para remover acentos
    def remover_acentos_unicodedata(self, texto):
        texto_sem_acentos = ''.join(c for c in unicodedata.normalize('NFD', texto)
                                    if unicodedata.category(c) != 'Mn')
        return texto_sem_acentos

    # Usar unicode para remover acentos
    def remover_acentos(self, texto):
        return unidecode(texto)

    def substituir_abreviaturas(self, texto):
        padrao = re.compile(r"\bDr\.")
        return padrao.sub(lambda m: m.group().replace(".", ""), texto)

    def substituir_iniciais(self, texto):
        padrao = re.compile(r"\b[A-Z]\.")  # Expressão regular para qualquer letra maiúscula seguida de ponto
        return padrao.sub(lambda m: m.group()[:-1], texto)  # Remove o ponto da inicial

    # Define a função para verificar se a string contém un dos termos da lista de detecção (case insensitive)
    def contem_fiocruz(self, instituicao):
        termos = ['fiocruz', 'fundação oswaldo cruz']
        for termo in termos:
            if termo.lower() in instituicao.lower():
                return 'interno'
        return 'externo'

    def organizar_orientacoes_geral(self, dict_list_docents):
        tipos = []
        orientacoes_set = set()
        orientacoes_lista = []

        ano = None
        papel = None
        orientando = None
        instituicao = ''

        for curriculo in dict_list_docents:
            nome = curriculo.get('Identificação').get('Nome')
            orientacoes_curriculo = curriculo.get('Orientações')
            if orientacoes_curriculo:
                for secao in orientacoes_curriculo: # Itera sobre as seções principais
                    for subsecao in secao['subsecoes']:
                        subsec_name = subsecao['nome']
                        if 'concluídas' in subsec_name:
                            status = 'concluídas'
                        else:
                            status = 'em andamento'
                        for orientacao in subsecao['orientacoes']:
                            tipo = orientacao['tipo']
                            detalhes = orientacao['detalhes']

                            detalhes = self.substituir_iniciais(detalhes)
                            detalhes = self.substituir_abreviaturas(detalhes)
                            if tipo == 'Supervisão de pós-doutorado':
                                orientando = detalhes.split('. ')[0].title()
                                ano = detalhes.split('. ')[1].replace('Início: ', '')
                                papel = 'supervisor'
                            else:
                                orientando = detalhes.split('. ')[0].title()
                                titulo = detalhes.split('. ')[1]
                                ano = detalhes.split('. ')[2].replace('Início: ', '')
                                try:
                                    hifens = len(detalhes.split('. ')[-2].split('-'))
                                    inst = detalhes.split('. ')[-2].split('-')[-1].strip()
                                    if len(inst) == 2:
                                        instituicao = detalhes.split('. ')[-3].split('-')[1].strip().replace('Ã¡', 'á').replace('Ã§Ã£', 'çã').replace('(', '').replace(').', '').replace(')', '')
                                    elif len(inst) == 4:
                                        instituicao = detalhes.split('. ')[-2].split('-')[-1].strip().replace('Ã¡', 'á').replace('Ã§Ã£', 'çã').replace('(', '').replace(').', '').replace(')', '')
                                    # elif len(inst) != 2 and hifens > 1:
                                    #   instituicao = detalhes.split('. ')[-2].split('-')[-1].strip().replace('Ã¡', 'á').replace('Ã§Ã£', 'çã').replace('(', '').replace(').', '').replace(')', '')
                                    else:
                                        instituicao = detalhes.split('. ')[-2].split('-')[-1].strip().replace('Ã¡', 'á').replace('Ã§Ã£', 'çã').replace('(', '').replace(').', '').replace(')', '')
                                except:
                                    instituicao = detalhes.split('. ')[-2].strip().replace('Ã¡', 'á').replace('Ã§Ã£', 'çã').replace('(', '').replace(').', '').replace(')', '')
                                papel = detalhes.split('. ')[-1].split(':')[0].strip().replace('(', '').replace(').', '').replace(')', '')

                            if len(ano) != 4:
                                resultado = re.search(r"\d{4}", detalhes)
                                if resultado:
                                    ano = resultado.group().strip()

                            if tipo not in tipos:
                                tipos.append(tipo)

                            padrao = re.compile(r" Associação| Avaliação Dos| Síndrome Da| Prevalência| Atendimento| Avaliação Da| Avaliação| Atenção| Uso Farmacológico| Assinaturas| : ?Estratégias| Construção| Caracterização| Efeitos")

                            if len(orientando.split(' ')) > 5:
                                orientando = padrao.sub("", orientando)

                            # Verifica se a instituição foi extraída corretamente, se não, atribui 'Não Informada'
                            if not instituicao:
                                instituicao = 'Não Informada'

                            # Cria uma tupla com as informações da orientação
                            orientacao_info = (nome, ano, papel, tipo, orientando, instituicao, status)

                            # Adiciona a orientação à lista apenas se for única
                            if orientacao_info not in orientacoes_set:
                                orientacoes_set.add(orientacao_info)
                                orientacoes_lista.append({
                                    'Docente': nome,
                                    'ano': ano,
                                    'papel': papel,
                                    'tipo': tipo,
                                    'orientando': orientando,
                                    'instituicao': instituicao,
                                    'status': status
                                })

        # Converte o conjunto de orientações de volta para uma lista
        orientacoes_lista = list({'Docente': nome, 'ano': ano, 'papel': papel, 'tipo': tipo, 'orientando': orientando, 'instituicao': instituicao, 'status': status}
            for (nome, ano, papel, tipo, orientando, instituicao, status) in orientacoes_set
        )

        # Ordena a lista de orientações
        orientacoes_lista = sorted(orientacoes_lista, key=lambda x: (x['Docente'], x['papel'], x['status'], x['tipo'], x['ano']))

        return orientacoes_lista

    def criar_dataframe_contagens(self, orientacoes_lista, ano_inicio, ano_final, tipos_aceitaveis):
        # Cria um DataFrame a partir da lista de orientações
        df_orientacoes = pd.DataFrame(orientacoes_lista)

        # Filter the DataFrame based on 'Ano' and 'Tipo', criando uma cópia explícita
        df_orientacoes_filtrado = df_orientacoes[
            (df_orientacoes['ano'].astype(int) >= ano_inicio) &
            (df_orientacoes['ano'].astype(int) <= ano_final) &
            (df_orientacoes['tipo'].isin(tipos_aceitaveis))
        ].copy()

        # Cria a coluna 'Interno Fiocruz' usando .loc para evitar o SettingWithCopyWarning
        df_orientacoes_filtrado.loc[:, 'Interno Fiocruz'] = df_orientacoes_filtrado['instituicao'].astype(str).apply(self.contem_fiocruz)

        # Calcula o total de orientações 'interno', incluindo a coluna 'status'
        total_interno = df_orientacoes_filtrado[df_orientacoes_filtrado['Interno Fiocruz'] == 'interno'].groupby(['Docente', 'status']).size().reset_index(name='Interno Fiocruz')

        # Calcula o total de orientações 'externo', incluindo a coluna 'status'
        total_externo = df_orientacoes_filtrado[df_orientacoes_filtrado['Interno Fiocruz'] == 'externo'].groupby(['Docente', 'status']).size().reset_index(name='Externo à Fiocruz')

        # Faz o merge dos resultados usando um outer join, incluindo a coluna 'status'
        resultado_final = total_interno.merge(total_externo, on=['Docente', 'status'], how='outer')

        # Ordena o DataFrame
        resultado_final = resultado_final.sort_values('Docente')

        # Preenche os valores ausentes com zero
        resultado_final = resultado_final.fillna(0)

        # Redefine o índice para ter 'Docente' como uma coluna
        resultado_final = resultado_final.reset_index()

        # Reordena as colunas
        cols = ['Docente', 'status', 'Interno Fiocruz', 'Externo à Fiocruz']
        resultado_final = resultado_final[cols]

        filepath = os.path.join(LattesScraper.find_repo_root(),'_data','in_xls','contagens_orientacoes.xlsx')

        # Salvar o DataFrame em uma planilha Excel
        resultado_final.to_excel(filepath, index=False)

        return resultado_final

    def listar_orientacoes_docente(self, orientacoes_lista, lista_docentes, lista_tipos, lista_status):
        # Filtrar a lista de orientações com base na lista de docentes
        orientacoes_docente = [
            orientacao
            for orientacao in orientacoes_lista
            if self.remover_acentos(orientacao['Docente']).lower() in [docente.lower() for docente in lista_docentes]
        ]

        # Criar uma lista para armazenar os dados das orientações filtradas
        dados_orientacoes = []

        # Iterar sobre as orientações do docente
        for orientacao in orientacoes_docente:
            if orientacao.get('tipo') in lista_tipos and orientacao.get('status') in lista_status:
                dados_orientacoes.append({
                    'Ano': orientacao.get('ano'),
                    'Tipo de Orientação': orientacao.get('tipo'),
                    'Instituição': orientacao.get('instituicao').split(',')[0],
                    'Orientando': orientacao.get('orientando'),
                    'Status': orientacao.get('status')
                })

        # Criar um DataFrame a partir dos dados das orientações filtradas
        df_orientacoes = pd.DataFrame(dados_orientacoes)
        print(df_orientacoes.keys())
        df_orientacoes.sort_values(by=['Status','Tipo de Orientação', 'Ano'])

        # Adicionar informações de resumo ao DataFrame
        df_orientacoes.loc['Total', :] = ''  # Adiciona uma linha vazia para o total
        df_orientacoes.loc['Total', 'Ano'] = len(orientacoes_docente)  # Total de orientações do docente
        df_orientacoes.loc['Total', 'Tipo de Orientação'] = f"Sendo {len(df_orientacoes) - 1} em {' e '.join(lista_tipos)}"

        return df_orientacoes

    # Gerar o relatório de orientações em HTML
    def generate_html_report(self, orientacoes_lista, ano_inicial=None, ano_final=None, tipos_orientacao=None):

        # Filtrar a lista de orientações com base nos parâmetros de entrada de período e tipos
        if ano_inicial and ano_final:
            orientacoes_lista = [o for o in orientacoes_lista if ano_inicial <= int(o.get('ano', 0)) <= ano_final]
        if tipos_orientacao:
            orientacoes_lista = [o for o in orientacoes_lista if o.get('tipo') in tipos_orientacao]

        # Encontrar o ano inicial e final pelos máximos e mínimos da coluna ano
        anos = [int(orientacao.get('ano', '')) for orientacao in orientacoes_lista if orientacao.get('ano', '')]
        if not ano_inicial:
            ano_inicial = min(anos) if anos else ''
        if not ano_final:
            ano_final = max(anos) if anos else ''
        
        # Criar a estrutura da tabela HTML, incluindo os novos cabeçalhos
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        table {
            font-family: arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
        }

        td, th {
            border: 1px solid #dddddd;
            text-align: left;
            padding: 8px;
        }

        tr:nth-child(even) {
            background-color:  
        #dddddd;
        }

        th {
            cursor: pointer; /* Indica que o cabeçalho é clicável */
        }

        th.sorted-asc::after {
            content: " ▲"; /* Adiciona um triângulo para cima */
        }

        th.sorted-desc::after {
            content: " ▼"; /* Adiciona um triângulo para baixo */
        }

        #total-row-concluidas, #total-row-andamento { /* Seleciona as linhas de totais gerais */
            font-weight: bold; 
        }        

        </style>
        </head>
        <body>

        <h2>Relatório das Orientações Concluídas/Em_andamento dos servidores Fiocruz Ceará</h2>
        """
        # Adicionar a informação do período usando f-string fora das aspas triplas
        html_content += f"<h2>Período: {ano_inicial} - {ano_final}</h2>\n\n"

        # Continuar a construção da tabela HTML com <tbody>
        html_content += """
        <table>
        <thead>
        <tr>
            <th>Docente</th>
            <th>Papel</th>
            <th>Ano</th>
            <th>Orientando</th>
            <th>Instituição</th>
            <th>Tipo de Orientação</th> 
            <th>Status</th>               
        </tr>
        </thead>
        <tbody>
        """

        # Iterar sobre a lista de dicionários e extrai as informações relevantes
        for orientacao in orientacoes_lista:
            docente = orientacao.get('Docente', '')
            papel = orientacao.get('papel', '')
            ano = orientacao.get('ano', '')
            orientando = orientacao.get('orientando', '')
            instituicao = orientacao.get('instituicao', '')
            tipo = orientacao.get('tipo', '')       
            status = orientacao.get('status', '')   

            # Popula a tabela HTML com os dados extraídos
            html_content += f"""
            <tr>
                <td>{docente}</td>
                <td>{papel}</td>
                <td>{ano}</td>
                <td>{orientando}</td>
                <td>{instituicao}</td>
                <td>{tipo}</td>
                <td>{status}</td>
            </tr>
            """

        # Fechar a tag <tbody> e adicionar as linhas de total
        html_content += """
            </tbody>
        """

        # Adicionar as duas linhas de total GERAL no final da tabela
        html_content += """
            </table>

            <script>
            // Ordenar valores por rótulo de coluna
            function sortTable(n) {
            var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
            table = document.querySelector("table"); 
            // Seleciona a tabela
            switching = true;
            // Definir a direção inicial como ascendente
            dir = "asc"; 
            // Loop até que nenhuma troca seja necessária
            while (switching) {
                switching = false;
                rows = table.rows;
                // Loop por todas as linhas da tabela (exceto a primeira, que contém os cabeçalhos)
                for (i = 1; i < (rows.length - 2); i++) { // Exclui as duas últimas linhas de total geral
                shouldSwitch = false;
                // Obtém as duas células a serem comparadas
                x = rows[i].getElementsByTagName("TD")[n];
                y = rows[i + 1].getElementsByTagName("TD")[n];
                // Verifica se as células devem ser trocadas
                if (dir == "asc") {
                    if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {
                    shouldSwitch= true;
                    break;
                    }
                } else if (dir == "desc") {
                    if (x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()) {
                    shouldSwitch = true;
                    break;
                    }
                }
                }
                if (shouldSwitch) {
                // Se uma troca for necessária, faz a troca e marca que uma troca foi feita
                rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                switching = true;
                switchcount ++;      
                } else {
                // Se nenhuma troca foi feita e a direção é ascendente, define a direção como descendente e reinicia o loop externo
                if (switchcount == 0 && dir == "asc") {
                    dir = "desc";
                    switching = true;
                }
                }
            }

            // Remover a classe de ordenação de todos os cabeçalhos
            var headers = table.querySelectorAll("th");
            headers.forEach(function(header) {
                header.classList.remove("sorted-asc", "sorted-desc");
            });

            // Adicionar a classe de ordenação apropriada ao cabeçalho clicado
            var clickedHeader = headers[n];
            clickedHeader.classList.add(dir === "asc" ? "sorted-asc" : "sorted-desc");
            }

            // Calcular e exibir os totais (com verificações e mensagens de erro)
            function calculateTotals() {
                var table = document.querySelector("table");
                var rows = table.rows;
                var tableBody = table.querySelector("tbody"); 

                if (!tableBody) {
                    console.error("Erro: Elemento <tbody> não encontrado na tabela.");
                    return;
                }

                // Dicionários para armazenar os totais por tipo e status
                var totals = {
                    'concluídas': {},
                    'em andamento': {}
                };

                // Loop para calcular os totais
                for (var i = 1; i < rows.length; i++) { 
                    var tipoCell = rows[i].getElementsByTagName("TD")[5]; // Coluna de tipo de orientação
                    var statusCell = rows[i].getElementsByTagName("TD")[6]; // Coluna de status

                    // Ignora as linhas de total geral
                    if (tipoCell === undefined || statusCell === undefined) {
                        continue;
                    }

                    var tipo = tipoCell.innerHTML;
                    var status = statusCell.innerHTML.toLowerCase();

                    // Inicializa o total para o tipo e status, se necessário
                    if (!totals[status][tipo]) {
                        totals[status][tipo] = 0;
                    }

                    // Incrementa o total
                    totals[status][tipo]++;
                }

                // Remove as linhas de total existentes, se houver (incluindo as de total geral)
                var existingTotalRows = table.querySelectorAll("tr.total-row");
                existingTotalRows.forEach(function(row) {
                    row.remove();
                });

                // Adiciona as novas linhas de total, primeiro as concluídas
                for (var tipo in totals['concluídas']) {
                    var newRow = tableBody.insertRow();
                    newRow.classList.add("total-row");

                    var cell1 = newRow.insertCell(0); 
                    cell1.colSpan = 5; 
                    cell1.innerHTML = "Total " + tipo + " (concluídas)";

                    var cell2 = newRow.insertCell(1); // Nova célula para o subtotal (vazia)

                    var cell3 = newRow.insertCell(2); // Total geral 
                    cell3.innerHTML = totals['concluídas'][tipo]; // Insere o valor na última coluna
                }

                // Adiciona as linhas de total em andamento
                for (var tipo in totals['em andamento']) {
                    var newRow = tableBody.insertRow(); 
                    newRow.classList.add("total-row");

                    var cell1 = newRow.insertCell(0);
                    cell1.colSpan = 5; 
                    cell1.innerHTML = "Total " + tipo + " (em andamento)";

                    var cell2 = newRow.insertCell(1); // Nova célula para o subtotal (vazia)

                    var cell3 = newRow.insertCell(2); // Total geral
                    cell3.innerHTML = totals['em andamento'][tipo]; // Insere o valor na última coluna
                }

                // Adiciona as duas linhas de total GERAL no final da tabela
                var newRowConcluidas = tableBody.insertRow();
                newRowConcluidas.id = "total-row-concluidas";
                newRowConcluidas.classList.add("total-row");
                var cellConcluidas1 = newRowConcluidas.insertCell(0);
                cellConcluidas1.colSpan = 5;
                cellConcluidas1.innerHTML = "Total de Orientações Concluídas";
                var cellConcluidas2 = newRowConcluidas.insertCell(1);
                cellConcluidas2.id = "total-concluidas";
                var cellConcluidas3 = newRowConcluidas.insertCell(2);

                var newRowAndamento = tableBody.insertRow();
                newRowAndamento.id = "total-row-andamento";
                newRowAndamento.classList.add("total-row");
                var cellAndamento1 = newRowAndamento.insertCell(0);
                cellAndamento1.colSpan = 5;
                cellAndamento1.innerHTML = "Total de Orientações Em andamento";
                var cellAndamento2 = newRowAndamento.insertCell(1);
                cellAndamento2.id = "total-em-andamento";
                var cellAndamento3 = newRowAndamento.insertCell(2);

                // Atualiza os totais gerais
                var totalGeralAndamento = Object.values(totals['em andamento']).reduce((a, b) => a + b, 0);
                var totalGeralConcluidas = Object.values(totals['concluídas']).reduce((a, b) => a + b, 0);

                document.getElementById("total-em-andamento").innerHTML = totalGeralAndamento;
                document.getElementById("total-concluidas").innerHTML = totalGeralConcluidas;

                if (Object.keys(totals['concluídas']).length === 0 && Object.keys(totals['em andamento']).length === 0) {
                    console.warn("Nenhuma orientação encontrada para calcular os totais.");
                }
            }

            // Chamar a função para calcular os totais quando a página carrega
            window.onload = calculateTotals;
            </script>

            </body>
            </html>
            """

        # try:
        #     # Tenta importar a biblioteca pdfkit
        #     import pdfkit

        #     # Converte o conteúdo HTML para PDF usando pdfkit
        #     pdfkit.from_string(html_content, 'relatorio_orientacoes.pdf')

        #     # Imprime uma mensagem de sucesso
        #     print("Relatório em PDF gerado com sucesso: relatorio_orientacoes.pdf")

        # except ImportError:
        #     # Lidar com o caso em que pdfkit não está instalado
        #     print("Erro: A biblioteca pdfkit não está instalada. Instale-a usando o comando 'pip install pdfkit' para gerar o relatório em PDF.")
        
        return html_content

    # Gerar o relatório de orientações em PDF
    def generate_pdf_report(self, html_content):
        pdfkit.from_string(html_content, 'relatorio_orientacoes.pdf')
        print("Relatório em PDF gerado com sucesso: relatorio_orientacoes.pdf")

    def imprimir_chaves_recursivo(self, dados, nivel=1, prefixo="", nivel1_filtro=None):
        if isinstance(dados, dict):
            for chave, valor in dados.items():
                if nivel == 1 and nivel1_filtro is not None and chave != nivel1_filtro:
                    continue  # Ignorar chaves que não correspondem ao filtro no nível 1

                print(f"{'  ' * nivel}{prefixo}n{nivel}: {chave}")

                # Chamada recursiva para o valor, mas somente se o filtro for None ou se já estivermos dentro do nível filtrado
                if nivel1_filtro is None or chave == nivel1_filtro:
                    self.imprimir_chaves_recursivo(valor, nivel + 1, prefixo + f"{chave}.", nivel1_filtro)

        elif isinstance(dados, list):
            for i, item in enumerate(dados):
                print(f"{'  ' * nivel}{prefixo}n{nivel}[{i}]")
                self.imprimir_chaves_recursivo(item, nivel + 1, prefixo + f"[{i}].", nivel1_filtro)
        else:
            print(f"{'  ' * nivel}{prefixo}n{nivel}: {dados}")

    def dias_desde_atualizacao(self, data_atualizacao_str):
        # Converte a data de atualização em um objeto datetime
        data_atualizacao = datetime.strptime(data_atualizacao_str, '%d/%m/%Y')
        
        # Obtém a data atual
        data_atual = datetime.now()
        
        # Calcula a diferença em dias
        diferenca_dias = (data_atual - data_atualizacao).days if data_atualizacao else None
        return diferenca_dias

    # Função para normalizar os nomes
    def normalizar_nome(self, nome):
        """Normaliza um nome, removendo acentos e caracteres especiais, convertendo para minúsculas e padronizando espaços."""

        # Remove acentos e caracteres especiais
        nome_sem_acentos = unidecode(nome)

        # Converte para minúsculas e remove espaços extras
        nome_normalizado = nome_sem_acentos.lower().strip().replace("  ", " ")

        return nome_normalizado

    def extrair_data_atualizacao(self, dict_list):
        ids_lattes_grupo=[]
        nomes_curriculos=[]
        dts_atualizacoes=[]
        tempos_defasagem=[]
        qtes_artcomplper=[]
        for dic in dict_list:
            info_nam = dic.get('Identificação').get('Nome',{})
            nomes_curriculos.append(info_nam)
            id_lattes = dic.get('Identificação', {}).get('ID Lattes',{})
            ids_lattes_grupo.append(id_lattes)
            dt_atualizacao = dic.get('Identificação', {}).get('Última atualização',{})
            dts_atualizacoes.append(dt_atualizacao)
            info_art = dic.get('Produções', {}).get('Artigos completos publicados em periódicos', {})
            qtes_artcomplper.append(len(info_art))
        for dt in dts_atualizacoes:
            tempo_atualizado = self.dias_desde_atualizacao(dt)
            tempos_defasagem.append(tempo_atualizado)

        # print(len(ids_lattes_grupo))
        # print(len(nomes_curriculos))
        # print(len(dts_atualizacoes))
        # print(len(tempos_defasagem))
        # print(len(qtes_artcomplper))
        
        dtf_atualizado = pd.DataFrame({"id_lattes": ids_lattes_grupo,
                                       "curriculos": nomes_curriculos, 
                                       "ultima_atualizacao": dts_atualizacoes,
                                       "dias_defasagem": tempos_defasagem,
                                       "qte_artigos_periodicos": qtes_artcomplper,
                                       })
        return dtf_atualizado

    def extrair_autor_qualis(self, dict_list):
        ids_lattes_grupo=[]
        nomes_curriculos=[]
        estratos_qualis=[]
        for dic in dict_list:
            info_nam = dic.get('Identificação').get('Nome',{})
            nomes_curriculos.append(info_nam)
            qualis = dic.get('Produções', {}).get('Artigos completos publicados em periódicos', {})['Qualis']
            estratos_qualis.append(qualis)
        print(len(ids_lattes_grupo))
        print(len(nomes_curriculos))
        print(len(estratos_qualis))
        dtf_atualizado = pd.DataFrame({"id_lattes": ids_lattes_grupo,
                                       "curriculos": nomes_curriculos, 
                                       "qualis": estratos_qualis,
                                       })
        return dtf_atualizado

    def contar_qualis(self, dict_list, ano_inicio, ano_final):
            lista_pubqualis = []
            for dic in dict_list:
                autor = dic.get('Identificação',{}).get('Nome',{})
                artigos = dic.get('Produções', {}).get('Artigos completos publicados em periódicos', {})
                current_year = datetime.now().year
                for i in artigos:
                    ano = i.get('ano') 
                    if not ano or pd.isna(ano):
                        ano = current_year
                    else:
                        try:
                            ano = int(ano)
                        except ValueError:
                            ano = current_year

                    # Garantindo conversão para string
                    ano = str(ano)

                    qualis = i.get('Qualis',{})
                    lista_pubqualis.append((autor, qualis))

            # **Print total count without year filter**
            total_count = len(lista_pubqualis)
            print(f'Total de publicações (sem filtro de ano): {total_count}')

            # Criar um DataFrame a partir da lista_pubqualis
            df_qualis_autores = pd.DataFrame(lista_pubqualis, columns=['Autor', 'Qualis'])
            
            # Concatenar os Qualis de cada autor em uma única string
            df_qualis_autores = df_qualis_autores.groupby('Autor')['Qualis'].agg(lambda x: ', '.join(x.astype(str))).reset_index()

            # Renomear a coluna Qualis para Publicações Qualis
            df_qualis_autores = df_qualis_autores.rename(columns={'Qualis': 'Publicações Qualis'})

            return df_qualis_autores

    def apurar_qualis_periodo(self, dict_list, ano_inicio, ano_final):
        lista_pubqualis = []
        for dic in dict_list:
            autor = dic.get('Identificação',{}).get('Nome',{})
            artigos = dic.get('Produções', {}).get('Artigos completos publicados em periódicos', {})
            for i in artigos:
                ano = i.get('ano',{})
                qualis = i.get('Qualis',{})
                lista_pubqualis.append((ano, autor, qualis))

        # Criar um DataFrame a partir da lista_pubqualis
        df_qualis_autores_anos = pd.DataFrame(lista_pubqualis, columns=['Ano','Autor', 'Qualis'])
                
        # Criar uma tabela pivot com base no DataFrame df_qualis_autores_anos
        pivot_table = df_qualis_autores_anos.pivot_table(index='Autor', columns='Ano', values='Qualis', aggfunc=lambda x: ', '.join(x))

        # Selecionar as colunas (anos) que estão dentro do intervalo de anos
        anos_interesse = [Ano for Ano in pivot_table.columns if Ano.isdigit() and ano_inicio <= int(Ano) <= ano_final and Ano != '']

        # Filtrar a tabela pivot pelos anos de interesse
        pivot_table_filtrada = pivot_table[anos_interesse]

        # Mostrar a tabela pivot filtrada
        return pivot_table_filtrada

    def apurar_publicacoes_periodo(self, dict_list, ano_inicio, ano_final):
        # Mapeamento de pontos por cada Estrato Qualis para PPGCS
        mapeamento_pontos = {
            'A1': 1,
            'A2': 1,
            'A3': 1,
            'A4': 1,
            'B1': 1,
            'B2': 1,
            'B3': 1,
            'B4': 1,
            'C': 1,
            'NA': 1
        }
        import pandas as pd
        lista_pubqualis = []
        for dic in dict_list:
            autor = dic.get('Identificação',{}).get('Nome',{})
            artigos = dic.get('Produções', {}).get('Artigos completos publicados em periódicos', {})
            for i in artigos:
                ano = i.get('ano',{})
                qualis = i.get('Qualis',{})
                lista_pubqualis.append((ano, autor, qualis))

        # Criar um DataFrame a partir da lista_pubqualis
        df_qualis_autores_anos = pd.DataFrame(lista_pubqualis, columns=['Ano','Autor', 'Qualis'])
        # df_qualis_autores_anos

        # Criar uma tabela pivot com base no DataFrame df_qualis_autores_anos
        pivot_table = df_qualis_autores_anos.pivot_table(index='Autor', columns='Ano', values='Qualis', aggfunc=lambda x: ', '.join(x))

        # Selecionar as colunas (anos) que estão dentro do intervalo de anos
        anos_interesse = [Ano for Ano in pivot_table.columns if Ano and ano_inicio <= int(Ano) <= ano_final]

        # Filtrar a tabela pivot pelos anos de interesse
        pivot_table_filtrada = pivot_table[anos_interesse]

        # Aplicar o mapeamento de pontos à tabela pivot filtrada apenas para valores do tipo str
        pivot_table_qte = pivot_table_filtrada.applymap(lambda x: sum(mapeamento_pontos[q] for q in x.split(', ') if q in mapeamento_pontos) if isinstance(x, str) else 0)

        # Adicionar uma coluna final com a soma dos pontos no período
        pivot_table_qte['Contagem'] = pivot_table_qte.sum(axis=1)

        # Ordenar a tabela pivot pela soma de pontos de forma decrescente
        pivot_table_qte_sorted = pivot_table_qte.sort_values(by='Contagem', ascending=False)

        # Mostrar a tabela pivot ordenada pela soma de pontos decrescente
        return pivot_table_qte_sorted
    
    def apurar_pontos_periodo(self, dict_list, ano_inicio, ano_final):
        # Mapeamento de pontos por cada Estrato Qualis para PPGCS
        mapeamento_pontos = {
            'A1': 90,
            'A2': 80,
            'A3': 60,
            'A4': 40,
            'B1': 20,
            'B2': 15,
            'B3': 10,
            'B4': 5,
            'C': 0,
            'NA': 0
        }
        import pandas as pd
        lista_pubqualis = []
        for dic in dict_list:
            autor = dic.get('Identificação',{}).get('Nome',{})
            artigos = dic.get('Produções', {}).get('Artigos completos publicados em periódicos', {})
            for i in artigos:
                ano = i.get('ano',{})
                qualis = i.get('Qualis',{})
                lista_pubqualis.append((ano, autor, qualis))

        # Criar um DataFrame a partir da lista_pubqualis
        df_qualis_autores_anos = pd.DataFrame(lista_pubqualis, columns=['Ano','Autor', 'Qualis'])
        # df_qualis_autores_anos

        # Criar uma tabela pivot com base no DataFrame df_qualis_autores_anos
        pivot_table = df_qualis_autores_anos.pivot_table(index='Autor', columns='Ano', values='Qualis', aggfunc=lambda x: ', '.join(x))

        # Selecionar as colunas (anos) que estão dentro do intervalo de anos
        anos_interesse = [Ano for Ano in pivot_table.columns if Ano and ano_inicio <= int(Ano) <= ano_final]

        # Filtrar a tabela pivot pelos anos de interesse
        pivot_table_filtrada = pivot_table[anos_interesse]

        # Aplicar o mapeamento de pontos à tabela pivot filtrada apenas para valores do tipo str
        pivot_table_pontos = pivot_table_filtrada.map(lambda x: sum(mapeamento_pontos[q] for q in x.split(', ') if q in mapeamento_pontos) if isinstance(x, str) else 0)

        # Adicionar uma coluna final com a soma dos pontos no período
        pivot_table_pontos['Soma de Pontos'] = pivot_table_pontos.sum(axis=1)

        # Ordenar a tabela pivot pela soma de pontos de forma decrescente
        pivot_table_pontos_sorted = pivot_table_pontos.sort_values(by='Soma de Pontos', ascending=False)

        # Mostrar a tabela pivot ordenada pela soma de pontos decrescente
        return pivot_table_pontos_sorted

    def obter_ano_por_doi(self, doi):
        """Obtém o ano de publicação de um artigo a partir do DOI.

        Args:
            doi: O DOI do artigo.

        Returns:
            int: O ano de publicação, ou None se não encontrado.
        """
        try:
            url = f"http://dx.doi.org/{doi}"
            response = requests.get(url)
            response.raise_for_status()  # Verifica se a requisição foi bem-sucedida

            soup = BeautifulSoup(response.content, 'html.parser')

            # Procura o ano na página (adapte o seletor CSS para o site específico)
            ano_elemento = soup.select_one('meta[name="citation_date"]')  # Exemplo para Springer
            if ano_elemento:
                return int(ano_elemento['content'].split('/')[0])  # Extrai o ano

            # Se não encontrar, tenta outros seletores ou sites (Crossref, etc.)
            # ...

        except requests.exceptions.RequestException as e:
            print(f"Erro ao obter ano por DOI: {e}")

    def nome_para_iniciais(self, nome):
        """Converte um nome completo em iniciais em maiúsculas, mantendo preposições e conjunções em minúsculas.

        Args:
            nome: O nome completo a ser convertido.

        Returns:
            str: O nome convertido em iniciais.
        """
        palavras = nome.split()
        iniciais = ""
        for palavra in palavras:
            if palavra.lower() not in ["de", "da", "do", "dos", "das", "e", "o", "a"]:
                iniciais += palavra[0].upper() + ". "
            else:
                iniciais += palavra.lower() + " "
        return iniciais.strip()

    def nome_iniciais_maiusculas(self, nome):
        """Converte um nome completo para maiúsculas, mantendo preposições e conjunções em minúsculas.

        Args:
            nome: O nome completo a ser convertido.

        Returns:
            str: O nome convertido com as primeiras letras em maiúsculas.
        """
        palavras = nome.split()
        nome_formatado = ""
        for palavra in palavras:
            if palavra.lower() not in ["de", "da", "do", "dos", "das", "e", "o", "a"]:
                nome_formatado += palavra.capitalize() + " "
            else:
                nome_formatado += palavra.lower() + " "
        return nome_formatado.strip()

    def buscar_palavras_chave_scielo(self, palavras_chave):
        """
        Busca trabalhos relacionados a uma lista de palavras-chave na SciELO.

        Args:
            palavras_chave (list): Uma lista de palavras-chave para a busca.

        Returns:
            list: Uma lista de tuplas (título, link) dos artigos encontrados.
        """

        # URL de busca da SciELO
        url_busca = "https://search.scielo.org/?"

        # https://search.scielo.org/?
        # q=Alzheimer+Diag*
        # &lang=en
        # &count=50
        # &from=0
        # &output=site
        # &sort=
        # &format=summary
        # &fb=
        # &page=1

        # Construir a query com as palavras-chave
        query = "+".join([f'"{palavra}"' for palavra in palavras_chave])
        params = {
            "q": query,
            "lang": "en",   # Remover a restrição de idioma
            "count": 100,    # Número de resultados por página
            "from": 0,
            "output": "site"
        }

        print(f"Query montada: {params}")

        # Definir headers com um User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3' 
        }

        # Fazer a requisição HTTP com headers
        response = requests.get(url_busca, params=params, headers=headers)

        # Extrair os títulos e links dos artigos
        soup = BeautifulSoup(response.content, 'html.parser')
        resultados = soup.find_all('div', class_='result')
        print(f"Resultados na base Scielo: {resultados}")
        artigos = []
        try:
            for resultado in resultados:
                titulo = resultado.find('a', class_='doc-title').text.strip()
                link = resultado.find('a', class_='doc-title')['href']
                artigos.append((titulo, link))
        except Exception as e:
            print(f"Erro ao tentar recuperar resultados:")
            print(e)
        return artigos

    def verificar_scielo_por_titulo_ano(self, titulo, ano):
        """
        Verifica se um artigo está indexado na SciELO usando web scraping (busca por título e ano).

        Args:
            titulo (str): O título do artigo.
            ano (int): O ano de publicação do artigo.

        Returns:
            bool: True se o artigo for encontrado na SciELO, False caso contrário.
        """

        # URL de busca da SciELO
        search_url = "https://search.scielo.org/?"
        query_params = {
            "q": titulo,
            # "lang": "pt",  # Para filtrar por linguagem, conforme necessário
            "count": 10,    # Número de resultados a serem buscados (ajuste conforme necessário)
            "from": 0,
            "output": "site"
        }

        # Definir headers e user-agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3' 
        }

        try:
            # Incluir os headers na requisição
            response = requests.get(search_url, params=query_params, headers=headers)
            response.raise_for_status()  # Lançar exceção se houver erro na requisição

            # Analisar os resultados da busca usando expressões regulares
            resultados = re.findall(r'<div class="result">.*?<a class="doc-title".*?>(.*?)</a>.*?<span class="doc-year">(.*?)</span>', response.text, re.DOTALL)
            for titulo_encontrado, ano_encontrado in resultados:
                if titulo_encontrado.lower().strip() == titulo.lower() and ano_encontrado.strip() == str(ano):
                    return True

        except requests.exceptions.RequestException as e:
            print(f"Erro na requisição à SciELO: {e}")

        return False

    def verificar_indexacao(self, doi=None, titulo=None, ano=None):
        """
        Verifica se um artigo está indexado nas bases de dados PubMed e SciELO.
        Adiciona validação do DOI com o título usando similaridade de Jaccard.

        Args:
            doi (str, optional): O DOI do artigo.
            titulo (str, optional): O título do artigo.
            ano (int, optional): O ano de publicação do artigo.

        Returns:
            tuple: Uma tupla com dois booleanos (PubMed, SciELO) indicando se o artigo está indexado em cada base de dados.
        """

        pubmed_indexado = False
        scielo_indexado = False
        doi_valido = True  # Assume que o DOI é válido por padrão

        if doi:
            # Verificar PubMed via Entrez
            pubmed_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={doi}&retmode=json"
            pubmed_response = requests.get(pubmed_url)
            pubmed_data = pubmed_response.json()
            # print(f"Dados PubMed\nDOI: {doi}\nQte resultados: {pubmed_data['esearchresult']['count']}")
            pubmed_indexado = pubmed_data['esearchresult']['count'] != '0'

        ## Fazer busca por dados detalhados usando API Crossref
        #     try:
        #         cr = Crossref()
        #         trabalho = cr.works(ids=doi)["message"]
        #         titulo_doi = trabalho['title'][0]
        #         print(f"Título: {titulo_doi}")
        #         print(f"\nDados Crossref")
        #         for i,j in trabalho.items():
        #             print(f"{i}: {j}")

        #         # Calcular a similaridade de Jaccard
        #         conjunto_titulo = set(titulo.lower().split())
        #         # print(conjunto_titulo)
        #         conjunto_titulo_doi = set(titulo_doi.lower().split())
        #         # print(conjunto_titulo_doi)
        #         intersecao = conjunto_titulo.intersection(conjunto_titulo_doi)
        #         uniao = conjunto_titulo.union(conjunto_titulo_doi)
        #         jaccard_similarity = len(intersecao) / len(uniao)

        #         doi_valido = jaccard_similarity > 0.9
        #         print(f"DOI corresponde ao título informado: {doi_valido}")

        #     except Exception as e:
        #         print(f"O Título informado não corresponde ao título do artigo do DOI informado")
        #         print(e)
        #         doi_valido = False  # Se houver erro ao obter o título, assume que o DOI não é válido

        #     # Verificar SciELO via Crossref (indiretamente)
        #     if doi_valido:
        #         try:
        #             if "link" in trabalho:
        #                 for link in trabalho["link"]:
        #                     if "scielo" in link["URL"]:
        #                         scielo_indexado = True
        #                         break
        #         except:
        #             pass # Se não encontrar via Crossref, assumir não indexado na SciELO

        elif titulo and ano:
            # Verificar PubMed via Entrez
            pubmed_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={titulo}[Title]+AND+{ano}[Publication Date]&retmode=json"
            pubmed_response = requests.get(pubmed_url)
            pubmed_data = pubmed_response.json()
            pubmed_indexado = pubmed_data['esearchresult']['count'] != '0'
        else:
            print('Verificar se o DOI, ou o título e o ano da publicação foram inseridas corretamente')

        if titulo and ano:
            # Verificar SciELO por título e ano
            scielo_indexado = self.verificar_scielo_por_titulo_ano(titulo, ano)
        else:
            print(f"Títlulo ou ano do arquivo indisponível impedindo buscar indexação na base scielo")

        # Invalidar resultados se o DOI não for válido
        if not doi_valido:
            pubmed_indexado = False
            scielo_indexado = False

        return pubmed_indexado, scielo_indexado

    def calcular_pontuacoes(self, dict_list_docents, ano_inicio, ano_final, orientadores_filtro=None):
        """Calcula as pontuações de produção e orientação para cada docente e retorna um DataFrame.

        Args:
            dict_list_docents: Lista de dicionários contendo dados dos docentes.
            ano_inicio: Ano inicial do período de análise.
            ano_final: Ano final do período de análise.

        Returns:
            DataFrame: Tabela com todos os docentes, ordenada pela pontuação total.
            DataFrame: Tabela filtrada apenas com os docentes da lista de orientadores, ordenada pela pontuação total.
         """
        pontuacoes = {}
        for m,docente in enumerate(dict_list_docents):
            clear_output(wait=True)
            print(f"Calculando pontuações no período {ano_inicio} a {ano_final} docente {m+1}/{len(dict_list_docents)}")
            nome = docente['Identificação']['Nome']
            pontuacoes[nome] = {
                'Somatório_Pontos': 0,
                'Pnt_Public.Artigos': 0,
                'Pnt_Livros/Capítulos': 0,
                'Pnt_Public.Congresso': 0,
                'Pnt_Orientações': 0,
                'Pnt_Patentes':0,
                'Pnt_Software':0,
                'Pnt_PubMed': 0,
                'Pnt_Scielo': 0,
            }

            # Cálculo de pontuação para artigos
            artigos = docente.get('Produções', {}).get('Artigos completos publicados em periódicos')
            if artigos:
                qte_jcr_faltante = 0
                qte_jcr_inferior = 0
                qte_jcr_superior = 0
                qte_index_pubmed = 0
                qte_index_scielo = 0
                is_pubmed = False
                is_scielo = False

                for n, artigo in enumerate(artigos):
                    try:
                        ano_artigo = artigo.get('ano')
                        if not ano_artigo or ano_artigo == '':  # Verifica se o ano está vazio ou não é um número
                            doi = artigo.get('DOI')
                            if doi:
                                ano_artigo = self.obter_ano_por_doi(doi)  # Consulta o ano pelo DOI
                                if ano_artigo is None:
                                    ano_artigo = ano_inicio  # Usa o ano inicial se não encontrar o ano

                        if ano_inicio <= int(ano_artigo) <= ano_final:
                            try:
                                # Cálculo de pontuações por indexação no PubMed e Scielo
                                doi = artigo.get('DOI')
                                titulo_artigo = artigo.get('titulo')
                                is_pubmed, is_scielo = self.verificar_indexacao(doi, titulo_artigo, ano_artigo)
                                if is_pubmed:
                                    qte_index_pubmed += 1
                                if is_scielo:
                                    qte_index_scielo += 1
                                print(f"Artigo {n+1:2} [ PubMed: {is_pubmed} | Scielo: {is_scielo} ] Acumulado Indexados: ({qte_index_pubmed} Pubmed | {qte_index_scielo} Scielo)")
                            except Exception as e:
                                print('Não foi possível consultar Scielo ou PubMed')
                                print(e)
                            try:
                                # Cálculo de pontuações por faixa de fator de impacto
                                impacto = float(artigo.get('fator_impacto_jcr', 0.0))  # Usar 0.0 se o fator de impacto não for encontrado
                                if impacto >= 1.1:
                                    qte_jcr_superior += 1
                                elif impacto >= 0.5:
                                    qte_jcr_inferior += 1
                            except:
                                qte_jcr_faltante += 1
                    except Exception as e:
                        print(f'Erro ao extrair artigo: {e}')
                        print(f'Artigo com problema: {artigo}')

                pontuacoes[nome]['Pnt_Artigos'] = qte_jcr_superior * 8 + qte_jcr_inferior * 6
                pontuacoes[nome]['Pnt_PubMed'] = qte_index_pubmed * 2
                pontuacoes[nome]['Pnt_Scielo'] = qte_index_scielo * 1

            else:
                pontuacoes[nome]['Pnt_Artigos'] = 0
                pontuacoes[nome]['Pnt_PubMed'] = 0
                pontuacoes[nome]['Pnt_Scielo'] = 0
                
            # Subdicionários dos dicionários de produções, orientações e patentes
            livros_publicados = docente.get('Produções', {}).get('Livros publicados/organizados ou edições')
            capitulos_livros = docente.get('Produções', {}).get('Capítulos de livros publicados')
           
            # Cálculo de pontuação para patentes e softwares
            patentes = docente.get('Patentes e registros', {}).get('Patente')
            softwares = docente.get('Patentes e registros', {}).get('Programa de computador')
            marcas = docente.get('Patentes e registros', {}).get('Marca registrada')

            if patentes:
                qte_patentes_concedidas = 0
                qte_patentes_depositadas = 0
                for patente in patentes.values():
                    texto = patente.get('texto')
                    ano_deposito = patente.get('ano')

                    # Encontrar todas as datas no formato dd/mm/aaaa
                    datas_encontradas = re.findall(r'\d{2}/\d{2}/\d{4}', texto)
                    if datas_encontradas:
                        ano_concessao = int(datas_encontradas[-1].split('/')[-1])  # Pegar o ano da última data
                        if ano_inicio <= ano_concessao <= ano_final:
                            qte_patentes_concedidas += 1
                    elif ano_deposito and ano_inicio <= ano_deposito <= ano_final:  # Se não houver concessão, usa o ano de depósito
                        qte_patentes_depositadas += 1
                
                pontuacoes[nome]['Pnt_Patentes'] = qte_patentes_concedidas * 4 + qte_patentes_depositadas * 3

            if softwares:
                qte_software = 0
                for software in softwares.values():
                    ano = software.get('ano')
                    if ano and ano_inicio <= int(ano) <= ano_final:
                        qte_software += 1
                pontuacoes[nome]['Pnt_Software'] = qte_software * 2  # Adiciona a pontuação de software

            # Cálculo de pontuação para livros e capítulos (COM FILTRO POR ANO)
            qte_livros = 0
            qte_capitulos = 0
            if livros_publicados:
                for livro in livros_publicados:
                    ano_match = re.search(r'\b(\d{4})\b', livro)
                    if ano_match and ano_inicio <= int(ano_match.group(1)) <= ano_final:
                        qte_livros += 1
            
            if capitulos_livros:
                for capitulo in capitulos_livros:
                    ano_match = re.search(r'\b(\d{4})\b', capitulo)
                    if ano_match and ano_inicio <= int(ano_match.group(1)) <= ano_final:
                        qte_capitulos += 1

            pontuacoes[nome]['Pnt_Livros/Capítulos'] = qte_livros * 3 + qte_capitulos * 2

            # Cálculo de pontuação para trabalhos em congressos
            trabalho_congresso = docente.get('Produções', {}).get('Trabalhos completos publicados em anais de congressos')
            if trabalho_congresso:
                qte_trabcongresso = 0
                for _, valor in trabalho_congresso.items():
                    ano_match = re.search(r'\b(\d{4})\b', valor)
                    if ano_match and ano_inicio <= int(ano_match.group(1)) <= ano_final:
                        qte_trabcongresso += 1
                pontuacoes[nome]['Pnt_Public.Congressos'] = min(qte_trabcongresso, 5) * 2  # Limitar a 10 pontos
            else:
                pontuacoes[nome]['Pnt_Public.Congressos'] = 0  # Define a pontuação como 0 se não houver trabalhos

            # Cálculo de pontuação para orientações
            concluidas = docente.get('Orientações', {}).get('Orientações e supervisões concluídas')
            if concluidas:
                for tipo_dict in concluidas:
                    for tipo, dados in tipo_dict.items():
                        for _, orientacao in dados.items():
                            ano_match = re.search(r'\b(\d{4})\b', orientacao)
                            if ano_match and ano_inicio <= int(ano_match.group(1)) <= ano_final:
                                if tipo == 'Iniciação científica':
                                    pontuacoes[nome]['Pnt_Orientações'] += 1
                                elif tipo in ['Trabalho de conclusão de curso de graduação', 'Monografia de conclusão de curso de aperfeiçoamento/especialização']:
                                    pontuacoes[nome]['Pnt_Orientações'] += 1
                                    if pontuacoes[nome]['Pnt_Orientações'] > 10:
                                        pontuacoes[nome]['Pnt_Orientações'] = 10  # Limitar a 10 pontos
                                elif tipo == 'Dissertação de mestrado':
                                    pontuacoes[nome]['Pnt_Orientações'] += 2
                                elif tipo == 'Tese de doutorado':
                                    pontuacoes[nome]['Pnt_Orientações'] += 2

            # Somar todos os subtotais para obter o total
            pontuacoes[nome]['Somatório_Pontos'] = sum(pontuacoes[nome].values())

        # Criar DataFrame a partir do dicionário
        df_pontuacoes = pd.DataFrame(pontuacoes).T

        # Reordenar colunas para que o Total fique em primeiro lugar
        df_pontuacoes = df_pontuacoes[['Somatório_Pontos', 'Pnt_Artigos', 'Pnt_Orientações', 'Pnt_Public.Congressos', 'Pnt_Livros/Capítulos', 'Pnt_Patentes', 'Pnt_Software']]

        # Ordenar o DataFrame em ordem decrescente pelo total de pontos
        df_pontuacoes_ordenado = df_pontuacoes.sort_values(by='Somatório_Pontos', ascending=False)

        # Normalizar os nomes dos orientadores no DataFrame
        df_pontuacoes_ordenado.index = df_pontuacoes_ordenado.index.map(self.normalizar_nome)

        # Filtrar por orientadores, se a lista for fornecida
        if orientadores_filtro:
            orientadores_filtro_norm = [self.normalizar_nome(nome) for nome in orientadores_filtro]
            df_filtrado = df_pontuacoes_ordenado.loc[df_pontuacoes_ordenado.index.isin(orientadores_filtro_norm)]
        else:
            df_filtrado = df_pontuacoes_ordenado.copy()  # Se não houver filtro, retorna o DataFrame completo

        # Converter nomes para iniciais no DataFrame geral
        df_pontuacoes_ordenado.index = df_pontuacoes_ordenado.index.map(self.nome_iniciais_maiusculas)
        df_filtrado.index = df_filtrado.index.map(self.nome_iniciais_maiusculas)

        return df_pontuacoes_ordenado, df_filtrado

    def avaliar_tipo_instancias_arvore(self, estrutura_dados, nivel=1, identacao=""):
        """
        Função recursiva que avalia o tipo e a quantidade de instâncias em cada nível da estrutura de dados, exibindo-a em formato de árvore.

        Args:
            estrutura_dados (list|dict): A estrutura de dados a ser avaliada.
            nivel (int): Nível atual da recurssão (inicia em 1).
            identacao (str): String de indentação para cada nível (inicia vazia).

        Returns:
            None: A função imprime os resultados na tela e não retorna nada.
        """

        if isinstance(estrutura_dados, list):
            print(f"{identacao}N{nivel}. Lista: {len(estrutura_dados)} elementos")
            for item in estrutura_dados:
                avaliar_tipo_instancias_arvore(item, nivel + 1, identacao + "    ")

        elif isinstance(estrutura_dados, dict):
            print(f"{identacao}N{nivel}. Mapa: {estrutura_dados.keys()}")
            for chave, valor in estrutura_dados.items():
                print(f"{identacao}  {chave}:")
                avaliar_tipo_instancias_arvore(valor, nivel + 1, identacao + "    ")

        elif isinstance(estrutura_dados, str):
            print(f"{identacao}N{nivel}. String: {estrutura_dados}")

    # ESTRATÉGIAS ANTIGAS NÃO MAIS UTILIZADAS

    # # Função para apurar pontos de publicações (correção completa)
    # def apurar_jcr_publicacoes(self, dict_list, keylevel_one, keylevel_two, data_measure, class_mapping, year_ini, year_end):
    #     # Dicionário para armazenar os resultados
    #     dict_points = {}

    #     # Expressão regular para extrair valores numéricos da chave (corrigida)
    #     pattern = re.compile(r'(\d+)p \((?:(.*?) < )?JCR (<= |> )(\d+(?:\.\d+)?)\)')

    #     # Converter as chaves do dicionário de mapeamento de pontuação para o formato desejado
    #     class_mapping_parsed = {}
    #     for key, value in class_mapping.items():
    #         match = pattern.match(key)
    #         if match:
    #             pontos = int(match.group(1))
    #             if match.group(2) is None:  # Caso "JCR > x"
    #                 limites = (float(match.group(4)), float('inf'))
    #             else:  # Caso "x < JCR <= y"
    #                 limites = (float(match.group(2)), float(match.group(4)))
    #             class_mapping_parsed[key] = (pontos, limites)
    #         else:  # Caso "0p (sem JCR)"
    #             pass

    #     for curriculum in dict_list:  # Iterar sobre os currículos
    #         id_lattes = curriculum.get('Identificação', {}).get('ID Lattes')
    #         name = curriculum.get('Identificação', {}).get('Nome')

    #         # Inicializar dicionário para o pesquisador atual
    #         dict_points[name] = {
    #             'Total Pontos': 0,
    #             **{k: [0, 0] for k in class_mapping_parsed.keys()}
    #         }

    #         try:
    #             publicacoes = curriculum[keylevel_one][keylevel_two]
    #             for publicacao in publicacoes:
    #                 try:
    #                     ano_public = int(publicacao.get('ano'))
    #                 except:
    #                     ano_public = 2021

    #                 # Filtrar para apurar somente publicações dentro do período em análise
    #                 if ano_public >= year_ini and ano_public <= year_end:
    #                     for jcr_value in publicacao[data_measure]:
    #                         try:
    #                             jcr_value = float(jcr_value.strip('.'))
    #                         except:
    #                             jcr_value = 0.0

    #                         for faixa, (pontos_faixa, (faixa_min, faixa_max)) in class_mapping_parsed.items():
    #                             if faixa_min < jcr_value <= faixa_max:
    #                                 dict_points[name][faixa][0] += 1
    #                                 dict_points[name][faixa][1] += pontos_faixa
    #                                 dict_points[name]['Total Pontos'] += pontos_faixa
    #                                 break

    #         except (KeyError, TypeError) as e:
    #             dict_points[name]['Total Pontos'] += 0

    #     # Criar DataFrame e renomear colunas
    #     df_resultado = pd.DataFrame(dict_points).T

    #     # Separar quantidade e soma de pontos
    #     df_qtd = df_resultado[[col for col in df_resultado.columns if isinstance(df_resultado[col][0], list)]].applymap(lambda x: x[0])
    #     df_soma = df_resultado[[col for col in df_resultado.columns if isinstance(df_resultado[col][0], list)]].applymap(lambda x: x[1])

    #     # Renomear colunas
    #     df_qtd = df_qtd.rename(columns={col: f"Quantidade_{col}" for col in df_qtd.columns})
    #     df_soma = df_soma.rename(columns={col: f"Soma_{col}" for col in df_soma.columns})

    #     # Concatenar DataFrames
    #     df_resultado = pd.concat([df_qtd, df_soma, df_resultado['Total Pontos']], axis=1)

    #     # Remover tudo após 6p ou 8p nos rótulos das colunas
    #     df_resultado = df_resultado.rename(columns=lambda x: re.sub(r'(0p|6p|8p).*', r'\1', x))

    #     return dict_points, df_resultado  # Retornar o dicionário com os resultados

    # def apurar_jcr_orientadores(self, df_resultado, lista_orientadores):
    #     # scraper = LattesScraper()
    #     orient_norm = [self.normalizar_nome(x) for x in lista_orientadores]

    #     # Normalizar os nomes no DataFrame
    #     df_resultado.index = df_resultado.index.map(self.normalizar_nome)

    #     # Normalizar os nomes na lista de filtro
    #     orient_norm = [self.normalizar_nome(nome) for nome in lista_orientadores]

    #     # Filtrar o DataFrame pelos nomes na lista (usando nomes normalizados)
    #     df_orientadores = df_resultado.loc[orient_norm]

    #     # Ordenar o DataFrame filtrado pela coluna "Total Pontos" em ordem decrescente
    #     df_orientadores = df_orientadores.sort_values(by='Total Pontos', ascending=False)

    #     # retornar o DataFrame filtrado
    #     return df_orientadores

    # ## Formato de dataframes de saída correto, porém está repetindo dados de diferentes tipos de orientação indevidamente
    # def gerar_tabela_pontuacao_old(self, dict_list_docents, pontos_ic, ano_inicio, ano_final):
    #     """
    #     Gera uma tabela com a pontuação de orientações concluídas por ano para cada orientador,
    #     considerando um valor por orientação e filtrando por intervalo de anos, incluindo o ano de início e fim.

    #     Args:
    #         dict_list_docents: Lista de dicionários contendo dados de docentes.
    #         pontos_ic: Pontos a serem atribuídos por orientação concluída.
    #         ano_inicio: Ano inicial do intervalo a ser considerado (inclusive).
    #         ano_final: Ano final do intervalo a ser considerado (inclusive).

    #     Returns:
    #         DataFrame: Tabela com orientadores como índices, anos como colunas em ordem crescente e uma coluna de somatório na primeira coluna.
    #     """
    #     orientacoes_por_orientador = {}

    #     for docente in dict_list_docents:
    #         orientacoes = docente.get('Orientações', {}).get('Orientações e supervisões concluídas')
    #         if orientacoes:
    #             for tipo in orientacoes:
    #                 qic = tipo.get('Iniciação científica')
    #                 if qic:
    #                     for orientacao in qic.values():
    #                         # Extrair ano da orientação
    #                         ano_match = re.search(r'\b(\d{4})\b', orientacao)
    #                         if ano_match:
    #                             ano = int(ano_match.group(1))

    #                             # Filtrar por intervalo de anos (inclusive ano_inicio e ano_final)
    #                             if ano_inicio <= ano <= ano_final:
    #                                 # Extrair nome do orientador após "Orientador:"
    #                                 orientador_match = re.search(r'Orientador:\s*(.*)', orientacao)
    #                                 if orientador_match:
    #                                     orientador = orientador_match.group(1).strip().strip('.')

    #                                     # Adicionar ao dicionário
    #                                     if orientador not in orientacoes_por_orientador:
    #                                         orientacoes_por_orientador[orientador] = {}
    #                                     orientacoes_por_orientador[orientador][ano] = orientacoes_por_orientador[orientador].get(ano, 0) + 1

    #     # Criar DataFrame a partir do dicionário, preenchendo anos faltantes com zero
    #     df = pd.DataFrame(orientacoes_por_orientador).fillna(0)

    #     # Multiplicar valores por pontos_ic
    #     df = df * pontos_ic

    #     # Transpor o DataFrame para ter os anos como colunas
    #     df = df.T

    #     # Verificar se os anos existem como colunas no DataFrame (após a transposição)
    #     anos_existentes = set(df.columns)
    #     anos_intervalo = set(range(ano_inicio, ano_final + 1))
    #     anos_a_somar = sorted(list(anos_existentes.intersection(anos_intervalo)))
        
    #     # Reordenar colunas para que anos fiquem em ordem crescente
    #     df = df.reindex(sorted(df.columns), axis=1)

    #     # Calcular o somatório dos valores dentro do período para cada orientador, somente se houver anos no intervalo
    #     if anos_a_somar:
    #         df.insert(0, 'Somatório', df[anos_a_somar].sum(axis=1))  # Calcula a soma para cada linha (orientador)
    #         # Converter a coluna 'Somatório' para string
    #         df['Somatório'] = df['Somatório'].astype(str)

    #     return df

    # ## Formato de dataframes de saída correto, porém está repetindo dados de diferentes tipos de orientação indevidamente
    # def apurar_orientacoes_old(self, dict_list_docents, ano_inicio, ano_final):
    #     """
    #     Gera uma tabela com a pontuação de orientações concluídas por ano para cada orientador,
    #     considerando um valor por orientação e filtrando por intervalo de anos, incluindo o ano de início e fim.

    #     Args:
    #         dict_list_docents: Lista de dicionários contendo dados de docentes.
    #         tipos: Dicionário com os tipos de orientação e seus respectivos pesos.
    #         ano_inicio: Ano inicial do intervalo a ser considerado (inclusive).
    #         ano_final: Ano final do intervalo a ser considerado (inclusive).

    #     Returns:
    #         DataFrame: Tabela com orientadores como índices, anos como colunas em ordem crescente,
    #                 colunas de somatório por tipo de orientação e uma coluna de somatório total.
    #     """
    #     tipos = {'Iniciação científica': 1,
    #             #  'Monografia de conclusão de curso de aperfeiçoamento/especialização': 1,
    #             #  'Monografias de conclusão de curso de aperfeiçoamento/especialização': 1,
    #             'Trabalho de conclusão de curso de graduação': 1,
    #             'Dissertação de mestrado': 2,
    #             'Tese de doutorado': 2,
    #             #  'Supervisão de pós-doutorado': 0,
    #             #  'Orientações de outra natureza': 0
    #             }

    #     siglas = {'Iniciação científica': 'IC',
    #             'Monografia de conclusão de curso de aperfeiçoamento/especialização': 'Grad',
    #             'Monografias de conclusão de curso de aperfeiçoamento/especialização': 'Grad',
    #             'Trabalho de conclusão de curso de graduação': 'Grad',
    #             'Dissertação de mestrado': 'Mest',
    #             'Tese de doutorado': 'Dout',
    #             'Supervisão de pós-doutorado': 'PosDoc',
    #             'Orientações de outra natureza': 'Outras'
    #             }    
        
    #     dfs_por_tipo = {}  # Dicionário para armazenar DataFrames por tipo de orientação
    #     colunas_convertidas = False # Variável para controlar a conversão

    #     for tipo_orientacao, pontos_orientacao in tipos.items():
    #         df_tipo = self.gerar_tabela_pontuacao_old(dict_list_docents, pontos_orientacao, ano_inicio, ano_final)
    #         sigla = siglas.get(tipo_orientacao, tipo_orientacao)  # Se não houver sigla, usa o nome completo
    #         df_tipo = df_tipo.rename(columns={'Somatório': f'Pts_Orient_{sigla}'})
    #         dfs_por_tipo[tipo_orientacao] = df_tipo

    #     # Combinar DataFrames de todos os tipos de orientação
    #     df = pd.concat(dfs_por_tipo.values(), axis=1)

    #     # Verificar se os anos existem como colunas no DataFrame
    #     anos_existentes = set(df.columns)
    #     anos_intervalo = set(range(ano_inicio, ano_final + 1))
    #     anos_a_somar = sorted(list(anos_existentes.intersection(anos_intervalo)))

    #     # Converter colunas de anos para inteiro para ordenação correta, somente se ainda não foram convertidas e se forem numéricas
    #     if not colunas_convertidas:
    #         df.columns = [int(col) if isinstance(col, str) and col.isnumeric() else col for col in df.columns]
    #         colunas_convertidas = True

    #     # Calcular somatório total (considerando todas as colunas, mesmo as de string)
    #     if anos_a_somar:
    #         df.insert(0, 'Somatório_Total', df.select_dtypes(include=['number']).sum(axis=1))  # Soma apenas colunas numéricas
    #         df['Somatório_Total'] = df['Somatório_Total'].astype(str)

    #     tabela_pivot = df.T

    #     # Converter índices de volta para string para a filtragem
    #     tabela_pivot.index = tabela_pivot.index.astype(str)    

    #     # Filtrar linhas de subtotais e total geral
    #     linhas_a_manter = [index for index in tabela_pivot.index if index.startswith('Pts_Orient_') or index == 'Somatório_Total']
    #     tabela_filtrada = tabela_pivot.loc[linhas_a_manter]

    #     return tabela_filtrada.T, tabela_pivot.T

    # def gerar_tabela_pontuacao_new(self, dict_list_docents, pontos_ic, ano_inicio, ano_final):
    #     orientacoes_por_orientador = {}
    #     anos_intervalo = list(range(ano_inicio, ano_final + 1))

    #     for docente in dict_list_docents:
    #         nome_orientador = docente.get('Identificação', {}).get('Nome')
    #         orientacoes = docente.get('Orientações', {}).get('Orientações e supervisões concluídas', [])

    #         if orientacoes:
    #             for tipo_dict in orientacoes:  # Itera sobre a lista de dicionários de tipos de orientação
    #                 tipo = list(tipo_dict.keys())[0]  # Extrai a chave (tipo de orientação) do dicionário
    #                 dados_orientacao = tipo_dict.get(tipo, {})  # Obtém os dados da orientação para o tipo específico

    #                 if tipo == pontos_ic[0]:  # Filtrar pelo tipo de orientação específico
    #                     for _, orientacao in dados_orientacao.items():
    #                         ano_match = re.search(r'\b(\d{4})\b', orientacao)
    #                         if ano_match:
    #                             ano = int(ano_match.group(1))
    #                             if ano_inicio <= ano <= ano_final:
    #                                 orientador_match = re.search(r'Orientador:\s*(.*)', orientacao)
    #                                 if orientador_match:
    #                                     orientador = orientador_match.group(1).strip().strip('.')
    #                                 else:
    #                                     try:
    #                                         orientador = nome_orientador
    #                                     except:
    #                                         print(f"Aviso: Não foi possível extrair o nome do orientador da orientação: {orientacao}")

    #                                 if orientador not in orientacoes_por_orientador:
    #                                     orientacoes_por_orientador[orientador] = {ano: 0 for ano in anos_intervalo}

    #                                 orientacoes_por_orientador[orientador][ano] += 1

    #     # Criar DataFrame a partir do dicionário, preenchendo anos faltantes com zero
    #     df = pd.DataFrame(orientacoes_por_orientador).fillna(0).T  # Transpor o DataFrame

    #     # Garantir que todas as colunas (anos) estejam presentes
    #     for ano in anos_intervalo:
    #         if ano not in df.columns:
    #             df[ano] = 0

    #     # Multiplicar valores por pontos_ic SOMENTE NAS COLUNAS NUMÉRICAS
    #     for col in df.select_dtypes(include=['number']).columns:
    #         df[col] = df[col] * pontos_ic[1]

    #     return df

    # def apurar_orientacoes_new(self, dict_list_docents, ano_inicio, ano_final, orientadores_filtro=None):
    #     """
    #     Gera uma tabela com a pontuação de orientações concluídas por ano para cada orientador,
    #     considerando um valor por orientação e filtrando por intervalo de anos, incluindo o ano de início e fim.

    #     Args:
    #         dict_list_docents: Lista de dicionários contendo dados de docentes.
    #         ano_inicio: Ano inicial do intervalo a ser considerado (inclusive).
    #         ano_final: Ano final do intervalo a ser considerado (inclusive).
    #         orientadores_filtro: Lista opcional de nomes de orientadores para filtrar os resultados.

    #     Returns:
    #         DataFrame: Tabela com orientadores como colunas, tipos de orientação e somatório total como linhas.
    #         DataFrame: Tabela com orientadores como índices, anos como colunas, e valores de orientações por tipo e ano.
    #     """
    #     tipos = {
    #         'Iniciação científica': 1,
    #         'Trabalho de conclusão de curso de graduação': 1,
    #         'Dissertação de mestrado': 2,
    #         'Tese de doutorado': 2
    #     }

    #     siglas = {
    #         'Iniciação científica': 'IC',
    #         'Trabalho de conclusão de curso de graduação': 'Grad',
    #         'Dissertação de mestrado': 'Mest',
    #         'Tese de doutorado': 'Dout'
    #     }

    #     dfs_por_tipo = {}
    #     colunas_convertidas = False
    #     anos_existentes = set()
    #     anos_intervalo = set(range(ano_inicio, ano_final + 1))

    #     for tipo_orientacao, pontos_orientacao in tipos.items():
    #         df_tipo = self.gerar_tabela_pontuacao_new(dict_list_docents, (tipo_orientacao, pontos_orientacao), ano_inicio, ano_final)
    #         sigla = siglas.get(tipo_orientacao, tipo_orientacao)
    #         df_tipo = df_tipo.rename(columns={c: f'Pts_Orient_{sigla}_{c}' for c in df_tipo.columns})
    #         # print(df_tipo.columns)
    #         # Calcular somatório por tipo de orientação
    #         df_tipo[f'Pts_Orient_{sigla}'] = df_tipo.sum(axis=1)
    #         dfs_por_tipo[f'Pts_Orient_{sigla}'] = df_tipo
    #         anos_existentes.update(df_tipo.columns)

    #     # Combinar DataFrames de todos os tipos de orientação
    #     df = pd.concat(dfs_por_tipo.values(), axis=1)

    #     # Normalizar os nomes dos orientadores no DataFrame
    #     df.index = df.index.map(self.normalizar_nome)

    #     # Filtrar por orientadores, se a lista for fornecida
    #     if orientadores_filtro:
    #         orientadores_filtro = [self.normalizar_nome(nome) for nome in orientadores_filtro]
    #         df = df.reindex(orientadores_filtro, fill_value=0)  # Reindexar e preencher com 0

    #     # Converter colunas de anos para inteiro para ordenação correta, somente se ainda não foram convertidas e se forem numéricas
    #     if not colunas_convertidas:
    #         df.columns = [int(col.split('_')[-1]) if isinstance(col, str) and col.split('_')[-1].isnumeric() else col for col in df.columns]
    #         colunas_convertidas = True

    #     # Calcular anos a somar após a conversão para numérico
    #     anos_a_somar = [col for col in df.columns if isinstance(col, int) and ano_inicio <= col <= ano_final]
    #     # print(anos_a_somar)

    #     # Calcular somatório total DEPOIS de calcular somatório por tipo de orientação (considerando apenas colunas numéricas)
    #     if anos_a_somar:
    #         df.insert(0, 'Somatório_Total', df.select_dtypes(include=['number']).sum(axis=1))  # Soma apenas colunas numéricas
    #         df['Somatório_Total'] = df['Somatório_Total'].astype(str)

    #     # Agrupar por nome do orientador e somar os valores das colunas de anos
    #     df_agrupado = df.groupby(df.index).sum()
    #     # print(df_agrupado.columns)

    #     # Converter colunas de anos para inteiro para ordenação correta
    #     df_agrupado.columns = [int(col) if str(col).isdigit() else col for col in df_agrupado.columns]

    #     # Filtrar colunas de subtotais e total geral DEPOIS de converter colunas para inteiro
    #     colunas_a_manter = [col for col in df_agrupado.columns if (isinstance(col, str) and col.startswith('Pts_Orient_')) or col == 'Somatório_Total']
    #     # print(colunas_a_manter)
    #     df_filtrado = df_agrupado[colunas_a_manter]

    #     # Transpor o DataFrame DEPOIS de agrupar E filtrar
    #     tabela_filtrada = df_filtrado.T

    #     # Renomear as colunas para remover o prefixo "Pts_Orient_"
    #     tabela_filtrada = tabela_filtrada.rename(columns=lambda x: x.replace('Pts_Orient_', ''))

    #     # Renomear a linha do somatório total
    #     tabela_filtrada = tabela_filtrada.rename(index={'Somatório_Total': 'Soma_Total_Geral'})

    #     return tabela_filtrada.T, df_agrupado


    # def gerar_tabela_pontuacao(self, dict_list_docents, pontos_ic, ano_inicio, ano_final):
    #     orientacoes_por_orientador = {}
    #     anos_intervalo = list(range(ano_inicio, ano_final + 1))

    #     for docente in dict_list_docents:
    #         nome_orientador = docente.get('Identificação', {}).get('Nome')
    #         orientacoes = docente.get('Orientações', {}).get('Orientações e supervisões concluídas', [])

    #         # print(f"Docente: {nome_orientador}")  # DEBUG
    #         # print(f"Orientacoes: {orientacoes}")  # DEBUG

    #         if orientacoes:
    #             for tipo_dict in orientacoes:  # Itera sobre a lista de dicionários de tipos de orientação
    #                 for tipo, dados_orientacao in tipo_dict.items():  # Itera sobre os tipos de orientação e seus dados
    #                     # print(f"Tipo: {tipo}")  # DEBUG
    #                     # print(f"Dados da orientação: {dados_orientacao}")  # DEBUG

    #                     if tipo == pontos_ic[0]:  # Filtrar pelo tipo de orientação específico
    #                         for _, orientacao in dados_orientacao.items():
    #                             ano_match = re.search(r'\b(\d{4})\b', orientacao)
    #                             if ano_match:
    #                                 ano = int(ano_match.group(1))
    #                                 # print(f"Ano: {ano}")  # DEBUG
    #                                 if ano_inicio <= ano <= ano_final:
    #                                     orientador_match = re.search(r'Orientador:\s*(.*)', orientacao)
    #                                     if orientador_match:
    #                                         orientador = orientador_match.group(1).strip().strip('.')
    #                                         # print(f"Orientador: {orientador}")  # DEBUG

    #                                         if orientador not in orientacoes_por_orientador:
    #                                             orientacoes_por_orientador[orientador] = {ano: 0 for ano in anos_intervalo}

    #                                         orientacoes_por_orientador[orientador][ano] += 1

    #     # Criar DataFrame a partir do dicionário, preenchendo anos faltantes com zero
    #     df = pd.DataFrame(orientacoes_por_orientador).fillna(0).T  # Transpor o DataFrame

    #     # print(f"DataFrame antes da multiplicação: \n{df}")  # DEBUG

    #     # Garantir que todas as colunas (anos) estejam presentes
    #     for ano in anos_intervalo:
    #         if ano not in df.columns:
    #             df[ano] = 0

    #     # Multiplicar valores por pontos_ic SOMENTE NAS COLUNAS NUMÉRICAS
    #     for col in df.select_dtypes(include=['number']).columns:
    #         df[col] = df[col] * pontos_ic[1]

    #     # print(f"DataFrame final: \n{df}")  # DEBUG

    #     return df

    # def apurar_orientacoes(self, dict_list_docents, ano_inicio, ano_final):
    #     tipos = {'Iniciação científica': 1,
    #              'Trabalho de conclusão de curso de graduação': 1,
    #              'Dissertação de mestrado': 2,
    #              'Tese de doutorado': 2}

    #     siglas = {'Iniciação científica': 'IC',
    #               'Trabalho de conclusão de curso de graduação': 'Grad',
    #               'Dissertação de mestrado': 'Mest',
    #               'Tese de doutorado': 'Dout'}

    #     dfs_por_tipo = {}
    #     colunas_convertidas = False

    #     # Verificar se os anos existem como colunas no DataFrame
    #     anos_existentes = set()
    #     anos_intervalo = set(range(ano_inicio, ano_final + 1))

    #     for tipo_orientacao, pontos_orientacao in tipos.items():
    #         df_tipo = self.gerar_tabela_pontuacao_old(dict_list_docents, pontos_orientacao, ano_inicio, ano_final)
    #         sigla = siglas.get(tipo_orientacao, tipo_orientacao)  # Se não houver sigla, usa o nome completo
    #         df_tipo = df_tipo.rename(columns={'Somatório': f'Pts_Orient_{sigla}'})
    #         dfs_por_tipo[tipo_orientacao] = df_tipo

    #     for tipo_orientacao, pontos_orientacao in tipos.items():
    #         df_tipo = self.gerar_tabela_pontuacao(dict_list_docents, (tipo_orientacao, pontos_orientacao), ano_inicio, ano_final)
    #         sigla = siglas.get(tipo_orientacao, tipo_orientacao)
    #         df_tipo = df_tipo.rename(columns={c: f'Pts_Orient_{sigla}_{c}' for c in df_tipo.columns})
    #         dfs_por_tipo[tipo_orientacao] = df_tipo
    #         anos_existentes.update(df_tipo.columns)

    #     anos_a_somar = sorted(list(anos_existentes.intersection(anos_intervalo)))

    #     # Combinar DataFrames de todos os tipos de orientação
    #     df = pd.concat(dfs_por_tipo.values(), axis=1)

    #     # Verificar se os anos existem como colunas no DataFrame
    #     anos_existentes = set(df.columns)
    #     anos_intervalo = set(range(ano_inicio, ano_final + 1))
    #     anos_a_somar = sorted(list(anos_existentes.intersection(anos_intervalo)))

    #     # Converter colunas de anos para inteiro para ordenação correta, somente se ainda não foram convertidas e se forem numéricas
    #     if not colunas_convertidas:
    #         df.columns = [int(col) if isinstance(col, str) and col.isnumeric() else col for col in df.columns]
    #         colunas_convertidas = True

    #     # Calcular somatório total (considerando todas as colunas, mesmo as de string)
    #     if anos_a_somar:
    #         df.insert(0, 'Somatório_Total', df.select_dtypes(include=['number']).sum(axis=1))  # Soma apenas colunas numéricas
    #         df['Somatório_Total'] = df['Somatório_Total'].astype(str)

    #     tabela_pivot = df.T

    #     # Converter índices de volta para string para a filtragem
    #     tabela_pivot.index = tabela_pivot.index.astype(str)    
    #     # print(tabela_pivot.index)

    #     # Filtrar linhas de subtotais e total geral
    #     linhas_a_manter = [index for index in tabela_pivot.index if index.startswith('Pts_Orient_') or index == 'Somatório_Total']
    #     tabela_filtrada = tabela_pivot.loc[linhas_a_manter]

    #     return tabela_filtrada.T, tabela_pivot.T

    # def apurar_orientacoes_orientadores(self, df_resultado, lista_orientadores):
    #     # scraper = LattesScraper()
    #     orient_norm = [self.normalizar_nome(x) for x in lista_orientadores]

    #     # Normalizar os nomes no DataFrame
    #     df_resultado.index = df_resultado.index.map(self.normalizar_nome)

    #     # Normalizar os nomes na lista de filtro
    #     orient_norm = [self.normalizar_nome(nome) for nome in lista_orientadores]

    #     # Filtrar o DataFrame pelos nomes na lista (usando nomes normalizados)
    #     df_orientadores = df_resultado.loc[orient_norm]

    #     # Ordenar o DataFrame filtrado pela coluna "Total Pontos" em ordem decrescente
    #     df_orientadores = df_orientadores.sort_values(by='Total Pontos', ascending=False)

    #     # retornar o DataFrame filtrado
    #     return df_orientadores

    # def apurar_orientacoes_ic(self, dict_list_docents, pontos_ic, ano_inicio, ano_final):
    #     """
    #     Gera uma tabela com a pontuação de orientações concluídas por ano para cada orientador,
    #     considerando um valor por orientação e filtrando por intervalo de anos, incluindo o ano de início e fim.

    #     Args:
    #         dict_list_docents: Lista de dicionários contendo dados de docentes.
    #         pontos_ic: Pontos a serem atribuídos por orientação concluída.
    #         ano_inicio: Ano inicial do intervalo a ser considerado (inclusive).
    #         ano_final: Ano final do intervalo a ser considerado (inclusive).

    #     Returns:
    #         DataFrame: Tabela com orientadores como índices, anos como colunas em ordem crescente e uma coluna de somatório na primeira coluna.
    #     """
    #     orientacoes_por_orientador = {}
    #     colunas_convertidas = False

    #     for docente in dict_list_docents:
    #         orientacoes = docente.get('Orientações', {}).get('Orientações e supervisões concluídas', {})
    #         if orientacoes:
    #             for tipo, dados_orientacao in orientacoes.items():
    #                 for orientacao in dados_orientacao.values():
    #                     ano_match = re.search(r'\b(\d{4})\b', orientacao)
    #                     if ano_match:
    #                         ano = int(ano_match.group(1))

    #                         if ano_inicio <= ano <= ano_final:
    #                             orientador_match = re.search(r'Orientador:\s*(.*)', orientacao)
    #                             if orientador_match:
    #                                 orientador = orientador_match.group(1).strip().strip('.')

    #                                 # Correção: usar o tipo de orientação como chave no segundo nível
    #                                 if orientador not in orientacoes_por_orientador:
    #                                     orientacoes_por_orientador[orientador] = {}
    #                                 if ano not in orientacoes_por_orientador[orientador]:
    #                                     orientacoes_por_orientador[orientador][ano] = {}
    #                                 orientacoes_por_orientador[orientador][ano][tipo] = orientacoes_por_orientador[orientador][ano].get(tipo, 0) + 1

    #     # Criar DataFrame a partir do dicionário, preenchendo anos faltantes com zero
    #     df = pd.DataFrame(orientacoes_por_orientador).fillna(0)

    #     # Multiplicar valores por pontos_ic
    #     df = df * pontos_ic
    #     df = df.T  # transpor o DataFrame para deixar nomes como linhas e anos como colunas

    #     # Verificar se os anos existem como colunas no DataFrame
    #     anos_existentes = set(df.columns)
    #     anos_intervalo = set(range(ano_inicio, ano_final + 1))
    #     anos_a_somar = sorted(list(anos_existentes.intersection(anos_intervalo)))

    #     # Converter colunas de anos para inteiro para ordenação correta, somente se ainda não foram convertidas e se forem numéricas
    #     if not colunas_convertidas:
    #         df.columns = [int(col) if isinstance(col, str) and col.isnumeric() else col for col in df.columns]
    #         colunas_convertidas = True

    #     # Reordenar colunas para que anos fiquem em ordem crescente
    #     df = df.reindex(sorted(df.columns), axis=1)

    #     # Calcular o somatório dos valores dentro do período para cada orientador, somente se houver anos no intervalo
    #     if anos_a_somar:
    #         df.insert(0, 'Pts_Orient_IC', df[anos_a_somar].sum(axis=1))  # Calcula a soma para cada linha (orientador)
    #         # Converter a coluna 'Somatório' para string
    #         df['Pts_Orient_IC'] = df['Pts_Orient_IC'].astype(str)

    #     # Reordenar as linhas para que nomes fiquem em ordem alfabética
    #     df = df.reindex(sorted(df.index), axis=0)
        
    #     return df

    # def apurar_jcr_periodo(self, dict_list, ano_inicio, ano_final):
    #     # Mapeamento de pontos por cada Estrato Qualis
    #     mapeamento_pontos = {
    #         'A1': 90,
    #         'A2': 80,
    #         'A3': 60,
    #         'A4': 40,
    #         'B1': 20,
    #         'B2': 15,
    #         'B3': 10,
    #         'B4': 5,
    #         'C': 0,
    #         'NA': 0
    #     }
    #     import pandas as pd
    #     lista_pubqualis = []
    #     for dic in dict_list:
    #         autor = dic.get('Identificação',{}).get('Nome',{})
    #         artigos = dic.get('Produções', {}).get('Artigos completos publicados em periódicos', {})
    #         for i in artigos:
    #             ano = i.get('ano',{})
    #             qualis = i.get('Qualis',{})
    #             lista_pubqualis.append((ano, autor, qualis))

    #     # Criar um DataFrame a partir da lista_pubqualis
    #     df_qualis_autores_anos = pd.DataFrame(lista_pubqualis, columns=['Ano','Autor', 'Qualis'])
    #     df_qualis_autores_anos

    #     # Criar uma tabela pivot com base no DataFrame df_qualis_autores_anos
    #     pivot_table = df_qualis_autores_anos.pivot_table(index='Autor', columns='Ano', values='Qualis', aggfunc=lambda x: ', '.join(x))

    #     # Selecionar as colunas (anos) que estão dentro do intervalo de anos
    #     anos_interesse = [Ano for Ano in pivot_table.columns if Ano and ano_inicio <= int(Ano) <= ano_final]

    #     # Filtrar a tabela pivot pelos anos de interesse
    #     pivot_table_filtrada = pivot_table[anos_interesse]

    #     # Mostrar a tabela pivot filtrada
    #     pivot_table_filtrada

    #     # Aplicar o mapeamento de pontos à tabela pivot filtrada apenas para valores do tipo str
    #     pivot_table_pontos = pivot_table_filtrada.applymap(lambda x: sum(mapeamento_pontos[q] for q in x.split(', ') if q in mapeamento_pontos) if isinstance(x, str) else 0)

    #     # Adicionar uma coluna final com a soma dos pontos no período
    #     pivot_table_pontos['Soma de Pontos'] = pivot_table_pontos.sum(axis=1)

    #     # Ordenar a tabela pivot pela soma de pontos de forma decrescente
    #     pivot_table_pontos_sorted = pivot_table_pontos.sort_values(by='Soma de Pontos', ascending=False)

    #     # Mostrar a tabela pivot ordenada pela soma de pontos decrescente
    #     return pivot_table_pontos_sorted
    
    
                    ## Trecho extrair página única com vários currículos, sem paginar (extrair_elementos_sem_paginacao)
                    # linhas = i.text.split('\n\n')
                    # # if verbose:
                    # #     print(f'qte_lin_result: {len(linhas):02}')
                    # if self.is_stale_file_handler_present():
                    #     raise StaleElementReferenceException
                    #     # return np.NaN, NOME, np.NaN, 'Stale file handle', self.driver
                    # for m,linha_multipla in enumerate(linhas):
                    #     nome_achado = linhas[m].split('\n')[0]
                    #     linha = linha_multipla.replace("\n", " ")
                    #     if verbose:
                    #         width = 7
                    #         print(f'       Currículo {m+1:02}/{len(linhas):02}: {linha.lower()}')
                    #         print(f'       {instituicao.lower():>25} | {instituicao.lower() in linha.lower()} | {linha.lower()}')
                    #         print(f'       {termo1.lower():>25} |{str(termo1.lower() in linha.lower()).center(width)}| {linha.lower()}')
                    #         print(f'       {termo2.lower():>25} |{str(termo2.lower() in linha.lower()).center(width)}| {linha.lower()}')
                    #         print(f'       {termo3.lower():>25} |{str(termo3.lower() in linha.lower()).center(width)}| {linha.lower()}')
                    #     # print(f'\nOrdem da linha: {m+1}, de total de linhas {len(linhas)}')
                    #     # print('Conteúdo da linha:',linha.lower())
                    #     if instituicao.lower() in linha.lower() or termo1.lower() in linha.lower() or termo2.lower() in linha.lower() or termo3.lower() in linha.lower():
                    #         count=m
                    #         while get_jaro_distance(nome_achado.lower(), str(NOME).lower()) < 0.85 and count>0:
                    #             count-=1
                    #             print(f'       Contador decrescente: {count}')
                    #         found = m+1
                    #         # nome_vinculo = linhas[count].replace('\n','\n       ').strip()
                    #         # print(f'       Achado: {nome_vinculo}')
                    #         css_vinculo = f".resultado > ol:nth-child(1) > li:nth-child({m+1}) > b:nth-child(1) > a:nth-child(1)"
                    #         # print('\nCSS_SELECTOR usado:', css_vinculo)
                    #         WebDriverWait(self.driver, self.delay).until(
                    #             EC.presence_of_element_located((By.CSS_SELECTOR, css_vinculo)))            
                    #         elm_vinculo  = self.driver.find_element(By.CSS_SELECTOR, css_vinculo)
                    #         nome_vinculo = elm_vinculo.text
                            
                    #         ## Tentar repetidamente clicar no elemento encontrado
                    #         self.retry(ActionChains(self.driver).click(elm_vinculo).perform(),
                    #             wait_ms=500,
                    #             limit=limite,
                    #             on_exhaust=(f'  Problema ao clicar no link do nome. {limite} tentativas sem sucesso.'))
                    #         force_break_loop = True
                    #         break
                        
                    #     ## Caso percorra toda lista e não encontre vínculo adiciona à dúvidas quanto ao nome
                    #     if m==(qte_res):
                    #         print(f'Nenhuma referência à {instituicao} ou aos termos {termo1} ou {termo2} ou {termo3}')
                    #         duvidas.append(NOME)
                    #         # clear_output(wait=True)
                    #         # driver.quit()
                    #         continue        

    # def navegar_paginas(self, url_inicial, termos_busca):
    #     """
    #     Navega pelas páginas de resultados e coleta dados relevantes.

    #     Args:
    #         url_inicial (str): URL da página inicial de resultados.
    #         termos_busca (list): Lista de termos para buscar nos links.

    #     Returns:
    #         list: Lista de dicionários, onde cada dicionário contém o nome, a descrição e a
    #             pontuação de compatibilidade.
    #     """
    #     pessoas = []
    #     url_atual = url_inicial
    #     while True:
    #         response = requests.get(url_atual)
    #         response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
    #         soup = BeautifulSoup(response.text, 'html.parser')
    #         resultados = soup.find_all('div', class_='resultado')
    #         for resultado in resultados:
    #             nome, descricao = self.extrair_dados_pessoa(resultado)
    #             pontuacao_compatibilidade = self.calcular_pontuacao_compatibilidade(termos_busca, nome, descricao)

    #             pessoas.append({
    #                 'nome': nome,
    #                 'descricao': descricao,
    #                 'pontuacao_compatibilidade': pontuacao_compatibilidade
    #             })
    #         # Extrai os parâmetros para a próxima página
    #         parametros_paginacao = self.extrair_parametros_paginacao(response.text)
    #         if parametros_paginacao:
    #             url_atual = self.gerar_url_proxima_pagina(url_inicial, parametros_paginacao)
    #         else:
    #             break
    #     return pessoas
