import os
import time
import difflib
import platform
import requests
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import urllib, h5py, logging, traceback, pytz
import os, re, bs4, time, json, warnings, requests, platform
import stat, shutil, psutil, subprocess, csv, string, torch, sqlite3, asyncio, nltk, sys, glob

from PIL import Image
from io import BytesIO
from pathlib import Path
from pprint import pprint
from zipfile import ZipFile
from string import Formatter
from PyPDF2 import PdfReader
from neo4j import GraphDatabase
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
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
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import exceptions
from selenium.common.exceptions import (
    NoSuchElementException, 
    StaleElementReferenceException,
    ElementNotInteractableException,
    TimeoutException,
    WebDriverException
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class JSONFileManager:
    # Carregar arquivo 'dict_list.json' para a variável dict_list
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

class Neo4jPersister:
    def __init__(self, uri, username, password):
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self._driver.close()

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
    def __init__(self, institution, term1, term2, term3, neo4j_uri, neo4j_user, neo4j_password):
        self.verbose = False
        self.configure_logging()
        self.driver = self.connect_driver()
        self.institution = institution
        self.unit = term1
        self.term = term2
        self.term3 = term3
        self.session = requests.Session()
        self.delay = 30
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

    def scrape_and_persist(self, data):
        self._scrape(data)
        self._persist(data)

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
    def connect_driver(only_doctors=False):
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
        only_doctors = True
        if only_doctors:
            print('Buscando currículos apenas entre nível de doutorado')
            url_docts = 'http://buscatextual.cnpq.br/buscatextual/busca.do?buscarDoutores=true&buscarDemais=false&textoBusca='
            driver.get(url_docts) # acessa a url de busca somente de doutores 
        else:
            print('Buscando currículos com qualquer nível de formação')
            url_busca = 'http://buscatextual.cnpq.br/buscatextual/busca.do?buscarDoutores=true&buscarDemais=true&textoBusca='
            driver.get(url_busca) # acessa a url de busca do CNPQ
        driver.set_window_position(-20, -10)
        driver.set_window_size(170, 1896)
        driver.mouse = webdriver.ActionChains(driver)
        return driver

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

    def handle_pagination_and_collect_profiles(self):
        profiles = []
        while True:
            WebDriverWait(self.driver, self.delay).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".resultado")))
            results = self.driver.find_elements(By.CSS_SELECTOR, ".resultado > ol > li")
            for result in results:
                profile_link = result.find_element(By.TAG_NAME, "a")
                profiles.append(profile_link.get_attribute('href'))

            next_button = self.driver.find_elements(By.LINK_TEXT, "próximo")
            if next_button:
                next_button[0].click()
            else:
                break
        return profiles

    def handle_stale_file_error(self, max_retries=5, retry_interval=10):
        for attempt in range(max_retries):
            try:
                error_div = self.driver.find_element(By.CSS_SELECTOR, 'resultado')
                linha1 = error_div.fidChild('li')
                if 'Stale file handle' in linha1.text:
                    time.sleep(retry_interval)
                else:
                    return True
            except NoSuchElementException:
                return True
        return False

    def extract_data_from_cvuri(element) -> dict:
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

    # Usar só em caso onde a busca for somente na basde de doutores, pois carrega página de busca com check 'demais pesquisadores' desabilitado
    def new_search(self):
        try:
            WebDriverWait(self.driver, self.delay).until(
                EC.element_to_be_clickable((By.ID, "botaoBuscaFiltros"))
                ).click()
            if self.verbose:
                print("Efetuando nova busca...")
        except Exception as e:
            print(f"Erro ao clicar em Nova consulta: {e}")

    # Usar para retonar para página de buscas por todos os níveis de formação e nacionalidades        
    def return_search_page(self):
        # url_busca = 'http://buscatextual.cnpq.br/buscatextual/busca.do?buscarDoutores=true&buscarDemais=true&textoBusca='
        # self.driver.get(url_busca) # acessa a url de busca do CNPQ
        url_docts = 'http://buscatextual.cnpq.br/buscatextual/busca.do?buscarDoutores=true&buscarDemais=false&textoBusca='
        self.driver.get(url_docts) # acessa a url de busca somente de doutores 

    def switch_to_new_window(self):
        # Espera até que uma nova janela seja aberta
        WebDriverWait(self.driver, self.delay).until(EC.number_of_windows_to_be(2))
        original_window = self.driver.current_window_handle
        new_window = [window for window in self.driver.window_handles if window != original_window][0]
        self.driver.switch_to.window(new_window)
        return original_window

    def switch_back_to_original_window(self):
        # Armazena o handle da janela original (primeira aba aberta)
        original_window = self.driver.window_handles[0]

        # Verifica se existem mais de uma aba aberta
        if len(self.driver.window_handles) > 1:
            # Muda o foco para a última aba aberta
            self.driver.switch_to.window(self.driver.window_handles[-1])
            # Fecha a última aba
            self.driver.close()

        # Volta o foco para a janela original
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
        # Muda para a aba que deseja fechar
        self.driver.switch_to.window(self.driver.window_handles[-1])
        # Envia a combinação de teclas para fechar a aba
        try:
            print("Tentando fechar aba corrente com Ctrl+w...")
            self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.CONTROL + 'w')
        except:
            print("Tentando fechar aba corrente com Ctrl+F4...")
            self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.CONTROL + Keys.F4)

        # Aguarde um momento para o fechamento da aba
        time.sleep(1)

        # Opcional: volte para a aba principal se necessário
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
        # Extrai o nome
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
            abs_text = abs_par.strip()
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
        div = soup.find('div', class_='')
        atuacao_profissional_section = soup.find('a', name='AtuacaoProfissional')
        if atuacao_profissional_section:
            all_experiences = atuacao_profissional_section.find_next_siblings('div', limit=1)[0].find_all('div', class_='inst_back')
            for exp in all_experiences:
                experience_info = {}
                experience_info['instituicao'] = exp.get_text(strip=True)
                experiences.append(experience_info)
        return experiences

    def get_productions(self, soup):
        logging.debug("Extraindo produções do currículo.")
        try:
            productions = {
                "artigos_completos": self.get_section_productions("ArtigosCompletos"),
                "livros_publicados": self.get_section_productions("LivrosCapitulos"),
                "capitulos_de_livros_publicados": self.get_section_productions("LivrosCapitulos"),
                "trabalhos_completos_em_anais": self.get_section_productions("TrabalhosPublicadosAnaisCongresso"),
                "apresentacoes_de_trabalho": self.get_section_productions("ApresentacoesTrabalho"),
                "outras_producoes_bibliograficas": self.get_section_productions("OutrasProducoesBibliograficas"),
                "assessoria_e_consultoria": self.get_section_productions("AssessoriaConsultoria"),
                "produtos_tecnologicos": self.get_section_productions("ProdutosTecnologicos"),
                "trabalhos_tecnicos": self.get_section_productions("TrabalhosTecnicos"),
                "demais_tipos_de_producao_tecnica": self.get_section_productions("DemaisProducaoTecnica"),
                "demais_trabalhos": self.get_section_productions("DemaisTrabalhos")
            }
            return productions
        except Exception as e:
            logging.error("Erro ao extrair produções: %s", e)

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

    def paginar(self, driver):
        '''
        Helper function to page results on the search page
        '''
        numpaginas = []
        css_paginacao = "div.paginacao:nth-child(2)"
        try:
            WebDriverWait(self.driver, self.delay).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_paginacao)))
            paginacao = self.driver.find_element(By.CSS_SELECTOR, css_paginacao)
            paginas = paginacao.text.split(' ')
            remover = ['', 'anterior', '...']
            numpaginas = [x for x in paginas if x not in remover]
        except Exception as e:
            print('  ERRO!! Ao rodar função paginar():', e)
        return numpaginas

    def medir_tempo_resposta(self):
        try:
            response = requests.get(self.base_url)
            tempo_resposta = response.elapsed.total_seconds()
            print(f"Tempo de resposta do servidor: {tempo_resposta:.2f} segundos")
        except requests.exceptions.RequestException as e:
            print(f"Erro ao fazer solicitação HTTP: {e}")

    ## Está com erro na escolha quando há homônimos, tem a ver aparentemente com manipulação da variável count
    ## Mas só ocorre quando não há paginação nos resultados, quando há escolhe corretamente e mostra corretamente
    def find_terms(self, NOME, instituicao, termo1, termo2, termo3, delay, limite=5):
        """
        Função para manipular o HTML até abir a página HTML de cada currículo   
        Parâmeteros:
            - NOME: É o nome completo de cada pesquisador
            - Instituição, termo1 e termo2: Strings a buscar no currículo para reduzir duplicidades
            - driver (webdriver object): The Selenium webdriver object.
            - limite (int): Número máximo de tentativas em casos de erro.
            - delay (int): tempo em milisegundos a esperar nas operações de espera.
        Retorna:
            elm_vinculo, np.NaN, np.NaN, np.NaN, driver.
        Em caso de erro retorna:
            None, NOME, np.NaN, e, driver
        """
        ignored_exceptions=(NoSuchElementException,StaleElementReferenceException,)
        # Inicializando variáveis para evitar UnboundLocalError
        verbose = False
        elm_vinculo = None
        qte_resultados = 0
        force_break_loop = False
        duvidas = []
        try:
            # Wait and fetch the number of results
            css_resultados = ".resultado"
            WebDriverWait(self.driver, delay, ignored_exceptions=ignored_exceptions).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_resultados)))
            resultados = self.driver.find_elements(By.CSS_SELECTOR, css_resultados)
            if self.is_stale_file_handler_present():
                raise Exception
            ## Ler quantidade de resultados apresentados pela busca de nome
            css_qteresultados = ".tit_form > b:nth-child(1)"
            WebDriverWait(self.driver, delay).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_qteresultados)))                       
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            div_element = soup.find('div', {'class': 'tit_form'})
            match = re.search(r'<b>(\d+)</b>', str(div_element))
            if match:
                qte_resultados = int(match.group(1))
                # print(f'{qte_resultados} resultados para {NOME}')
            else:
                return None, NOME, np.NaN, 'Currículo não encontrado', self.driver
            ## Escolher função a partir da quantidade de resultados da lista apresentada na busca
            ## Ao achar clica no elemento elm_vinculo com link do nome para abrir o currículo
            numpaginas = self.paginar(self.driver)
            if numpaginas == [] and qte_resultados==1:
                # capturar link para o primeiro nome resultado da busca
                css_linknome = ".resultado > ol:nth-child(1) > li:nth-child(1) > b:nth-child(1) > a:nth-child(1)"
                WebDriverWait(self.driver, delay).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, css_linknome)))            
                elm_vinculo  = self.driver.find_element(By.CSS_SELECTOR, css_linknome)
                nome_vinculo = elm_vinculo.text
                # print('Clicar no nome único:', nome_vinculo)
                try:
                    self.retry(ActionChains(self.driver).click(elm_vinculo).perform(),
                        wait_ms=200,
                        limit=limite,
                        on_exhaust=(f'  Problema ao clicar no link do nome. {limite} tentativas sem sucesso.'))
                except:
                    print('  Erro ao clicar no único nome encontrado anteriormente')
                    return None, NOME, np.NaN, None, self.driver
            ## Quantidade de resultados até 10 currículos, acessados sem paginação
            else:
                print(f'       {qte_resultados:>2} currículos homônimos: {NOME}')
                numpaginas = self.paginar(self.driver)
                numpaginas.append('próximo')
                iteracoes=0
                ## iterar em cada página de resultados
                pagin = qte_resultados//10+1
                count = None
                found = None
                for i in range(pagin+1):
                    # print(i,'/',pagin)
                    iteracoes+=1
                    numpaginas = self.paginar(self.driver)
                    # print(f'       Iteração: {iteracoes}. Páginas sendo lidas: {numpaginas}')
                    css_resultados = ".resultado"
                    WebDriverWait(self.driver, delay).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, css_resultados)))
                    resultados = self.driver.find_elements(By.CSS_SELECTOR, css_resultados)
                    if verbose:
                        print(f'qte_div_result: {len(resultados):02}')
                    if self.is_stale_file_handler_present():
                        raise Exception
                    ## iterar em cada resultado
                    for n,i in enumerate(resultados):
                        linhas = i.text.split('\n\n')
                        if verbose:
                            print(f'qte_lin_result: {len(linhas):02}')
                        if 'Stale file handle' in str(linhas):
                            raise Exception
                            # return np.NaN, NOME, np.NaN, 'Stale file handle', self.driver
                        for m,linha_multipla in enumerate(linhas):
                            nome_achado = linhas[m].split('\n')[0]
                            linha = linha_multipla.replace("\n", " ")
                            if verbose:
                                width = 7
                                print(f'Linha {m+1:02}/{len(linhas):02}: {type(linha)}| {linha.lower()}')
                                print(f'{instituicao.lower():>10} | {instituicao.lower() in linha.lower()} | {linha.lower()}')
                                print(f'{termo1.lower():>10} |{str(termo1.lower() in linha.lower()).center(width)}| {linha.lower()}')
                                print(f'{termo2.lower():>10} |{str(termo2.lower() in linha.lower()).center(width)}| {linha.lower()}')
                                print(f'{termo3.lower():>10} |{str(termo3.lower() in linha.lower()).center(width)}| {linha.lower()}')
                            # print(f'\nOrdem da linha: {m+1}, de total de linhas {len(linhas)}')
                            # print('Conteúdo da linha:',linha.lower())
                            if instituicao.lower() in linha.lower() or termo1.lower() in linha.lower() or termo2.lower() in linha.lower() or termo3.lower() in linha.lower():
                                count=m
                                while get_jaro_distance(nome_achado.lower(), str(NOME).lower()) < 0.85 and count>0:
                                    count-=1
                                    print(f'       Contador decrescente: {count}')
                                found = m+1
                                # nome_vinculo = linhas[count].replace('\n','\n       ').strip()
                                # print(f'       Achado: {nome_vinculo}')
                                css_vinculo = f".resultado > ol:nth-child(1) > li:nth-child({m+1}) > b:nth-child(1) > a:nth-child(1)"
                                # print('\nCSS_SELECTOR usado:', css_vinculo)
                                WebDriverWait(self.driver, delay).until(
                                    EC.presence_of_element_located((By.CSS_SELECTOR, css_vinculo)))            
                                elm_vinculo  = self.driver.find_element(By.CSS_SELECTOR, css_vinculo)
                                nome_vinculo = elm_vinculo.text
                                ## Tentar repetidamente clicar no elemento encontrado
                                self.retry(ActionChains(self.driver).click(elm_vinculo).perform(),
                                    wait_ms=500,
                                    limit=limite,
                                    on_exhaust=(f'  Problema ao clicar no link do nome. {limite} tentativas sem sucesso.'))
                                force_break_loop = True
                                break
                            ## Caso percorra toda lista e não encontre vínculo adiciona à dúvidas quanto ao nome
                            if m==(qte_resultados):
                                print(f'Nenhuma referência à {instituicao} ou aos termos {termo1} ou {termo2} ou {termo3}')
                                duvidas.append(NOME)
                                # clear_output(wait=True)
                                # driver.quit()
                                continue
                        if force_break_loop:
                            break
                    try:
                        prox = self.driver.find_element(By.PARTIAL_LINK_TEXT, 'próximo')
                        prox.click()
                    except:
                        continue
                if count:
                    nome_vinculo = linhas[count].replace('\n','\n       ').strip()
                    print(f'        Escolhido o homônimo {found}: {nome_vinculo}')
            if self.is_stale_file_handler_present():
                print("       Erro 'Stale File Handler' detectado na página. Tentando novamente em 10 segundos...")
                time.sleep(1)
                raise Exception
        except Exception as e:
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            base = 2  # Fator de multiplicação exponencial (pode ser ajustado)
            max_wait_time = 120  # Tempo máximo de espera em segundos
            for i in range(1, 12):  # Tentativas máximas com espera exponencial
                wait_time = min(base ** i, max_wait_time)  # Limita o tempo máximo de espera
                print(f"  Erro ao recuperar dados do servidor CNPq, tentando novamente em {wait_time} segundos...")
                time.sleep(wait_time)
                try:
                    self.retry_click_vinculo(elm_vinculo)
                    break  # Se o clique for bem-sucedido, saia do loop de retry
                except TimeoutException as se:
                    logging.error(f"Tentativa {i} falhou: {traceback_str}.")
                    limite+=1
            if limite <= 0:
                print("       Tentativas esgotadas. Abortando ação.")
        # Verifica antes de retornar para garantir que elm_vinculo foi definido
        if elm_vinculo is None:
            print("       Vínculo não foi definido.")
            return None, NOME, np.NaN, 'Vínculo não encontrado', self.driver
        # Retorna a saída de sucesso
        return elm_vinculo, np.NaN, np.NaN, np.NaN, self.driver

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
                print(f"       Erro em fill_name() ao inserir o nome. Tentando novamente...")
                self.return_search_page()
                self.fill_name(NOME, retry_count - 1)
            else:
                print("       Tentativas esgotadas. Abortando ação.")

    def search_profile(self, name, instituicao, termo1, termo2, termo3, retry_count=3):
        '''
        Usa a função find_terms para assegurar escolha do homônimo correto
        '''
        try:
            # Find terms to interact with the web page and extract the profile
            profile_element, _, _, _, _ = self.find_terms(
                name,
                instituicao,
                termo1,
                termo2,
                termo3,
                10,  # delay extração tooltips (10 funciona sem erros em dia normal)
                3
            )
            # print('Elemento encontrado:', profile_element)
            if profile_element:
                return profile_element
            else:
                logging.info(f'Currículo não encontrado: {name}')
                self.return_search_page()

        except requests.HTTPError as e1:
            logging.error(f"HTTPError occurred: {str(e1)}")
            return None
        except Exception as e2:
            logging.error(f"Erro inesperado ao buscar: {str(e2)}")
            return None

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
                            # Se necessário adicionar mais dados ao tooltip_data, acrescentar aqui
                            break  # Saíndo do loop após sucesso na captura dos dados

                tooltip_data_list.append(tooltip_data)

            print(f'       {len(tooltip_data_list):>003} artigos extraídos')

        except TimeoutException:
            print("       Servidor CNPq demorou demais a responder durante extração de tooltips o que impede extrair dados das publicações. \n       Tentar novamente mais tarde, caso persistir o erro é preciso manutenção no código de extração")
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
                            artigo['Qualis'] = 'Não encontrado'

    def encontrar_qualis_por_issn(self, issn):
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

    def scrape_retry(self, name, instituicao, termo1, termo2, termo3, retry_count):
        dict_list = []
        try:
            for _ in range(retry_count):
                try:
                    dict_list.extend(self.scrape_single(name, instituicao, termo1, termo2, termo3))
                    break  # Se o a extração for bem-sucedido, saia do loop de retry
                except TimeoutException as se:
                    logging.error(f"Tentativa de extrair currículo de {name} em scrape_retry() falhou: {se}. Tentando novamente...")
                    time.sleep(1)  # Aguarda um segundo antes de tentar novamente
            else:
                logging.error(f"Todas as tentativas em em scrape_retry() falharam para {name}.")
        except Exception as e:
            raise TimeoutException(f"Erro inesperado em scrape_retry() para {name}")
        return dict_list

    def scrape_single(self, name, instituicao, termo1, termo2, termo3):
        dict_list = []  # Inicialize a lista de dicionários vazia
        try:
            self.fill_name(name)
            elm_vinculo = self.search_profile(name, instituicao, termo1, termo2, termo3)
            if elm_vinculo:
                if self.verbose:
                    print(f"       {name}: vínculo encontrado no currículo, tentando abrir...")
                self.retry_click_vinculo(elm_vinculo) # Tentativas recorrentes para clicar no abrirCurrículo
                if self.verbose:
                    print(f"       {name}: mudando para nova janela após clique para abrir currículo...")
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
                        print('       Disparado return_search_page()...')
                else:
                    print(f"{name}: página de resultado vazia.")
        except Exception as e:
            raise TimeoutException(f"       Erro ao realizar a extração para {name}: {e}")
        return dict_list

    # Realizar chamada recursiva para processar cada nome da lista
    def scrape(self, name_list, instituicao, termo1, termo2, termo3, retry_count=5):
        dict_list = []
        for k, name in enumerate(name_list):
            print(f'{k+1:>2}/{len(name_list)}: {name}')
            try:
                dict_list.extend(self.scrape_retry(name, instituicao, termo1, termo2, termo3, retry_count))
            except TimeoutException:
                logging.error(f"Erro de Timeout ao extrair {name}")
                if retry_count > 0:
                    logging.info(f"Tentando novamente para {name}...")
                    # Realiza novar tentativa para o mesmo nome passando número de tentativas decrementado de 1
                    dict_list.extend(self.scrape([name], instituicao, termo1, termo2, termo3, retry_count-1))
                else:
                    logging.error(f"Todas as tentativas falharam para {name}")
            except Exception as e:
                logging.error(f"Erro inesperado ao extrair {name}: {e}")
                traceback_str = ''.join(traceback.format_tb(e.__traceback__))
                logging.error(traceback_str)
        self.driver.quit()
        return dict_list

    # def retry_click_vinculo_junto(self, elm_vinculo, retry_count=3):
    #     if elm_vinculo is None:
    #         logging.info("       Nenhum dos vínculos esperados encontrado no currículo...")
    #         self.return_search_page()
    #         return
    #     try:
    #         # Aguardar até que o botão esteja presente na página
    #         botao_abrir_curriculo = self.driver.find_element(By.ID, "idbtnabrircurriculo")
    #         botao_abrir_curriculo = WebDriverWait(self.driver, 10).until(
    #             EC.visibility_of_element_located((By.CSS_SELECTOR, '#idbtnabrircurriculo'))
    #             # EC.visibility_of_element_located((By.XPATH, '//*[@id="idbtnabrircurriculo"]'))
    #         )
    #         botao_abrir_curriculo = WebDriverWait(self.driver, 10).until(
    #             EC.element_to_be_clickable((By.CSS_SELECTOR, '#idbtnabrircurriculo'))
    #             # EC.element_to_be_clickable((By.XPATH, '//*[@id="idbtnabrircurriculo"]'))
    #         )
    #         actions = ActionChains(self.driver)
    #         actions.move_to_element(botao_abrir_curriculo).perform()
    #         botao_abrir_curriculo.click()
    #         # Checar se a página foi carregada
    #         # WebDriverWait(self.driver, self.delay).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, "#artigos-completos img.ajaxJCR")))
    #         layout_cells = self.driver.find_elements(By.CSS_SELECTOR, '#artigos-completos .layout-cell-11')

    #     except TimeoutException:
    #         if retry_count > 0:
    #             print("       Elemento não encontrado. Tentando novamente...")
    #             self.check_and_click_vinculo(elm_vinculo, retry_count - 1)
    #         else:
    #             print("       Tentativas esgotadas. Abortando ação.")

    #     except WebDriverException as e:
    #         print(f"       Erro ao clicar no botão 'Abrir Currículo'")
    #         traceback_str = ''.join(traceback.format_tb(e.__traceback__))
    #         print(f'       {traceback_str}')
    #         logging.info(f"Erro ao abrir currículo: {elm_vinculo.text}: {e} {traceback_str}")
    #         self.return_search_page()

    ## Extrai perfeitamente mas com erro nos retry
    # def scrape(self, name_list, instituicao, termo1, termo2, retry_count=5):
    #     dict_list = []
    #     for k, name in enumerate(name_list):
    #         try:
    #             print(f'{k+1:>2}/{len(name_list)}: {name}')
    #             # Preencher o nome
    #             self.fill_name(name)
    #             # Realizar a busca
    #             elm_vinculo = self.search_profile(name, instituicao, termo1, termo2)
    #             if elm_vinculo:
    #                 if self.verbose:
    #                     print(f"       Vínculo encontrado no currículo, tentando abrir...")
    #                 # self.check_and_click_vinculo(elm_vinculo) # sem recursão para nova tentativa
    #                 self.retry_click_vinculo(elm_vinculo)
    #                 if self.verbose:
    #                     print(f"       Mudando para janela nova após clique para abrir currículo...")
    #                 # Muda para a nova janela aberta com o currículo
    #                 window_before = self.switch_to_new_window()
    #                 tooltip_data_list = self.extract_tooltip_data()
    #                 if self.verbose:
    #                     print(f'       {len(tooltip_data_list):>003} tooltips encontrados')
    #                 # Verifica se há o erro "Stale File Handler" na página
    #                 if self.is_stale_file_handler_present():
    #                     print("       Erro 'Stale File Handler' detectado na página. Tentando novamente em 10 segundos...")
    #                     time.sleep(10)
    #                     raise TimeoutException
    #                 # Continua com o processamento normal se o erro não foi detectado
    #                 page_source = self.driver.page_source
    #                 if page_source is not None:
    #                     # Usar classe para fazer a extração a partir do HTML carregado
    #                     parser = HTMLParser(page_source)
    #                     data = parser.to_json()
    #                     data['JCR2'] = tooltip_data_list
    #                     dict_list.append(data)
    #                     self.switch_back_to_original_window()
    #                     self.close_popup()
    #                     self.return_search_page()
    #                     print(f'       {len(dict_list):>003}/{len(name_list):>003} currículos extraídos com sucesso')
    #                 else:
    #                     print("       Página de resultado vazia.")
    #         except Exception as e:
    #             print(f"       Erro ao realizar o scrapping: {e}")
    #             traceback_str = ''.join(traceback.format_tb(e.__traceback__))
    #             print(f'       {traceback_str}')
    #             logging.info(f"Erro ao realizar o scrapping de: {name}: {traceback_str}")
    #             if retry_count > 0:
    #                 print("       Tentando novamente...")
    #                 # Chamada recursiva para uma nova tentativa
    #                 dict_list.extend(self.scrape([name], instituicao, termo1, termo2, retry_count-1))
    #             else:
    #                 print("       Todas as tentativas falharam.")
    #                 logging.info(f"Todas as tentativas falharam para: {name}")
    #                 continue
    #     return dict_list

    #     # Melhor adicionar o Estrato Qualis depois para evitar erros na geração da lista de dicionários
    #     self.buscar_qualis(dict_list)
    #     self.driver.quit()
    #     return dict_list

    ## VERSÕES ANTIGAS COM EXTRAÇÃO POR SEÇÕES INDIVIDUALIZADAS

    # Checar e clicar sem retry
    # def check_and_click_vinculo(self, elm_vinculo):
    #     if elm_vinculo is None:
    #         logging.info("Nenhum dos vínculos esperados encontrado no currículo...")
    #         self.return_search_page()
    #         return
    #     # Clicar no botão para abrir o currículo
    #     try:
    #         # Aguardar até que o botão esteja presente na página
    #         botao_abrir_curriculo = self.driver.find_element(By.ID, "idbtnabrircurriculo")
    #         botao_abrir_curriculo = WebDriverWait(self.driver, 10).until(
    #             EC.visibility_of_element_located((By.CSS_SELECTOR, '#idbtnabrircurriculo'))
    #             # EC.visibility_of_element_located((By.XPATH, '//*[@id="idbtnabrircurriculo"]'))
    #         )
    #         actions = ActionChains(self.driver)
    #         actions.move_to_element(botao_abrir_curriculo).perform()
    #         botao_abrir_curriculo.click()
    #         # Checar se a página foi carregada

    #     except WebDriverException as e:
    #         print(f"       Erro ao clicar no botão 'Abrir Currículo'")
    #         traceback_str = ''.join(traceback.format_tb(e.__traceback__))
    #         print(f'       {traceback_str}')
    #         logging.info(f"Erro ao abrir currículo: {elm_vinculo.text}: {e} {traceback_str}")
    #         self.return_search_page()

    # def scrape(self, name_list, instituicao, termo1, termo2, json_filename, hdf5_filename):
    #     dict_list = []
    #     for k, name in enumerate(name_list):
    #         try:
    #             print(f'{k+1:>2}/{len(name_list)}: {name}')
    #             # Preencher o nome
    #             self.fill_name(name)
    #             # Realizar a busca
    #             elm_vinculo = self.search_profile(name, instituicao, termo1, termo2)
    #             # current_url = self.driver.current_url
    #             if elm_vinculo:
    #                 if self.verbose:
    #                     print(f"       Vínculo encontrado no currículo, tentando abrir...")
    #                 try:
    #                     self.check_and_click_vinculo(elm_vinculo)
    #                     if self.verbose:
    #                         print(f"       Mudando para janela nova após clique para abrir currículo...")
    #                 except Exception as e:
    #                     print(f"       Erro ao tentar clicar no botão no iframe de abrir currículo...")
    #                     traceback_str = ''.join(traceback.format_tb(e.__traceback__))
    #                     print(f'       {traceback_str}')
    #                     continue
    #                 # Muda para a nova janela aberta com o currículo
    #                 window_before = self.switch_to_new_window()
    #                 try:
    #                     tooltip_data_list = self.extract_tooltip_data()
    #                     if self.verbose:
    #                         print(f'       {len(tooltip_data_list):>003} tooltips encontrados')
    #                 except Exception as e:
    #                     print(f"Erro ao extrair tooltips: {e}")
    #                     # Se ocorrer um erro, tenta extrair os dados novamente
    #                     tooltip_data_list = self.extract_tooltip_data()
    #                     if self.verbose:
    #                         print(f"       Segunda tentativa extrair tooltips: {tooltip_data_list}")
                    
    #                 page_source = self.driver.page_source
    #                 if page_source is not None:
    #                     try:
    #                         # Usar classe para fazer a extração a partir do HTML carregado
    #                         parser = HTMLParser(page_source)
    #                         data = parser.to_json()
    #                         data['JCR2'] = tooltip_data_list
    #                         dict_list.append(data)
    #                         self.switch_back_to_original_window()
    #                         self.close_popup()
    #                         self.return_search_page()
    #                         print(f'       {len(dict_list):>003}/{len(name_list):>003} currículos extraídos com sucesso')
    #                     except Exception as e:
    #                         print("       Erro ao processar com método Parser")
    #                         print(e)

    #                     ## EM CASO DE EXTRAÇÃO APENAS DE SEÇÕES
    #                     # soup = BeautifulSoup(page_source, 'html.parser')
    #                     # if self.verbose:
    #                     #     if len(soup) > 1:
    #                     #         des_numero = 's'
    #                     #     else:
    #                     #         des_numero = ''
    #                     #     print(f'       {len(soup):>003} elemento{des_numero} encontrado{des_numero} no objeto soup')
    #                     # soup.attrs['tooltips'] = tooltip_data_list
    #                     # data={}
    #                     # if soup:
    #                     #    # print(soup.get_text())
    #                     #    # Extrai informações básicas do currículo
    #                     #     try:
    #                     #         if self.extrair_informacoes_infpessoa(soup):
    #                     #             data["general_data"] = self.extrair_informacoes_infpessoa(soup)
    #                     #         else:
    #                     #             print('Dados gerais não encontrados')
    #                     #     except Exception as e:
    #                     #         logging.error(f"Erro em get_general_data(soup): {e}")
    #                     #     try:
    #                     #         if self.get_abstract(soup):
    #                     #             data["abstract"] = self.get_abstract(soup).get_text.strip()
    #                     #     except Exception as e:
    #                     #         logging.error(f"Erro ao extrair resumo com get_abstract(): {e}")
    #                     #     try:
    #                     #         formation_div = self.get_formation(soup)
    #                     #         if formation_div:
    #                     #             formation_txt = formation_div.get_text.strip()
    #                     #             data["get_formation"] = formation_txt
    #                     #     except Exception as e:
    #                     #         logging.error(f"Erro em get_formation(soup): {e}")
    #                     #     try:
    #                     #         data["get_professional_experience"] = self.get_professional_experience(soup)
    #                     #     except Exception as e:
    #                     #         logging.error(f"Erro em get_professional_experience(soup): {e}")
    #                     #     try:
    #                     #         data["productions"] = self.get_productions(soup)
    #                     #     except Exception as e:
    #                     #         logging.error(f"Erro em get_productions(soup): {e}")
    #                     #     try:
    #                     #         data["technical_productions"] = self.get_technical_productions(soup)
    #                     #     except Exception as e:
    #                     #         logging.error(f"Erro em get_technical_productions(soup): {e}")
    #                     #     try:
    #                     #         data["guidance"] = self.get_guidance(soup)
    #                     #     except Exception as e:
    #                     #         logging.error(f"Erro em get_guidance(soup): {e}")
    #                     #     try:
    #                     #         data["projects"] = self.get_projects(soup)
    #                     #     except Exception as e:
    #                     #         logging.error(f"Erro em get_projects(soup): {e}")
    #                     #     try:
    #                     #         data["courses"] = self.get_courses(soup)
    #                     #     except Exception as e:
    #                     #         logging.error(f"Erro em get_courses(soup): {e}")
    #                     #     try:
    #                     #         data["events"] = self.get_events(soup)
    #                     #     except Exception as e:
    #                     #         logging.error(f"Erro em get_events(soup): {e}")
    #                     #     try:
    #                     #         data["patents"] = self.get_patents(soup)
    #                     #     except Exception as e:
    #                     #         logging.error(f"Erro em get_patents(soup): {e}")
    #                     #     # Adicionar outras seções que sejam relevantes
    #                     #     # data["other_section"] = self.other_extraction_function(soup)
    #                     #     if self.verbose:
    #                     #         try:
    #                     #             print(f'       {len(data):>003} elementos extract_data encontrados no objeto soup')
    #                     #         except Exception as e:
    #                     #             print(f'       {e}')
    #                     #     # Chama métodos de conversão de dicionário individual
    #                     #     # parse_soup_instance.to_json(data, json_filename)
    #                     #     # parse_soup_instance.to_hdf5(data, hdf5_filename)
    #                     #     dict_list.append(data)
    #                     #     print(f'       {len(dict_list):>003} subdicionários adicionados ao objeto data: {dict_list.keys()}')
    #                     # else:
    #                     #     print(f"Não foi gerado objeto soup para: {name}")
    #                     #     logging.error(f"Não foi possível extrair dados do currículo: {name}")
    #             else:
    #                 print(f"       Elemento com vínculo encontrado retornado vazio...")
    #                 logging.info(f"Currículo não encontrado para: {name}")
    #         except Exception as e:
    #             print(f"       Erro ao extrair Elemento com vínculo")
    #             traceback_str = ''.join(traceback.format_tb(e.__traceback__))
    #             print(f'       Um erro impediu de extrair a produção:')
    #             print(f'       {traceback_str}')
    #             logging.info(f"Timeout antes de carregar o currículo: {name}: {traceback_str}")

    #     self.driver.quit()
    #     return dict_list
  
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
                print("Container de idiomas não encontrado")
        else:
            print("Seção de idiomas não encontrada")

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

    def extract_info(self):
        soup = BeautifulSoup(self.html_element, 'html.parser')
        qualis_info = self.extract_qualis(soup)

        # Extrai os autores
        autores = soup.find_all('span', class_='informacao-artigo', data_tipo_ordenacao='autor')
        primeiro_autor = autores[0].text if autores else None
        # Considera todos os textos após o autor como parte da lista de autores até um elemento estrutural significativo (<a>, <b>, <sup>, etc.)
        autores_texto = self.html_element.split('autor">')[-1].split('</span>')[0] if autores else ''

        ano_tag = soup.find('span', {'class': 'informacao-artigo', 'data-tipo-ordenacao': 'ano'})
        ano = int(ano_tag.text) if ano_tag else 'Ano não disponível'

        # Extrai o título, periódico, e outras informações diretamente do texto
        texto_completo = soup.get_text(separator=' ', strip=True)
        
        # Assume que o título vem após os autores e termina antes de uma indicação de periódico ou volume
        titulo_match = re.search(r'; ([^;]+?)\.', texto_completo)
        titulo = titulo_match.group(1) if titulo_match else None

        # Periódico e detalhes como volume, página, etc., 
        periodico_match = re.search(r'(\. )([^.]+?),( v\. \d+, p\. \d+, \d+)', texto_completo)
        periodico = periodico_match.group(2) if periodico_match else None
        detalhes_periodico = periodico_match.group(3) if periodico_match else None

        # Extrai citações se disponível
        citacoes = soup.find('span', class_='numero-citacao')
        citacoes = int(citacoes.text) if citacoes else 0

        # Extrai ISSN
        issn = soup.find('img', class_='ajaxJCR')
        issn = issn['data-issn'] if issn else None

        # Qualis/CAPES pode ser extraído se existir um padrão identificável
        qualis_capes = "quadriênio 2017-2020"  # Neste exemplo, hardcoded, mas pode ser ajustado

        # Monta o dicionário de resultados
        resultado = {
            "dados_gerais": texto_completo,
            "primeiro_autor": primeiro_autor,
            "ano": ano,
            "autores": autores_texto,
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
        planilha_excel = os.path.join(self.find_repo_root(), 'data', file_name)        
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
                                chave = divs_indices[i].get_text(strip=True).replace('\t','').replace('\n',' ')
                                valor = divs_ocorrencias[i].get_text(strip=True).replace('\t','').replace('\n',' ')

                                # Adiciona o par chave-valor ao dicionário
                                ocorrencias[chave] = valor
                        # geral.append(ocorrencias)
                        self.estrutura["Bancas"][subsec_name] = ocorrencias

    def process_orientacoes(self):
        self.estrutura["Orientações"]={}
        secoes = self.soup.find_all('div', class_='title-wrapper')
        for secao in secoes:
            titulo_h1 = secao.find('h1')
            if titulo_h1 and 'Orientações' in titulo_h1.get_text(strip=True):
                data_cell = secao.find('div', class_='layout-cell layout-cell-12 data-cell')
                subsecoes = data_cell.find_all('div', class_='inst_back')
                for subsecao in subsecoes:
                    ocorrencias = {}
                    if subsecao:
                        subsec_name = subsecao.get_text(strip=True)
                        # print(f'Seção: {subsec_name}')
                        if subsec_name not in self.estrutura["Orientações"]:                       
                            self.estrutura["Orientações"][subsec_name] = []
                    
                    ## Extrair cada tipo de orientação
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
                            # Verificar se o irmão tem a classe "cita-artigos" ou "inst_back"
                            if 'cita-artigos' in sibling.get('class', []) or 'inst_back' in sibling.get('class', []):
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

                        self.estrutura["Orientações"][subsec_name].append({subsubsecao_name: ocorrencias})

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
                            artigo['Qualis'] = 'Não encontrado'

    def encontrar_qualis_por_issn(self, issn):
        qualis = self.dados_planilha[self.dados_planilha['ISSN'].str.replace('-','') == issn]['Estrato'].tolist()
        if qualis:
            return qualis[0]
        else:
            return None

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
        ## EDUCAÇÃO
        self.process_bancas()                   # Ok!
        self.process_orientacoes()              # Ok!

        # TO-DO-LATER em Produções extraindo vazio
        # "Citações": {},                           
        # "Resumos publicados em anais de congressos (artigos)": {} 

    def to_json(self):
        self.process_all()
        json_string = json.dumps(self.estrutura, ensure_ascii=False, indent=4)
        return json.loads(json_string)

class GetQualis:
    def __init__(self):
        self.dados_planilha = pd.read_excel(os.path.join(LattesScraper.find_repo_root(),'_data','in_xls','classificações_publicadas_todas_as_areas_avaliacao1672761192111.xlsx'))

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
                            artigo['Qualis'] = 'Não encontrado'

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

    def get_articles(self, dict_list):
        # Extrair cada artigo de cada dicionário de currículo
        colaboracoes = []
        for dic in dict_list:
            autor = dic.get('Identificação',{}).get('Nome',{})
            artigos = dic.get('Produções', {}).get('Artigos completos publicados em periódicos', {})
            lista_coautores_artigo = []
            for i in artigos:
                ano = i.get('ano',{})
                qualis = i.get('Qualis',{})
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
                lista_coautores_artigo.append(coautores)
                colaboracoes.append({autor: coautores})
        return colaboracoes

    def get_coauthors(self, string):
        """
        Verifica e retorna a posição de cada padrão buscado individualmente.

        Argumentos:
            string (str): A string que contém todos dados de cada artigo no formato: 
            * prim_autor|ano|autores|demaisdados.

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
        regex_pntofinal_esp = r"\.\s+(\w+)"
        regex_restante_strg = r"\.\s+(\w+)+.*$" 

        # Encontrar as primeiras ocorrências de cada padrão
        string = str(string)
        match_quatrodigitos = re.search(regex_quatrodigitos, string)
        match_pontos_duplos = re.search(regex_pontos_duplos, string)
        match_ponto_esp_pnt = re.search(regex_ponto_esp_pnt, string)
        # match_ponto_fim_esp = re.search(regex_pntofinal_esp, string)
        # Encontrar todas ocorrências de padrões de abreviação/separação 
        match_ponto_virgula = re.finditer(regex_ponto_virgula, string)
        match_pnto_iniciais = re.finditer(regex_pnto_iniciais, string)
        match_ponto_fim_esp = re.finditer(regex_pntofinal_esp, string)

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
  
        # if 'Citações' in lista_coautores[-1]:
        #     lista_coautores[-1] = lista_coautores[-1].split('. ')[0]
            
        if self.verbose:
            print(f"quatrodigitos: {padroes['quatrodigitos']:003} | {type(padroes['quatrodigitos'])}")
            print(f"pontos_duplos: {padroes['pontos_duplos']:003} | {type(padroes['pontos_duplos'])}")
            print(f"ponto_esp_pnt: {padroes['ponto_esp_pnt']:003} | {type(padroes['ponto_esp_pnt'])}")
            print(f"ponto_inicial: {padroes['ponto_inicial']} | {type(padroes['ponto_inicial'])}")
            print(f"ponto_virgula: {padroes['ponto_virgula']} | {type(padroes['ponto_virgula'])}")
            print(f"pnto_espfinal: {padroes['ponto_final']} | {type(padroes['ponto_final'])}")
            print(f"       inicio: {beg:003} | final: {end:003}")

        return lista_coautores, padroes

    def discent_direct_counter(self, coauthors_list, discent_list):
        count=0
        for name in coauthors_list:
            if name in discent_list:
                count+=1

        return qte_discent_collaborations

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

    def is_similar(self, name_discent, name_coauthor):
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
        distancia_levenshtein = difflib.levenshtein(name_discent_norm, name_coauthor_norm)

        # Cálcular índice de similaridade
        max_length = max(len(name_discent_norm), len(name_coauthor_norm))
        similarity_index = 1 - (distancia_levenshtein / max_length)

        return similarity_index

    def discent_similar_counter(self, coauthors_list, discent_list):
        count=0
        threshold=0.85

        for name in coauthors_list:
            if self.is_similar(name_discent, name_coauthor) >= threshold:
                count+=1

        return qte_discent_collaborations

class ArticlesCounter:
    def __init__(self, dict_list):
        self.data_list = dict_list

    def dias_desde_atualizacao(self, data_atualizacao_str):
        # Converte a data de atualização em um objeto datetime
        data_atualizacao = datetime.strptime(data_atualizacao_str, '%d/%m/%Y')
        
        # Obtém a data atual
        data_atual = datetime.now()
        
        # Calcula a diferença em dias
        diferenca_dias = (data_atual - data_atualizacao).days if data_atualizacao else None
        return diferenca_dias

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

    def contar_qualis(self, dict_list):
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
        minimo = min(df_qualis_autores_anos['Ano'])
        maximo = max(df_qualis_autores_anos['Ano'])
        print(f'Contagem de Publicações por Qualis Periódicos no período {minimo} a {maximo}')

        # Contar as ocorrências de cada combinação de Autor e Qualis
        pivot_table = df_qualis_autores_anos.pivot_table(index='Autor', columns='Qualis', aggfunc='size', fill_value=0)
    
        return pivot_table

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

    def apurar_pontos_periodo(self, dict_list, ano_inicio, ano_final):
        # Mapeamento de pontos por cada Estrato Qualis
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
            'Não encontrado': 0
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
        df_qualis_autores_anos

        # Criar uma tabela pivot com base no DataFrame df_qualis_autores_anos
        pivot_table = df_qualis_autores_anos.pivot_table(index='Autor', columns='Ano', values='Qualis', aggfunc=lambda x: ', '.join(x))

        # Selecionar as colunas (anos) que estão dentro do intervalo de anos
        anos_interesse = [Ano for Ano in pivot_table.columns if Ano and ano_inicio <= int(Ano) <= ano_final]

        # Filtrar a tabela pivot pelos anos de interesse
        pivot_table_filtrada = pivot_table[anos_interesse]

        # Mostrar a tabela pivot filtrada
        pivot_table_filtrada

        # Aplicar o mapeamento de pontos à tabela pivot filtrada apenas para valores do tipo str
        pivot_table_pontos = pivot_table_filtrada.applymap(lambda x: sum(mapeamento_pontos[q] for q in x.split(', ') if q in mapeamento_pontos) if isinstance(x, str) else 0)

        # Adicionar uma coluna final com a soma dos pontos no período
        pivot_table_pontos['Soma de Pontos'] = pivot_table_pontos.sum(axis=1)

        # Ordenar a tabela pivot pela soma de pontos de forma decrescente
        pivot_table_pontos_sorted = pivot_table_pontos.sort_values(by='Soma de Pontos', ascending=False)

        # Mostrar a tabela pivot ordenada pela soma de pontos decrescente
        return pivot_table_pontos_sorted