import os
import time
import logging
import platform
import requests
import traceback
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import re, sys, time, json, glob, warnings
import psutil, urllib, h5py, logging, traceback 
import stat, shutil, requests, platform, subprocess
import os, csv, string, torch, sqlite3, asyncio, nltk

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
        lattes_id = self.extract_lattes_id(data_dict.get("InfPes", []))
        
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
            key, _, value = part.partition(":")
            area_data[key.strip()] = value.strip()
        return area_data

    def process_all_person_nodes(self):
        """Iterates over all Person nodes and persists secondary nodes and relationships."""
        with self._driver.session() as session:
            result = session.run("MATCH (p:Person) RETURN p.name AS name, p.`Áreas de atuação` AS areas")

            for record in result:
                person_name = record["name"]
                
                # Check if name or areas is None
                if person_name is None or record["areas"] is None:
                    print(f"Skipping record for name {person_name} due to missing name or areas.")
                    continue

                # Check if the areas data is already in dict form
                if isinstance(record["areas"], dict):
                    areas = record["areas"]
                else:
                    # Attempt to convert from a string representation (e.g., JSON)
                    try:
                        areas = json.loads(record["areas"])
                    except Exception as e:
                        print(f"Failed to parse areas for name {person_name}. Error: {e}")
                        continue
                
                self.persist_secondary_nodes(person_name, areas)

class SoupParser:
    def __init__(self, driver):
        self.configure_logging()
        self.base_url = 'http://buscatextual.cnpq.br'
        self.session = requests.Session()
        self.failed_extractions = []
        self.driver = driver
        self.delay = 10
        self.soup = None

    def configure_logging(self):
        # Configura o logging para usar um novo arquivo de log, substituindo o antigo
        logging.basicConfig(filename='lattes_scraper.log', level=logging.INFO, filemode='w')

    def __enter__(self):
        return self  # the object to bind to the variable in the `as` clause

    def __exit__(self, exc_type, exc_value, traceback):
        self.driver.quit()
        self.driver = None

    def to_json(self, data_dict: Dict, filename: str) -> None:
        try:
            with open(filename, 'w') as f:
                json.dump(data_dict, f)
        except Exception as e:
            logging.error(f"An error occurred while saving to JSON: {e}")

    def to_hdf5(self, processed_data: List[Dict], hdf5_filename: str) -> None:
        try:
            with h5py.File(hdf5_filename, 'w') as hdf5_file:
                for i, data in enumerate(processed_data):
                    # Serializa o dicionário como uma string JSON antes de armazená-lo.
                    serialized_data = json.dumps(data)
                    hdf5_file.create_dataset(str(i), data=serialized_data)
        except Exception as e:
            logging.error(f"An error occurred while saving to HDF5: {e}")

    def dictlist_to_json(self, data_list: List[Dict], filename: str) -> None:
        try:
            with open(filename, 'w') as f:
                json.dump(data_list, f)
        except Exception as e:
            logging.error(f"An error occurred while saving to JSON: {e}")

    def dictlist_to_hdf5(self, data_list: List[Dict], filename: str, directory=None) -> None:
        try:
            converter = DictToHDF5(data_list)
            converter.create_dataset(filename, directory)
        except Exception as e:
            logging.error(f"An error occurred while saving to HDF5: {e}")
    
    def format_string(self, input_str):
        # Verifica se a entrada é uma string de oito dígitos
        if input_str and len(input_str) == 9:
            return input_str
        elif input_str and len(input_str) == 8:
            # Divide a string em duas partes
            part1 = input_str[:4]
            part2 = input_str[4:]
            # Concatena as duas partes com um hífen
            formatted_str = f"{part1}-{part2}"
            return formatted_str
        else:
            return input_str
            
    def extract_tit1_soup(self, soup, data_dict=None, verbose=True):
        if data_dict is None:
            data_dict = {}
        elm_main_cell = soup.find("div", class_="layout-cell-pad-main")
        divs_title_wrapper = elm_main_cell.find_all('div', class_='title-wrapper')
        # Títulos contendo subseções
        tit1a = ['Identificação','Endereço','Formação acadêmica/titulação','Pós-doutorado','Formação Complementar',
                'Linhas de pesquisa','Projetos de pesquisa','Projetos de extensão','Projetos de desenvolvimento', 'Revisor de periódico','Revisor de projeto de fomento','Áreas de atuação','Idiomas','Inovação']
        tit1b = ['Atuação Profissional'] # dados com subseções
        for div_title_wrapper in divs_title_wrapper:
            # Encontre o título do bloco
            try:
                titulo = div_title_wrapper.find('h1').text.strip()
            except:
                titulo = 'Não disponível na tag h1 do Currículo Lattes'
            data_cells = div_title_wrapper.find_all("div", class_="data-cell")
            # Verifique se o título está na lista 'tit1'
            if titulo in tit1a:
                if verbose:
                    print(titulo)
                data_dict[titulo] = {}  # Inicialize o dicionário para o título 'Eventos'
                for data_cell in data_cells:
                    divs_layout_cell_3 = data_cell.find_all('div', class_='layout-cell-3')
                    divs_layout_cell_9 = data_cell.find_all('div', class_='layout-cell-9')
                    keys = []
                    vals = []
                    for i, j in zip(divs_layout_cell_3, divs_layout_cell_9):
                        if divs_layout_cell_3 and divs_layout_cell_9:
                            key = i.find('div', class_='layout-cell-pad-6 text-align-right')
                            key_text = key.get_text().strip().replace('\n', ' ').replace('\t', '')
                            keys.append(key_text)
                            val = j.find('div', class_='layout-cell-pad-6')
                            val_text = val.get_text().strip().replace('\n', ' ').replace('\t', '')
                            vals.append(val_text)
                            if verbose:
                                print(f'      {key_text:>3}: {val_text}')
                    agg_dict = {key: val for key, val in zip(keys, vals)}
                    data_dict[titulo] = Neo4jPersister.convert_to_primitives(agg_dict)
            if titulo in tit1b:
                if verbose:
                    print(titulo)
                data_dict[titulo] = {}  # Inicialize o dicionário para o título 'Eventos'
                for data_cell in data_cells:
                    sections = data_cell.find_all("div", class_="inst_back")               
                    if verbose:
                        print(len(sections), 'seções')
                    for section in sections:
                        section_name = section.find('b').get_text().strip()
                        data_dict[titulo][section_name] = []
                        if verbose:
                            print(section_name)
                        sibling = section.find_next_sibling()
                        current_data = {}  # Criamos um dicionário para armazenar os dados da subseção atual
                        while sibling:
                            classes = sibling.get('class', [])
                            if 'layout-cell-3' in classes:  # Data key
                                key = sibling.find("div", class_="layout-cell-pad-6 text-align-right").get_text().strip()
                                sibling = sibling.find_next_sibling()

                                if sibling and 'layout-cell-9' in sibling.get('class', []):  # Check if value is present
                                    val = sibling.find("div", class_="layout-cell-pad-6").get_text().strip().replace('\n', '').replace('\t','')
                                    current_data[key] = val
                                    if verbose:
                                        print(len(current_data.values()), key, val)
                            elif sibling.name == 'br' and 'clear' in sibling.get('class', []):  # Fim de seção/subseção
                                next_sibling = sibling.find_next_sibling()
                                if next_sibling and 'clear' in next_sibling.get('class', []):
                                    sibling = None
                                else:
                                    if current_data:
                                        data_dict[titulo][section_name].append(current_data)  # Armazenamos os dados em uma lista
                            if sibling:
                                sibling = sibling.find_next_sibling()
        return data_dict

    def extract_tit2_soup(self, soup, data_dict=None, verbose=False):
        if data_dict is None:
            data_dict = {}
        database = ''
        total_trab_text = 0
        total_cite_text = 0
        num_fator_h = 0
        data_wos_text = ''
        elm_main_cell = soup.find("div", class_="layout-cell-pad-main")
        divs_title_wrapper = elm_main_cell.find_all('div', class_='title-wrapper')
        tit2 = ['Produções', 'Bancas', 'Orientações']
        for div_title_wrapper in divs_title_wrapper:
            # Encontre o título do bloco
            try:
                titulo = div_title_wrapper.find('h1').text.strip()
            except:
                titulo = 'Não disponível na tag h1 do Currículo Lattes'
            data_cells = div_title_wrapper.find_all("div", class_="data-cell")
            # Verifique se o título está na lista 'tit2'
            if titulo in tit2:
                if verbose:
                    print(f'Título: {titulo}')
                data_dict[titulo] = {}  # Inicialize o dicionário para o título 'Eventos'
                for data_cell in data_cells:
                    sections = data_cell.find_all("div", class_="inst_back")
                    if verbose:
                        print(len(sections), 'seções')
                    for section in sections:
                        section_name = section.find('b').get_text().strip()
                        data_dict[titulo][section_name] = {}
                        if verbose:
                            print(f'Seção: {section_name}')
                        sibling = section.find_next_sibling()
                        current_subsection = None
                        current_data = {}  # Criamos um dicionário para armazenar os dados da subseção atual
                        if section_name == 'Produção bibliográfica':
                            subsections = section.find_next_siblings('div', class_='cita-artigos')
                            if verbose:
                                print(len(subsections), 'subseções')                       
                            for subsection in subsections:                            
                                if subsection:
                                    subsection_name = subsection.find('b').get_text().strip()
                                    if verbose:
                                        print(f'    Subseção: {subsection_name}') # nomes de subseção como ocorrências 
                                        print(f'    {len(subsection)} divs na subseção {subsection_name}')                                
                                    if subsection_name == 'Citações':
                                        current_subsection = subsection_name
                                        data_dict[titulo][section_name]['Citações'] = {}
                                        sub_section_list = []  
                                        ## Extrair quantidade de citações e fator H das divs de subseção com classe lyout-cell-12
                                        next_siblings = subsection.find_next_siblings("div", class_="layout-cell-12") #acha os irmãos da Subseção
                                        for sibling in next_siblings:
                                            citation_counts = sibling.findChildren("div", class_="web_s")  # Encontra as divs que tem os Valores de Citações
                                            if citation_counts:
                                                for i in citation_counts:
                                                    database = i.get_text()
                                                    total_trab = i.find_next_sibling("div", class_="trab")
                                                    if total_trab:
                                                        total_trab_text = total_trab.get_text().split("Total de trabalhos:")[1]
                                                    total_cite = i.find_next_sibling("div", class_="cita")
                                                    if total_cite:
                                                        total_cite_text = total_cite.get_text().split("Total de citações:")[1]
                                                    fator_h = i.find_next_sibling("div", class_="fator").get_text() if i.find_next_sibling("div", class_="fator") else None
                                                    num_fator_h = float(fator_h.replace('Fator H:', '')) if fator_h else None
                                                    data_wos = i.find_next_sibling("div", class_="detalhes")
                                                    if data_wos:
                                                        try:
                                                            data_wos_text = data_wos.get_text().split("Data:")[1].strip()
                                                        except:
                                                            data_wos_text = data_wos.get_text()
                                                    # Converta os valores para tipos de dados adequados
                                                    total_trab = int(total_trab_text)
                                                    total_cite = int(total_cite_text)
                                                    citation_numbers = {
                                                        "Database": database,
                                                        "Total de trabalhos": total_trab,
                                                        "Total de citações": total_cite,
                                                        "Índice_H": num_fator_h,
                                                        "Data": data_wos_text
                                                    }
                                                    # Verifique se a subseção atual já existe no dicionário
                                                    if 'Citações' not in data_dict[titulo][section_name]:
                                                        data_dict[titulo][section_name]['Citações'] = {}  # Inicialize como uma lista vazia
                                                    data_dict[titulo][section_name]['Citações'] = citation_numbers
                                                    if verbose:
                                                        print(f'        {database:>15}: {total_trab:>3} trabalhos, {total_cite:>3} citações, {fator_h}, {data_wos}')

                                    ## TODO: Segmentar Livros
                                    # if subsection_name == 'Capítulos de livros publicados':
                                    #     current_subsection = subsection_name
                                    #     data_dict[titulo][section_name]['Capítulos de livros publicados'] = {}
                                    #     sibling = div_artigo.findChild()


                            ## Encontrar a div irmã de div subseção com classe layout-cell-12 com artigos
                            vals_jcr = []
                            div_artigo_geral = data_cell.findChild("div", id="artigos-completos")
                            if verbose:
                                print(f'Encontrada {len(div_artigo_geral)} div geral de artigos')
                            if div_artigo_geral:
                                divs_artigos = div_artigo_geral.find_all('div', class_='artigo-completo')
                                if verbose:
                                    print(len(divs_artigos), 'divs de artigos')
                                current_data = {}  # Criamos um dicionário para armazenar os dados da subseção atual
                                if divs_artigos:                              
                                    for div_artigo in divs_artigos:
                                        data_dict[titulo][section_name]['Artigos completos publicados em periódicos'] = {}
                                        ## Extrair filhos da classes de artigos completos que estão à frente
                                        sibling = div_artigo.findChild()
                                        while sibling:
                                            classes = sibling.get('class', [])
                                            if 'layout-cell-1' in classes:  # Data key
                                                key = sibling.find("div", class_="layout-cell-pad-6 text-align-right").get_text().strip()
                                                sibling = sibling.find_next_sibling()
                                                if sibling and 'layout-cell-11' in sibling.get('class', []):  # Check if value is present
                                                    val = sibling.find("div", class_="layout-cell-pad-6").get_text().strip().replace('\n', '').replace('\t','')
                                                    info_dict = {
                                                        'data-issn': 'NULL',
                                                        'impact-factor': 'NULL',  
                                                        'jcr-year': 'NULL',
                                                    }
                                                    # Remova as tags span da div
                                                    for span in sibling.find_all('span'):
                                                        span.extract()
                                                    val_text = sibling.get_text(strip=True).strip().replace('\n',' ').replace('\t','')
                                                    current_data[key] = val_text
                                                    if verbose:
                                                        print(len(current_data.values()), key, val)
                                                    sup_element = sibling.find('sup')
                                                    if sup_element:
                                                        raw_jcr_data = sup_element.get_text()
                                                        # print('sup_element:',sup_element)
                                                        img_element = sup_element.find('img')
                                                        # print('img_element:',img_element)
                                                        if img_element:
                                                            original_title = img_element.get('original-title')
                                                            if original_title:
                                                                info_list = original_title.split('<br />') if original_title.split('<br />') else original_title
                                                                if info_list != 'NULL':
                                                                    issn = self.format_string(img_element.get('data-issn'))
                                                                    if verbose:
                                                                        print(f'impact-factor: {info_list[1].split(": ")[1]}')
                                                                    info_dict = {
                                                                        'data-issn': issn,
                                                                        'impact-factor': info_list[1].split(': ')[1],
                                                                        'jcr-year': info_list[1].split(': ')[0].replace('Fator de impacto ','').replace('(','').replace(')',''),
                                                                        'journal': info_list[0],
                                                                    }
                                                            else:
                                                                if verbose:
                                                                    print('Entrou no primeiro Else')
                                                                issn = self.format_string(img_element.get('data-issn'))
                                                                info_dict = {
                                                                    'data-issn': issn,
                                                                    'impact-factor': 'NULL',
                                                                    'jcr-year': 'NULL',
                                                                    'journal': 'NULL',
                                                                }
                                                    else:
                                                        if verbose:
                                                                    print('Entrou no segundo Else')
                                                        info_dict = {
                                                            'data-issn': 'NULL',
                                                            'impact-factor': 'NULL',
                                                            'jcr-year': 'NULL',
                                                            'journal': 'NULL',
                                                        }
                                                    vals_jcr.append(info_dict)
                                                    if verbose:
                                                        print(f'         {info_dict}')
                                                if 'JCR' not in data_dict:
                                                    data_dict['JCR'] = []
                                                if verbose:
                                                    print(len(vals_jcr))
                                                data_dict['JCR'] = vals_jcr
                                            elif sibling.name == 'br' and 'clear' in sibling.get('class', []):  # Fim de seção/subseção
                                                next_sibling = sibling.find_next_sibling()
                                                if next_sibling and 'clear' in next_sibling.get('class', []):
                                                    sibling = None
                                                else:
                                                    if current_data:
                                                        converted_data = Neo4jPersister.convert_to_primitives(current_data)
                                                        data_dict[titulo][section_name]['Artigos completos publicados em periódicos'] = converted_data
                                            if sibling:
                                                sibling = sibling.find_next_sibling()
                        else:
                            while sibling:
                                classes = sibling.get('class', [])
                                if 'cita-artigos' in classes:  # Subsection start
                                    subsection_name = sibling.find('b').get_text().strip()
                                    current_subsection = subsection_name
                                    if verbose:
                                        print(f'    Subseção: {subsection_name}')
                                    data_dict[titulo][section_name][current_subsection] = {}
                                    current_data = {}  # Inicializamos o dicionário de dados da subseção atual
                                elif 'layout-cell-1' in classes:  # Data key
                                    key = sibling.find("div", class_="layout-cell-pad-6 text-align-right").get_text().strip()
                                    sibling = sibling.find_next_sibling()
                                    if sibling and 'layout-cell-11' in sibling.get('class', []):  # Check if value is present
                                        val = sibling.find("div", class_="layout-cell-pad-6").get_text().strip().replace('\n', '').replace('\t','')
                                        current_data[key] = val
                                elif sibling.name == 'br' and 'clear' in sibling.get('class', []):  # Subsection or section end
                                    next_sibling = sibling.find_next_sibling()
                                    if next_sibling and 'clear' in next_sibling.get('class', []):
                                        sibling = None
                                    else:
                                        if current_subsection:
                                            data_dict[titulo][section_name][current_subsection] = Neo4jPersister.convert_to_primitives(current_data)  # Armazenamos os dados da subseção atual
                                if sibling:
                                    sibling = sibling.find_next_sibling()
        
        # Verifique se os dados dos tooltips estão presentes no objeto soup
        if 'tooltips' in soup.attrs:
            tooltips_data = soup.attrs['tooltips']
            agg = []
            for tooltip in tooltips_data:
                agg_data = {}
                # Extração do ano JCR a partir do "original_title"
                if tooltip.get("original_title"):
                    jcr_year = tooltip["original_title"].split(': ')[0].replace('Fator de impacto ','').replace('(','').replace(')','')
                    agg_data["jcr-ano"] = jcr_year
                # Adicionar todas as chaves e valores do tooltip ao dicionário agg_data
                for key, value in tooltip.items():
                    agg_data[key] = value
                agg.append(agg_data)
            data_dict['JCR2'] = agg
        else:
            print('Não foram achados os dados de tooltip')
            print(soup.attrs)
        return data_dict

    def extract_tit3_soup(self, soup, data_dict=None, verbose=True):
        if data_dict is None:
            data_dict = {}
        elm_main_cell = soup.find("div", class_="layout-cell-pad-main")
        divs_title_wrapper = elm_main_cell.find_all('div', class_='title-wrapper')
        # Títulos da seção 'Eventos'
        tit3 = ['Eventos']
        for div_title_wrapper in divs_title_wrapper:
            # Encontre o título do bloco
            try:
                titulo = div_title_wrapper.find('h1').text.strip()
            except:
                titulo = 'Não disponível na tag h1 do Currículo Lattes'
            data_cells = div_title_wrapper.find_all("div", class_="data-cell")
            # Verifique se o título está na lista 'tit3'
            if titulo in tit3:
                if verbose:
                    print(f'Título: {titulo}')
                data_dict[titulo] = {}  # Inicialize o dicionário para o título 'Eventos'
                for data_cell in data_cells:
                    sections = data_cell.find_all("div", class_="inst_back")
                    if verbose:
                        print(len(sections), 'seções')
                    for section in sections:
                        section_name = section.find('b').get_text().strip()
                        data_dict[titulo][section_name] = []
                        if verbose:
                            print(section_name)
                        sibling = section.find_next_sibling()
                        current_data = {}  # Criamos um dicionário para armazenar os dados da subseção atual
                        while sibling:
                            classes = sibling.get('class', [])
                            if 'layout-cell-1' in classes:  # Data key
                                key = sibling.find("div", class_="layout-cell-pad-6 text-align-right").get_text().strip()
                                sibling = sibling.find_next_sibling()

                                if sibling and 'layout-cell-11' in sibling.get('class', []):  # Check if value is present
                                    val = sibling.find("div", class_="layout-cell-pad-6").get_text().strip().replace('\n', '').replace('\t','')
                                    current_data[key] = val
                                    if verbose:
                                        print(len(current_data.values()), key, val)
                            elif sibling.name == 'br' and 'clear' in sibling.get('class', []):  # Fim de seção/subseção
                                next_sibling = sibling.find_next_sibling()
                                if next_sibling and 'clear' in next_sibling.get('class', []):
                                    sibling = None
                                else:
                                    if current_data:
                                        converted_data = Neo4jPersister.convert_to_primitives(current_data)
                                        data_dict[titulo][section_name] = converted_data
                            if sibling:
                                sibling = sibling.find_next_sibling()
        return data_dict

    def extract_data(self, soup):
        """
        Aggregates data from various dictionary sources into a consolidated nested dictionary, 
        ensuring that all nested lists within the dictionaries are transformed into nested dictionaries.
        Parameters:
        - soup: BeautifulSoup object, representing the parsed HTML content.
        Returns:
        - dict: An aggregated dictionary containing the consolidated data.
        """
        self.soup = soup
        
        def convert_list_to_dict(lst):
            """
            Converts a list into a dictionary with indices as keys.
            
            Parameters:
            - lst: list, input list to be transformed.
            
            Returns:
            - dict: Transformed dictionary.
            """
            return {str(i): item for i, item in enumerate(lst)}

        def merge_dict(d1, d2):
            """
            Recursively merges two dictionaries, transforming nested lists into dictionaries.
            Parameters:
            - d1: dict, the primary dictionary into which data is merged.
            - d2: dict or list, the secondary dictionary or list from which data is sourced.
            Returns:
            - None
            """
            # If d2 is a list, convert it to a dictionary first
            if isinstance(d2, list):
                d2 = convert_list_to_dict(d2)
            
            for key, value in d2.items():
                if isinstance(value, list):
                    d2[key] = convert_list_to_dict(value)
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    merge_dict(d1[key], value)
                else:
                    d1[key] = value

        # Extract necessary information from soup
        elm_main_cell = soup.find("div", class_="layout-cell-pad-main")
        info_list = [x.strip() for x in elm_main_cell.find("div", class_="infpessoa").get_text().split('\n') if x.strip() !='']
        name = info_list[0]

        # Initialization of the aggregated_data dictionary
        aggregated_data = {"labels": "Person", "name": name, "InfPes": info_list, "Resumo": [elm_main_cell.find("p", class_="resumo").get_text().strip()]}

        # Data extraction and merging
        for data_extraction_func in [self.extract_producoes, self.extract_tit1_soup, self.extract_tit2_soup, self.extract_tit3_soup]:
            extracted_sections = data_extraction_func(soup)
            for title, data in extracted_sections.items():
                if title not in aggregated_data:
                    aggregated_data[title] = {}
                merge_dict(aggregated_data[title], data)
        return aggregated_data

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
    def __init__(self, institution, unit, term):
        self.driver = LattesScraper.connect_driver()
        self.session = requests.Session()
        self.institution = institution
        self.unit = unit
        self.term = term
        self.base_url = 'http://buscatextual.cnpq.br/buscatextual/busca.do?buscarDoutores=true&buscarDemais=true&textoBusca='
        self.delay = 30

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
    def connect_driver():
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
        url_busca = 'http://buscatextual.cnpq.br/buscatextual/busca.do?buscarDoutores=true&buscarDemais=true&textoBusca='
        driver.get(url_busca) # acessa a url de busca do CNPQ   
        driver.set_window_position(-20, -10)
        driver.set_window_size(170, 1896)
        driver.mouse = webdriver.ActionChains(driver)
        return driver

    def wait_for_element(self, css_selector: str, ignored_exceptions=None):
        """
        Waits for the element specified by the CSS selector to load.
        :param css_selector: CSS selector of the element to wait for
        :param ignored_exceptions: List of exceptions to ignore
        """
        WebDriverWait(self.driver, self.delay, ignored_exceptions=ignored_exceptions).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)))

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

    def find_terms(self, NOME, instituicao, unidade, termo, delay, limite):
        """
        Função para manipular o HTML até abir a página HTML de cada currículo   
        Parâmeteros:
            - NOME: É o nome completo de cada pesquisador
            - Instituição, unidade e termo: Strings a buscar no currículo para reduzir duplicidades
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
        elm_vinculo = None
        qte_resultados = 0
        ## Receber a quantidade de opções ao ler elementos de resultados
        duvidas   = []
        force_break_loop = False
        try:
            # Wait and fetch the number of results
            css_resultados = ".resultado"
            WebDriverWait(self.driver, delay, ignored_exceptions=ignored_exceptions).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_resultados)))
            resultados = self.driver.find_elements(By.CSS_SELECTOR, css_resultados)
            ## Ler quantidade de resultados apresentados pela busca de nome
            try:
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
            except Exception as e1:
                print('  ERRO!! Currículo não disponível no Lattes')
                return None, NOME, np.NaN, e1, self.driver
            
            ## Escolher função a partir da quantidade de resultados da lista apresentada na busca
            ## Ao achar clica no elemento elm_vinculo com link do nome para abrir o currículo
            numpaginas = self.paginar(self.driver)
            if numpaginas == [] and qte_resultados==1:
                # capturar link para o primeiro nome resultado da busca
                try:
                    css_linknome = ".resultado > ol:nth-child(1) > li:nth-child(1) > b:nth-child(1) > a:nth-child(1)"
                    WebDriverWait(self.driver, delay).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, css_linknome)))            
                    elm_vinculo  = self.driver.find_element(By.CSS_SELECTOR, css_linknome)
                    nome_vinculo = elm_vinculo.text
                except Exception as e2:
                    print('  ERRO!! Ao encontrar o primeiro resultado da lista de nomes:', e2)
                    
                    # Call the handle stale file_error function
                    if self.handle_stale_file_error(self.driver):
                        # If the function returns True, it means the error was resolved.
                        # try to get the nome_vinculo again:
                        try:
                            elm_vinculo  = self.driver.find_element(By.CSS_SELECTOR, css_linknome)
                            nome_vinculo = elm_vinculo.text
                        except Exception as e3:
                            print('  ERRO!! Servidor CNPq indisponível no momento, tentar em alguns minutos:', e3)
                            return None, NOME, np.NaN, e3, self.driver
                    else:
                        # If the function returns False, it means the error was not resolved within the given retries.
                        return None, NOME, np.NaN, e2, self.driver

                    print('  Não foi possível extrair por falha no servidor do CNPq:',e)
                    return None, NOME, np.NaN, e2, self.driver
                # print('Clicar no nome único:', nome_vinculo)
                try:
                    self.retry(ActionChains(self.driver).click(elm_vinculo).perform(),
                        wait_ms=20,
                        limit=limite,
                        on_exhaust=(f'  Problema ao clicar no link do nome. {limite} tentativas sem sucesso.'))   
                except Exception as e4:
                    print('  ERRO!! Ao clicar no único nome encontrado anteriormente',e)
                    return None, NOME, np.NaN, e4, self.driver
            
            ## Quantidade de resultados até 10 currículos, acessados sem paginação
            else:
                print(f'       {qte_resultados:>3} homônimos de: {NOME}')
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
                    try:
                        numpaginas = self.paginar(self.driver)
                        # print(f'       Iteração: {iteracoes}. Páginas sendo lidas: {numpaginas}')
                        css_resultados = ".resultado"
                        WebDriverWait(self.driver, delay).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, css_resultados)))
                        resultados = self.driver.find_elements(By.CSS_SELECTOR, css_resultados)
                    except Exception as e:
                        print('  ERRO!! Ao paginar:',e)
                    ## iterar em cada resultado
                    for n,i in enumerate(resultados):
                        linhas = i.text.split('\n\n')
                        # print(linhas)
                        if 'Stale file handle' in str(linhas):
                            return np.NaN, NOME, np.NaN, 'Stale file handle', self.driver
                        for m,linha in enumerate(linhas):
                            # print(f'\nOrdem da linha: {m+1}, de total de linhas {len(linhas)}')
                            # print('Conteúdo da linha:',linha.lower())
                            # print(linha)
                            try:
                                if instituicao.lower() in linha.lower() or unidade.lower() in linha.lower() or termo.lower() in linha.lower():
                                    # print('Vínculo encontrado!')
                                    count=m
                                    while get_jaro_distance(linhas[count].split('\n')[0], str(NOME)) < 0.75:
                                        count-=1
                                    # print('       Identificado vínculo no resultado:', m+1)
                                    found = m+1
                                    # nome_vinculo = linhas[count].replace('\n','\n       ').strip()
                                    # print(f'       Achado: {nome_vinculo}')
                                    try:
                                        css_vinculo = f".resultado > ol:nth-child(1) > li:nth-child({m+1}) > b:nth-child(1) > a:nth-child(1)"
                                        # print('\nCSS_SELECTOR usado:', css_vinculo)
                                        WebDriverWait(self.driver, delay).until(
                                            EC.presence_of_element_located((By.CSS_SELECTOR, css_vinculo)))            
                                        elm_vinculo  = self.driver.find_element(By.CSS_SELECTOR, css_vinculo)
                                        nome_vinculo = elm_vinculo.text
                                        # print('Elemento retornado:',nome_vinculo)
                                        self.retry(ActionChains(self.driver).click(elm_vinculo).perform(),
                                            wait_ms=200,
                                            limit=limite,
                                            on_exhaust=(f'  Problema ao clicar no link do nome. {limite} tentativas sem sucesso.'))            
                                    except Exception as e5:
                                        print('  ERRO!! Ao achar o link do nome com múltiplos resultados')
                                        return np.NaN, NOME, np.NaN, e5, self.driver
                                    force_break_loop = True
                                    break
                            except Exception as e6:
                                traceback_str = ''.join(traceback.format_tb(e6.__traceback__))
                                print('  ERRO!! Ao procurar vínculo com currículos achados')    
                                print(e6,traceback_str)
                            ## Caso percorra toda lista e não encontre vínculo adiciona à dúvidas quanto ao nome
                            if m==(qte_resultados):
                                print(f'Nenhuma referência à {instituicao} ou ao {unidade} ou ao termo {termo}')
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
                    print(f'       Escolhido homônimo {found}: {nome_vinculo}')
                else:
                    print(f'       Não foi possível identificar o vínculo de: {NOME}')
                    duvidas.append(NOME)
            try:
                elm_vinculo.text
                # print(f'Nomes: {NOME} | {elm_vinculo.text}')
            except:
                return None, NOME, np.NaN, 'Vínculo não encontrado', self.driver
        except exceptions.TimeoutException:
            print("  ERRO 8a!! O tempo limite de espera foi atingido.")
            return None, NOME, np.NaN, "TimeoutException", self.driver
        except exceptions.WebDriverException as e7:
            print("  ERRO 8b!! Problema ao interagir com o driver.")
            return None, NOME, np.NaN, e7, self.driver
        except Exception as e8:
            print("  ERRO 8c!! Erro no servidor do CNPq (Stale file handler).")
            print(f'  {e8}')
            
            # Call the handle stale file_error function
            if self.handle_stale_file_error(self.driver):
                # If the function returns True, it means the error was resolved.
                # try to get the nome_vinculo again:
                try:
                    elm_vinculo  = self.driver.find_element(By.CSS_SELECTOR, css_linknome)
                    nome_vinculo = elm_vinculo.text
                except Exception as e9:
                    print('  ERRO 9!! Servidor CNPq indisponível no momento, tentar em alguns minutos:', e9)
                    return None, NOME, np.NaN, e9, self.driver
            else:
                # If the function returns False, it means the error was not resolved within the given retries.
                return None, NOME, np.NaN, e8, self.driver

        # Verifica antes de retornar para garantir que elm_vinculo foi definido
        if elm_vinculo is None:
            print("Vínculo não foi definido.")
            return None, NOME, np.NaN, 'Vínculo não encontrado', self.driver
        # Retorna a saída de sucesso
        return elm_vinculo, np.NaN, np.NaN, np.NaN, self.driver

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

    def return_search_page(self):
        url_busca = 'http://buscatextual.cnpq.br/buscatextual/busca.do?buscarDoutores=true&buscarDemais=true&textoBusca='
        self.driver.get(url_busca) # acessa a url de busca do CNPQ    

    def check_and_click_vinculo(self, elm_vinculo):
        if elm_vinculo is None:
            logging.info("Nenhum dos vínculos esperados encontrado no currículo...")
            self.return_search_page()
            return

        try:
            time.sleep(0.8)  # Um pequeno atraso para garantir que a página tenha carregado

            # Esperar até que o botão esteja visível e clicável
            btn_abrir_curriculo = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "#idbtnabrircurriculo"))
            )

            # Clicar no botão para abrir o currículo
            btn_abrir_curriculo.click()
        except WebDriverException as e:
            print(f"Erro ao clicar no botão 'Abrir Currículo': {e}")
            self.return_search_page()

    # Clicar no nome achado e clicar no botão "Abrir Currículo" da janela pop-up
    # def check_and_click_vinculo(self, elm_vinculo):
    #     if elm_vinculo is None:
    #         logging.info("Nenhum dos vínculos esperados encontrado no currículo...")
    #         self.return_search_page()
    #     # try:
    #     #     logging.info(f'Vínculo encontrado no currículo de nome: {elm_vinculo.text}')
    #     # except AttributeError:
    #     #     logging.error("Vínculo não encontrado, passando para o próximo nome...")
    #     #     self.return_search_page()

    #     try:
    #         time.sleep(0.8) # para evitar erros em conexão lenta
    #         # Aguardar até estar pronto para ser clicado       
    #         btn_abrir_curriculo = WebDriverWait(self.driver, delay).until(
    #             EC.element_to_be_clickable((By.CSS_SELECTOR, "#idbtnabrircurriculo")))
    #         time.sleep(0.3) # para evitar erros em conexão lenta
    #         try:
    #             # Clicar no botão para abrir o currículo       
    #             ActionChains(self.driver).click(btn_abrir_curriculo).perform()
    #         except TimeoutException:
    #             print("       New window did not open. Clicking again.")
    #             ActionChains(self.driver).click(btn_abrir_curriculo).perform()
    #     except WebDriverException:
    #         print('       Currículo não encontrado para o nome buscado.')
    #         self.return_search_page()

    def extract_tooltip_data(self, retries=3, delay=1, verbose=False):
        """
        Extracts tooltip data from articles section using Selenium with retry logic.
        :param retries: Number of retries if element is not interactable.
        :param delay: Wait time before retrying.
        :return: List of dictionaries containing the extracted tooltip data.
        """
        tooltip_data_list = []

        try:
            WebDriverWait(self.driver, 20).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, "#artigos-completos img.ajaxJCR")))
            layout_cells = self.driver.find_elements(By.CSS_SELECTOR, '#artigos-completos .layout-cell-11')
            if verbose:
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
                    time.sleep(delay)  # Dando tempo para o tooltip ser carregado
                    
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

    def search_profile(self, name, instituicao, unidade, termo):
        try:
            # Find terms to interact with the web page and extract the profile
            profile_element, _, _, _, _ = self.find_terms(
                name, 
                instituicao,  
                unidade,  
                termo,  
                10,  
                3  
            )
            # print('Elemento encontrado:', profile_element)
            if profile_element:
                return profile_element
            else:
                logging.info(f'Currículo não encontrado: {name}')
                self.return_search_page()

        except requests.HTTPError as e:
            logging.error(f"HTTPError occurred: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Erro inesperado ao buscar: {str(e)}")
            return None

    def switch_to_new_window(self):
        # Espera até que uma nova janela seja aberta
        WebDriverWait(self.driver, 10).until(EC.number_of_windows_to_be(2))
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

    def close_popup(self, verbose=False):
        try:
            WebDriverWait(self.driver, self.delay).until(
                EC.element_to_be_clickable((By.ID, "idbtnfechar"))
                ).click()
            if verbose:
                print("Pop-up fechado com sucesso.")
        except Exception as e:
            print(f"Erro ao fechar o pop-up: {e}")

    def fill_name(self, NOME):
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

            search_button_selector = "#botaoBuscaFiltros"  # Atualizar conforme necessidade do tipo de perfil buscado
            WebDriverWait(self.driver, self.delay).until(EC.element_to_be_clickable((By.CSS_SELECTOR, search_button_selector)))
            search_button = self.driver.find_element(By.CSS_SELECTOR, search_button_selector)
            search_button.click()

        except Exception as e:
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            print(f'  ERRO!! Ao colar nome {NOME}: {e}')
            # traceback.print_exc()

    def scrape(self, name_list, instituicao, unidade, termo, json_filename, hdf5_filename, verbose=False):
        dict_list = []
        for k, name in enumerate(name_list):
            try:
                print(f'{k+1:>2}/{len(name_list)}: {name}')
                # Preenche o nome e realiza a busca
                self.fill_name(name)
                elm_vinculo = self.search_profile(name, instituicao, unidade, termo)
                if elm_vinculo:
                    # Clica no link do nome e no botão Abrir Currículo
                    self.check_and_click_vinculo(elm_vinculo)
                    # Muda para a nova janela aberta com o currículo
                    window_before = self.switch_to_new_window()
                    try:
                        tooltip_data_list = self.extract_tooltip_data()
                        if verbose:
                            print(f'       {len(tooltip_data_list):>003} tooltips encontrados')
                    except Exception as e:
                        print(f"Erro ao extrair tooltips: {e}")
                        # Se ocorrer um erro, tenta extrair os dados novamente
                        tooltip_data_list = self.extract_tooltip_data()
                        if verbose:
                            print(f"       Adicionando tooltips: {tooltip_data_list}")
                    page_source = self.driver.page_source
                    if page_source is not None:
                        soup = BeautifulSoup(page_source, 'html.parser')
                        if verbose:
                            print(f'       {len(soup):>003} elementos encontrados no objeto soup') ## ATÉ AQUI FUNCIONANDO BEM
                        soup.attrs['tooltips'] = tooltip_data_list
                        if soup:
                            # print('Extraindo dados do objeto Soup...')
                            try:
                                parse_soup_instance = SoupParser(self.driver)
                                data = parse_soup_instance.extract_data(soup)
                            except Exception as e:
                                print(f'       Erro com SoupParser: {e}')
                            if verbose:
                                try:
                                    print(f'       {len(data):>003} elementos extract_data encontrados no objeto soup')
                                except Exception as e:
                                    print(f'       {e}')
                            # Chama métodos de conversão de dicionário individual
                            # parse_soup_instance.to_json(data, json_filename)
                            # parse_soup_instance.to_hdf5(data, hdf5_filename)
                            dict_list.append(data)
                            print(len(dict_list))
                        else:
                            logging.error(f"Não foi possível extrair dados do currículo: {name}")
                            print(f"Não foi gerado objeto soup para: {name}")
                else:
                    logging.info(f"Currículo não encontrado para: {name}")
            except Exception as e:
                logging.info(f"Currículo inexistente: {name}: {e}")

        self.driver.quit()
        return dict_list