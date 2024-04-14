# curriculumScrapper
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

## Configurar exibição dos dataframes do pandas na tela
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('colheader_justify', 'left')
pd.set_option('display.max_rows', 600)

delay = 10
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Utils:
    def __init__(self, data):
        """
        Inicializa a classe com os dados dos experimentos.
        :param data: DataFrame com os dados dos experimentos.
        """
        self.data = data
        

    def strfdelta(self, tdelta, fmt='{H:02}h {M:02}m {S:02}s', inputtype='timedelta'):
        from string import Formatter
        
        """Convert a datetime.timedelta object or a regular number to a custom-formatted string, 
        just like the stftime() method does for datetime.datetime objects.

        The fmt argument allows custom formatting to be specified.  Fields can 
        include seconds, minutes, hours, days, and weeks.  Each field is optional.

        Some examples:
            '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
            '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
            '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
            '{H}h {S}s'                       --> '72h 800s'

        The inputtype argument allows tdelta to be a regular number instead of the  
        default, which is a datetime.timedelta object.  Valid inputtype strings: 
            's', 'seconds', 
            'm', 'minutes', 
            'h', 'hours', 
            'd', 'days', 
            'w', 'weeks'
        """

        # Convert tdelta to integer seconds.
        if inputtype == 'timedelta':
            remainder = int(tdelta.total_seconds())
        elif inputtype in ['s', 'seconds']:
            remainder = int(tdelta)
        elif inputtype in ['m', 'minutes']:
            remainder = int(tdelta)*60
        elif inputtype in ['h', 'hours']:
            remainder = int(tdelta)*3600
        elif inputtype in ['d', 'days']:
            remainder = int(tdelta)*86400
        elif inputtype in ['w', 'weeks']:
            remainder = int(tdelta)*604800

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
        from datetime import timedelta
            
        t=end-start

        tempo = timedelta(
            weeks   = t//(3600*24*7),
            days    = t//(3600*24),
            seconds = t,
            minutes = t//(60),
            hours   = t//(3600),
            microseconds=t//1000000,
            )
        fmt='{H:2}:{M:02}:{S:02}'
        return self.strfdelta(tempo)

    # https://sh-tsang.medium.com/tutorial-cuda-cudnn-anaconda-jupyter-pytorch-installation-in-windows-10-96b2a2f0ac57

    def check_path(self):
        try:
            # Tenta obter a variável de ambiente PATH
            path_output = subprocess.check_output("echo $PATH", shell=True).decode('utf-8').strip()
            return path_output
        except Exception as e:
            print("Erro ao obter PATH:", e)
        return path_output

    def check_nvcc(self):
        # Identify the operating system
        os_type = platform.system()

        # Depending on the operating system, alter the command
        if os_type == "Linux":
            nvcc_path = "/usr/local/cuda/bin/nvcc"
            # Check if the nvcc executable exists in the path
            if not os.path.exists(nvcc_path):
                print("NVCC not found in the expected location for Linux.")
                return
        elif os_type == "Windows":
            # Construct the potential paths where NVCC could be located
            cuda_paths = glob.glob('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin/nvcc.exe')
            if cuda_paths:
                nvcc_path = cuda_paths[0]  # Take the first match
            else:
                print("NVCC not found in the default installation paths for Windows.")
                return
        else:
            print("Unsupported Operating System.")
            return

        # Try to retrieve the NVCC version using the found path
        try:
            nvcc_output = subprocess.check_output([nvcc_path, '-V'], stderr=subprocess.STDOUT).decode()
            print(nvcc_output)
        except subprocess.CalledProcessError as e:
            print(f"Failed to execute NVCC: {e.output.decode()}")
        except OSError as e:
            print(f"NVCC not found: {e.strerror}")

    def try_amb(self):
        ## Visualizar versões dos principais componentes
        
        pyVer      = sys.version
        pipVer     = pip.__version__
        
        print('\nVERSÕES DAS PRINCIPAIS BIBLIOTECAS INSTALADAS NO ENVIROMENT')
        print('Interpretador em uso:', sys.executable)
        
        # Improved handling of the 'CONDA_DEFAULT_ENV' environment variable
        try:
            print('    Ambiente Conda ativado:', os.environ['CONDA_DEFAULT_ENV'])
        except KeyError:
            print('    Ambiente Conda ativado: Não disponível')
        
        print('     Python: ' + pyVer, '\n        Pip:', pipVer, '\n')

    def get_cpu_info_windows(self):
        import subprocess

        try:
            return subprocess.check_output("wmic cpu get Name", shell=True).decode('utf-8').split('\n')[1].strip()
        except:
            return "Informação não disponível"

    def get_cpu_info_unix(self):
        import subprocess
        try:
            return subprocess.check_output("lscpu", shell=True).decode('utf-8')
        except:
            try:
                return subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode('utf-8').strip()
            except:
                return "Informação não disponível"

    def try_cpu(self):
        # Métricas da CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        cpu_times_percent = psutil.cpu_times_percent(interval=1)

        # Informação específica do modelo do processador
        if platform.system() == "Windows":
            cpu_model = self.get_cpu_info_windows()
        else:
            cpu_model = self.get_cpu_info_unix()

        # Informações adicionais sobre o Processador
        cpu_brand = platform.processor()
        cpu_architecture = platform.architecture()[0]
        cpu_machine_type = platform.machine()
        
        # Métricas da Memória RAM
        ram = psutil.virtual_memory()
        total_ram = ram.total / (1024 ** 3)  # Em GB
        used_ram = ram.used / (1024 ** 3)  # Em GB
        
        # Métricas do Espaço em Disco
        disk = psutil.disk_usage('/')
        total_disk = disk.total / (1024 ** 3)  # Em GB
        used_disk = disk.used / (1024 ** 3)  # Em GB
        free_disk = (total_disk - used_disk)
        used_disk_percent = (used_disk / total_disk) * 100
        free_disk_percent = (1 - (used_disk / total_disk)) * 100

        # Exibição das Métricas
        print(f"\nMarca do Processador: {cpu_brand}")
        print(f"Modelo do Processador: {cpu_model}")
        print(f"Frequência da CPU: {np.round(cpu_freq.current,2)} MHz")
        # print(f"Tipo de Máquina: {cpu_machine_type}")
        print(f"Arquitetura do Processador: {cpu_architecture}")
        print(f"Número de CPUs físicas: {cpu_count_physical}")
        print(f"Número de CPUs lógicas: {cpu_count_logical}")
        print(f"Uso atual CPU: {cpu_percent}%")
        print(f"Tempos de CPU: user={cpu_times_percent.user}%, system={cpu_times_percent.system}%, idle={cpu_times_percent.idle}%")
        print(f"\nTotal de RAM: {total_ram:>5.2f} GB")
        print(f"Usado em RAM: {used_ram:>5.2f} GB")
        print(f"Espaço Total em disco: {total_disk:>7.2f} GB")
        print(f"Espaço em disco usado: {used_disk:>7.2f} GB {used_disk_percent:>4.1f}%")
        print(f"Espaço em disco livre: {free_disk:>7.2f} GB {free_disk_percent:>4.1f}%")

    def try_gpu(self):
        print('\nVERSÕES DOS DRIVERS CUDA, PYTORCH E GPU')
        try:
            self.check_nvcc()
        except Exception as e:
            print("NVCC não encontrado:",e,"\n")
        try:
            import torch
            print('    PyTorch:',torch.__version__)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print('Dispositivo:',device)
            print('Disponível :',device,torch.cuda.is_available(),' | Inicializado:',torch.cuda.is_initialized(),'| Capacidade:',torch.cuda.get_device_capability(device=None))
            print('Nome GPU   :',torch.cuda.get_device_name(0),'         | Quantidade:',torch.cuda.device_count())
        except Exception as e:
            print('  ERRO!! Ao configurar a GPU:',e,'\n')

    def try_folders(self, drives, pastas, pastasraiz):
        """
        Function to search for chromedriver in specified folders on both Windows and Unix-like systems.

        Parameters:
        drives (list): List of drives to search in.
        pastas (list): List of folder names to look into.
        pastasraiz (list): List of root folder names to look into.

        Returns:
        str: Valid path if found, otherwise raises a FileNotFoundError.

        Raises:
        FileNotFoundError: If chromedriver is not found in any of the specified locations.
        """
        caminho = None  
        for drive in drives:
            for i in pastas:
                for j in pastasraiz:
                    try:
                        caminho_testado = os.path.join(drive, i, j)
                        chromedriver_path = os.path.join(caminho_testado, 'chromedriver', 'chromedriver.exe' if os.name == 'nt' else 'chromedriver')

                        if os.path.isfile(chromedriver_path):
                            print(f"Listing files in: {caminho_testado}")
                            print(os.listdir(caminho_testado))
                            caminho = os.path.join(caminho_testado, '')
                            return caminho

                    except FileNotFoundError as e:
                        print(f"File not found: {e}")
                        print('Could not locate the working folder.')
                        
        if caminho is None:
            caminho='./home/mak/fioce/'
            raise FileNotFoundError("Chromedriver could not be located in the specified directories.")

        return caminho

    def try_browser(self, raiz):
        print('\nVERSÕES DO BROWSER E DO CHROMEDRIVER INSTALADAS')
        import platform
        from selenium import webdriver
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service

        try:
            # Caminho para o seu chromedriver
            if platform.system() == "Windows":
                driver_path=raiz+'\\chromedriver\\chromedriver.exe'
            else:
                driver_path=raiz+'chromedriver/chromedriver'
            print(driver_path)
            service = Service(driver_path)
            driver = webdriver.Chrome(service=service)
            str1 = driver.capabilities['browserVersion']
            str2 = driver.capabilities['chrome']['chromedriverVersion'].split(' ')[0]
            print(f'     Versão do browser: {str1}')
            print(f'Versão do chromedriver: {str2}')
            print()
            driver.quit()

            if str1[0:3] != str2[0:3]: 
                print(f"Versões incompatíveis, atualizar chromedriver!")
                print(f'  Baixar versão atualizada do Chromedriver em:')
                print(f'  https://googlechromelabs.github.io/chrome-for-testing/#stable')
                print(f'     Ex:. Chromedriver Versão 119 para Windows:')
                print(f'	   https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/119.0.6045.105/win64/chromedriver-win64.zip')
                print(f'     Ex:. Chromedriver Versão 119 para Linux:')
                print(f'       https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/119.0.6045.105/linux64/chromedriver-linux64.zip')
        except Exception as e:
            print(e)

    def try_chromedriver(self, caminho):
        try:
            import os
            os.listdir(caminho)
            raiz=caminho
        except Exception as e:
            raiz=caminho

        finally:
            print(raiz)
        return raiz    
    
    def definir_sistema(self, pastaraiz):
        """
        Function to define the system and prepare local folders.

        Parameters:
        pastaraiz (str): Root folder for the operation.

        Returns:
        tuple: A tuple containing the path (caminho), drive, and user (usuario).

        Raises:
        EnvironmentError: If the operating system is not recognized.
        """
        # Initialize variables
        caminho = ''
        drive = ''
        usuario = ''

        sistema_operacional = sys.platform

        try:
            if 'linux' in sistema_operacional:
                print('Sistema operacional Linux...')
                linux_users = ['mak/', 'marcos/']
                drive = '/home/'
                for user in linux_users:
                    temp_caminho = os.path.join(drive, user, pastaraiz)
                    if os.path.isdir(temp_caminho):
                        usuario = user
                        caminho = os.path.join(drive, usuario, pastaraiz, '')
                        break

            elif 'win32' in sistema_operacional:
                print('Sistema operacional Windows...')
                windows_users = ['\\Users\\marco\\', '\\Users\\marcos.aires\\']
                drive = 'C:'
                # print('Procurando em:')
                for user in windows_users:
                    temp_caminho = os.path.join(drive, user, pastaraiz)
                    print(f"  {temp_caminho}")
                    if os.path.isdir(temp_caminho):
                        usuario = user
                        caminho = os.path.join(drive, usuario, pastaraiz, '')
                        break
                    else:
                        pathzip, pathcsv, pathjson, pathfig, caminho, pathout = self.preparar_pastas(pastaraiz)
                        if os.path.isdir(caminho):
                            usuario = user
            else:
                raise EnvironmentError('SO não reconhecido')

        except FileNotFoundError as e:
            print('  ERRO!! Diretório não encontrado!')
            print(e)
        except EnvironmentError as e:
            print('  ERRO!! Sistema Operacional não suportado!')
            print(e)

        if not caminho:
            print('  ERRO!! Caminho não foi definido!')

        print(f'Pasta armazenagem local {caminho}\n')

        return caminho, drive, usuario

    def preparar_pastas(self, caminho):
        # caminho, drive, usuario = definir_sistema(pastaraiz)
        # caminho = drive+':/'+usuario+pastaraiz
        # caminho = drive+':/'+pastaraiz

        # Check if caminho is empty or None
        if not caminho:
            raise ValueError("Variável 'caminho' vazia. Não é possível criar os diretórios.")

        if os.path.isdir(caminho) is False:
            os.mkdir(caminho)
            if os.path.isdir(caminho+'/xml_zip'):
                print ('Pasta para os arquivo xml já existe!')
            else:
                os.mkdir(caminho+'/xml_zip')
                print ('Pasta para arquivo xml criada com sucesso!')
            if os.path.isdir(caminho+'/csv'):
                print ('Pasta para os arquivo CSV já existe!')
            else:
                os.mkdir(caminho+'/csv')
                print ('Pasta para arquivo CSV criada com sucesso!')
            if os.path.isdir(caminho+'/json'):
                print ('Pasta para os arquivo JSON já existe!')
            else:
                os.mkdir(caminho+'/json')
                print ('Pasta para JSON criada com sucesso!')
            if os.path.isdir(caminho+'/fig'):
                print ('Pasta para figuras já existe!')
            else:
                os.mkdir(caminho+'/fig')
                print ('Pasta para JSON criada com sucesso!')
        else:
            if os.path.isdir(caminho+'/xml_zip'):
                print ('Pasta para os xml já existe!')
            else:
                os.mkdir(caminho+'/xml_zip')
                print ('Pasta para xml criada com sucesso!')
            if os.path.isdir(caminho+'/csv'):
                print ('Pasta para os CSV já existe!')
            else:
                os.mkdir(caminho+'/csv')
                print ('Pasta para CSV criada com sucesso!')
            if os.path.isdir(caminho+'/json'):
                print ('Pasta para os JSON já existe!')
            else:
                os.mkdir(caminho+'/json')
                print ('Pasta para JSON criada com sucesso!')
            if os.path.isdir(caminho+'/fig'):
                print ('Pasta para figuras já existe!')
            else:
                os.mkdir(caminho+'/fig')
                print ('Pasta para figuras criada com sucesso!')
            if os.path.isdir(caminho+'/output'):
                print ('Pasta para saídas já existe!')
            else:
                os.mkdir(caminho+'/output')
                print ('Pasta para saídas criada com sucesso!')            

        pathzip  = caminho+'xml_zip/'
        pathcsv  = caminho+'csv/'
        pathjson = caminho+'json/'
        pathfig  = caminho+'fig/'
        pathaux  = caminho
        pathout  = caminho+'output/'

        print('\nCaminho da pasta raiz', pathaux)
        print('Caminho arquivos  XML', pathzip)
        print('Caminho arquivos JSON', pathjson)
        print('Caminho arquivos  CSV', pathcsv)
        print('Caminho para  figuras', pathfig)
        print('Pasta arquivos saídas', pathout)
        
        return pathzip, pathcsv, pathjson, pathfig, pathaux, pathout

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

class ParseSoup:
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
            
    def extract_tit1_soup(self, soup, data_dict=None, verbose=False):
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
                            key = i.find('div', class_='layout-cell-pad-5 text-align-right')
                            key_text = key.get_text().strip().replace('\n', ' ').replace('\t', '')
                            keys.append(key_text)
                            val = j.find('div', class_='layout-cell-pad-5')
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
                                key = sibling.find("div", class_="layout-cell-pad-5 text-align-right").get_text().strip()
                                sibling = sibling.find_next_sibling()

                                if sibling and 'layout-cell-9' in sibling.get('class', []):  # Check if value is present
                                    val = sibling.find("div", class_="layout-cell-pad-5").get_text().strip().replace('\n', '').replace('\t','')
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
                                                key = sibling.find("div", class_="layout-cell-pad-5 text-align-right").get_text().strip()
                                                sibling = sibling.find_next_sibling()
                                                if sibling and 'layout-cell-11' in sibling.get('class', []):  # Check if value is present
                                                    val = sibling.find("div", class_="layout-cell-pad-5").get_text().strip().replace('\n', '').replace('\t','')
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
                                    key = sibling.find("div", class_="layout-cell-pad-5 text-align-right").get_text().strip()
                                    sibling = sibling.find_next_sibling()
                                    if sibling and 'layout-cell-11' in sibling.get('class', []):  # Check if value is present
                                        val = sibling.find("div", class_="layout-cell-pad-5").get_text().strip().replace('\n', '').replace('\t','')
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

    def extract_tit3_soup(self, soup, data_dict=None, verbose=False):
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
                                key = sibling.find("div", class_="layout-cell-pad-5 text-align-right").get_text().strip()
                                sibling = sibling.find_next_sibling()

                                if sibling and 'layout-cell-11' in sibling.get('class', []):  # Check if value is present
                                    val = sibling.find("div", class_="layout-cell-pad-5").get_text().strip().replace('\n', '').replace('\t','')
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
        for data_extraction_func in [self.extract_tit1_soup, self.extract_tit2_soup, self.extract_tit3_soup]:
            extracted_sections = data_extraction_func(soup)
            for title, data in extracted_sections.items():
                if title not in aggregated_data:
                    aggregated_data[title] = {}
                merge_dict(aggregated_data[title], data)
        return aggregated_data

class LattesScraper:
    def __init__(self, driver, institution, unit, term):
        self.base_url = 'http://buscatextual.cnpq.br'
        self.session = requests.Session()
        self.driver = driver
        self.delay = 30

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

    def fill_name(self, NOME):
        '''
        Move cursor to the search field and fill in the specified name.
        '''
        if self.driver is None:
            logging.error("O driver não foi inicializado corretamente.")
            return
        try:
            nome = lambda: self.driver.find_element(By.CSS_SELECTOR, ("#textoBusca"))
            nome().send_keys(Keys.CONTROL + "a")
            nome().send_keys(NOME)
        except Exception as e:
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            print(f'  ERRO!! Ao colar nome para buscar.') #, {traceback_str}
        try:            
            seletorcss = 'div.layout-cell-12:nth-child(8) > div:nth-child(1) > div:nth-child(1) > a:nth-child(1)'
            WebDriverWait(self.driver, self.delay).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, seletorcss))).click()
            
            seletorcss = "#botaoBuscaFiltros"
            WebDriverWait(self.driver, self.delay).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, seletorcss)))
        except Exception as e:
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            print(f'  ERRO!! Ao clicar no botão Buscar.\n{e}, {traceback_str}')

    def return_search_page(self):
        url_busca = 'http://buscatextual.cnpq.br/buscatextual/busca.do?buscarDoutores=true&buscarDemais=true&textoBusca='
        self.driver.get(url_busca) # acessa a url de busca do CNPQ        

    # # Clicar no nome achado e clicar no botão "Abrir Currículo" da janela pop-up
    # def check_and_click_vinculo(self, elm_vinculo, max_retries=5, retry_interval=10):
    #     for attempt in range(max_retries):
    #         if elm_vinculo is None:
    #             logging.info("Nenhum dos vínculos esperados encontrado no currículo...")
    #             self.return_search_page()
    #         try:
    #             btn_abrir_curriculo = WebDriverWait(self.driver, delay).until(
    #                 EC.element_to_be_clickable((By.CSS_SELECTOR, "#idbtnabrircurriculo")))                    
    #             self.retry(ActionChains(self.driver).click(btn_abrir_curriculo).perform(),
    #                 wait_ms=1000,
    #                 limit=max_retries,
    #                 on_exhaust=(f'  Problema ao clicar no link do nome. {max_retries} tentativas sem sucesso.'))
    #             return
    #         except WebDriverException:
    #             print('       Não foi possível abrir o currículo.')
    #             self.return_search_page()

    # Clicar no nome achado e clicar no botão "Abrir Currículo" da janela pop-up
    def check_and_click_vinculo(self, elm_vinculo):
        if elm_vinculo is None:
            logging.info("Nenhum dos vínculos esperados encontrado no currículo...")
            self.return_search_page()
        # try:
        #     logging.info(f'Vínculo encontrado no currículo de nome: {elm_vinculo.text}')
        # except AttributeError:
        #     logging.error("Vínculo não encontrado, passando para o próximo nome...")
        #     self.return_search_page()

        try:
            time.sleep(0.4) # para evitar erros em conexão lenta
            # Aguardar até estar pronto para ser clicado       
            btn_abrir_curriculo = WebDriverWait(self.driver, delay).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "#idbtnabrircurriculo")))
            time.sleep(0.3) # para evitar erros em conexão lenta
            try:
                # Clicar no botão para abrir o currículo       
                ActionChains(self.driver).click(btn_abrir_curriculo).perform()
            except TimeoutException:
                print("       New window did not open. Clicking again.")
                ActionChains(self.driver).click(btn_abrir_curriculo).perform()
        except WebDriverException:
            print('       Currículo não encontrado para o nome buscado.')
            self.return_search_page()
        
    def switch_to_new_window(self):
        window_before = self.driver.current_window_handle
        WebDriverWait(self.driver, self.delay).until(EC.number_of_windows_to_be(2))
        window_after = self.driver.window_handles
        new_window = [x for x in window_after if x != window_before][0]
        self.driver.switch_to.window(new_window)

    def switch_back_to_original_window(self):
        current_window = self.driver.current_window_handle
        original_window = [x for x in self.driver.window_handles if x != current_window][0]
        self.driver.close()
        self.driver.switch_to.window(original_window) # Voltar para janela original

    def extract_tooltip_data(self, retries=3, delay=0.2) -> list[dict]:
        """
        Extracts tooltip data from articles section using Selenium with retry logic.
        :param retries: Number of retries if element is not interactable.
        :param delay: Wait time before retrying.
        :return: List of dictionaries containing the extracted tooltip data.
        """
        tooltip_data_list = []

        try:
            self.wait_for_element("#artigos-completos img.ajaxJCR", [TimeoutException])
            layout_cells = self.driver.find_elements(By.CSS_SELECTOR, '#artigos-completos .layout-cell-11 .layout-cell-pad-5')
            for cell in layout_cells:
                tooltip_data = {}
                try:
                    elem_citado = cell.find_element(By.CSS_SELECTOR, '.citado')
                    tooltip_data.update(self.extract_data_from_cvuri(elem_citado))
                except (ElementNotInteractableException, NoSuchElementException):
                    pass

                try:
                    doi_elem = cell.find_element(By.CSS_SELECTOR, "a.icone-producao.icone-doi")
                    tooltip_data["doi"] = doi_elem.get_attribute("href")
                except NoSuchElementException:
                    tooltip_data["doi"] = None

                current_retries = retries
                while current_retries > 0:
                    try:
                        self.wait_for_element("img.ajaxJCR", [TimeoutException])
                        tooltip_elem = cell.find_element(By.CSS_SELECTOR, "img.ajaxJCR")
                        if tooltip_elem.is_displayed() and tooltip_elem.size['height'] > 0:
                            ActionChains(self.driver).move_to_element(tooltip_elem).perform()
                            original_title = tooltip_elem.get_attribute("original-title")
                            match = re.search(r"Fator de impacto \(JCR \d{4}\): (\d+\.\d+)", original_title)
                            tooltip_data["impact-factor"] = match.group(1) if match else None
                            tooltip_data["original_title"] = original_title.split('<br />')[0].strip()
                            break  # Saída do loop se a ação foi bem-sucedida
                    except (ElementNotInteractableException, NoSuchElementException, TimeoutException):
                        time.sleep(delay)  # Espera antes de tentar novamente
                        current_retries -= 1  # Decrementa a contagem de retentativas

                tooltip_data_list.append(tooltip_data)

            print(f'       {len(tooltip_data_list):>003} artigos extraídos')
            logging.info(f'{len(tooltip_data_list):>003} artigos extraídos')

        except TimeoutException as e:
            logging.info("Publicações de artigos não detectada no currículo")
        except Exception as e:
            logging.error(f"Erro inesperado ao extrair tooltips: {e}")

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
        
    def scrape(self, driver, name_list, instituicao, unidade, termo, json_filename, hdf5_filename):
        dict_list=[]
        for k, name in enumerate(name_list):
            try:
                print(f'{k+1:>2}/{len(name_list)}: {name}')
                self.fill_name(name)
                elm_vinculo = self.search_profile(name, instituicao, unidade, termo)
                
                # Clica no link do nome e no botão Abrir Currículo
                self.check_and_click_vinculo(elm_vinculo)
                
                if elm_vinculo:
                    # Muda para a nova janela aberta com o currículo
                    self.switch_to_new_window()

                    try:
                        tooltip_data_list = self.extract_tooltip_data()
                    except:
                        print(f"Erro ao extrair tooltips, tentando novamente...")
                        tooltip_data_list = self.extract_tooltip_data()
                    
                    page_source = driver.page_source
                    if page_source is not None:
                        soup = BeautifulSoup(page_source, 'html.parser')
                        soup.attrs['tooltips'] = tooltip_data_list                 
                        if soup:
                            # print('Extraindo dados do objeto Soup...')
                            parse_soup_instance = ParseSoup(driver)
                            data = parse_soup_instance.extract_data(soup)
                            # Chama métodos de conversão de dicionário individual
                            # parse_soup_instance.to_json(data, json_filename)
                            # parse_soup_instance.to_hdf5(data, hdf5_filename)
                            dict_list.append(data)
                    else:
                        logging.error(f"Não foi possível extrair dados do currículo: {name}")
                        continue

                    # Fechar janela do currículo e voltar para página de busca
                    self.switch_back_to_original_window()

                    # Clicar no botão para fechar janela pop-up
                    btn_fechar_curriculo = WebDriverWait(driver, delay).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "#idbtnfechar")))
                    ActionChains(driver).click(btn_fechar_curriculo).perform()    

                else:
                    logging.info(f"Currículo não encontrado para: {name}")
                    continue

                self.return_search_page()
                # logging.info('Successfully restarded extraction.')
            # except TimeoutException as e:
            #     logging.error(f"Sem resposta antes do timeout para: {name}: {str(e)}")
            except Exception as e:
                logging.info(f"Currículo inexistente: {name}: {str(e)}")
        driver.quit()
        return dict_list
    
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
            info_nam = dic.get('name',{})
            nomes_curriculos.append(info_nam)
            info_pes = dic.get('InfPes', {})
            if type(info_pes) == dict:
                processar = info_pes.values()
            elif type(info_pes) == list:
                processar = info_pes
            for line in processar:
                try:
                    id_pattern = re.search(r'ID Lattes: (\d+)', line)
                    dt_pattern = re.search(r'\d{2}/\d{2}/\d{4}', line)
                    id_lattes =  id_pattern.group(1) if id_pattern else None
                    if id_lattes:
                        ids_lattes_grupo.append(id_lattes)
                    data_atualizacao = dt_pattern.group() if dt_pattern else None
                    if data_atualizacao:
                        dts_atualizacoes.append(data_atualizacao)
                        tempo_atualizado = self.dias_desde_atualizacao(data_atualizacao)
                        tempos_defasagem.append(tempo_atualizado)                    
                except Exception as e:
                    pass
                    # print(e)

            info_art = dic.get('Produções', {}).get('Produção bibliográfica', {}).get('Artigos completos publicados em periódicos', {})
            qtes_artcomplper.append(len(info_art.values()))

        dtf_atualizado = pd.DataFrame({"id_lattes": ids_lattes_grupo,
                                       "curriculos": nomes_curriculos, 
                                       "ultima_atualizacao": dts_atualizacoes,
                                       "dias_defasagem": tempos_defasagem,
                                       "qte_artigos_periodicos": qtes_artcomplper,
                                       })
        return dtf_atualizado