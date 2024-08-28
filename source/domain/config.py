
from pathlib import Path
from getpass import getpass
from datetime import datetime
from IPython.display import clear_output
import pandas as pd, os, re, sys, time, json, subprocess

# Configurações para o modo de depuração
DEBUG = True  # Defina como False em produção


## Configurar exibição do pandas para melhor visualizar os dados
pd.set_option('display.max_colwidth', None)
pd.set_option('colheader_justify', 'left')
pd.set_option('display.max_rows', 600)

def find_repo_root(path='.', depth=10):
    ''' 
    Busca o arquivo .git e retorna string com a pasta raiz do repositório
    '''
    # Prevent infinite recursion by limiting depth
    if depth < 0:
        return None
    path = Path(path).absolute()
    if (path / '.git').is_dir():
        return path
    return find_repo_root(path.parent, depth-1)

delay = 10

## Definir a pasta de base do repositório local
base_repo_dir = find_repo_root()

# Configurações para arquivos estáticos
STATIC_ROUTE = "/static"  # Rota para servir arquivos estáticos
# STATIC_DIRECTORY = "static"  # Diretório onde os arquivos estáticos estão localizados
STATIC_DIRECTORY = os.path.join(base_repo_dir, 'static')
STATIC_NAME = "static"  # Nome para a montagem dos arquivos estáticos

## Sempre construir os caminhos usando os.path.join para compatibilidade WxL
folder_utils = os.path.join(base_repo_dir, 'utils')
folder_domain = os.path.join(base_repo_dir, 'source', 'domain')
folder_data_input = os.path.join(base_repo_dir, '_data', 'in_csv')
folder_data_output = os.path.join(base_repo_dir, '_data', 'out_json')

## Adicionar pastas locais ao sys.path para importar pacotes criados localmente
sys.path.append(folder_utils)
sys.path.append(folder_domain)
from scraper_pasteur import PasteurScraper
from environment_setup import EnvironmentSetup
from scraper_sucupira import SucupiraScraper
from scraper_sucupira_edge import SucupiraScraperEdge
from chromedriver_manager import ChromeDriverManager
from lattes_scrapper import JSONFileManager, LattesScraper, HTMLParser, SoupParser, GetQualis, ArticlesCounter, DictToHDF5, attribute_to_be_non_empty