import os
import re
import sys
import time
import json
import torch
import psutil
import logging
import platform
import requests
import threading
import subprocess
import numpy as np
import concurrent.futures
import multiprocessing as mp
import plotly.graph_objects as go

from tqdm import tqdm
from numba import jit
from datetime import datetime
from tqdm.notebook import tqdm
from plotly.subplots import make_subplots
from multiprocessing import Pool, cpu_count
from bs4 import BeautifulSoup, Tag, NavigableString
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
from experiment_profiler import ExperimentProfiler

'''
Implementação para montar datasets de artigos e monitorar threads e processos e cargas na CPU e GPU enquanto processa requisições com a classe DatasetArticlesGenerator. Registra o estado inicial do sistema antes dos experimentos e identifica mudanças durante e após a execução dos experimentos, para isolar as threads e medir quantitativamente os tempos de execução e recursos consumidos.

Os métodos start_thread_monitoring e start_process_monitoring iniciam o monitoramento de threads e processos, respectivamente, quando um novo monitoramento de segmento no fluxo de execução é disparado o anterior é encerrado automaticamente. O método finalize_monitoring consolida os dados coletados, fornecendo uma visão clara das threads e processos criados e remanescentes após a conclusão dos experimentos. Este design ajuda a identificar recursos que foram iniciados durante a execução dos experimentos separando dos demais recursos destinados à outras atividades que podem estar interferindo nas métricas de desempenho.

Os métodos de monitoramento de threads e processos são executados em suas próprias threads para não interferirem no desempenho dos experimentos que estão sendo monitorados. Isso garante que a sobrecarga de monitoramento seja mínima e não afete significativamente os resultados dos experimentos. Finalmente, as informações coletadas por esses métodos podem ser usadas para analisar o impacto dos experimentos no sistema como um todo, permitindo uma avaliação mais precisa de seu desempenho e eficiência.
'''

class DatasetArticlesGenerator:
    def __init__(self, base_repo_dir, profiler: ExperimentProfiler):
        self.profiler = profiler
        self.profiler.start_profiling_segment('T01_io_prepare')
        # Definindo os caminhos baseando-se no diretório do repositório
        self.folder_utils = os.path.join(base_repo_dir, 'utils')
        self.folder_domain = os.path.join(base_repo_dir, 'source', 'domain')
        self.folder_data_input = os.path.join(base_repo_dir, 'data', 'input')
        self.folder_data_output = os.path.join(base_repo_dir, 'data', 'output')    
        self.failed_extractions = []
        self.request_count = 0
        self.initial_threads = set()
        self.initial_processes = set()
        self.threads_created = []
        self.processes_created = []
        self.time_spent = {'scraping': 0, 'crossref': 0, 'data_processing': 0}
        self.gpu_transfer_time = 0.0
        self.gpu_load_time = 0
        self.gpu_calc_time = 0
        self.cpu_calc_time = 0

        # Criar a pasta de logs se não existir
        log_directory = "logs"
        os.makedirs(log_directory, exist_ok=True)

        # Configurar o logging
        log_filename = os.path.join(log_directory, 'dataset_generator.log')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=log_filename,
                            filemode='w')

        self.logger = logging.getLogger(__name__)
        self.logger.info("DatasetArticlesGenerator initialized")

        log_filename = os.path.join(log_directory, 'crossref_requests.log')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=log_filename,
                            filemode='w')

        # Inicializar contador de requisições
        self.crossref_request_count = 0

    def convert_list_to_dict(self, lst: List[Any]) -> Dict[str, Any]:
        return {str(i): item for i, item in enumerate(lst)}

    def _handle_set_value(self, value: set) -> list:
        return list(value)
    
    def _handle_string_value(self, value: str) -> Any:
        return value.strip() if value else None  # Melhoria no tratamento de strings

    def _handle_list_value(self, value: list) -> Dict[str, Any]:
        return self.convert_list_to_dict(value)
    
    def _handle_string_value(self, value: str) -> Any:
        return value.strip() if value else None  # Melhoria no tratamento de strings

    def _handle_list_value(self, value: list) -> Dict[str, Any]:
        return self.convert_list_to_dict(value)

    def preprocess_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        processed_data = self._recursive_preprocessing(extracted_data)
        return processed_data

    def _recursive_preprocessing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.profiler.start_profiling_segment('T03_processing')
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = self._handle_string_value(value)
            elif isinstance(value, list):
                data[key] = self._handle_list_value(value)
            elif isinstance(value, dict):
                data[key] = self._recursive_preprocessing(value)
            elif isinstance(value, set):
                data[key] = self._handle_set_value(value)
        self.profiler.start_profiling_segment('T01_io_prepare')
        return data

    ## Parte 2 Busca e Extração de dados
    def extrair_idlattes(self, dic: Dict[str, Any]) -> Optional[str]:
        self.profiler.start_profiling_segment('T03_processing')
        try:
            id_lattes_info = dic.get('InfPes', {}).get('2', '') or dic.get('InfPes', {}).get('3', '')
            id_lattes = self._buscar_id_lattes(id_lattes_info)
            self.profiler.start_profiling_segment('T01_io_prepare')
            return id_lattes
        except Exception as e:
            self.logger.error(f"Erro ao extrair ID Lattes: {e}")
            self.profiler.start_profiling_segment('T01_io_prepare')
            return None

    def _buscar_id_lattes(self, info: str) -> Optional[str]:
        # Buscar o ID Lattes em uma string
        padrao = r'ID Lattes: (\d+)'
        correspondencia = re.search(padrao, info)
        if correspondencia:
            self.profiler.start_profiling_segment('T01_io_prepare')
            return correspondencia.group(1)

        padrao_cv = r'Endereço para acessar este CV: http%3A//lattes.cnpq.br/(\d+)'
        correspondencia_cv = re.search(padrao_cv, info)
        if correspondencia_cv:
            self.profiler.start_profiling_segment('T01_io_prepare')
            return correspondencia_cv.group(1)
        self.profiler.start_profiling_segment('T01_io_prepare')
        return None

    def _extract_areas_of_expertise(self, areas_info: Dict[str, str]) -> List[Dict[str, str]]:
        self.profiler.start_profiling_segment('T03_processing')
        extracted_areas = []

        for key, area_str in areas_info.items():
            area_parts = area_str.split('/')
            extracted_area = {
                'GrandeÁrea': area_parts[0].replace('Grande área:', '').strip() if len(area_parts) > 0 else '',
                'Área': area_parts[1].replace('Área:', '').strip() if len(area_parts) > 1 else '',
                'Subárea': area_parts[2].replace('Subárea:', '').strip() if len(area_parts) > 2 else '',
                'Especialidade': area_parts[3].strip() if len(area_parts) > 3 else ''
            }
            extracted_areas.append(extracted_area)
        self.profiler.start_profiling_segment('T01_io_prepare')
        return extracted_areas

    def _extract_area_info(self, article_data: Dict[str, Any]) -> Dict[str, str]:
        # Extrair informações da área de atuação
        return {
            'GrandeÁrea': article_data.get('GrandeÁrea', ''),
            'Área': article_data.get('Área', ''),
            'Subárea': article_data.get('Subárea', ''),
            'Especialidade': article_data.get('Especialidade', '')
        }

    def _extract_jcr2_info(self, article_data: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        # Extrair informações de publicação (títulos, DOI, JCR)
        self.profiler.start_profiling_segment('T03_processing')
        subdict_titulos = {}
        subdict_doi = {}
        subdict_jci = {}

        if 'JCR2' in article_data:
            for key, value in article_data['JCR2'].items():
                subdict_titulos[key] = value.get('titulo', '')
                subdict_doi[key] = value.get('doi', '')
                subdict_jci[key] = value.get('impact-factor', '')

        if not subdict_titulos:
            producoes = article_data.get('Produções', {}).get('Produção bibliográfica', {}).get('Artigos completos publicados em periódicos', {})
            for key, value in producoes.items():
                title_match = re.search(r'^(.*?)\s+\.\s+', value)
                if title_match:
                    subdict_titulos[key] = title_match.group(1)
        self.profiler.start_profiling_segment('T07_postproces')
        return subdict_titulos, subdict_doi, subdict_jci

    def fetch_article_info_from_crossref(self, doi):
        self.logger.info(f"Fetching article info from CrossRef for DOI: {doi}")
        self.profiler.start_profiling_segment('T05_networkcom')
        self.request_count += 1
        url = f"https://api.crossref.org/works/{doi}"
        try:
            start_time = time.time()
            response = requests.get(url)
            if response.status_code == 200:
                self.profiler.start_profiling_segment('T03_processing')
                data = response.json()
                article_info = data.get('message', {})

                # Extrair o título
                title = article_info.get('title', ["Title not found"])
                if isinstance(title, list) and title:
                    title = title[0]
                else:
                    title = "Title not found"

                # Buscar e extrair o resumo no crossref
                abstract_html = article_info.get('abstract', None)
                abstract_text = "Abstract not found"
                if abstract_html:
                    soup = BeautifulSoup(abstract_html, "html.parser")
                    abstract_text = soup.get_text(separator=" ", strip=True)
                    if abstract_text.startswith('Abstract'):
                        abstract_text = abstract_text[len('Abstract'):].strip()
                        if 'Resumo' in abstract_text:
                            abstract_text = abstract_text.replace('Resumo','').strip()

                self.time_spent['crossref'] += time.time() - start_time
                self.profiler.start_profiling_segment('T01_io_prepare')
                return title, abstract_text, self.request_count
            else:
                self.profiler.start_profiling_segment('T01_io_prepare')
                return "Title not found", "Abstract not found"

        except Exception as e:
            self.logger.error(f"Error fetching data from CrossRef for DOI {doi}: {e}")
            self.profiler.start_profiling_segment('T01_io_prepare')
            return "Error", "Error"

    def fetch_article_info_from_crossref_by_title(self, title_to_search):
        self.profiler.start_profiling_segment('T05_networkcom')
        url = "https://api.crossref.org/works?query.title=" + title_to_search
        try:
            response = requests.get(url)
            if response.status_code == 200:
                self.profiler.start_profiling_segment('T03_processing')
                logging.info(f"CrossRef request successful for title: {title_to_search} | Request Count: {self.crossref_request_count}")

                data = response.json()
                items = data.get('message', {}).get('items', [])

                for item in items:
                    crossref_title = item.get('title', [])
                    if crossref_title:
                        crossref_title = crossref_title[0]
                        similarity = self.calculate_jaccard_similarity(title_to_search, crossref_title)
                        # if similarity > 0.85:
                        #     abstract_html = item.get('abstract', None)
                        #     abstract_text = "Abstract not found"
                        #     if abstract_html:
                        #         soup = BeautifulSoup(abstract_html, "html.parser")
                        #         abstract_text = soup.get_text(separator=" ", strip=True)
                        #         if abstract_text.startswith('Abstract'):
                        #             abstract_text = abstract_text[len('Abstract'):].strip()
                        #     return crossref_title, abstract_text
                        return crossref_title
                
                self.profiler.start_profiling_segment('T01_io_prepare')
                return "Title not found"
            else:
                self.profiler.start_profiling_segment('T01_io_prepare')
                # Log do insucesso da requisição
                logging.info(f"CrossRef request failed for title: {title_to_search} | Request Count: {self.crossref_request_count} | Status Code: {response.status_code}")
                return "Title not found"               

        except Exception as e:
            # Log de erro na requisição
            logging.error(f"Error fetching data from CrossRef for title {title_to_search}: {e} | Request Count: {self.crossref_request_count}")
            print(f"Error fetching data from CrossRef for title {title_to_search}: {e}")
            self.profiler.start_profiling_segment('T01_io_prepare')
            return "Error", "Error"

    def _complement_article_info(self, jcr2_article_info: Dict, producoes_article: Dict) -> Dict:
        # Complementar informações dos artigos de JCR2 com os dados de 'Produções', se necessário
        self.profiler.start_profiling_segment('T03_processing')
        if not jcr2_article_info['subdict_titulos']:
            title_match = re.search(r'^(.*?)\s+\.\s+', producoes_article)
            if title_match:
                jcr2_article_info['subdict_titulos'] = title_match.group(1)
            year_match = re.search(r',\s+(\d{4})\.', producoes_article)
            if year_match:
                year = year_match.group(1)
                jcr2_article_info['subdict_years'] = int(year)        
        self.profiler.start_profiling_segment('T01_io_prepare')
        return jcr2_article_info

    # Buscar resumos por scrap da página do artigo
    def scrape_article_info(self, doi):
        self.logger.info(f"Scraping article info for DOI: {doi}")
        self.profiler.start_profiling_segment('T05_networkcom')        
        start_time = time.time()
        self.request_count += 1
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
        url = f"https://doi.org/{doi}"

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                self.profiler.start_profiling_segment('T03_processing')
                soup = BeautifulSoup(response.content, 'html.parser')
                title = soup.find_all(['title', 'título'])
                # abstract = soup.find_all(['Abstract', 'resumo'])

                extracted_title = title[0].get_text().strip() if title else "Title not found"
                # extracted_abstract = abstract[0].get_text().strip() if abstract else "Abstract not found"

                self.time_spent['scraping'] += time.time() - start_time
                self.profiler.start_profiling_segment('T01_io_prepare')
                return extracted_title
        except Exception as e:
            print(f"Error scraping DOI {doi}: {e}")
            self.profiler.start_profiling_segment('T01_io_prepare')
            return "Error", "Error"

    ## Cálculos de similaridade de Jaccard
    def calculate_jaccard_similarity(self, str1, str2):
        self.profiler.start_profiling_segment('T03_processing')
        a = set(str1.split()) 
        b = set(str2.split())
        c = a.intersection(b)
        self.profiler.start_profiling_segment('T01_io_prepare')
        return float(len(c)) / (len(a) + len(b) - len(c))

    def calculate_jaccard_similarity_cpu(self, str1, str2):
        self.profiler.start_profiling_segment('T03_processing')
        cpu_calc_time_start = time.time()

        # Implementação em CPU do cálculo de similaridade de Jaccard
        a = set(str1.split()) 
        b = set(str2.split())
        c = a.intersection(b)
        similarity = float(len(c)) / (len(a) + len(b) - len(c))

        # Registrar o tempo de cálculo na CPU
        cpu_calc_time_end = time.time()
        self.cpu_calc_time += cpu_calc_time_end - cpu_calc_time_start
        self.profiler.start_profiling_segment('T01_io_prepare')
        return similarity

    def calculate_jaccard_similarity_gpu(self, str1, str2):
        if torch.cuda.is_available():
            self.logger.info("Iniciando cálculo de similaridade Jaccard com GPU")

            # Sincronização antes de iniciar operações na GPU
            self.profiler.start_profiling_segment('T06_syncronize')
            torch.cuda.synchronize()
            
            # Preparação dos dados para carregar na GPU
            self.profiler.start_profiling_segment('T02_loadto_dev')
            device = torch.device("cuda")
            set1 = set(str1.split())
            set2 = set(str2.split())
            vocab = list(set1.union(set2))
            vocab_index = {word: idx for idx, word in enumerate(vocab)}
            vec1 = torch.zeros(len(vocab), device=device)
            vec2 = torch.zeros(len(vocab), device=device)
            for word in set1:
                vec1[vocab_index[word]] = 1
            for word in set2:
                vec2[vocab_index[word]] = 1

            # Sincronização após carregamento de dados na GPU
            self.profiler.start_profiling_segment('T06_syncronize')                
            torch.cuda.synchronize()

            # Processamento na GPU
            self.profiler.start_profiling_segment('T03_processing')
            intersection = torch.sum(vec1 * vec2).float()
            union = torch.sum(vec1 + vec2 - (vec1 * vec2)).float()
            similarity = intersection / union if union != 0 else 0

            # Sincronização após o cálculo na GPU
            self.profiler.start_profiling_segment('T06_syncronize')
            torch.cuda.synchronize()
            self.logger.info("Cálculo de similaridade Jaccard com GPU concluído")
            self.profiler.start_profiling_segment('T01_io_prepare')
            return similarity.item()
        else:
            self.logger.info("GPU não disponível, utilizando CPU para cálculo de similaridade.")
            self.profiler.start_profiling_segment('T01_io_prepare')
            return self.calculate_jaccard_similarity_cpu(str1, str2)

         
    @jit(nopython=True)
    def calculate_jaccard_similarity_numba(self, set1, set2):
        self.profiler.start_profiling_segment('T03_processing')
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        self.profiler.start_profiling_segment('T01_io_prepare')
        return float(intersection) / union if union != 0 else 0


    def calculate_jaccard_similarity_numba_wrapper(self, str1, str2):
        self.profiler.start_profiling_segment('T03_processing')
        set1 = set(str1.split()) 
        set2 = set(str2.split())
        self.profiler.start_profiling_segment('T01_io_prepare')
        return self.calculate_jaccard_similarity_numba(set1, set2)

    def _extract_article_info(self, article_data: Dict[str, Any], use_gpu=False) -> Dict[str, Any]:
        self.logger.info(f"Extracting article info, use_gpu={use_gpu}")
        self.profiler.start_profiling_segment('T03_processing')        
        # print("ArticleData:",article_data)
        # print(article_data.keys())

        # Inicializar dicionários para armazenar as informações
        subdict_titulos = {}
        subdict_years = {}
        subdict_autores = {}
        subdict_doi = {}
        subdict_jci = {}

        # Primeiro, processar 'Produções'
        producoes_data = article_data.get('Produções', {})
        # print(producoes_data)
        for key, value in producoes_data.items():
            year_match = re.search(r',\s+(\d{4})\.', value)
            if year_match:
                subdict_years[key] = int(year_match.group(1))

            authors_match = re.search(r'^(.*?)\s+\.\s+', value)
            if authors_match:
                subdict_autores[key] = authors_match.group(1)

            title_match = re.search(r'\s\.\s(.*)', value)
            if title_match:
                subdict_titulos[key] = title_match.group(1)
            else:
                subdict_titulos[key] = ''  # Caso não encontre um título

        # Complementar com informações de JCR2, se disponíveis
        jcr2_data = article_data.get('JCR2', {})
        for key, value in jcr2_data.items():
            subdict_doi[str(int(key)+1)] = value.get('doi', '')
            subdict_jci[str(int(key)+1)] = value.get('impact-factor', '')
            subdict_titulos[str(int(key)+1)] = value.get('titulo', '')
        
        self.profiler.start_profiling_segment('T01_io_prepare')
        return {
            'subdict_areas': {},  # vazio para receber a inferência futuramente
            'subdict_years': subdict_years,
            'subdict_titulos': subdict_titulos,
            'subdict_autores': subdict_autores,
            'subdict_doi': subdict_doi,
            'subdict_jci': subdict_jci
        }

    ## Processamento básico de montagem do dataset de artigos repetido pelos vários métodos de processamento
    def _basic_data_processing(self, extracted_data: Dict, use_gpu=False) -> Optional[Dict]:
        self.logger.info("Processing single result, use_gpu={use_gpu}")
        try:
            articles_info = []
            preprocessed_data = self.preprocess_data(extracted_data)

            id_lattes = self.extrair_idlattes(preprocessed_data)
            
            producoes_articles = preprocessed_data.get('Produções', {}).get('Produção bibliográfica', {}).get('Artigos completos publicados em periódicos', {})
            
            jcr2_articles = extracted_data.get('JCR2', {})
            for key, producoes_article in producoes_articles.items():
                article_data = {'Produções': {key: producoes_article}}
                key = str(int(key.strip('.')) - 1)
                if key in jcr2_articles:
                    article_data['JCR2'] = {key: jcr2_articles[key]}

                article_info = self._extract_article_info(article_data, use_gpu)
                
                articles_info.append(article_info)

            # Preencher dados adicionais para cada artigo
            for article in articles_info:
                for key, art_info in article.items():
                    if 'doi' in art_info and art_info['doi']:
                        
                        title, abstract, _ = self.fetch_article_info_from_crossref(art_info['doi'])
                        
                        art_info['titulo'] = title
                        art_info['abstract'] = abstract

                    # Caso o título ainda não conste nos dados
                    elif not art_info.get('titulo'):
                        if 'doi' in art_info:
                            doi = art_info['doi'].replace('http://dx.doi.org/', '')
                            
                            scraped_title = self.scrape_article_info(doi)
                            
                            art_info['titulo'] = scraped_title
                            # art_info['abstract'] = scraped_abstract
                    else:
                        print('Erro ao preencher título faltante a partir do doi')

            # Retorna os dados processados juntamente com os dados de profiling
            self.profiler.start_profiling_segment('T07_postproces')
            return {
                'id_lattes': id_lattes,
                'name': preprocessed_data.get('name', 'N/A'),
                'areas_of_expertise': self._extract_areas_of_expertise(preprocessed_data.get('Áreas de atuação', {})),
                'articles': articles_info,
                'profiling_data': self.profiler.get_profile_data()
            }
        except Exception as e:
            self.logger.error(f"Error in basic data processing: {e}")
            self.profiler.start_profiling_segment('T07_postproces')
            return None

    def _fill_missing_data(self, articles_info, use_gpu=False):
        """
        Preenche os dados faltantes nos artigos, buscando informações adicionais
        via scraping ou CrossRef. Podendo utilizar GPU para cálculos de similaridade.

        :param articles_info: Lista com informações dos artigos.
        :param use_gpu: Indica se deve usar GPU para cálculos de similaridade.
        """
        self.logger.info("Filling missing data")
        for art_info in articles_info:
            # Preencher informações faltantes via scraping
            if 'doi' in art_info['subdict_doi']:
                doi = art_info['subdict_doi']['doi'].replace('http://dx.doi.org/', '')
                
                scraped_title = self.scrape_article_info(doi)
                
                if scraped_title != "Title not found" and scraped_title != "Error":
                    art_info['subdict_titulos']['title'] = scraped_title
                # if scraped_abstract != "Abstract not found" and scraped_abstract != "Error":
                #     art_info['abstract'] = scraped_abstract

            # Preencher informações faltantes via CrossRef
            # elif 'subdict_titulos' in art_info and art_info['subdict_titulos']:
            #     title = list(art_info['subdict_titulos'].values())[0]
                
            #     new_title, abstract = self.fetch_article_info_from_crossref_by_title(title)
                
            #     if new_title != "Title not found" and new_title != "Error":
                    
            #         similarity = self.calculate_jaccard_similarity_gpu(title, new_title) if use_gpu else self.calculate_jaccard_similarity(title, new_title)
                    
            #         if similarity > 0.85:
            #             art_info['abstract'] = abstract

    ## Processar artigo individual serializado em CPU
    def process_single_result_with_cpu(self, extracted_data: Dict) -> Optional[Dict]:
        self.start_thread_monitoring()
        self.start_process_monitoring()

        # Chamar a função de processamento básico dos dados em CPU
        processed_data = self._basic_data_processing(extracted_data)
        if processed_data is None:
            # Finaliza o monitoramento se houver falha no processamento
            self.stop_thread_monitoring()
            self.stop_process_monitoring()
            return None

        # Preenchimento de dados com CrossRef e scraping
        self._fill_missing_data(processed_data['articles'], use_gpu=False)

        # Finaliza o monitoramento
        self.stop_thread_monitoring()
        self.stop_process_monitoring()

        # Retorna os dados processados e informações de monitoramento
        return {
            'processed_data': processed_data,
            'monitoring_data': {
                'threads_created': self.threads_created,
                'processes_created': self.processes_created
            }
        }

    ## Processar artigo individual serializado em GPU
    def process_single_result_with_gpu(self, extracted_data: Dict) -> Optional[Dict]:
        # Inicia o monitoramento
        self.start_thread_monitoring()
        self.start_process_monitoring()

        # Processamento básico dos dados em GPU
        processed_data = self._basic_data_processing(extracted_data, use_gpu=True)
        if processed_data is None:
            # Finaliza o monitoramento se houver falha no processamento
            self.stop_thread_monitoring()
            self.stop_process_monitoring()
            return None

        # Preenchimento de dados com CrossRef e scraping com GPU
        self._fill_missing_data(processed_data['articles'], use_gpu=True)

        # Finaliza o monitoramento
        self.stop_thread_monitoring()
        self.stop_process_monitoring()

        # Retorna os dados processados e informações de monitoramento
        return {
            'processed_data': processed_data,
            'monitoring_data': {
                'threads_created': self.threads_created,
                'processes_created': self.processes_created
            }
        }

    ## Processar artigos em multiprocessos em GPU
    def process_single_result_multiprocess_gpu(self, args):
        """
        Método para processar um único resultado usando multiprocessing com suporte a GPU.
        """
        extracted_data, _, _ = args
        return self.process_single_result_with_gpu(extracted_data)

    ## Monitoramento
    def record_initial_state(self):
        # Registra o estado inicial de threads e processos
        self.initial_threads = set([t.ident for t in threading.enumerate()])
        self.initial_processes = set(psutil.pids())

    def start_thread_monitoring(self):
        # Inicia o monitoramento de threads
        self.monitoring_threads = True
        self.threads_created.clear()
        self.thread_monitor_thread = threading.Thread(target=self.monitor_threads)
        self.thread_monitor_thread.start()
                        
    def monitor_threads(self):
        while self.monitoring_threads:
            current_threads = set([t.ident for t in threading.enumerate()])
            new_threads = current_threads - self.initial_threads
            self.threads_created.extend(new_threads - set(self.threads_created))
            time.sleep(1)

    def stop_thread_monitoring(self):
        # Stop monitoring threads
        self.monitoring_threads = False
        if self.thread_monitor_thread:
            self.thread_monitor_thread.join()

    def start_process_monitoring(self):
        # Inicia o monitoramento de processos
        self.monitoring_processes = True
        self.processes_created.clear()
        self.process_monitor_thread = threading.Thread(target=self.monitor_processes)
        self.process_monitor_thread.start()

    def monitor_processes(self):
        while self.monitoring_processes:
            current_processes = set(psutil.pids())
            new_processes = current_processes - self.initial_processes
            self.processes_created.extend(new_processes - set(self.processes_created))
            time.sleep(1)

    def stop_process_monitoring(self):
        # Stop monitoring processes
        self.monitoring_processes = False
        if self.process_monitor_thread:
            self.process_monitor_thread.join()

    def finalize_monitoring(self):
        # Finaliza o monitoramento e retorna informações sobre threads e processos criados e remanescentes
        self.stop_thread_monitoring()
        self.stop_process_monitoring()

        remaining_threads = set([t.ident for t in threading.enumerate()]) - self.initial_threads
        remaining_processes = set(psutil.pids()) - self.initial_processes

        return {
            'created_threads': list(self.threads_created),
            'remaining_threads': list(remaining_threads),
            'created_processes': list(self.processes_created),
            'remaining_processes': list(remaining_processes)
        }

    ## Processar grupo completo de artigos
    # Processar com singlethread em CPU (Experimento 1)
    def process_dicts(self, all_extracted_data: List[Dict], json_filename: str) -> List[Dict]:
        self.logger.info("Processing all results")        
        successful_processed_data = []
        for extracted_data in tqdm(all_extracted_data, 
                                   desc="Singlethread CPU-Dataset Preparation in Python"):
            
            processed_data = self.process_single_result_with_cpu(extracted_data)
            
            if processed_data is not None:
                successful_processed_data.append(processed_data)
            else:
                self.failed_extractions.append(extracted_data)

        # Salvar dados processados
        self.to_json(successful_processed_data, json_filename)

        return successful_processed_data

    # Processar com singlethread em GPU (Experimento 2)
    def process_dicts_with_gpu(self, all_extracted_data: List[Dict], 
                               json_filename: str) -> List[Dict]:
        self.logger.info("Iniciado processamento singlethread com Python em GPU")
        start_time = time.time()
        successful_processed_data = []
        for extracted_data in tqdm(all_extracted_data, 
                                   desc="Singlethread GPU-Dataset Preparation in Python"):
            
            processed_data = self.process_single_result_with_gpu(extracted_data)
            
            if processed_data is not None:
                successful_processed_data.append(processed_data)
            else:
                self.failed_extractions.append(extracted_data)

        # Buscar informações adicionais para artigos que têm título mas não DOI
        # for article in successful_processed_data:
        #     if article['articles']:
        #         for art_info in article['articles']:
        #             if art_info['subdict_titulos'] and not art_info['subdict_doi']:
        #                 title = list(art_info['subdict_titulos'].values())[0]
                        
        #                 new_title, abstract = self.fetch_article_info_from_crossref_by_title(title)
                        
        #                 if new_title != "Title not found" and new_title != "Error":
        #                     art_info['abstract'] = abstract

        # Salvar dados processados
        self.to_json(successful_processed_data, json_filename)

        end_time = time.time()
        self.logger.info(f"Tempo total para processamento com GPU: {end_time - start_time:.4f} segundos")

        return successful_processed_data
    
    # Processar com multithread (paralelismo) em CPU (Experimento 3)
    def process_dicts_multithread(self, all_extracted_data: List[Dict], 
                                  json_filename: str) -> List[Dict]:
        successful_processed_data = []
        # failed_extractions = []

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submeter todas as tarefas
            
            futures = [executor.submit(self.process_single_result_with_cpu, data) for data in all_extracted_data]

            # Processar tarefas concluídas com tqdm
            for future in tqdm(as_completed(futures), total=len(futures), 
                               desc="Multithreads CPU-Dataset Preparation in Python"):
                try:
                    processed_data = future.result()
                    if processed_data is not None:
                        successful_processed_data.append(processed_data)
                except Exception as e:
                    # Registrar ou armazenar informação de erro para debug.
                    logging.error(f"Error processing data on CPU multithreading experiment: {e}")

        # Salvar os dados processados no final
        self.to_json(successful_processed_data, json_filename)

        return successful_processed_data

    # Processar com multithread em GPU com Python (Experimento 4)
    def process_dicts_multithreads_with_gpu(self, 
                                            all_extracted_data: List[Dict], 
                                            json_filename: str) -> List[Dict]:
        successful_processed_data = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self.process_single_result_with_gpu, data) for data in all_extracted_data]
            for future in tqdm(as_completed(futures), total=len(futures), 
                               desc="Multithreads GPU-Dataset Preparation in Python"):
                try:
                    processed_data = future.result()
                    if processed_data is not None:
                        successful_processed_data.append(processed_data)
                except Exception as e:
                    logging.error(f"Error processing data with GPU in Python: {e}")
        self.to_json(successful_processed_data, json_filename)
        return successful_processed_data

    ## Sem feedback tqdm
    # def process_dicts_multithread(self, all_extracted_data, json_filename):
    #     with ThreadPoolExecutor() as executor:
    #         futures = [executor.submit(self.process_single_result_with_cpu, data, json_filename) for data in all_extracted_data]
    #         results = [future.result() for future in futures]
    #     return results

    # Processar com multithread em CPU com Golang e Goroutines (Experimento 5)
    def process_data_multithreads_with_go(self,
                                          dict_list: List[Dict]):
        # Inicializar go_executable
        go_executable = None
        
        # Detecta o sistema operacional e ajusta o caminho do executável
        os_type = platform.system()
        if go_executable is None:
            if os_type == "Linux":
                go_executable = os.path.join(self.folder_domain, "dataset_articles_generator_linux")
            elif os_type == "Windows":
                go_executable = os.path.join(self.folder_domain, "dataset_articles_generator_windows.exe")
            else:
                raise Exception("Sistema operacional não suportado")

        # Restante do código permanece o mesmo
        start_time = time.time()
        # Caminhos atualizados para uso de variáveis de instância
        input_json = os.path.join(self.folder_data_input, "normalized_dict_list.json")
        output_json = os.path.join(self.folder_data_output, "output_go_cpu_multithreads.json")
        progress_file = os.path.join(self.folder_data_input, "progress.txt")

        json_normalizado = self.normalizar_para_json(dict_list)
        with open(input_json, "w", encoding='utf-8') as file:
            file.write(json_normalizado)

        with open(progress_file, "w") as file:
            pass

        try:
            go_process = subprocess.Popen([go_executable, input_json, output_json])
            progress_bar = tqdm(total=len(dict_list), 
                                desc="Multithreads CPU-Dataset Prepare on Goroutines")

            while go_process.poll() is None:
                try:
                    with open(progress_file, "r") as file:
                        progress = int(file.read().strip())
                    progress_bar.update(progress - progress_bar.n)  # Atualizar com o incremento
                except (FileNotFoundError, ValueError):
                    pass
                time.sleep(0.1)

            # Atualizar a barra de progresso para o total antes de fechá-la
            progress_bar.update(progress_bar.total - progress_bar.n)
            progress_bar.close()

        except Exception as e:
            print("Erro ao processar com executável Go:", e)
            return None, time.time() - start_time

        if go_process.returncode != 0:
            print("Processo Go terminou com erro. Código de retorno:", go_process.returncode)
            return None, time.time() - start_time

        if os.path.exists(output_json):
            with open(output_json, "r") as file:
                processed_data = json.load(file)
        else:
            print(f"Arquivo de saída não encontrado: {output_json}")
            processed_data = None

        return processed_data, time.time() - start_time

    # Processar multithread CPU com Golang, Goroutines/Semáforos de concorrência (Experimento 6)
    def process_data_multithreads_with_go_optim(self,
                                                dict_list: List[Dict]):
        # Inicializar go_executable
        go_executable = None

        # Detecta o sistema operacional e ajusta o caminho do executável
        os_type = platform.system()
        if go_executable is None:
            if os_type == "Linux":
                go_executable = os.path.join(self.folder_domain, 
                                             "dataset_articles_generator_optim_linux")
            elif os_type == "Windows":
                go_executable = os.path.join(self.folder_domain, 
                                             "dataset_articles_generator_optim_windows.exe")
            else:
                raise Exception("Sistema operacional não suportado")

        # Restante do código permanece o mesmo
        start_time = time.time()
        # Caminhos atualizados
        input_json = os.path.join(self.folder_data_input, "normalized_dict_list.json")
        output_json = os.path.join(self.folder_data_output, "output_go_cpu_mthreadoptim.json")
        progress_file = os.path.join(self.folder_data_input, "progress_optimized.txt")

        json_normalizado = self.normalizar_para_json(dict_list)
        with open(input_json, "w", encoding='utf-8') as file:
            file.write(json_normalizado)

        with open(progress_file, "w") as file:
            pass

        try:
            go_process = subprocess.Popen([go_executable, input_json, output_json])

            # Função para atualizar a barra de progresso
            def update_progress():
                progress_bar = tqdm(total=len(dict_list), 
                                    desc="Multithreads CPU-DsPrep on Goroutine-Semaphors")
                while go_process.poll() is None:
                    try:
                        with open(progress_file, "r") as file:
                            progress = int(file.read().strip())
                        progress_bar.update(progress - progress_bar.n)  # Atualizar com o incremento
                    except (FileNotFoundError, ValueError):
                        pass
                    time.sleep(0.1)
                progress_bar.update(progress_bar.total - progress_bar.n)
                progress_bar.close()

            # Iniciar thread para atualização da barra de progresso
            progress_thread = threading.Thread(target=update_progress)
            progress_thread.start()

            # Aguardar o subprocesso concluir
            go_process.wait()

            # Aguardar a thread de progresso concluir
            progress_thread.join()

        except Exception as e:
            print("Erro ao iniciar subprocesso Go:", e)
            return None, time.time() - start_time

        if go_process.returncode != 0:
            print("Processo Go terminou com erro. Código de retorno:", go_process.returncode)
            return None, time.time() - start_time

        if os.path.exists(output_json):
            with open(output_json, "r") as file:
                processed_data = json.load(file)
        else:
            print(f"Arquivo de saída não encontrado: {output_json}")
            processed_data = None

        return processed_data, time.time() - start_time

    def normalizar_recursivamente(self, valor, chave=None):
        if isinstance(valor, dict):
            # Especial tratamento para 'Atuacao'
            if chave == 'Atuação Profissional':
                return self.normalizar_atuacao(valor)
            return {str(k): self.normalizar_recursivamente(v, k) for k, v in valor.items()}
        elif isinstance(valor, list):
            return [self.normalizar_recursivamente(elem) for elem in valor]
        else:
            return str(valor)

    def normalizar_atuacao(self, atuacao):
        # Garantir que atuacao seja um mapa de listas
        if not isinstance(atuacao, dict):
            return {}
        
        resultado = {}
        for chave, valor in atuacao.items():
            if isinstance(valor, list):
                resultado[chave] = [item if isinstance(item, dict) else {} for item in valor]
            else:
                resultado[chave] = [{}]
        return resultado

    def normalizar_curriculum(self, item):
        # Converter o campo 'Resumo', que é uma lista, em um dicionário
        resumo = {str(i): v for i, v in enumerate(item.get("Resumo", []))}

        curriculum_normalizado = {
            "Labels": str(item.get("labels", "")),
            "Name": str(item.get("name", "")),
            "InfPes": item.get("InfPes", []),
            "Resumo": resumo,
            "Identificacao": self.normalizar_recursivamente(item.get("Identificação", {})),
            "Endereco": self.normalizar_recursivamente(item.get("Endereço", {})),
            "Formacao": self.normalizar_recursivamente(item.get("Formação acadêmica/titulação", {})),
            "Complementar": self.normalizar_recursivamente(item.get("Formação Complementar", {})),
            "Atuacao": self.normalizar_recursivamente(item.get("Atuação Profissional"), 'Atuação Profissional'),
            "Pesquisa": self.normalizar_recursivamente(item.get("Projetos de pesquisa", {})),
            "Desenvolvimento": self.normalizar_recursivamente(item.get("Projetos de desenvolvimento", {})),
            "AtuacaoAreas": self.normalizar_recursivamente(item.get("Áreas de atuação", {})),
            "Idiomas": self.normalizar_recursivamente(item.get("Idiomas", {})),
            "Inovacao": self.normalizar_recursivamente(item.get("Inovação", {})),
            "Producoes": self.normalizar_recursivamente(item.get("Produções", {})),
            "JCR2": self.normalizar_recursivamente(item.get("JCR2", {})),
            "Bancas": self.normalizar_recursivamente(item.get("Bancas", {}))
        }
        return curriculum_normalizado

    def normalizar_para_json(self, dict_list):
        lista_normalizada = [self.normalizar_curriculum(item) for item in dict_list]
        # Certificar que a lista normalizada seja o que está sendo convertida para JSON
        return json.dumps(lista_normalizada, ensure_ascii=False)

    def to_json(self, data: List[Dict], filename: str) -> None:
        # Salvar os dados em formato JSON
        with open(filename, 'w') as file:
            json.dump(data, file)

    def to_hdf5(self, data: List[Dict], filename: str) -> None:
        # Salvar os dados em formato HDF5
        with h5py.File(filename, 'w') as file:
            for i, item in enumerate(data):
                file.create_dataset(str(i), data=json.dumps(item))

    ## Outros experimentos com processamento paralelo, além do multitreading
    ## Processar com multiprocess apresenta Problemas com o Python
    ## Processar com multiprocess (concorrência)
    # def process_dicts_multiprocess(self, all_extracted_data: List[Dict], json_filename: str: str) -> List[Dict]:
    #     successful_processed_data = []

    #     # Determinar o número de processos a serem usados. Geralmente, é igual ao número de CPUs.
    #     num_processes = cpu_count()

    #     # Preparar os argumentos para cada tarefa
    #     tasks = [(data, json_filename) for data in all_extracted_data]

    #     # Usar um pool de processos para executar as tarefas em paralelo
    #     with Pool(processes=num_processes) as pool:
    #         # Mapear as tarefas para o pool e coletar os resultados
    #         for result in tqdm(pool.imap_unordered(self.process_single_result_multiprocess, tasks), total=len(tasks), desc="Processing with multiprocessing"):
    #             if result is not None:
    #                 successful_processed_data.append(result)
    #             else:
    #                 self.failed_extractions.append(result)

    #     # Salvar dados processados
    #     self.to_json(successful_processed_data, json_filename)
    #     self.to_hdf5(successful_processed_data)

    #     return successful_processed_data

    def process_dicts_multiprocess(self, all_extracted_data, json_filename, progress_callback=None):
        successful_processed_data = []
        num_processes = mp.cpu_count()

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(self.process_single_result_with_cpu, data) for data in all_extracted_data]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        successful_processed_data.append(result)
                        if progress_callback:
                            progress_callback()  # Atualiza a barra de progresso
                except Exception as e:
                    print(f"Error processing data: {e}")

        self.to_json(successful_processed_data, json_filename)
        self.to_hdf5(successful_processed_data)
        return successful_processed_data

    # Método process_dicts_multiprocess_gpu similar, apenas mudando a função chamada no executor.submit
    def process_dicts_multiprocess_gpu(self, all_extracted_data, 
                                       json_filename, 
                                       progress_callback=None):
        successful_processed_data = []
        num_processes = mp.cpu_count()

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(self.process_single_result_with_gpu, data) for data in all_extracted_data]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        successful_processed_data.append(result)
                        if progress_callback:
                            progress_callback()  # Atualiza a barra de progresso
                except Exception as e:
                    print(f"Error processing data with GPU: {e}")

        self.to_json(successful_processed_data, json_filename)
        self.to_hdf5(successful_processed_data)
        return successful_processed_data

    ## Plotagem comparativas do desempenho
    def plot_comparision(self, results):
        # Desempacotar resultados
        single_thread_results = results['single_thread']
        single_thread_gpu_results = results['single_thread_gpu']
        multi_thread_results = results['multi_thread']
        multi_thread_gpu_results = results['multi_thread_gpu']
        multi_process_results = results['multi_process']
        multi_process_gpu_results = results['multi_process_gpu']

        # Dados para plotagem
        times = [single_thread_results['time'], single_thread_gpu_results['time'],
                multi_thread_results['time'], multi_thread_gpu_results['time'],
                multi_process_results['time'], multi_process_gpu_results['time']]
        avg_threads = [single_thread_results['avg_threads'], single_thread_gpu_results['avg_threads'],
                    multi_thread_results['avg_threads'], multi_thread_gpu_results['avg_threads'],
                    multi_process_results['avg_threads'], multi_process_gpu_results['avg_threads']]
        avg_processes = [single_thread_results['avg_processes'], single_thread_gpu_results['avg_processes'],
                        multi_thread_results['avg_processes'], multi_thread_gpu_results['avg_processes'],
                        multi_process_results['avg_processes'], multi_process_gpu_results['avg_processes']]

        # Criar gráfico de barras
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Execution Time", "Average Threads & Processes"))
        fig.add_trace(go.Bar(name='Time', x=['ST', 'ST-GPU', 'MT', 'MT-GPU', 'MP', 'MP-GPU'], y=times), row=1, col=1)
        fig.add_trace(go.Bar(name='Threads', x=['ST', 'ST-GPU', 'MT', 'MT-GPU', 'MP', 'MP-GPU'], y=avg_threads), row=1, col=2)
        fig.add_trace(go.Bar(name='Processes', x=['ST', 'ST-GPU', 'MT', 'MT-GPU', 'MP', 'MP-GPU'], y=avg_processes), row=1, col=2)

        # Atualizar layout
        fig.update_layout(title='Comparative Performance Analysis', barmode='group')
        fig.show(renderer="notebook")

    def plot_scatterplot_comparision(self, results):
        # Converter resultados em listas separadas para cada configuração, incluindo dados de monitoramento
        single_thread_times = [res['time'] for res in results['single_thread']]
        single_thread_gpu_times = [res['time'] for res in results['single_thread_gpu']]
        multi_thread_times = [res['time'] for res in results['multi_thread']]
        multi_thread_gpu_times = [res['time'] for res in results['multi_thread_gpu']]
        multi_process_times = [res['time'] for res in results['multi_process']]
        multi_process_gpu_times = [res['time'] for res in results['multi_process_gpu']]

        # Criar scatter plot com Plotly
        fig = go.Figure()

        # Adicionar cada série de dados
        fig.add_trace(go.Scatter(x=list(range(len(single_thread_times))), y=single_thread_times, mode='markers', name='Single Thread without GPU'))
        fig.add_trace(go.Scatter(x=list(range(len(single_thread_gpu_times))), y=single_thread_gpu_times, mode='markers', name='Single Thread with GPU'))
        fig.add_trace(go.Scatter(x=list(range(len(multi_thread_times))), y=multi_thread_times, mode='markers', name='Multi Thread without GPU'))
        fig.add_trace(go.Scatter(x=list(range(len(multi_thread_gpu_times))), y=multi_thread_gpu_times, mode='markers', name='Multi Thread with GPU'))
        fig.add_trace(go.Scatter(x=list(range(len(multi_process_times))), y=multi_process_times, mode='markers', name='Multi Process without GPU'))
        fig.add_trace(go.Scatter(x=list(range(len(multi_process_gpu_times))), y=multi_process_gpu_times, mode='markers', name='Multi Process with GPU'))

        # Configurar layout
        fig.update_layout(
            title='Comparação Visual dos Tempos de Execução',
            legend_title='Configuração',
            xaxis_title='Número do Teste',
            yaxis_title='Tempo de Execução (s)',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1)
        )

        # Mostrar gráfico na célula do notebook jupyter
        fig.show(renderer="notebook")

    def plot_boxplot_comparision(self, results):
        # Desempacotar tempos de execução e dados de monitoramento
        single_thread_times = [res['time'] for res in results['single_thread']]
        single_thread_gpu_times = [res['time'] for res in results['single_thread_gpu']]
        multi_thread_times = [res['time'] for res in results['multi_thread']]
        multi_thread_gpu_times = [res['time'] for res in results['multi_thread_gpu']]
        multi_process_times = [res['time'] for res in results['multi_process']]
        multi_process_gpu_times = [res['time'] for res in results['multi_process_gpu']]

        # Criar subplots
        fig = make_subplots(rows=1, cols=6, subplot_titles=("Single Thread", "Single Thread GPU", "Multi Thread", "Multi Thread GPU", "Multi Process", "Multi Process GPU"))

        # Adicionar boxplots
        fig.add_trace(go.Box(y=single_thread_times, name="Single Thread"), row=1, col=1)
        fig.add_trace(go.Box(y=single_thread_gpu_times, name="Single Thread GPU"), row=1, col=2)
        fig.add_trace(go.Box(y=multi_thread_times, name="Multi Thread"), row=1, col=3)
        fig.add_trace(go.Box(y=multi_thread_gpu_times, name="Multi Thread GPU"), row=1, col=4)
        fig.add_trace(go.Box(y=multi_process_times, name="Multi Process"), row=1, col=5)
        fig.add_trace(go.Box(y=multi_process_gpu_times, name="Multi Process GPU"), row=1, col=6)

        # Atualizar layout
        fig.update_layout(height=600, width=1800, title_text="Comparação Estatística dos Tempos de Execução")
        fig.show(renderer="notebook")

    ## Função de comparação substituída pela classe especializada em Monitoramento
    def compare_execution_times(self, all_extracted_data, json_filename):
        self.logger.info("Comparing execution times")        

        print("1. Medir tempo de execução serial em CPU")
        start_time = time.time()
        self.process_dicts(all_extracted_data, json_filename)
        single_thread_time = time.time() - start_time

        print("2. Medir tempo de execução serial com GPU")
        start_time = time.time()
        self.process_dicts_with_gpu(all_extracted_data, json_filename)
        single_thread_gpu_time = time.time() - start_time

        print("3. Medir tempo de execução com multithreading em CPU")
        start_time = time.time()
        self.process_dicts_multithread(all_extracted_data, json_filename)
        multi_thread_time = time.time() - start_time

        print("4. Medir tempo de execução com multithreading e com GPU")
        start_time = time.time()
        self.process_dicts_multithreads_with_gpu(all_extracted_data, json_filename)
        multi_thread_gpu_time = time.time() - start_time

        print("5. Medir tempo de execução com multiprocessing em CPU com Python")
        start_time = time.time()
        self.process_dicts_multiprocess(all_extracted_data, json_filename)
        multi_process_time = time.time() - start_time

        print("6. Medir tempo de execução com multiprocessing em CPU com Golang, Goroutines e Semáforos")
        start_time = time.time()
        self.process_dicts_multiprocess_gpu(all_extracted_data, json_filename)
        multi_process_gpu_time = time.time() - start_time

        # Plotar os tempos de execução com Plotly
        fig = go.Figure(data=[
            go.Bar(name='Single-thread', x=['Without GPU', 'With GPU'], y=[single_thread_time, single_thread_gpu_time]),
            go.Bar(name='Multi-thread', x=['Without GPU', 'With GPU'], y=[multi_thread_time, multi_thread_gpu_time]),
            go.Bar(name='Multi-process', x=['Without GPU', 'With GPU'], y=[multi_process_time, multi_process_gpu_time])
        ])
        fig.update_layout(barmode='group', title='Comparison of Execution Times', yaxis_title='Execution Time (seconds)')
        fig.show(renderer="notebook")

        return {
            "single_thread": single_thread_time,
            "single_thread_gpu": single_thread_gpu_time,
            "multi_thread": multi_thread_time,
            "multi_thread_gpu": multi_thread_gpu_time,
            "multi_process": multi_process_time,
            "multi_process_gpu": multi_process_gpu_time
        }
        
    def plot_time_spent_by_activity(self, experiment_data):
        labels = [f'Experimento {i+1}' for i in range(len(experiment_data))]

        # Dados para plotagem
        scraping_times = [data['time_spent']['scraping'] for data in experiment_data]
        crossref_times = [data['time_spent']['crossref'] for data in experiment_data]
        data_processing_times = [data['time_spent']['data_processing'] for data in experiment_data]
        thread_monitoring_times = [data['time_spent']['thread_monitoring'] for data in experiment_data]
        process_monitoring_times = [data['time_spent']['process_monitoring'] for data in experiment_data]

        # Criar gráfico de barras
        fig = go.Figure(data=[
            go.Bar(name='Scraping', x=labels, y=scraping_times),
            go.Bar(name='CrossRef', x=labels, y=crossref_times),
            go.Bar(name='Data Processing', x=labels, y=data_processing_times),
            go.Bar(name='Thread Monitoring', x=labels, y=thread_monitoring_times),
            go.Bar(name='Process Monitoring', x=labels, y=process_monitoring_times)
        ])

        fig.update_layout(barmode='stack', title='Time Spent by Activity in Each Experiment')
        fig.show(renderer="notebook")