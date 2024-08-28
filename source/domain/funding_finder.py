import pandas as pd, numpy as np
import xml.etree.ElementTree as ET
import os, sys, json, time, html, requests, logging
import platform, urllib.error, urllib.request, urllib.parse

from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from pprint import pprint
from datetime import datetime
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.expected_conditions import presence_of_element_located
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.common.exceptions import WebDriverException
from jinja2 import Environment, FileSystemLoader

class CarlosChagasAPI:
    def __init__(self, api_key):
        self.base_url = 'http://efomento.cnpq.br/efomento/login.do?metodo=apresentar'
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def buscar_editais(self, params=None):
        """
        Busca editais na Plataforma Carlos Chagas com base nos parâmetros fornecidos.

        Args:
            params (dict, optional): Dicionário com os parâmetros da busca (e.g., área, modalidade, etc.).

        Returns:
            list: Lista de editais encontrados em formato JSON.
        """
        endpoint = "/editais"
        url = self.base_url + endpoint

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Erro na busca de editais: {response.status_code} - {response.text}")
            return []

class FundingFinder:
    def __init__(self):
        self.base_repo_dir = self._get_base_repo()
        self.folder_utils = os.path.join(self.base_repo_dir, 'utils')
        self.folder_assets = os.path.join(self.base_repo_dir, 'assets')        
        self.folder_domain = os.path.join(self.base_repo_dir, 'source', 'domain')
        self.folder_data_input = os.path.join(self.base_repo_dir, '_data', 'in_csv')
        self.folder_data_output = os.path.join(self.base_repo_dir, '_data', 'out_json')            
        
        # Chaves de API (substitua pelos seus valores reais)
        self.carlos_chagas_api_key = "SUA_CHAVE_API_CARLOS_CHAGAS"
        self.cordis_api_key = "SUA_CHAVE_API_CORDIS"
        self.nih_api_key = "SUA_CHAVE_API_NIH"
        self.nsf_api_key = "SUA_CHAVE_API_NSF"

        self.driver = self._setup_driver()
        self.todos_programas = []
        self.falhas_totais = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        # Configuração do logger
        logging.basicConfig(filename=os.path.join(self.base_repo_dir,'source','logs','founding_finder.log'),
                            level=logging.ERROR,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # para usar plataforma financiar
        # self.base_url = "https://www.financiar.org.br/index.php"
        # self._start()

    def _start(self):
        try:
            self._login(self.driver)
            if self.driver is None:
                print("Falha no login. Encerrando a execução.")
                return
        except Exception as e:
            print(e)

    def _logout(self):
        if self.driver:
            self.driver.quit()
            self.driver = None
            return

    def buscar_editais(self, palavras_chave, fontes=["carlos_chagas", "cordis", "nih", "nsf", "finep", "capes", "cnpq"]):
        """
        Busca editais em diferentes fontes com base nas palavras-chave fornecidas.
        """
        resultados = []

        if "financiar" in fontes:
            try:
                self._login(self.driver)
                if self.driver is not None:  # Checar sucesso no login
                    resultados_financiar, _ = self.search_funding([palavras_chave], filtros=False)
                    # Certifique-se de que resultados_financiar não é None antes de iterar
                    if resultados_financiar is not None:
                        resultados.extend(resultados_financiar.to_dict('records'))
                else:
                    print("Falha no login na Financiar. Pulando busca.")
            except Exception as e:
                print(f"Erro ao buscar editais na Financiar: {e}")
            finally:
                self._logout()

        plat = "cnpq"
        if plat in fontes:
            try:
                resultados_cnpq = self._buscar_editais_cnpq(palavras_chave)
                # Certificar de que resultados não é None antes de iterar
                if resultados_cnpq is not None:
                    resultados.extend(resultados_cnpq)
            except Exception as e:
                plataforma = plat.replace("_"," ").title()
                print(f"Erro ao buscar editais na Plataforma {plataforma}: {e}")

        plat = "finep"
        if plat in fontes:
            try:
                resultados_finep = self._buscar_editais_finep(palavras_chave)
                # Certificar de que resultados não é None antes de iterar
                if resultados_finep is not None:
                    resultados.extend(resultados_finep)
            except Exception as e:
                plataforma = plat.replace("_"," ").title()
                print(f"Erro ao buscar editais na Plataforma {plataforma}: {e}")

        plat = "capes"
        if plat in fontes:
            try:
                resultados_capes = self._buscar_editais_capes(palavras_chave)
                # Certificar de que resultados não é None antes de iterar
                if resultados_capes is not None:
                    resultados.extend(resultados_capes)
            except Exception as e:
                plataforma = plat.replace("_"," ").title()
                print(f"Erro ao buscar editais na Plataforma {plataforma}: {e}")
    
        plat = "carlos_chagas"
        if plat in fontes:
            try:
                resultados_carlos_chagas = self._buscar_editais_carlos_chagas(palavras_chave)
                # Certificar de que resultados não é None antes de iterar
                if resultados_carlos_chagas is not None:
                    resultados.extend(resultados_carlos_chagas)
            except Exception as e:
                plataforma = plat.replace("_"," ").title()
                print(f"Erro ao buscar editais na Plataforma {plataforma}: {e}")

        plat = "cordis"
        if plat in fontes:
            try:
                resultados_cordis = self._buscar_editais_cordis(palavras_chave)
                # Certificar de que resultados não é None antes de iterar
                if resultados_cordis is not None:
                    resultados.extend(resultados_cordis)
            except Exception as e:
                plataforma = plat.replace("_"," ").title()
                print(f"Erro ao buscar editais na Plataforma {plataforma}: {e}")

        plat = "nih"
        if plat in fontes:
            try:
                resultados_nih = self._buscar_editais_cordis(palavras_chave)
                # Certificar de que resultados não é None antes de iterar
                if resultados_nih is not None:
                    resultados.extend(resultados_nih)
            except Exception as e:
                plataforma = plat.replace("_"," ").title()
                print(f"Erro ao buscar editais na Plataforma {plataforma}: {e}")

        plat = "nsf"
        if plat in fontes:
            try:
                resultados_nsf = self._buscar_editais_cordis(palavras_chave)
                # Certificar de que resultados não é None antes de iterar
                if resultados_nsf is not None:
                    resultados.extend(resultados_nsf)
            except Exception as e:
                plataforma = plat.replace("_"," ").title()
                print(f"Erro ao buscar editais na Plataforma {plataforma}: {e}")

        return resultados

    ## Buscar em APIs de fomento internacional
    def buscar_editais_em_apis(self, palavras_chave, apis=["cordis", "nih", "nsf"]):
        """
        Busca editais em diferentes APIs de plataformas internacionais com base nas palavras-chave fornecidas.

        Args:
            palavras_chave (str): Palavras-chave para a busca de editais.
            apis (list, optional): Lista de APIs a serem utilizadas na busca. 
                                   Valores possíveis: "cordis", "nih", "nsf".
                                   Padrão: Todas as APIs.

        Returns:
            list: Lista de editais encontrados em todas as APIs.
        """
        resultados = []

        # if "carlos_chagas" in apis:
        #     resultados += self._buscar_editais_carlos_chagas(palavras_chave)

        if "cordis" in apis:
            resultados += self._buscar_editais_cordis(palavras_chave)

        if "nih" in apis:
            resultados += self._buscar_editais_nih(palavras_chave)

        if "nsf" in apis:
            resultados += self._buscar_editais_nsf(palavras_chave)

        return resultados

    def _buscar_editais_cordis(self, palavras_chave):
        """
        Busca editais na API da CORDIS com base nas palavras-chave fornecidas.

        Args:
            palavras_chave (str): Palavras-chave para a busca de editais.

        Returns:
            list: Lista de editais encontrados em formato JSON.
        """
        base_url = "https://cordis.europa.eu/api/en/cordis-h2020/v1/fundingOpportunities"
        params = {
            "q": palavras_chave,
            "type": "call",  # Filtrar apenas por chamadas (editais)
            "pageSize": 100  # Número máximo de resultados por página (ajuste conforme necessário)
        }

        resultados = []
        pagina_atual = 1

        while True:
            params["page"] = pagina_atual

            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()  # Lança uma exceção se o status code não for 2xx

                data = response.json()

                if not data.get("payload", {}).get("fundingOpportunities", []):
                    break  # Não há mais resultados

                resultados.extend(data["payload"]["fundingOpportunities"])
                pagina_atual += 1

            except requests.exceptions.RequestException as e:
                print(f"Erro ao buscar editais na CORDIS: {e}")
                break  # Interrompe a busca em caso de erro

        return resultados

    def _buscar_editais_nih(self, palavras_chave):
        """
        Busca editais na API do NIH (Grants Search API) com base nas palavras-chave fornecidas.

        Args:
            palavras_chave (str): Palavras-chave para a busca de editais.

        Returns:
            list: Lista de editais encontrados em formato JSON.
        """
        base_url = "https://api.reporter.nih.gov/v1/projects/search"
        params = {
            "query": f"terms={palavras_chave}",
            "fields": "core_project_num,project_title,project_start_date,project_end_date,award_notice_date,organization_name,total_cost",
            "size": 100  # Número máximo de resultados por página (ajuste conforme necessário)
        }

        resultados = []

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Lança uma exceção se o status code não for 2xx

            data = response.json()

            if not data.get("results", []):
                print("Nenhum edital encontrado na API do NIH.")
                return []

            resultados.extend(data["results"])

        except requests.exceptions.RequestException as e:
            print(f"Erro ao buscar editais na API do NIH: {e}")

        return resultados

    def _buscar_editais_nsf(self, palavras_chave):
        """
        Busca editais na API da NSF com base nas palavras-chave fornecidas.

        Args:
            palavras_chave (str): Palavras-chave para a busca de editais.

        Returns:
            list: Lista de editais encontrados em formato JSON.
        """
        base_url = "https://api.nsf.gov/services/v1/awards.json"
        params = {
            "keyword": palavras_chave,
            "printFields": "id,title,startDate,expDate,date,awardeeName,awardeeCity,awardeeState,abstractText,poName,programOfficer",
            "offset": 0,  # Deslocamento para paginação (começando do 0)
            "pageSize": 100  # Número máximo de resultados por página (ajuste conforme necessário)
        }

        resultados = []

        while True:
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()  # Lança uma exceção se o status code não for 2xx

                data = response.json()

                if not data.get("response", {}).get("award", []):
                    break  # Não há mais resultados

                resultados.extend(data["response"]["award"])
                params["offset"] += params["pageSize"]  # Atualiza o offset para a próxima página

            except requests.exceptions.RequestException as e:
                print(f"Erro ao buscar editais na API da NSF: {e}")
                break  # Interrompe a busca em caso de erro

        return resultados 

    ## Acessar Plataformas de fomento específicas nacionais
    def _buscar_editais_carlos_chagas(self, palavras_chave):
        """
        Implementação da busca de editais na API da Plataforma Carlos Chagas.
        """
        # Lógica para buscar editais na API da Plataforma Carlos Chagas usando 'palavras_chave'
        # ...
        pass  # Substitua por sua implementação real

    def _buscar_editais_finep(self, palavras_chave, situacao="aberta"):
        """
        Busca editais na página da FINEP com base nas palavras-chave fornecidas.

        Args:
            palavras_chave (str): Palavras-chave para a busca de editais.
            situacao (str, optional): Situação dos editais a serem buscados ("aberta" ou "todas"). 
                                      Padrão: "aberta".

        Returns:
            list: Lista de editais encontrados em formato adequado.
        """
        base_url = "https://www.finep.gov.br/chamadas-publicas"

        try:
            response = requests.get(base_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Verificar a situação selecionada no dropdown
            situacao_selecionada = soup.find('select', id='situacao').find('option', selected=True)['value']

            # Extrair os dados
            items = soup.find_all('div', class_='item')
            editais_encontrados = []

            for item in items:
                data_pub_div = item.find('div', class_='data_pub')
                prazo_div = item.find('div', class_='prazo')
                fonte_div = item.find('div', class_='fonte')
                publico_alvo_div = item.find('div', class_='publico')
                tema_div = item.find('div', class_='tema')

                data_publicacao = data_pub_div.text.strip().split(': ')[1] if data_pub_div else None
                prazo_envio = prazo_div.text.strip().split(': ')[1] if prazo_div else None
                fonte_recurso = fonte_div.text.strip().split(': ')[1] if fonte_div else None
                publico_alvo = publico_alvo_div.text.strip().split(': ')[1] if publico_alvo_div else None
                temas = tema_div.text.strip().split(': ')[1].split('; ') if tema_div else []

                edital = {
                    'data_publicacao': data_publicacao,
                    'prazo_envio': prazo_envio,
                    'fonte_recurso': fonte_recurso,
                    'publico_alvo': publico_alvo,
                    'temas': temas
                }

                # Verificar se o edital corresponde à situação desejada
                if situacao == "todas" or situacao_selecionada == situacao:
                    editais_encontrados.append(edital)

            return editais_encontrados

        except requests.exceptions.RequestException as e:
            print(f"Erro ao buscar editais na FINEP: {e}")
            return []

    def _buscar_editais_capes(self, palavras_chave):
        """
        Busca editais na página da CAPES com base nas palavras-chave fornecidas.
        """
        base_url = "https://www.gov.br/capes/pt-br/acesso-a-informacao/editais-de-programas"  # URL da página de editais da CAPES

        try:
            response = requests.get(base_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Lógica de scraping para extrair os editais da página, utilizando BeautifulSoup
            # ... (sua implementação)

            return editais_encontrados  # Retorna uma lista de editais em formato adequado

        except requests.exceptions.RequestException as e:
            print(f"Erro ao buscar editais na CAPES: {e}")
            return []

    def _buscar_editais_cnpq(self, palavras_chave):
        """
        Busca editais na página do CNPq com base nas palavras-chave fornecidas.
        """
        base_url = "https://www.cnpq.br/web/guest/chamadas-publicas"  # URL da página de chamadas públicas do CNPq

        try:
            response = requests.get(base_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Lógica de scraping para extrair os editais da página, utilizando BeautifulSoup
            # ... (sua implementação)

            return editais_encontrados  # Retorna uma lista de editais em formato adequado

        except requests.exceptions.RequestException as e:
            print(f"Erro ao buscar editais no CNPq: {e}")
            return []


    ## Acessar Plataforma Financiar
    def _login(self, driver):
        """Realiza o login no sistema, tratando possíveis erros e verificando o sucesso da operação."""

        if self._verificar_login(driver):
            print("Usuário já está logado.")
            return  # Já logado, não precisa fazer mais nada

        try:
            self._clicar_botao_avancado(driver)  # Lidar com certificado SSL, se necessário
        except NoSuchElementException:
            print("Erro ao lidar com certificado SSL.")
            return

        # Se não estiver logado, prosseguir com o login
        try:
            self._inserir_credenciais(driver)

            # Verificar se o login foi bem-sucedido
            try:
                # Aguardar elemento específico disponível após login bem-sucedido
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='logout']"))
                )
                print("Login realizado com sucesso.")
            except TimeoutException:

                # Verificar se o login foi negado
                try:
                    div_usuario_inativo = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'div.alert.alert-danger'))
                    )
                    if div_usuario_inativo:
                        msg_usuario_inativo = div_usuario_inativo.text.strip()
                        print(f"Login negado pela plataforma com a mensagem: {msg_usuario_inativo}")
                        print("Entrar em contato com administrador da plataforma financiar para reativar o login de usuário")
                        self._logout()
                        return  # Encerra a execução após o logout em caso de login negado
                except TimeoutException:
                    print("Erro ao verificar o status do login. Tente novamente mais tarde.")
                    self._logout()  # Ensure logout even if there's an error checking login status
                    return

        except Exception as e:
            self.logger.error("Erro durante o processo de login: %s", e)
            print(f"Descrição do erro: {e}")
            driver.close()
            return

    def search_funding(self, palavras_chave_lista, filtros=False):
        if self.driver is None:  # Check if driver is available
            # print("O driver não está disponível. Certifique-se de que o login foi realizado com sucesso.")
            return pd.DataFrame(), self.falhas_totais

        for palavras_chave in palavras_chave_lista:
            # Determina se a palavra-chave é um termo composto
            termo_composto = " " in palavras_chave or palavras_chave.startswith('"') and palavras_chave.endswith('"')

            try:
                if self.driver is None:  # Check again before navigating
                    # print("O driver não está disponível. Certifique-se de que o login foi realizado com sucesso.")
                    return pd.DataFrame(), self.falhas_totais
                self._navigate_to_search_page(self.driver)
            except Exception as e:  # Captura qualquer exceção durante a navegação
                print(f"Erro ao navegar para a página de busca: {e}")
                self._logout()
                return pd.DataFrame(), self.falhas_totais

            if filtros:
                if self.driver is None:  # Check again before filtering
                    # print("O driver não está disponível. Certifique-se de que o login foi realizado com sucesso.")
                    return pd.DataFrame(), self.falhas_totais
                try:
                    self._filter_pdi_ceara(self.driver)
                except Exception as e:  # Captura qualquer exceção durante a filtragem
                    print(f"Erro ao aplicar filtros: {e}")
                    self._logout()
                    return pd.DataFrame(), self.falhas_totais

            resultados_encontrados = self._execute_and_extract(palavras_chave)
            if not resultados_encontrados and termo_composto:
                # Remove as aspas se houver e divide o termo composto para buscar cada termo individualmente
                termos_individuais = palavras_chave.replace('"', '').split()
                for termo_individual in termos_individuais:
                    try:
                        self._navigate_to_search_page(self.driver)
                        if filtros:
                            self._filter_pdi_ceara(self.driver)
                        self._execute_and_extract(termo_individual)
                    except Exception as e:  # Captura qualquer exceção durante a busca individual
                        print(f"Erro ao realizar busca individual para '{termo_individual}': {e}")
                        # Continua a execução para os próximos termos, se houver

        df_todos_programas = pd.DataFrame(self.todos_programas) if self.todos_programas else pd.DataFrame()

        # Check if df_todos_programas is empty before generating the report
        if not df_todos_programas.empty:
            self.mount_foment_report(df_todos_programas)

        self._logout()
        return df_todos_programas, self.falhas_totais

    def _get_base_repo(self):
        """Retorna o caminho absoluto quatro níveis acima do diretório atual."""
        current_directory = os.getcwd()
        # Construir o caminho para subir quatro níveis
        path_five_levels_up = os.path.join(current_directory, '../../../../')
        # Normalizar o caminho para o formato absoluto
        absolute_path = os.path.abspath(path_five_levels_up)
        return absolute_path

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
        
        # Usar FundingFinder.find_repo_root de forma recursiva para buscar arquivo .git
        return FundingFinder.find_repo_root(path.parent, depth-1)

    def _setup_driver(self):
        '''
        Gera o objeto driver para manipular o acesso ao Navegador, usando Google Chrome
        '''
        # print(f'Conectando com o servidor do CNPq...')
        chrome_options = Options()
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")

        driver_path = None
        try:
            # Caminho para o chromedriver no sistema local
            if platform.system() == "Windows":
                driver_path=FundingFinder.find_repo_root(os.getcwd())/'chromedriver'/'chromedriver.exe'
            else:
                driver_path=FundingFinder.find_repo_root(os.getcwd())/'chromedriver'/'chromedriver'
        except Exception as e:
            print("Não foi possível estabelecer uma conexão, verifique o chromedriver")
            print(e)
        
        # print(driver_path)
        service = ChromeService(driver_path)
        driver = webdriver.Chrome(service=service)
        driver.set_page_load_timeout(10)  # timeout carregar página
        driver.set_window_position(-20, -10)
        driver.set_window_size(900, 900) # Largura (necessariamente no máximo 900 para gerar BuscaAvançada), Altura

        return driver

    def _ler_apikeys(self):
        # Obter caminho absoluto para a pasta home do usuário linux ou windows
        home_dir = os.path.expanduser("~")

        # Criar caminho completo para o arquivo secrets.json
        secrets_file_path = os.path.join(home_dir, "secrets_founding.json")

        # Verificar se o arquivo existe
        if os.path.exists(secrets_file_path):
            # Abra o arquivo secrets.json para leitura
            with open(secrets_file_path, 'r', encoding='utf-8') as secrets_file:
                secrets = json.load(secrets_file)
                return secrets
        else:
            print(f"Credenciais não disponíveis em: {home_dir}.")

    def clicar_botao_avancado(self, driver):
        '''
        Busca de várias formas o botão 'Avançado' que indica que o usuário já está logado.
            Seletor preciso por ID único "details-button".
            Seletor por classe: .secondary-button.small-link
            Seletor por texto: button:contains("Avançado")
        '''
        tempo_maximo_espera = 2 
        # Tentativa 1: Buscar diretamente botão 'Avançado'
        try:
            # print("  Buscando botão 'Avançado' por ID...")
            btn_avancado = WebDriverWait(driver, tempo_maximo_espera).until(
                EC.element_to_be_clickable((By.ID, "details-button")) 
            )
            btn_avancado.click()
        except TimeoutException:
            # print("  Buscando botão 'Avançado' em iframe...")
            # Tentativa 2: Busca do Botão Dentro de um iframe (Ajustar quando necessário por mudança na página)
            try:
                # print("  Buscando botão 'Avançado' por XPath...")
                btn_avancado = driver.find_element(
                    By.XPATH, "//*[contains(text(), 'Avançado')]")
                btn_avancado.click() 
            except (TimeoutException, NoSuchElementException):
                try:
                    # Último Recurso:  JavaScript ou ação do usuário
                    btn_avancado =  WebDriverWait(driver, tempo_maximo_espera).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, ".secondary-button.small-link")))
                    driver.execute_script("arguments[0].click();", btn_avancado)
                    # driver.execute_script("window.scrollTo(0, posição_y_do_botão);")
                    # btn_avancado.send_keys(Keys.ENTER)
                except (TimeoutException, NoSuchElementException):
                    return

    # def _login(self, driver):
    #     # Verificar se já está logado.
    #     try:
    #         driver.get(self.base_url)
    #     except:
    #         print("não foi possível alcançar a página verifique sua conexão.")
    #         return
    #     try:
    #         # Tentar primeiro encontrar o link de logout, se presente está logado.
    #         logout_link = WebDriverWait(driver, 2).until(
    #             EC.presence_of_element_located((
    #                 By.CSS_SELECTOR, "a[href*='logout']"))
    #             )
    #         # print("Usuário já está logado.")
    #         return  # Se o link de logout for encontrado, retorna cedo.
    #     except TimeoutException:
    #         # print("Tentando logar no servidor...")
    #         try:
    #             self.clicar_botao_avancado(driver)
    #             WebDriverWait(driver, 2).until(
    #                 EC.visibility_of_element_located((
    #                     By.CSS_SELECTOR, "#final-paragraph #proceed-link"))
    #             )
    #             link_prosseguir = driver.find_element(
    #                 By.CSS_SELECTOR, "#final-paragraph #proceed-link")
    #             link_prosseguir.click()
    #         except NoSuchElementException:
    #             # TO-DO: Tratar erros de certificado SSL.
    #             print("Erro ao lidar com certificado SSL.")
    #             return
    #         except TimeoutException:
    #             pass
            
    #     # Se o link de logout não for encontrado, prosseguir com o login.
    #     try:
    #         print(f"Inserindo credenciais de acesso...")
    #         driver.get(self.base_url)
    #         keys = self._ler_apikeys()
    #         try:
    #             WebDriverWait(driver, 10).until(
    #                 EC.presence_of_element_located((By.ID, "j_username"))
    #             )
    #             username_field = driver.find_element(By.ID, "j_username")
    #             username_field.send_keys(keys["username"])

    #             password_field = driver.find_element(By.ID, "j_password")
    #             password_field.send_keys(keys["password"])

    #             login_button = driver.find_element(By.NAME, "Submit")
    #             login_button.click()
    #         except TimeoutException:
    #             try:
    #                 print("Timeout, tentando inserir credenciais novamente...")
    #                 driver.get(self.base_url)
    #                 keys = self._ler_apikeys()
    #                 WebDriverWait(driver, 20).until(
    #                     EC.presence_of_element_located((By.ID, "j_username"))
    #                 )
    #                 username_field = driver.find_element(By.ID, "j_username")
    #                 username_field.send_keys(keys["username"])

    #                 password_field = driver.find_element(By.ID, "j_password")
    #                 password_field.send_keys(keys["password"])

    #                 login_button = driver.find_element(By.NAME, "Submit")
    #                 login_button.click()
    #             except Exception as e:
    #                 print("Erro ao inserir credenciais de acesso:")
    #                 print(e)
    #                 return
    #         except Exception as e:
    #             print(f"  Erro ao clicar no login {e}")

    #         # Detectar se as credenciais foram validadas com sucesso
    #         ## Procurar mensagem de login inativo
    #         WebDriverWait(driver, 10).until(
    #             EC.presence_of_element_located((By.ID, "j_username"))
    #         )

    #         # Aguarde até que o elemento esteja presente na página
    #         div_usuario_inativo = WebDriverWait(driver, 10).until(
    #             EC.presence_of_element_located((By.CSS_SELECTOR, 'div.alert.alert-danger'))
    #         )

    #         if div_usuario_inativo:
    #             print("Login negado pela plataforma com a mensagem:")
    #             msg_usuario_inativo = div_usuario_inativo.text.strip()
    #             print(msg_usuario_inativo)
    #             print("\nEntrar em contato com administrador da plataforma financiar para reativar o login de usuário")
    #             driver.close()
    #             return

    #         # Aguardar elemento específico só disponível após o login bem-sucedido: link de logout.
    #         WebDriverWait(driver, 10).until(
    #             EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='logout']"))
    #         )
    #         # Somente se não cair em nenhuma das excessões é porque o login teve sucesso e vai seguir o resto da execução
    #         print("Login realizado com sucesso.")
    #     except Exception as e:
    #         self.logger.error("  Erro após clicar no login: %s", e)
    #         print(f"  Descrição: {e}")
    #         driver.close()
    #         return

    # def search_funding(self, palavras_chave_lista, filtros=False):
    #     if self.driver is None:  # Check if driver is available
    #         print("O driver não está disponível. Certifique-se de que o login foi realizado com sucesso.")
    #         return pd.DataFrame(), self.falhas_totais

    #     for palavras_chave in palavras_chave_lista:
    #         # Determina se a palavra-chave é um termo composto
    #         termo_composto = " " in palavras_chave or palavras_chave.startswith('"') and palavras_chave.endswith('"')

    #         try:
    #             if self.driver is None:  # Check again before navigating
    #                 print("O driver não está disponível. Certifique-se de que o login foi realizado com sucesso.")
    #                 return pd.DataFrame(), self.falhas_totais
    #             self._navigate_to_search_page(self.driver)
    #         except:
    #             self._logout()
    #             return pd.DataFrame(), self.falhas_totais

    #         if filtros:
    #             if self.driver is None:  # Check again before filtering
    #                 print("O driver não está disponível. Certifique-se de que o login foi realizado com sucesso.")
    #                 return pd.DataFrame(), self.falhas_totais
    #             self._filter_pdi_ceara(self.driver)

    #         resultados_encontrados = self._execute_and_extract(palavras_chave)
    #         if not resultados_encontrados and termo_composto:
    #             # Remove as aspas se houver e divide o termo composto para buscar cada termo individualmente
    #             termos_individuais = palavras_chave.replace('"', '').split()
    #             for termo_individual in termos_individuais:
    #                 self._navigate_to_search_page(self.driver)
    #                 if filtros:
    #                     self._filter_pdi_ceara(self.driver)
    #                 self._execute_and_extract(termo_individual)
                    
    #     df_todos_programas = pd.DataFrame(self.todos_programas) if self.todos_programas else pd.DataFrame()

    #     # Check if df_todos_programas is empty before generating the report
    #     if not df_todos_programas.empty:
    #         self.mount_foment_report(df_todos_programas)
        
    #     self._logout()
    #     return df_todos_programas, self.falhas_totais

    ## Funções acessórias pra realizar o login na plataforma, ou encerrar em caso de acesso negado
    def _verificar_login(self, driver):
        """Verifica se o usuário já está logado, buscando o link de logout."""
        try:
            driver.get(self.base_url)
            WebDriverWait(driver, 2).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='logout']"))
            )
            return True  # Link de logout encontrado, usuário está logado
        except (TimeoutException, Exception):
            return False  # Link de logout não encontrado ou outro erro ocorreu

    def _clicar_botao_avancado(self, driver):
        """Clica no botão 'Avançado' para lidar com certificado SSL, se necessário."""
        try:
            WebDriverWait(driver, 2).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "#final-paragraph #proceed-link"))
            )
            link_prosseguir = driver.find_element(By.CSS_SELECTOR, "#final-paragraph #proceed-link")
            link_prosseguir.click()
        except TimeoutException:
            pass  # Botão 'Avançado' não encontrado, possivelmente não é necessário

    def _inserir_credenciais(self, driver):
        """Insere as credenciais de acesso e clica no botão de login."""
        print("Inserindo credenciais de acesso...")
        driver.get(self.base_url)
        keys = self._ler_apikeys()
        try:
            WebDriverWait(driver, 20).until(  # Aumentado o tempo de espera para 20 segundos
                EC.presence_of_element_located((By.ID, "j_username"))
            )
            username_field = driver.find_element(By.ID, "j_username")
            username_field.send_keys(keys["username"])

            password_field = driver.find_element(By.ID, "j_password")
            password_field.send_keys(keys["password"])

            login_button = driver.find_element(By.NAME, "Submit")
            login_button.click()
        except TimeoutException:
            print("Timeout ao inserir credenciais. Verifique sua conexão ou tente novamente mais tarde.")
            raise  # Lançar a exceção para ser tratada na função _login
        except Exception as e:
            print(f"Erro ao inserir credenciais de acesso: {e}")
            raise 

    def _navigate_to_search_page(self, driver):
        try:
            # Localizar e aguardar pelo link para página de busca avançada após login
            link_busca_avancada = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '.fa-search-plus'))
            )
            link_busca_avancada.click()
            # print("Acessada página de busca avançada")
        except Exception as e:
            print("Não foi possível acessar a busca avançada")
            print(f"Erro: {e}")
            return

    # Acionar checkboxes de fomento em pesquisa e inovação elegíveis para o Ceará
    def _filter_pdi_ceara(self, driver):
            # Aguardar que o primeiro checkbox esteja clicável
        try:
            # Esperar checkbox e clicar "ctp_1" Financiamento para Projetos de Pesquisa
            checkbox_ctp_1 = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.ID, "ctp_1")))
            checkbox_ctp_1.click()
            # Esperar pelo checkbox e clicar "ctp_8" para Empreendedorismo e Inovação
            checkbox_ctp_8 = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.ID, "ctp_8")))
            checkbox_ctp_8.click()
            # Selecionar no dropdown '6' valor correspondente ao estado do Ceará
            select_estado = Select(driver.find_element(By.NAME, 'estado'))
            select_estado.select_by_value('6')
            # Adicionar aqui mais configurações conforme necessário
        except Exception as e:
            self.logger.error("Erro ao configurar a busca avançada: %s", e.msg)

    def _extract_details(self, driver):
        # Esperar até que os detalhes do programa estejam visíveis
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#meio"))
        )
        # Dicionário para armazenar os dados detalhados
        detalhes_programa = {}

        # Coletar dados básicos
        titulo_detalhado = driver.find_element(By.CSS_SELECTOR, "h1.hidden-print").text
        if titulo_detalhado:
            detalhes_programa['titulo_detalhado'] = titulo_detalhado
        # Coletar detalhes disponíveis na página
        secoes = [
                  'elegibilidade',
                  'valorfinanciado',
                  'datalimite',
                  'formasolicitacao',                  
                  'descricao',
                  'homepage',                  
                #   'contatos',
                #   'fonte',
                #   'revisao',
                  ]
        for secao in secoes:
            try:
                conteudo = driver.find_element(By.NAME, secao).find_element(By.XPATH, "./following-sibling::div").text
                detalhes_programa[secao] = conteudo
            except:
                detalhes_programa[secao] = 'Informação não disponível'
        # Informações de contato da financiadora
        financiadora_info = driver.find_elements(By.CSS_SELECTOR, ".list-group-item")
        for item in financiadora_info:
            text = item.text.split("\n", 1)
            if len(text) > 1:
                chave, valor = text
                detalhes_programa[chave.strip()] = valor.strip()
        return detalhes_programa

    def _extract_program(self, driver):
        programas = []
        falhas = []  # Lista para armazenar as falhas temporariamente
        divs_programas = driver.find_elements(By.CLASS_NAME, "bs-callout")       
        for div_index, div in enumerate(divs_programas):
            tentativas = 0
            max_tentativas = 3
            while tentativas < max_tentativas:
                try:
                    financiadora = div.find_element(By.TAG_NAME, 'h4').text.strip().replace('\nDestaques', '')
                    link = div.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    titulo = div.find_element(By.TAG_NAME, 'a').text.strip()
                    # Abrir nova aba/janela e mudar o foco para ela
                    driver.execute_script("window.open('{}');".format(link))
                    driver.switch_to.window(driver.window_handles[-1])
                    detalhes = self._extract_details(driver)  # Tentar extrair os detalhes
                    programa_info = {'financiadora': financiadora,
                                     'titulo': titulo,
                                     'detalhes': detalhes}
                    programas.append(programa_info)
                    driver.close()  # Fecha a aba de detalhes
                    driver.switch_to.window(driver.window_handles[0]) # Voltar aba
                    break  # Sair do loop de tentativas após sucesso
                except Exception as e:
                    tentativas += 1
                    self.logger.error(f"Falha extração {div_index+1}: {e.msg}, tentativa {tentativas}")
                    driver.close()  # Assegurar fechamento da aba de detalhes na falha
                    driver.switch_to.window(driver.window_handles[0])  # Voltar aba
                    if tentativas >= max_tentativas:
                        falhas.append({'financiadora': financiadora, 
                                       'titulo': titulo, 
                                       'link': link, 
                                       'tentativas': tentativas, 
                                       'erro': str(e)})
        return programas, falhas

    def _execute_search(self, driver, palavra_chave):
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.NAME, "query"))
            )
            input_query = driver.find_element(By.NAME, 'query')
        
            input_query.clear()
            input_query.send_keys(palavra_chave)

            WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.NAME, "Submit"))
            )
            search_button = driver.find_element(By.NAME, "Submit")
            search_button.click()

            print(f"\nBuscando programas de fomento para: {palavra_chave}, aguarde...")

            # Verifica se existe o alerta de "Nenhum resultado encontrado" após a busca
            try:
                driver.find_element(By.CSS_SELECTOR, "div.alert.alert-danger span#ResultadoBusca")
                print(f"Nenhum resultado encontrado para: {palavra_chave}")
                return False
            except NoSuchElementException:
                # Resultados existem, prosseguir com a extração
                return True
        except Exception as e:
            print("Erro ao executar a busca avançada")
            print(e)
            return False

    def _execute_and_extract(self, palavra_chave):
        """Busca por palavra-chave e extrai resultados, retorna True se resultados forem encontrados."""
        if self._execute_search(self.driver, palavra_chave):
            # Processo de extração se resultados existirem
            try:
                # Tentar extrair a informação sobre a quantidade de oportunidades
                texto_quantidade_oportunidades = WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.col-md-12 > h5"))).text
                partes_texto = texto_quantidade_oportunidades.split()
                # Ajustar de acordo com o formato exato do texto, assumindo "Mostrando X de Y oportunidades"
                total_oportunidades = int(partes_texto[-1])
                pbar = tqdm(desc=f"Extraindo oportunidades para: {palavra_chave}", total=total_oportunidades)
            except TimeoutException:
                # Caso não seja possível extrair a quantidade total, inicializa o tqdm sem total definido
                pbar = tqdm(desc=f"Extraindo oportunidades para: {palavra_chave}")
                total_oportunidades = 0  # Assume 0 se não conseguir extrair o total

            oportunidades_extraídas = 0 
            while True:
                programas_atual, falhas = self._extract_program(self.driver)
                for programa in programas_atual:
                    programa['palavras-chave'] = palavra_chave
                    programa['quantidade'] = total_oportunidades  # Adiciona a quantidade total de oportunidades
                    self.todos_programas.append(programa)
                    oportunidades_extraídas += 1
                self.falhas_totais.extend(falhas)
                pbar.update(len(programas_atual))
                if oportunidades_extraídas >= total_oportunidades:
                    pbar.close()
                    break
                try:
                    proxima_pagina = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Próxima')]"))
                    )
                    proxima_pagina.click()
                except TimeoutException:
                    pbar.close()
                    break
            return True
        else:
            # Caso de "Nenhum resultado encontrado"
            self.todos_programas.append({'palavras-chave': palavra_chave, 
                                         'quantidade': 0})
            return False

    def set_report_date(self, nome_base):
        from datetime import datetime
        dt = datetime.today()
        formatted_string = dt.strftime('%Y%m%d')
        filename = nome_base+'_'+formatted_string+'.html'
        return filename

    def mount_foment_report(self, df, nome_base='relatorio_fomento'):
        # %pip install pandas jinja2
        try:
            df['detalhes'] = df['detalhes'].apply(eval)
        except:
            pass
        base_repo_dir = self._get_base_repo()
        # Configuração do Jinja2
        template_folder=os.path.join(base_repo_dir,'source','template')
        env = Environment(loader=FileSystemLoader(template_folder))
        template = env.get_template('template_fioce.html')
        html_output = template.render(dados=df) # Renderizar template com os dados
        # html_output = html_output.replace('\u2192', '&rarr;') # Substituir caracteres não utf8 que causariam problemas
       
        # Salvar o relatório gerado em um arquivo HTML
        report_name = self.set_report_date(nome_base)
        filepath = os.path.join(base_repo_dir,'source', 'visualizations',
                                report_name)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_output)
        print(f"Relatório montado disponível em {filepath}")

    @staticmethod
    def convert_to_html(text):
        # Replace line breaks with <br>, and any other transformations needed
        html_content = text.replace('\n', '<br>')
        return f"<html><body>{html_content}</body></html>"
    
    # @staticmethod
    # def save_to_json(data, filename):
    #     """
    #     Save the given data to a JSON file.

    #     :param data: List of dictionaries to be saved.
    #     :param filename: Name of the file where the data will be saved.
    #     """
    #     try:
    #         with open(filename, 'w', encoding='utf-8') as f:
    #             json.dump(data, f, ensure_ascii=False, indent=4)
    #         print(f"Data successfully saved to {filename}")
    #     except Exception as e:
    #         print(f"Error saving data: {e}")

    # def clean_pprint_output(self, output):
    #     # Remove unwanted characters introduced by pprint
    #     pprinted = pprint(output, width=200)
    #     try:
    #         if "('" in pprinted:
    #             output = pprinted.replace("('", "")
    #         if "'" in pprinted:    
    #             output = pprinted.replace("'", "")
    #         if "')" in pprinted:    
    #             output = pprinted.replace("')", "")
    #     except:
    #         pass
    #     # Handle any other pprint artifacts as needed
    #     return output

    # def inserir_logotipo(self, imagem, alinhamento="center"):
    #     """
    #     Insere um logotipo em um html.

    #     Args:
    #         imagem: O caminho para o arquivo .png do logotipo.
    #         alinhamento: O alinhamento do logotipo no cabecalho de página. Os valores possíveis são "left", "center" e "right".

    #     Returns:
    #         O código html do logotipo.
    #     """

    #     if alinhamento not in ("left", "center", "right"):
    #         raise ValueError("O alinhamento deve ser 'left', 'center' ou 'right'.")

    #     return html.escape(f"""
    #         <img src="{imagem}" alt="Logotipo" align="{alinhamento}" width="300" height="200">
    #     """)

    # def inserir_logotipos(self, logotipo_esquerdo=None, logotipo_centro=None, logotipo_direito=None):
    #     """
    #     Insere três logotipos em um html.

    #     Args:
    #         logotipo_esquerdo: O caminho para o arquivo .png do logotipo esquerdo.
    #         logotipo_centro: O caminho para o arquivo .png do logotipo central.
    #         logotipo_direito: O caminho para o arquivo .png do logotipo direito.

    #     Returns:
    #         O código html dos logotipos.
    #     """

    #     html = ""

    #     if logotipo_esquerdo is not None:
    #         html += self.inserir_logotipo(logotipo_esquerdo, "left")

    #     if logotipo_centro is not None:
    #         html += self.inserir_logotipo(logotipo_centro, "center")

    #     if logotipo_direito is not None:
    #         html += self.inserir_logotipo(logotipo_direito, "right")

    #     return html