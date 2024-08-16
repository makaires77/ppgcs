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

class FundingFinder:
    def __init__(self):
        self.base_repo_dir = self._get_base_repo()
        self.folder_utils = os.path.join(self.base_repo_dir, 'utils')
        self.folder_assets = os.path.join(self.base_repo_dir, 'assets')        
        self.folder_domain = os.path.join(self.base_repo_dir, 'source', 'domain')
        self.folder_data_input = os.path.join(self.base_repo_dir, 'data', 'input')
        self.folder_data_output = os.path.join(self.base_repo_dir, 'data', 'output')            
        self.base_url = "https://www.financiar.org.br/index.php"
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
        self._start()

    def _start(self):
        self._login(self.driver)        

    def _logout(self):
        if self.driver:
            self.driver.quit()
            self.driver = None

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

    def _login(self, driver):
        # Verificar se já está logado.
        try:
            driver.get(self.base_url)
        except:
            print("não foi possível alcançar a página verifique sua conexão.")
            return
        try:
            # Tentar primeiro encontrar o link de logout, se presente está logado.
            logout_link = WebDriverWait(driver, 2).until(
                EC.presence_of_element_located((
                    By.CSS_SELECTOR, "a[href*='logout']"))
                )
            # print("Usuário já está logado.")
            return  # Se o link de logout for encontrado, retorna cedo.
        except TimeoutException:
            # print("Tentando logar no servidor...")
            try:
                self.clicar_botao_avancado(driver)
                WebDriverWait(driver, 2).until(
                    EC.visibility_of_element_located((
                        By.CSS_SELECTOR, "#final-paragraph #proceed-link"))
                )
                link_prosseguir = driver.find_element(
                    By.CSS_SELECTOR, "#final-paragraph #proceed-link")
                link_prosseguir.click()
            except NoSuchElementException:
                # TO-DO: Tratar erros de certificado SSL.
                print("Erro ao lidar com certificado SSL.")
                return
            except TimeoutException:
                pass
            
        # Se o link de logout não for encontrado, prosseguir com o login.
        try:
            print(f"Inserindo credenciais de acesso...")
            driver.get(self.base_url)
            keys = self._ler_apikeys()
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "j_username"))
                )
                username_field = driver.find_element(By.ID, "j_username")
                username_field.send_keys(keys["username"])

                password_field = driver.find_element(By.ID, "j_password")
                password_field.send_keys(keys["password"])

                login_button = driver.find_element(By.NAME, "Submit")
                login_button.click()
            except Exception as e:
                print(f"  Erro ao clicar no login {e}")
            
            # Aguardar elemento específico só disponível após o login bem-sucedido: link de logout.
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='logout']"))
            )
            print("Login realizado com sucesso.")
        except Exception as e:
            self.logger.error("  Erro após clicar no login: %s", e)
            print(f"  Descrição: {e}")
            driver.close()
            return
    
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

    def search_funding(self, palavras_chave_lista, filtros=False):
        for palavras_chave in palavras_chave_lista:
            # Determina se a palavra-chave é um termo composto
            termo_composto = " " in palavras_chave or palavras_chave.startswith('"') and palavras_chave.endswith('"')

            self._navigate_to_search_page(self.driver)
            if filtros:
                self._filter_pdi_ceara(self.driver)

            resultados_encontrados = self._execute_and_extract(palavras_chave)
            if not resultados_encontrados and termo_composto:
                # Remove as aspas se houver e divide o termo composto para buscar cada termo individualmente
                termos_individuais = palavras_chave.replace('"', '').split()
                for termo_individual in termos_individuais:
                    self._navigate_to_search_page(self.driver)
                    if filtros:
                        self._filter_pdi_ceara(self.driver)
                    self._execute_and_extract(termo_individual)
                    
        df_todos_programas = pd.DataFrame(self.todos_programas) if self.todos_programas else pd.DataFrame()
        self.mount_foment_report(df_todos_programas)
        self._logout()
        return df_todos_programas, self.falhas_totais

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