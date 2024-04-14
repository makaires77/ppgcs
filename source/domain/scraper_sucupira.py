import os, requests, platform
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class SucupiraScraper:
    def __init__(self, url):
        self.url = url
        self.driver = self.connect_driver()
        # self.options = webdriver.ChromeOptions()
        # self.options.add_argument('--ignore-certificate-errors')
        # self.options.add_argument('--ignore-ssl-errors')
        # self.driver = webdriver.Chrome(options=self.options)
        self.response = requests.get(self.url)
        self.soup = BeautifulSoup(self.response.text, 'html.parser')

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
        return SucupiraScraper.find_repo_root(path.parent, depth-1)
    
    @staticmethod
    def connect_driver():
        '''
        Conecta ao servidor da plataforma Sucupira
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
                driver_path=SucupiraScraper.find_repo_root(os.getcwd())/'chromedriver'/'chromedriver.exe'
            else:
                driver_path=SucupiraScraper.find_repo_root(os.getcwd())/'chromedriver'/'chromedriver'
        except Exception as e:
            print("Não foi possível estabelecer uma conexão, verifique o chromedriver")
            print(e)
        
        # print(driver_path)
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service)    
        url_busca = 'https://sucupira.capes.gov.br/sucupira/'
        driver.get(url_busca) # acessa a url da página inicial do sucupira   
        driver.set_window_position(-20, -10)
        driver.set_window_size(170, 1896)
        driver.mouse = webdriver.ActionChains(driver)
        return driver

    def access_qualis_page(self):
        self.driver.get(self.url)
        try:
            print("Acessar Qualis...")
            qualis_element = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'div[ui-popup="ui-popup"][target="#qualis"]'))
            )
            qualis_element.click()
        except Exception as e:
            print("Erro ao acessar lista de classificação de periódicos...")
            return

    def click_accept_button(self):
        try:
            # print("Clicar no botão ACEITO...")
            accept_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//div[@class="br-modal-footer actions"]//button[text()="ACEITO"]'))
            )
            accept_button.click()

            # Aguardar o carregamento da página após clicar no botão ACEITO
            WebDriverWait(self.driver, 30).until(
                EC.invisibility_of_element_located((By.XPATH, '//div[@class="br-modal-footer actions"]'))
            )
            print("Página carregada após clicar no botão ACEITO.")
        except Exception as e:
            print("Erro ao clicar no botão ACEITO:")
            print(e)

    def close_remanescent_popup(self):
        try:
            print("Fechando pop-up...")
            popup_close_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.close > a'))
            )
            popup_close_button.click()
            print("Pop-up fechado com sucesso.")
        except Exception as e:
            print("Erro ao fechar pop-up:", e)

    def close_pop_up(self):
        try:
            # print("Fechando pop-up...")
            # Esperar até que o pop-up seja visível
            pop_up = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, '.messages-container.top.center'))
            )
            # Localizar o botão de fechar e clicar nele
            close_button = pop_up.find_element(By.CLASS_NAME, 'message-close')
            close_button.click()
            print("Pop-up fechado com sucesso.")
        except Exception as e:
            print("Erro ao fechar o pop-up:", e)

    def close_second_pop_up(self):
        try:
            # print("Fechando segundo pop-up...")
            # Esperar até que o pop-up seja visível
            pop_up = WebDriverWait(self.driver, 30).until(
                EC.visibility_of_element_located((By.ID, 'myModal'))
            )
            # Localizar o botão de fechar e clicar nele
            close_button = pop_up.find_element(By.XPATH, '//button[@class="br-button primary small"]')
            close_button.click()
            print("Segundo pop-up fechado com sucesso.")
        except Exception as e:
            print("Erro ao fechar o segundo pop-up:", e)

    def close_third_pop_up(self):
        try:
            # print("Fechando terceiro pop-up...")
            # Esperar até que o pop-up seja visível
            pop_up = WebDriverWait(self.driver, 30).until(
                EC.visibility_of_element_located((By.CLASS_NAME, 'close'))
            )
            # Localizar o botão de fechar e clicar nele
            close_button = pop_up.find_element(By.TAG_NAME, 'a')
            close_button.click()
            print("Terceiro pop-up fechado com sucesso.")
        except Exception as e:
            print("Erro ao fechar o terceiro pop-up:", e)

    def click_qualis_banner(self):
        try:
            # Verificar se há um pop-up remanescente e fechá-lo, se encontrado
            try:
                pop_up = WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.XPATH, '//div[@class="popup-remanescente"]'))
                )
                close_button = pop_up.find_element(By.XPATH, '//button[@class="close"]')
                close_button.click()
                print("Pop-up remanescente fechado com sucesso.")
            except:
                print("Nenhum pop-up remanescente encontrado.")

            print("Clicando no banner do Qualis...")
            # Esperar até que o banner do Qualis seja clicável na página
            banner_qualis = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//div[contains(@class, "container-img-btn")]/parent::div[@ui-popup="ui-popup"][@target="#qualis"]'))
            )
            # Role a página para o banner do Qualis ficar visível
            self.driver.execute_script("arguments[0].scrollIntoView();", banner_qualis)
            # Aguarde um curto intervalo para garantir que o banner seja totalmente carregado e visível
            self.driver.implicitly_wait(2)
            # Clicar no elemento pai que contém o banner do Qualis
            banner_qualis.click()
            print("Banner do Qualis clicado com sucesso.")

        except Exception as e:
            print("Erro ao clicar no banner do Qualis:", e)

    def click_search_button(self):
        try:
            print("Clicar em Buscar...")
            buscar_button = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//div[@class="form-group"]/a[contains(@id, "formQualis:j_idt470")]'))
            )
            buscar_button.click()
        except Exception as e:
            print("Erro ao clicar em Buscar...")
            print(e)

    def select_option(self):
        try:
            print("Escolher período...")
            dropdown_menu = self.driver.find_element(By.ID, 'form:evento')
            select = Select(dropdown_menu)
            select.select_by_value('236')
        except Exception as e:
            print("Erro ao clicar no dropdown...")
            print(e)

    def click_consult_button(self):
        try:
            print("Clicar em Buscar...")
            consultar_button = self.driver.find_element(By.ID, 'form:consultar')
            consultar_button.click()
        except Exception as e:
            print("Erro ao clicar em buscar...")
            print(e)

    def download_file(self):
        link = self.soup.find('a', href=True)
        download_url = link['href']
        print("Fazendo o download do arquivo...")
        r = requests.get(download_url, stream=True)
        filename = os.path.join('data', download_url.split('/')[-1])
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Arquivo baixado para: {filename}")

if __name__ == "__main__":
    url = 'https://sucupira.capes.gov.br/sucupira/'
    scraper = SucupiraScraper(url)
    scraper.access_qualis_page()
    scraper.close_pop_up()
    scraper.close_second_pop_up()
    scraper.close_third_pop_up()
    scraper.click_qualis_banner()
    scraper.click_search_button()
    scraper.select_option()
    scraper.click_consult_button()
    scraper.download_file()