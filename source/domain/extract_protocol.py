import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd

class SaudeGovDataExtractor:
    def __init__(self, url):
        self.url = url
        self.soup = None
        self.df = None
        self.links = None

    def extract_data(self):
        response = requests.get(self.url)
        self.soup = BeautifulSoup(response.content, 'html.parser')

    # def extract_data(self):
    #     options = webdriver.ChromeOptions()
    #     options.add_argument('--headless')  # Para executar o Chrome sem interface gráfica
    #     driver = webdriver.Chrome(options=options)

    #     driver.get(self.url)

    #     # Espera que a div #parent-fieldname-text seja carregada
    #     wait = WebDriverWait(driver, 10)  # Tempo máximo de espera: 10 segundos
    #     data_div = wait.until(
    #         EC.presence_of_element_located((By.ID, 'parent-fieldname-text'))
    #     )

    #     # Extrai o HTML da div
    #     html_content = data_div.get_attribute('innerHTML')
    #     print(html_content)
    #     self.soup = BeautifulSoup(html_content, 'html.parser')
        # print(self.soup)

        # if response.status_code != 200:
        #     print('Erro na requisição na página do Ministério da Saúde')
        #     print(f'Código de resposta para requisição: {response.status_code}')
        #     print('Verifique conexão com a internet e tente novamente mais tarde.')
        # else:
        #     print('Requisição atendida com sucesso, lendo dados da página...')
        # self.soup = BeautifulSoup(response.content, 'html.parser')

        agravos = []
        doc_types = ["Diretrizes Brasileiras", "PCDT", "DDT", "Protocolo de Uso", "Linha de Cuidado"]
        doc_counts = {doc_type: 0 for doc_type in doc_types}
        self.links = []

        # Novo seletor CSS para a div #content-core
        data_div = self.soup.find('#wrapper #main-content #content #content-core #parent-fieldname-text') 
        print(data_div)

        # Itera sobre as listas de itens (ul) dentro da div
        for ul_tag in data_div.find_all('ul'):
            for item in ul_tag.find_all('li'):
                text = item.text.strip()
                agravo, doc_type = text.split(" (")
                doc_type = doc_type[:-1]  # Remove o parêntese final
                agravos.append(agravo)
                doc_counts[doc_type] += 1
                link_tag = item.find('a')
                if link_tag:
                    self.links.append(link_tag['href'])
                else:
                    print(f"Link não encontrado para: {agravo}")  # Para depuração

        self.df = pd.DataFrame(doc_counts, index=agravos)

    def get_dataframe(self):
        if self.df is None:
            self.extract_data()
        return self.df

    def get_links(self):
        if self.links is None:
            self.extract_data()
        return self.links
    
    def download_pdfs(self):
        if self.links is None:
            self.extract_data()

        os.makedirs("data/in_pdf", exist_ok=True)  # Cria a pasta se não existir

        for link in self.links:
            filename = os.path.join("data/in_pdf", link.split("/")[-1])
            response = requests.get(link)

            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"PDF baixado: {filename}")
            else:
                print(f"Erro ao baixar PDF: {link}")    