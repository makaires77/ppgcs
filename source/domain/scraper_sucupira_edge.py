import os, platform, urllib.request, requests
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class SucupiraScraperEdge:
    def __init__(self):
        self.driver_path = self.get_driver()
        self.driver = webdriver.Edge(service=Service(self.driver_path))

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
        return SucupiraScraperEdge.find_repo_root(path.parent, depth-1)
    
    def get_driver(self):
        driver_dir = os.path.join(self.find_repo_root(), 'edgewebdriver')
        driver_path = os.path.join(driver_dir, 'msedgedriver.exe')
        if not os.path.exists(driver_path):
            os.makedirs(driver_dir, exist_ok=True)
            self.download_driver(driver_dir)
        return driver_path

    def download_driver(self, driver_dir):
        system = platform.system()
        if system == 'Windows':
            driver_url = 'https://msedgedriver.azureedge.net/123.0.2420.81/edgedriver_win64.zip'  # URL for Edge driver for Windows
        elif system == 'Linux':
            driver_url = 'https://msedgedriver.azureedge.net/123.0.2420.81/edgedriver_linux64.zip'  # URL for Edge driver for Linux
        else:
            raise Exception('Unsupported OS')

        driver_zip = os.path.join(driver_dir, 'msedgedriver.zip')
        urllib.request.urlretrieve(driver_url, driver_zip)

        with ZipFile(driver_zip, 'r') as zip_ref:
            zip_ref.extractall(driver_dir)

        os.remove(driver_zip)

    def close_popups(self):
        pass  # Implement this method based on the specific popups on the site

    def download_spreadsheet(self, save_path):
        try:
            self.driver.get(self.url)
            self.close_popups()

            # Wait for the page to load and find the download link
            download_link = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//a[contains(text(), "classificações_publicadas_todas_as_areas_avaliacao")]'))
            )

            # Get the file name from the download link
            file_name = download_link.text

            # Click the download link to start the download
            download_link.click()

            # Save the file to the specified path
            with open(save_path, 'wb') as file:
                file.write(self.driver.page_source.encode('utf-8'))

            print(f"Spreadsheet downloaded successfully to {save_path}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            self.driver.quit()

    def scrape_sucupira(self):
        self.driver.get('https://sucupira.capes.gov.br/sucupira/public/consultas/coleta/veiculoPublicacaoQualis/listaConsultaGeralPeriodicos.jsf')

        self.close_popups()

        # Select the option from the dropdown
        dropdown = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, '//select[@id="form:evento"]')))
        for option in dropdown.find_elements(By.TAG_NAME, 'option'):
            if '2017-2020' in option.text:
                option.click()
                break

        # Click the 'Consultar' button
        button = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, '//input[@value="Consultar"]')))
        button.click()

        # Wait for the page to load and find the download link
        download_link = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, '//a[contains(text(), "classificações_publicadas_todas_as_areas_avaliacao")]')))

        # Get the file name from the download link
        file_name = download_link.text
        
        # Click the download link to start the download
        download_link.click()

        save_path = os.path.join(self.find_repo_root(), 'data', file_name)

        # Save the file to the specified path
        with open(save_path, 'wb') as file:
            file.write(self.driver.page_source.encode('utf-8'))

        print(f"Spreadsheet downloaded successfully to {save_path}")

        with open(save_path, 'w') as f:
            f.write(self.driver.page_source)

        self.driver.quit()

class GetQualis:
    def __init__(self, planilha_excel, arquivo_json):
        self.planilha = pd.read_excel(planilha_excel)  # Carrega a planilha Excel
        self.json_data = arquivo_json  # Carrega o arquivo JSON

    def complementar_informacoes(self):
        for artigo in self.json_data['Produções']['Artigos completos publicados em periódicos']:
            issn = artigo['ISSN']
            estrato = self.planilha.loc[self.planilha['ISSN'] == issn, 'Estrato'].values[0]
            artigo['Estrato'] = estrato

if __name__ == '__main__':
    # Exemplo de uso
    file_name = 'classificações_publicadas_todas_as_areas_avaliacao1672761192111.xls'
    planilha_excel = os.path.join(SucupiraScraperEdge.find_repo_root(), 'data', file_name)
    arquivo_json = {
        # Seu arquivo JSON aqui
    }

    complementador = GetQualis(planilha_excel, arquivo_json)
    complementador.complementar_informacoes()

    # Agora o arquivo JSON foi atualizado com os valores de Estrato
    print(arquivo_json)