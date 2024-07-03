import os
import requests
import pandas as pd
from git import Repo
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC

class SaudeGovDataExtractor:
    def __init__(self, url):
        self.url = url
        self.soup = None
        self.df = None
        self.links = None

    def padronizar_tipos(self):
        """Padroniza os tipos de documentos no DataFrame self.df."""

        # Mapa de padronização
        padronizacao = {
            "PCDT": ["PCDT"],
            "DDT": ["DDT"],
            "Protocolos de Uso": ["Protocolo de uso", "Protocolo de Uso", "Protocolos de Uso", "Protocolo para Diagnóstico Etiológico"],
            "Diretrizes Nacionais/Brasileiras": ["Diretriz Nacional", "Diretriz Brasileira", "Diretrizes Brasileiras", "Diretrizes Brasileiras para o Diagnóstico", "Diretrizes Brasileiras para Diagnóstico e Tratamento"],
            "Linhas de Cuidado": ["Linha de Cuidado"]
        }

        # Colunas originais da tabela pivô
        colunas_originais = self.df.columns

        # Lista para armazenar as novas colunas
        novas_colunas = []

        # Itera sobre as colunas originais
        for coluna in colunas_originais:
            # Encontra o tipo padronizado correspondente
            for tipo_padronizado, tipos_originais in padronizacao.items():
                if coluna in tipos_originais:
                    novas_colunas.append(tipo_padronizado)
                    break  # Sai do loop interno quando encontra o tipo padronizado

        # Renomeia as colunas do DataFrame
        self.df.columns = novas_colunas

    def extract_data(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)

        try:
            driver.get(self.url)

            # Espera que a div #parent-fieldname-text seja carregada com timeout
            wait = WebDriverWait(driver, 10)
            body_div = wait.until(
                EC.presence_of_element_located((By.ID, 'wrapper'))
            )
            
            marcador_css = 'parent-fieldname-text'
            print(f'Extraindo: {driver.title}')
            data_div = wait.until(
                EC.presence_of_element_located((By.ID, marcador_css))
            )

            html_content = data_div.get_attribute('innerHTML')
            self.soup = BeautifulSoup(html_content, 'html.parser')

            agravos = []
            tipodoc = []
            doc_types = ["Diretriz Nacional", "Diretriz Brasileira", "Diretrizes Brasileiras", "Diretrizes Brasileiras para o Diagnóstico", "Diretrizes Brasileiras para Diagnóstico e Tratamento", "PCDT", "DDT", "Protocolo de uso", "Protocolo de Uso", "Protocolos de Uso", "Protocolo para Diagnóstico Etiológico", "Linha de Cuidado"]
            doc_counts = {doc_type: 0 for doc_type in doc_types}
            self.links = []

            ul_divs = self.soup.find_all('ul')
            for ul_tag in ul_divs:
                for item in ul_tag.find_all('li'):
                    text = item.text.strip()
                    try:
                        tipo = text.split('(')[-1].split(')')[0]
                        agravo = text.split('(')[0].replace(tipo,'').strip()
                        if tipo == 'Linha de Cuidado do Infarto Agudo do Miocárdio e o Protocolo de Síndromes Coronarianas Agudas':
                            tipo = 'Linha de Cuidado'
                            agravo = 'Infarto Agudo do Miocárdio e o Protocolo de Síndromes Coronarianas Agudas'
                        if tipo == 'Hemofilia A – Uso do Emicizumabe\xa0*Publicado em 30/08/2021':
                            tipo = 'Protocolo de Uso'
                            agravo = 'Hemofilia A'
                        # print(agravo , tipo)
                        agravos.append(agravo)
                        tipodoc.append(tipo)
                        doc_counts[tipo] += 1
                        link_tag = item.find('a')
                        if link_tag:
                            self.links.append(link_tag['href'])
                        else:
                            print(f"Link não encontrado para: {agravo}")  # Para depuração
                    except Exception as e:
                        print(f'    Erro: {e}')
                        print(f'      Em: {text}')
                        print(f'  Agravo: {agravo}')
                        print(f'    Tipo: {tipo}')

            # Mapa de padronização
            padronizacao = {
                "PCDT": ["PCDT"],
                "DDT": ["DDT"],
                "Protocolos de Uso": ["Protocolo de uso", "Protocolo de Uso", "Protocolos de Uso", "Protocolo para Diagnóstico Etiológico"],
                "Diretrizes Nacionais/Brasileiras": ["Diretriz Nacional", "Diretriz Brasileira", "Diretrizes Brasileiras", "Diretrizes Brasileiras para o Diagnóstico", "Diretrizes Brasileiras para Diagnóstico e Tratamento"],
                "Linhas de Cuidado": ["Linha de Cuidado"]
            }

            # Padroniza os tipos de documentos na lista tipodoc
            tipodoc_padronizado = []
            for tipo in tipodoc:
                for tipo_padronizado, tipos_originais in padronizacao.items():
                    if tipo in tipos_originais:
                        tipodoc_padronizado.append(tipo_padronizado)
                        break

            # Cria um DataFrame com os dados extraídos e os tipos padronizados
            df = pd.DataFrame({'Agravo': agravos, 'Tipo': tipodoc_padronizado})

            # Gera a tabela pivô
            pivot_df = df.pivot_table(index='Agravo', columns='Tipo', aggfunc='size', fill_value=0)

            self.df = pivot_df

        except TimeoutException:
            print(f"Tempo limite excedido ao carregar a página: {self.url}")
            return None

        finally:
            driver.quit()  # Fecha o navegador em qualquer caso

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
        erros=[]
        sucessos=[]
        # Obter a pasta raiz do repositório Git
        try:
            repo = Repo(search_parent_directories=True)
            root_folder = repo.working_tree_dir
        except git.exc.InvalidGitRepositoryError:
            root_folder = os.getcwd()  # Usar o diretório de trabalho atual se não estiver em um repositório Git

        # Encontrar ou criar a pasta para salvar os PDFs
        os.makedirs(os.path.join(root_folder, "_data","in_pdf"), exist_ok=True)
        qte_total = len(self.links)

        for n,link in enumerate(self.links):
            if link[-1] == '/':
                link = link[:-1]
            document = link.split("/")[-1]
            if '#' in document:
                document = document.split('#')[0]
            filename = os.path.join(root_folder,"_data","in_pdf", document)

            if not filename:
                filename = filename = os.path.join(root_folder,"_data","in_pdf", document)
                if document == '':
                    raise Exception

            # Verifica se o arquivo já existe
            if not os.path.exists(filename):
                # Efetuar requisições e downloads
                try:
                    response = requests.get(link)
                    if response.status_code == 200:
                        # Verifica o tipo de conteúdo para determinar a extensão (somente se ainda não houver)
                        if not os.path.splitext(filename)[1]:
                            content_type = response.headers.get('content-type')
                            if 'html' in content_type:
                                filename += '.html'
                            elif 'pdf' in content_type:
                                filename += '.pdf'
                            else:
                                print(f"Tipo de conteúdo desconhecido: {content_type}")
                                erros.append(link)
                                continue  # Pula para o próximo link

                        with open(filename, 'wb') as f:
                            f.write(response.content)
                        print(f"{n+1:3}/{qte_total:3}: {document}")
                        sucessos.append(link)
                    else:
                        print(f"    Erro de requisição: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"    Erro ao baixar o arquivo: {e}")
                    erros.append(link)
                    print(f'    {document}')
                except Exception as e:
                    print('         Erro: Caminho vazio no site do MS')
                    erros.append(link)
                    print(f'         {document}')
            else:
                print(f"{n+1:3}/{qte_total:3}: {document} (arquivo já existe)")

            # # Efetuar requisições e downloads
            # try:
            #     response = requests.get(link)

            #     if response.status_code == 200:
            #         with open(filename, 'wb') as f:
            #             f.write(response.content)
            #         print(f"{n+1:3}/{qte_total:3}: {document}")
            #         sucessos.append(link)
            #     else:
            #         print(f"         Erro de requisição: {response.status_code}")
            # except requests.exceptions.RequestException as e:
            #     print(f"         Erro ao baixar o arquivo: {e}")
            #     erros.append(link)
            #     print(f'         {document}')
            # except Exception as e:
            #     print('         Erro: Caminho vazio no site do MS')
            #     erros.append(link)
            #     print(f'         {document}')
        print('Extração concluída!')
        print(f'{len(sucessos):3} PDFs extraídos com sucesso')
        print(f'{len(erros):3} links apresentaram erro ao extrair')
        return self.df, sucessos, erros