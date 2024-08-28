from tqdm import tqdm
from pprint import pprint
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from fitz import FileDataError
from jinja2 import Environment, FileSystemLoader
from requests.exceptions import RequestException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
import pandas as pd, numpy as np, os, re, json, requests, logging, tempfile, time, fitz

class FundingFinderCNPq:
    def __init__(self):
        self.base_repo_dir = self._get_base_repo()
        self.folder_utils = os.path.join(self.base_repo_dir, 'utils')
        self.folder_assets = os.path.join(self.base_repo_dir, 'assets')        
        self.folder_domain = os.path.join(self.base_repo_dir, 'source', 'domain')
        self.folder_data_input = os.path.join(self.base_repo_dir, 'data', 'input')
        self.folder_data_output = os.path.join(self.base_repo_dir, 'data', 'output')            
        self.driver = self._setup_driver()
        self.todos_programas = []
        self.falhas_totais = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        # Configuração do logger
        logging.basicConfig(filename=os.path.join(self.base_repo_dir,'source','logs','founding_finder.log'),
                            level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self._start()

    ## Métodos auxilires para preparar para execução do pipeline principal
    def _start(self):
        # self._login(self.driver)
        pass      

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

    def _setup_driver(self):
        chrome_options = Options()
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # Habilitar logs para diagnóstico
        filepath = os.path.join(self.base_repo_dir,'chromedriver','chromedriver.log')
        # service = ChromeService(log_path=filepath, enable_verbose_logging=True)
        service = ChromeService(log_path=filepath)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(60)  # Definir timeout para carregamento de página
        return driver

    def _ler_apikeys(self):
        # Obter caminho absoluto para a pasta home do usuário
        home_dir = os.path.expanduser("~")
        # Criar caminho completo para o arquivo secrets.json
        secrets_file_path = os.path.join(home_dir, "secrets_founding.json")
        # Verificar se o arquivo existe
        if os.path.exists(secrets_file_path):
            # Abra o arquivo secrets.json para leitura
            with open(secrets_file_path, 'r') as secrets_file:
                secrets = json.load(secrets_file)
                return secrets
        else:
            print("O arquivo secrets.json não foi encontrado na pasta home.")

    @staticmethod
    def convert_to_html(text):
        # Replace line breaks with <br>, and any other transformations needed
        html_content = text.replace('\n', '<br>')
        return f"<html><body>{html_content}</body></html>"

    def ler_pdf_link(self, url_pdf):
        """
        Função para ler um PDF a partir de um link usando PyMuPDF e retornar o texto extraído.

        Argumentos:
            url_pdf (str): URL do arquivo PDF.

        Retorno:
            str: Texto extraído do PDF.
        """
        substituicoes = {
            "\n": " ",
            "  ": " ",
            "Ɵ": "ti",
            "ơ": "ti",
            "‒": "-",
            "hƩp": "http",
            "LaƩes": "Lattes",
        }
        response = requests.get(url_pdf)
        # print(response.status_code)
        pdf_content = response.content
        texto_completo = ""
        with fitz.open(pdf_content, "rb") as pdf_documento:
            for pagina in pdf_documento.pages():
                texto_pagina = pagina.get_text("text")
                for caracter, substituto in substituicoes.items():
                    texto_pagina = texto_pagina.replace(caracter, substituto)
                texto_completo += texto_pagina.strip()
        return texto_completo

    def ler_pdf_local(self, caminho_pdf):
        """
        Função para ler um arquivo PDF usando PyMuPDF e retornar o texto extraído.

        Argumentos:
            caminho_pdf (str): Caminho para o arquivo PDF.

        Retorno:
            str: Texto extraído do PDF.
        """
        substituicoes = {
            "\n": " ",
            "  ": " ",
            "Ɵ": "ti",
            "ơ": "ti",
            "‒": "-",
            "hƩp": "http",
            "LaƩes": "Lattes",
        }
        pdf_documento = fitz.open(caminho_pdf)
        texto_completo = ""
        for pagina in pdf_documento.pages():
            texto_pagina = pagina.get_text("text")
            for caracter, substituto in substituicoes.items():
                texto_pagina = texto_pagina.replace(caracter, substituto)
            texto_completo += texto_pagina.strip()
        pdf_documento.close()
        return texto_completo

    def extract_inscricao_data(self, div_inscricao):
        """
        Extrai a data de início e a data de término das inscrições a partir de um elemento `div` com classe `inscricao`.
        Args:
            div_inscricao: Elemento BeautifulSoup da div com classe `inscricao`.
        Returns:
            Dicionário com as chaves "data_inicio_inscricao" e "data_termino_inscricao".
        """
        texto_li = div_inscricao.find("li").text.strip() # Extrair texto da li
        data_inicio, data_termino = texto_li.split(" a ") # Dividir texto em data de início e data de término
        # Formatar datas
        try:
            data_inicio = datetime.strptime(data_inicio.strip(), "%d/%m/%Y").date()
        except Exception as e:
            print(e)
            data_inicio = np.NaN
        try:
            data_termino = datetime.strptime(data_termino.strip(), "%d/%m/%Y").date()
        except:
            data_termino = np.NaN
        return {"data_inicio_inscricao": data_inicio, 
                "data_termino_inscricao": data_termino}

    # Receber as duas divs distintas e combinar seus dados
    def extract_data(self, div_content, div_bottom_content, max_retries=3):
        """Extrai dados da 'div_content' e o link do PDF da 'div_bottom_content'.

        Args:
            div_content: Objeto BeautifulSoup da 'div' com classe 'content'.
            div_bottom_content: Objeto BeautifulSoup da 'div' com classe 'bottom-content'.

        Returns:
            Dicionário contendo os dados extraídos ou None se não forem encontrados.
        """
        titulo = div_content.find("h4").text.strip()
        descricao = div_content.find("p").text.strip()
        data_inscricao = self.extract_inscricao_data(div_content)
        if not data_inscricao:
            data_inscricao = {"data_inicio_inscricao": None,
                            "data_termino_inscricao": None}
        link_pdf_element = div_bottom_content.find("a",
                                                   alt="Chamada",
                                                   class_="btn")
        if link_pdf_element:
            link_pdf = link_pdf_element.get('href')
            conteudo_pdf = self.ler_pdf_link_temp(link_pdf)
            return {
                "link": link_pdf,
                "titulo": titulo,
                "descricao": descricao,
                "conteudo": conteudo_pdf,
                **data_inscricao
            }
        else:
            print(f"Erro: Link do PDF não encontrado em {titulo}")
            link_pdf = None

    def mount_dfchamadas(self, url, max_retries=3):
        """Extrai e organiza dados de chamadas do CNPq em um DataFrame.
        Args:
            url: URL da página de resultados de chamadas do CNPq.
            max_retries: Número máximo de tentativas para recuperar PDFs.
        Returns:
            DataFrame pandas com as informações das chamadas.
        """
        chamadas_data = []  # Criar lista para armazenar dados das chamadas
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        # div_lista = soup.find("ol", class_="unstyled list-striped list-chamadas")
        # print(len(div_lista))
        divs_content = soup.find_all("div", class_="content", tabindex="0")
        divs_bottom_content = soup.find_all("div", class_="bottom-content")
        if len(divs_content) != len(divs_bottom_content):
            print("Erro: Quantidades diferentes de divs 'content' e 'bottom-content'")
            exit()
        for div_content, div_bottom_content in zip(divs_content, divs_bottom_content):
            print(len(div_content), len(div_bottom_content)) # Debug
            try:
                if div_content and div_bottom_content:
                    # Adicionar dados à lista, como um dicionário
                    chamadas_data.append(self.extract_data(div_content,
                                                        div_bottom_content))
            except Exception as e:
                print(f"Erro ao processar chamada")
                print(e)
        return pd.DataFrame(chamadas_data)

    # Funcionamento básico sem exception e retry
    def ler_pdf_link_temp(self, url_pdf):
        """
        Função para ler um PDF, gerando um arquivo temporário, a partir de um link usando PyMuPDF e retornar o texto extraído.
        Argumentos:
            url_pdf (str): URL do arquivo PDF.
        Retorno:
            str: Texto extraído do PDF.
        """
        substituicoes = {
            "\n": " ",
            "  ": " ",
            "Ɵ": "ti",
            "ơ": "ti",
            "‒": "-",
            "hƩp": "http",
            "LaƩes": "Lattes",
        }
        try:
            response = requests.get(url_pdf)
            # print(response.status_code)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_pdf: 
                temp_pdf.write(response.content)
                temp_pdf.seek(0)  # Volta ponteiro para início do arquivo
                texto_completo = ""
                with fitz.open(temp_pdf.name) as pdf_documento:
                    for pagina in pdf_documento.pages():
                        texto_pagina = pagina.get_text("text")
                        for caracter, substituto in substituicoes.items():
                            texto_pagina = texto_pagina.replace(caracter, substituto)
                        texto_completo += texto_pagina.strip()
                        # print(texto_completo)
            return texto_completo
        except requests.exceptions.RequestException as error:
            print(f"Error reading PDF: {error}")
            return None

    def substituir_erros(self, texto, corrigir):
        """Substitui as chaves do dicionário 'corrigir' pelo seu valor respectivo no 'texto'

        Args:
            texto (str): O texto no qual aplicar as correções
            corrigir (dict): dicionário de correções de erro

        Returns:
            str: O texto atualizado com as correções
        """
        padrao = re.compile("|".join(corrigir.keys()))
        return padrao.sub(lambda m: corrigir[m.group(0)], texto)

    def texto_exigibilidade(self, text):
        """
        Extrai o conteúdo entre o final de "Critérios de Elegibilidade" e o próximo match de separador.
        Argumentos:
            text: String contendo o texto a ser analisado.
        Retorna:
            String contendo o conteúdo extraído.
        """
        pattern_separador = r'\s\d{1,2}(?:\s|-(?:\s)?|‒\s)'
        pattern_exigibilidade = r'(?<=Critérios de Elegibilidade)\s*'
        matches = re.findall(pattern_exigibilidade, text)
        # Usar re.finditer para encontrar todos os matches como objetos com start e end
        matches = re.finditer(pattern_exigibilidade, text)
        if matches:
            # Iterar sobre os matches para encontrar o último
            ultimo_match = None
            for match in matches:
                ultimo_match = match
            match_separador = re.search(pattern_separador, text[ultimo_match.end():])
            if match_separador:
                return text[ultimo_match.end():match_separador.start() + ultimo_match.end()].strip()
            else:
                return text[ultimo_match.end():].strip()

    def texto_proponente(self, text):
        """
        Extrai o conteúdo entre o final de "Critérios de Elegibilidade" e o próximo match de separador.
        Argumentos:
            text: String contendo o texto a ser analisado.
        Retorna:
            String contendo o conteúdo extraído.
        """
        pattern_separador = r'\s\d{1,2}(?:\s|-(?:\s)?|‒\s|\.\s)'
        pattern_proponente = r'(?<=Quanto ao Proponente)\s*'
        if not isinstance(text, str):
            text = str(text)  # Converte explicitamente
        matches = re.findall(pattern_proponente, text)

        # Usar re.finditer para encontrar todos os matches caso1
        matches = re.finditer(pattern_proponente, text)
        if matches:
            # Iterar sobre os matches para encontrar o último
            ultimo_match = None
            for match in matches:
                ultimo_match = match
            if ultimo_match:
                match_separador = re.search(pattern_separador, text[ultimo_match.end():])
                if match_separador:
                    # print(f"Separador: {match_separador}")
                    return text[ultimo_match.end()+1:match_separador.start() + ultimo_match.end()].strip()
                else:
                    # print(f"Padrão separador não encontrado: {pattern_separador}")
                    return text[ultimo_match.end()+1:].strip()
            else:
                return None
        else:
            return None
        
    def texto_valores(self, text):
        # '(?:\s+\w+)?' procura zero ou uma sequência de espaços seguida de caracteres alfanuméricos não brancos (palavras) de forma opcional.
        pattern = r'valor global de(?:\s+\w+)?\s+([A-Z]{1,3})\$\s+(.*?)\)'
        # Usar re.search para encontrar apenas a primeira ocorrência
        match = re.search(pattern, text)
        if match:
            symbol, number = match.groups()
            return(f"{symbol}$ {number})")
        else:
            pass
            # print("Não foi possível encontrar o padrão de valor global.")

    def mount_final(self,df_content,palavrachave):
        df_content['financiadora'] = 'CNPq'
        # elegibilidades = []
        proponentes = []
        valores = []
        hyperlinks = []
        for n,linha in enumerate(df_content['conteudo']):
            # secao_elegibilidade = texto_exigibilidade(linha)
            # if secao_elegibilidade:
            #   elegibilidades.append(secao_elegibilidade)
            # else:
            #   elegibilidades.append(f"Padrão 'Critérios de Elegibilidade' não detectado no edital")
            secao_proponente = self.texto_proponente(linha)
            if secao_proponente:
                proponentes.append(secao_proponente)
            else:
                proponentes.append(f"Seção específica 'Quanto ao Proponente' não detectada no edital")
            secao_valores = self.texto_valores(linha)
            if secao_valores:
                valores.append(secao_valores)
            else:
                valores.append(f"Seção específica 'Valor Global' não detectada no edital")
            url = df_content['link'][n]
            ## Criar link só no template
            hyperlinks.append(url)
            ## Criar link já no dataframe
            # hyperlinks.append(f'<a href="{url}">Edital</a>')
        # df_content['elegibilidade'] = elegibilidades 
        # df_content['valor_global_edital'] = valores
        # df_content['hyperlink'] = hyperlinks
            
        df_content = df_content.assign(detalhes=pd.Series([{'titulo': t,
                                                            'eligibilidade_proponente': p,
                                                            'valor_global': v,
                                                            'ini_inscricao': i,
                                                            'fim_inscricao': f,
                                                            'objetivo': o,
                                                            'link_edital': h}
                                                        for t, p, v, i, f, o, h in zip(df_content['titulo'], proponentes, valores, pd.to_datetime(df_content['data_inicio_inscricao']).dt.strftime('%d/%m/%Y'), 
                                                                        pd.to_datetime(df_content['data_termino_inscricao']).dt.strftime('%d/%m/%Y'), df_content['descricao'], hyperlinks)]))
        df_content['palavras-chave'] = palavrachave
        # print(df_content.keys()) # Debug
        # pprint([x for x in df_content['detalhes']], width=110) # Debug
        df_new = df_content.drop(columns=['link',
                                          'descricao',
                                          'conteudo'], axis=1)
        self._logout()
        return df_new

    def mount_foment_report(self, df, filename):
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
        # Salvar o relatório gerado em um arquivo HTML
        filepath = os.path.join(base_repo_dir,'source','visualizations',filename)
        with open(filepath, 'w') as f:
            f.write(html_output)
        print(f"Relatório montado disponível em {filepath}")

    ## Extração que demandam paginação e tratamento de erro aprimorado
    def ler_pdf_retry(self, url_pdf):
        """
        Função para baixar e ler o conteúdo de um PDF a partir de um URL.
        Args:
            url_pdf (str): URL do PDF a ser baixado e lido.
        Returns:
            str: Conteúdo textual completo do PDF.
        Raises:
            RuntimeError: Se o número máximo de tentativas for excedido.
        """
        substituicoes = {
            "\n": " ",
            "  ": " ",
            "Ɵ": "ti",
            "ơ": "ti",
            "‒": "-",
            "hƩp": "http",
            "LaƩes": "Lattes",
        }
        max_retries = 5
        initial_wait = 1
        backoff_factor = 2
        for attempt in range(1, max_retries + 1):
            try:
                texto_completo = self.ler_pdf_link_temp(url_pdf)
                if texto_completo:
                    return texto_completo
                else:
                    continue
            except RequestException as error:
                print("Erro ao tentar ler PDF com arquivo temporário")
                logging.error(f"Erro no download do PDF (tentativa {attempt}/{max_retries}): {error}")
            except FileDataError as error:
                try:
                    with requests.get(url_pdf, stream=True) as response:
                        response.raise_for_status()
                        pdf_bytes = response.content
                    doc = fitz.open(pdf_bytes)
                    texto_completo = ""
                    for page in doc.pages:
                        for caracter, substituto in substituicoes.items():
                            texto_pagina = page.get_text().replace(caracter, substituto)
                        texto_completo += texto_pagina
                    return texto_completo
                except:
                    logging.error(f"Erro leitura/conteúdo no PDF (tentativa {attempt}/{max_retries}): {error}")
            except TypeError as error:
                logging.error(f"Erro no nome do arquivo (TypeError): {error} - URL utilizada: {url_pdf}")
            except RuntimeError as error:
                logging.error(f"Erro geral ao ler o PDF (RuntimeError): {error} - URL utilizada: {url_pdf}")
            finally:
                if attempt < max_retries:
                    time.sleep(initial_wait * backoff_factor ** attempt)
        # Se todas as tentativas falharem, lança uma exceção
        raise RuntimeError("Excedeu o número máximo de tentativas ao ler o PDF")

    ## Funcionamento básico sem exception e retry
    def mount_dfchamadas_encerradas(self,url):
        # Listas para links das chamadas e datas de inscrição
        chamada_urls = []
        chamada_titulo = []
        chamada_descricao = []
        chamada_inscricao = []
        # chamada_inicial = []
        # chamada_termino = []
        chamada_conteudo = []
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        numero_total_resultados = int(soup.find("li", class_="resultSearch").text.strip().split("de ")[1].split(" ")[0])
        res_por_pagina = int(soup.find("li", class_="resultSearch").text.strip().split("de ")[0].split(" ")[1].split("-")[1])
        numero_total_paginas = int(np.round(numero_total_resultados/res_por_pagina))
        # numero_total_paginas = 1
        print(f"Total de páginas com as chamadas do CNPq a extrair: {numero_total_paginas}")
        for pagina in tqdm(range(1, numero_total_paginas + 1), desc="Paginando"):
            # URL da página de resultados com paginação
            if numero_total_paginas > 1:
                url_resultados_pagina = f"{url}&startPage={pagina}&buscaChamada=&ano="
            else:
                url_resultados_pagina = f"{url}"
            # print(url_resultados_pagina)
            response = requests.get(url_resultados_pagina)
            soup = BeautifulSoup(response.content, "html.parser")
            for div in soup.find_all("div", class_="content"):
                titulo = div.find("h4").text
                chamada_titulo.append(titulo)
                descricao = div.find("p").text.replace('\n', ' ').replace('\r','').strip()
                chamada_descricao.append(descricao)        
            for div in soup.find_all("div", class_="inscricao"):
                datas = self.extract_inscricao_data(div)
                inicio = datas.get('data_inicio_inscricao')
                termino = datas.get('data_termino_inscricao')
                chamada_inscricao.append(f"{inicio} a {termino}")
                # chamada_inicial.append(datas.get('data_inicio_inscricao'))
                # chamada_termino.append(datas.get('data_termino_inscricao'))
            for div in soup.find_all("div", class_="links-normas pull-left"):
                link = div.find("a").get("href")
                chamada_urls.append(link)
                conteudo_completo = self.ler_pdf_retry(link)
                chamada_conteudo.append(conteudo_completo)
        # Criar dataframe inicial
        df = pd.DataFrame({
            "link": chamada_urls,
            "titulo": chamada_titulo,
            "descricao": chamada_descricao,
            "datas_inscricoes": chamada_inscricao,
            # "inscricao_inicio": chamada_inicial,
            # "inscricao_termino": chamada_termino,
            "conteudo": chamada_conteudo,
            })
        return df

    ## Versão melhorada mas ainda com bugs de geração de quantidades diferentes nas listas da extração
    # def mount_dfchamadas(self, url, max_retries=3):
    #     """
    #     Extrai dados das páginas de resultados de chamadas do CNPq e monta um DataFrame.
    #     Args:
    #         url (str): URL da página de resultados de chamadas do CNPq.
    #         max_retries (int): Número máximo de tentativas para recuperar o conteúdo de uma página.
    #     Returns:
    #         DataFrame: DataFrame contendo os dados das chamadas.
    #     """
    #     erro = False
    #     # Listas para links das chamadas e datas de inscrição
    #     chamada_urls = []
    #     chamada_titulo = []
    #     chamada_descricao = []
    #     chamada_inscricao = []
    #     chamada_conteudo = []
    #     # Extrair número total de resultados e páginas
    #     try:
    #         response = requests.get(url)
    #         soup = BeautifulSoup(response.content, "html.parser")
    #         numero_total_resultados = int(
    #             soup.find("li", class_="resultSearch").text.strip().split("de ")[1].split(" ")[0]
    #         )
    #         res_por_pagina = int(
    #             soup.find("li", class_="resultSearch").text.strip().split("de ")[0].split(" ")[1].split("-")[1]
    #         )
    #         numero_total_paginas = int(np.round(numero_total_resultados / res_por_pagina))
    #     except requests.exceptions.HTTPError as e:
    #         logging.error("Erro ao obter número total de resultados: %s", e)
    #         return None
    #     # Extrair dados de cada página
    #     for pagina in tqdm(range(1, numero_total_paginas + 1), desc="Paginando"):
    #         # URL da página de resultados com complemento para paginação se maior que 1 página
    #         if numero_total_paginas > 1:
    #             url_resultados_pagina = f"{url}&startPage={pagina}&buscaChamada=&ano="
    #         else:
    #             url_resultados_pagina = f"{url}"
    #         for tentativa in range(max_retries + 1):
    #             try:
    #                 response = requests.get(url_resultados_pagina)
    #                 if response.status_code == 200:
    #                     soup = BeautifulSoup(response.content, "html.parser")
    #                     # Extrair conteúdo de cada edital na página
    #                     for div in soup.find_all("div", class_="content"):
    #                         # Extrair Título
    #                         titulo = div.find("h4").text  
    #                         chamada_titulo.append(titulo)
    #                         # Extrair  Descrições e outras informações (Mesma Lógica)
    #                         descricao = div.find("p").text.replace('\n', ' ').replace('\r', '').strip()
    #                         chamada_descricao.append(descricao)
    #                     for div in soup.find_all("div", class_="inscricao"):
    #                         datas = self.extract_inscricao_data(div)
    #                         inicio = datas.get('data_inicio_inscricao')
    #                         termino = datas.get('data_termino_inscricao')
    #                         chamada_inscricao.append(f"{inicio} a {termino}")
    #                     for div in soup.find_all("div", class_="links-normas pull-left"):
    #                         link = div.find("a", class_="btn").get("href")
    #                         chamada_urls.append(link)
    #                         conteudo_completo = self.ler_pdf_link_temp(link)
    #                         chamada_conteudo.append(conteudo_completo)
    #                 else:
    #                     erro = True
    #                     logging.error("Erro ao ler página %d: %s", pagina, response.status_code)
    #             except requests.exceptions.HTTPError as e:
    #                 erro = True
    #                 logging.error("Erro ao ler página %d: %s", pagina, e)
    #             except Exception as e:
    #                 erro = True
    #                 logging.error("Erro inesperado na página %d: %s", pagina, e)
    #             # Interromper o loop se a página for lida com sucesso
    #             if not erro:
    #                 break
    #         if erro:
    #             logging.error("Falha ao ler página %d após %d tentativas", pagina, max_retries + 1)
    #     try:
    #         # Criar dataframe inicial
    #         df = pd.DataFrame({
    #             "link": chamada_urls,
    #             "titulo": chamada_titulo,
    #             "descricao": chamada_descricao,
    #             "datas_inscricoes": chamada_inscricao,
    #             "conteudo": chamada_conteudo,
    #         })
    #         return df
    #     except:
    #         ## DEBUG montagem das listas
    #         print('Erro na montagem do dataframe, quantide diferentes entre listas:')
    #         print(f'Qte Hyperlinks: {len(chamada_urls)}')
    #         print(f'Qte de Titulos: {len(chamada_titulo)}')
    #         print(f'Qte Descrições: {len(chamada_descricao)}')
    #         print(f'Qte Inscrições: {len(chamada_inscricao)}')
    #         print(f'Qte  Conteúdos: {len(chamada_conteudo)}')

    # def ler_pdf_link_temp(self, link, max_retries=0):
    #     """
    #     Lê o conteúdo de um PDF a partir de um link.
    #     Args:
    #         link: URL do PDF.
    #         max_retries: Número máximo de tentativas de download e leitura do PDF.
    #     Returns:
    #         Texto completo do PDF.
    #     """
    #     substituicoes = {
    #         "\n": " ",
    #         "  ": " ",
    #         "Ɵ": "ti",
    #         "ơ": "ti",
    #         "‒": "-",
    #         "hƩp": "http",
    #         "LaƩes": "Lattes",
    #     }
    #     for tentativa in range(max_retries + 1):
    #         try:
    #             response = requests.get(link, stream=True)
    #             with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_pdf:
    #                 temp_pdf.write(response.content)  # Salva conteúdo do PDF 
    #                 temp_pdf.seek(0)  # Volta o ponteiro para o início do arquivo
    #                 texto_completo = ""
    #                 with fitz.open(temp_pdf.name) as pdf_documento:
    #                     for pagina in pdf_documento.pages():
    #                         texto_pagina = pagina.get_text("text")
    #                         for caracter, substituto in substituicoes.items():
    #                             texto_pagina = texto_pagina.replace(caracter, substituto)
    #                         texto_completo += texto_pagina
    #             return texto_completo
    #         except:
    #             logging.error("Erro ao ler PDF (tentativa %d/%d): %s", tentativa + 1, max_retries, link)
    #     # Atingiu o limite máximo de tentativas
    #     logging.error("Erro ao ler PDF: %s", link)
    #     return None