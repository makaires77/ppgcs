import os
import re
import sys
import json
import time
import random
import locale
import logging
import requests
from pathlib import Path
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class MedicamentoAltoCustoScraper:
    def __init__(self, url_base):
        self.url_base = url_base
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.todos_precos={}
    
    def find_repo_root(self, path='.', depth=10):
        ''' 
        Busca o arquivo .git e retorna string com a pasta raiz do repositório
        '''
        # Prevent infinite recursion by limiting depth
        if depth < 0:
            return None
        path = Path(path).absolute()
        if (path / '.git').is_dir():
            return path
        return self.find_repo_root(str(path.parent), depth-1)

    def extrair_preco(self, texto_preco, verbose=False):
        # Remover quebras de linha e espaços extras
        texto_limpo = ' '.join(texto_preco.split())
        # Remover "A partir de" ou "À partir de" do início
        texto_limpo = re.sub(r'^[AÀ]\s+partir\s+de\s+', '', texto_limpo, flags=re.IGNORECASE)
        # Remover R$ e espaços
        texto_limpo = texto_limpo.replace('R$', '').strip()
        if verbose:
            print(f"Texto do preço extraído: {texto_limpo}")
        try:
            # Converte para float
            return float(texto_limpo.replace('.', '').replace(',', '.'))
        except ValueError:
            logging.warning(f"Não foi possível converter o preço: {texto_limpo}")
            return None

    def extrair_dados_pagina(self, url, verbose=False):
        if verbose:
            logging.debug(f"Iniciando requisição para: {url}")
        try:
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            if verbose:
                logging.debug(f"Requisição bem-sucedida. Status code: {response.status_code}")
        except requests.RequestException as e:
            logging.error(f"Erro na requisição: {e}")
            return [], None

        if verbose:
            logging.debug("Iniciando parsing do HTML")
        soup = BeautifulSoup(response.content, 'html.parser')
        
        medicamentos = []
        
        if verbose:
            logging.debug("Procurando elementos de produto")
        items = soup.find_all('li', class_='cr-border-divider')
        if verbose:
            logging.debug(f"Encontrados {len(items)} elementos de produto")
        
        for i, item in enumerate(items, 1):
            if verbose:
                logging.debug(f"Processando item {i}")
            
            # Extrai o nome do medicamento
            nome_element = item.find('h2', class_='col-md-12')
            nome = nome_element.text.strip() if nome_element else "Nome não disponível"
            if verbose:
                logging.debug(f"Nome extraído: {nome}")
            
            # Extrai o preço usando o seletor correto
            valor = None
            preco_element = item.find('p', class_='cr-typography-heading-h2')
            if preco_element:
                preco_texto = preco_element.text.strip()
                try:
                    valor = float(preco_texto.replace('.', '').replace(',', '.').strip())
                    if verbose:
                        logging.debug(f"Preço extraído e convertido: {valor}")
                except ValueError:
                    if verbose:
                        logging.warning(f"Não foi possível converter o preço: {preco_texto}")
            else:
                if verbose:
                    logging.warning("Elemento de preço não encontrado")
            
            # Extrai o fabricante
            fabricante_element = item.find('a', class_='d-flex cr-info-details')
            fabricante = fabricante_element.text.strip() if fabricante_element else "Fabricante não disponível"
            if verbose:
                logging.debug(f"Fabricante extraído: {fabricante}")
            
            medicamentos.append({
                "nome": nome,
                "valor": valor,
                "fabricante": fabricante
            })

        # Verifica se existe próxima página
        next_page = soup.find('a', class_='page-link next')
        next_url = None
        if next_page:
            next_url = next_page.get('href') # type: ignore
            if next_url and not str(next_url).startswith('http'):
                next_url = f"https://consultaremedios.com.br{next_url}"
            if verbose:
                logging.debug(f"Próxima página encontrada: {next_url}")
        
        return medicamentos, next_url
  

    def extrair_todos_dados(self):
        todos_medicamentos = []
        url_atual = self.url_base
        pagina = 1
        total_registros = 0
        
        while url_atual:            
            # logging.debug(f"Processando página {pagina}")
            medicamentos, proxima_url = self.extrair_dados_pagina(url_atual)
            
            if medicamentos:
                todos_medicamentos.extend(medicamentos)
                total_registros += len(medicamentos)
                logging.info(f"Página {pagina} processada. Total de registros até agora: {total_registros}")
            
            if not proxima_url:
                logging.info("Chegou à última página")
                break
                
            url_atual = proxima_url if str(proxima_url).startswith('http') else f"https://consultaremedios.com.br{proxima_url}"
            pagina += 1
            
            # Adiciona um pequeno delay entre as requisições
            time.sleep(random.uniform(1, 3))
        
        logging.info(f"Extração concluída. Total de {total_registros} medicamentos extraídos")
        self.todos_precos = todos_medicamentos
        return todos_medicamentos

    def salvar_json(self, nome_arquivo):
        dados = self.todos_precos
        pathfilename = os.path.join(str(self.find_repo_root()),'_data','in_json', nome_arquivo)
        with open(pathfilename, 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        print(f"Dados salvos em {nome_arquivo}")
        logging.info(f"Dados salvos em {nome_arquivo}")

# Uso da classe
if __name__ == "__main__":
    scraper = MedicamentoAltoCustoScraper("https://consultaremedios.com.br/campanha/medicamentos-alto-custo/c")
    scraper.salvar_json("medicamentos_alto_custo.json")