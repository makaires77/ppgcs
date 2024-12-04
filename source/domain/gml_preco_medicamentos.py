import os
import re
import sys
import json
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

    def extrair_preco(self, texto_preco):
        # Remover quebras de linha e espaços extras
        texto_limpo = ' '.join(texto_preco.split())
        # Remover "A partir de" ou "À partir de" do início
        texto_limpo = re.sub(r'^[AÀ]\s+partir\s+de\s+', '', texto_limpo, flags=re.IGNORECASE)
        # Remover R$ e espaços
        texto_limpo = texto_limpo.replace('R$', '').strip()
        try:
            # Converte para float
            return float(texto_limpo.replace('.', '').replace(',', '.'))
        except ValueError:
            logging.warning(f"Não foi possível converter o preço: {texto_limpo}")
            return None
    
    def extrair_dados_pagina(self, url):
        logging.debug(f"Iniciando requisição para: {url}")
        try:
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            logging.debug(f"Requisição bem-sucedida. Status code: {response.status_code}")
        except requests.RequestException as e:
            logging.error(f"Erro na requisição: {e}")
            return []

        logging.debug("Iniciando parsing do HTML")
        soup = BeautifulSoup(response.content, 'html.parser')
        
        medicamentos = []
        
        logging.debug("Procurando elementos de produto")
        items = soup.find_all('li', class_='cr-border-divider')
        logging.debug(f"Encontrados {len(items)} elementos de produto")
        
        for i, item in enumerate(items, 1):
            logging.debug(f"Processando item {i}")
            
            # Extrai o nome do medicamento
            nome_element = item.find('h2', class_='col-md-12')
            nome = nome_element.text.strip() if nome_element else "Nome não disponível"
            logging.debug(f"Nome extraído: {nome}")
            
            # Extrai o preço
            preco_element = item.find('div', class_='d-flex flex-column')
            preco_texto = preco_element.text.strip() if preco_element else "Preço não disponível"
            logging.debug(f"Preço extraído: {preco_texto}")
            
            # Extrai o fabricante
            fabricante_element = item.find('a', class_='d-flex cr-info-details')
            fabricante = fabricante_element.text.strip() if fabricante_element else "Fabricante não disponível"
            logging.debug(f"Fabricante extraído: {fabricante}")
            
            # Trata o preço
            valor = self.extrair_preco(preco_texto)
            if valor:
                logging.debug(f"Valor convertido: {valor}")
            else:
                logging.warning(f"Não foi possível converter o preço: {preco_texto}")
            
            medicamentos.append({
                "nome": nome,
                "valor": valor,
                "fabricante": fabricante
            })

        # Verifica se existe próxima página
        next_page = soup.find('a', class_='page-link', rel='next')
        next_url = next_page.get('href') if next_page else None
        
        return medicamentos, next_url
    
    def extrair_todos_dados(self):
        todos_medicamentos = []
        url_atual = self.url_base
        pagina = 1
        
        while url_atual:
            logging.debug(f"Processando página {pagina}")
            medicamentos, proxima_url = self.extrair_dados_pagina(url_atual)
            todos_medicamentos.extend(medicamentos)
            
            if not proxima_url:
                break
                
            url_atual = f"https://consultaremedios.com.br{proxima_url}"
            pagina += 1
            
        return todos_medicamentos

    def salvar_json(self, nome_arquivo):
        dados = self.extrair_todos_dados()
        pathfilename = os.path.join(str(self.find_repo_root()),'_data','in_json', nome_arquivo)
        with open(pathfilename, 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        print(f"Dados salvos em {nome_arquivo}")
        logging.info(f"Dados salvos em {nome_arquivo}")

# Uso da classe
if __name__ == "__main__":
    scraper = MedicamentoAltoCustoScraper("https://consultaremedios.com.br/campanha/medicamentos-alto-custo/c")
    scraper.salvar_json("medicamentos_alto_custo.json")