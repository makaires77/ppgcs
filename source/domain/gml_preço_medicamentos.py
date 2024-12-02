import os
import re
import json
import requests
from bs4 import BeautifulSoup


class MedicamentoAltoCustomScraper:
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def extrair_dados(self):
        response = self.session.get(self.url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        medicamentos = []
        
        # Encontrar todos os elementos que contêm informações dos medicamentos
        items = soup.find_all('div', class_='product-item')
        
        for item in items:
            nome = item.find('h2', class_='product-item__name').text.strip()
            
            preco_element = item.find('span', class_='price-box__best-price')
            preco = preco_element.text.strip() if preco_element else "Preço não disponível"
            
            fabricante_element = item.find('span', class_='product-item__brand')
            fabricante = fabricante_element.text.strip() if fabricante_element else "Fabricante não disponível"
            
            # Extrair o valor numérico do preço
            valor = re.search(r'R\$\s*([\d.,]+)', preco)
            valor = float(valor.group(1).replace('.', '').replace(',', '.')) if valor else None
            
            medicamentos.append({
                "nome": nome,
                "valor": valor,
                "fabricante": fabricante
            })
        
        return json.dumps(medicamentos, ensure_ascii=False, indent=2)

    def salvar_json(self, pasta_json, nome_arquivo):
        dados = self.extrair_dados()
        pathfilename = os.path.join(pasta_json, nome_arquivo)
        with open(pathfilename, 'w', encoding='utf-8') as f:
            f.write(dados)
        print(f"Dados salvos em {nome_arquivo}")


# import requests
# from bs4 import BeautifulSoup
# import csv

# class MedicamentosAltoCustomScraper:
#     def __init__(self, url):
#         self.url = url
#         self.headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#         }

#     def scrape(self):
#         response = requests.get(self.url, headers=self.headers)
#         soup = BeautifulSoup(response.content, 'html.parser')
        
#         medicamentos = []
#         items = soup.find_all('div', class_='product-item')
        
#         for item in items:
#             titulo = item.find('h2', class_='product-item__name').text.strip()
#             valor = item.find('span', class_='price-box__best-price').text.strip()
#             fabricante = item.find('span', class_='product-item__brand').text.strip()
#             nome_medicamento = item.find('span', class_='product-item__ean').text.strip()
            
#             medicamentos.append({
#                 'titulo': titulo,
#                 'valor': valor,
#                 'fabricante': fabricante,
#                 'nome_medicamento': nome_medicamento
#             })
        
#         return medicamentos

#     def save_to_csv(self, medicamentos, filename='medicamentos_alto_custo.csv'):
#         with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
#             fieldnames = ['titulo', 'valor', 'fabricante', 'nome_medicamento']
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
#             writer.writeheader()
#             for medicamento in medicamentos:
#                 writer.writerow(medicamento)

#     def run(self):
#         medicamentos = self.scrape()
#         self.save_to_csv(medicamentos)
#         print(f"Dados extraídos e salvos em medicamentos_alto_custo.csv")

# # Uso da classe
# if __name__ == "__main__":
#     url = "https://consultaremedios.com.br/campanha/medicamentos-alto-custo/c"
#     scraper = MedicamentosAltoCustomScraper(url)
#     scraper.run()


