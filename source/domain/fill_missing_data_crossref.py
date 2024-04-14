import requests
from bs4 import BeautifulSoup
import json
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_article_info_from_crossref(doi):
    """
    Busca informações do artigo usando CrossRef API.
    """
    url = f"https://api.crossref.org/works/{doi}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            article_info = data.get('message', {})

            title = article_info.get('title', ["Title not found"])[0]
            abstract_html = article_info.get('abstract', None)
            abstract_text = BeautifulSoup(abstract_html, "html.parser").get_text(strip=True) if abstract_html else "Abstract not found"

            return title, abstract_text
        else:
            return "Title not found", "Abstract not found"
    except Exception as e:
        logger.error(f"Error fetching data from CrossRef for DOI {doi}: {e}")
        return "Error", "Error"

def scrape_article_info(doi):
    """
    Busca informações do artigo por scraping.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"https://doi.org/{doi}"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('title').get_text().strip() if soup.find('title') else "Title not found"
            abstract = soup.find_all(text=lambda text: "abstract" in text.lower())
            abstract_text = abstract[0].strip() if abstract else "Abstract not found"

            return title, abstract_text
        else:
            return "Title not found", "Abstract not found"
    except Exception as e:
        logger.error(f"Error scraping DOI {doi}: {e}")
        return "Error", "Error"

def fill_missing_data(article_data):
    """
    Preenche dados faltantes para um conjunto de artigos.
    """
    for article in article_data:
        if 'doi' in article and not article.get('title'):
            title, abstract = fetch_article_info_from_crossref(article['doi'])
            if title != "Title not found" and title != "Error":
                article['title'] = title
                article['abstract'] = abstract
        elif not article.get('doi') and article.get('title'):
            title, abstract = scrape_article_info(article['title'])
            if abstract != "Abstract not found" and abstract != "Error":
                article['abstract'] = abstract

    return article_data

if __name__ == "__main__":
    # Carregar dados de exemplo
    with open('articles_data.json', 'r') as file:
        articles = json.load(file)

    # Preencher dados faltantes
    filled_articles = fill_missing_data(articles)

    # Salvar os dados atualizados
    with open('filled_articles_data.json', 'w') as file:
        json.dump(filled_articles, file)

    logger.info("Completed filling missing data.")
