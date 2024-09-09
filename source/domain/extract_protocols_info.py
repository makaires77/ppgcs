import os
import re
import spacy
import numpy as np
from git import Repo
from pprint import pprint
from tika import parser  # Para extrair texto de PDFs
from bs4 import BeautifulSoup  # Para analisar HTML
from tqdm.notebook import tqdm  # Importar tqdm para notebook
from spacy.matcher import PhraseMatcher

# Carregue o modelo de linguagem português do spaCy
nlp = spacy.load("pt_core_news_sm")

# Crie um PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

# Defina as frases que devem ser reconhecidas como entidades únicas
terms = [
    "Ministério da Saúde", 
    "Secretaria de Atenção à Saúde", 
    "SECRETÁRIO DE ATENÇÃO À SAÚDE",
    "SECRETARIA DE CIÊNCIA, TECNOLOGIA E INSUMOS ESTRATÉGICOS",
    "SECRETÁRIO DE CIÊNCIA, TECNOLOGIA E INSUMOS ESTRATÉGICOS"
]
patterns = [nlp.make_doc(text) for text in terms]
matcher.add("ORG_PHRASES", patterns)

# Função para processar o texto e aplicar o PhraseMatcher
def process_text_with_matcher(text):
    doc = nlp(text)
    matches = matcher(doc)
    
    # Adicionar as correspondências do PhraseMatcher como entidades
    with doc.retokenize() as retokenizer:
        for match_id, start, end in matches:
            retokenizer.merge(doc[start:end], attrs={"ENT_TYPE": "ORG"})

    return doc

def clean_raw(text):
  """
  Remove espaços em branco extras e quebras de linha do texto.

  Args:
      text: O texto a ser limpo.

  Returns:
      O texto limpo.
  """
  text = re.sub(r'\s+', ' ', text)  # Remove espaços em branco extras
  text = re.sub(r'\n+', '\n', text) # Remove múltiplas quebras de linha consecutivas 

  return text.strip()

def extract_raw_text(caminho_documento):
    """
    Extrai texto de documento pdf ou html
    """
    raw_text = None

    # Extrair o texto do documento (adapte para outros formatos se necessário)
    if caminho_documento.endswith(".pdf"):
        raw_text = parser.from_file(caminho_documento)["content"].replace('\n\n','\n').replace('\n','')
        raw_text = clean_raw(raw_text)
    elif caminho_documento.endswith(".html"):
        with open(caminho_documento, "r", encoding="latin-1") as f:  # ou 'iso-8859-1'
            html_content = f.read()
        soup = BeautifulSoup(html_content, "html.parser")
        raw_text = soup.get_text()
    else:
        raise ValueError("Formato de arquivo não suportado")
    
    return raw_text

def extract_raw_text_from_pdf(file_path):
    """
    Extrai texto de um documento PDF.

    Args:
        file_path: O caminho do arquivo PDF.

    Returns:
        O texto extraído do PDF.
    """
    raw_text = parser.from_file(file_path)["content"].replace('\n\n','\n').replace('\n','')
    raw_text = clean_raw(raw_text)
    return raw_text

def extract_raw_text_from_html(file_path):
    """
    Extrai texto de um documento HTML, considerando as marcações <h2> como delimitadoras de seções.

    Args:
        file_path: O caminho do arquivo HTML.

    Returns:
        O texto extraído do HTML, com as seções delimitadas por <h2>.
    """
    with open(file_path, "r", encoding="latin-1") as f:  # ou 'iso-8859-1'
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")

    # Encontrar todas as tags <h2>
    h2_tags = soup.find_all('h2')

    # Construir o texto com as seções delimitadas por <h2>
    raw_text = ""
    for h2_tag in h2_tags:
        raw_text += "\n" + h2_tag.text + "\n"  # Adiciona quebra de linha antes e depois do título da seção
        for sibling in h2_tag.next_siblings:
            if sibling.name == 'h2':
                break  # Para quando encontrar a próxima seção
            if sibling.name is not None:  # Ignora elementos que não são tags (como texto solto)
                raw_text += sibling.get_text() + "\n"

    raw_text = clean_raw(raw_text)
    return raw_text

def extract_initial_info(text):
    """
    Extrai informações do início do documento (tipo, título, agravos) usando expressões regulares.

    Args:
        text: O texto do documento.

    Returns:
        Um dicionário contendo as informações extraídas: 'fonte', 'tipo_nome', 'tipo_sigla', 'agravos_nome'.
    """

    # Padrões de busca
    fonte_pattern = r"^(PORTARIA\s+CONJUNTA\s+N° \d+, de \d+ de [a-z]+ de \d+)\."
    tipo_agravos_pattern = r"Aprova (?:o|as) (.+) d[ao] (.+)\."

    # Busca no texto (com re.IGNORECASE para ignorar maiúsculas/minúsculas)
    fonte_match = re.search(fonte_pattern, text, re.MULTILINE | re.IGNORECASE)
    tipo_match = re.search(tipo_agravos_pattern, text, re.IGNORECASE)
    # tipo_match = re.search(tipo_agravos_pattern, text, re.MULTILINE | re.IGNORECASE)

    # Extração das informações
    info = {}
    if fonte_match:
        info['fonte'] = fonte_match.group(1)
    else:
        info['fonte'] = None

    if tipo_match:
        info['tipo_nome'] = tipo_match.group(1).strip()
        info['tipo_sigla'] = "".join([p[0] for p in info['tipo_nome'].split() if p[0].isupper()])

        # Extrair agravos, considerando separadores como "e", ",", "ou" e ";"
        agravos_raw = tipo_match.group(2).strip()
        separadores = r" e |, | ou |; "
        info['agravos_nome'] = [a.strip() for a in re.split(separadores, agravos_raw)]
    else:
        info['tipo_nome'] = None
        info['tipo_sigla'] = None
        info['agravos_nome'] = []

    return info

def extract_initial_info_spacy(text):
    """
    Extrai informações do início do documento (tipo, título, agravos) usando spaCy.

    Args:
        text: O texto do documento.

    Returns:
        Um dicionário contendo as informações extraídas: 'fonte', 'tipo_nome', 'tipo_sigla', 'agravos_nome'.
    """
    doc = nlp(text)
    print(f"Entidades identificadas: {doc.ents}")
    info = {}

    # Encontrar a entidade "PORTARIA"
    # for ent in doc.ents:
    #     if ent.label_ == "ORG" and "PORTARIA" in ent.text:
    #         info['fonte'] = ent.text

    # Encontrar a frase que começa com "Aprova o" ou "Aprova as"
    for sent in doc.sents:
        if sent.text.startswith("PORTARIA"):
            info['fonte'] = sent.text
        if sent.text.startswith("Aprova o") or sent.text.startswith("Aprova as"):
            # Extrair o tipo de documento e os agravos (expressão regular generalizada)
            tipo_match = re.search(r"Aprova (?:o|as) (.+) d[ao] (.+)\.", sent.text)
            if tipo_match:
                info['tipo_nome'] = tipo_match.group(1).strip()
                info['tipo_sigla'] = "".join([p[0] for p in info['tipo_nome'].split() if p[0].isupper()])

                # Extrair agravos, considerando separadores como "e", ",", "ou" e ";"
                agravos_raw = tipo_match.group(2).strip()
                separadores = r" e | ou |; "
                info['agravos_nome'] = [a.strip() for a in re.split(separadores, agravos_raw)]

            break  # Interrompe o loop após encontrar a primeira ocorrência

    return info


def process_documents_in_folder(folder_path):
    """
    Processa todos os arquivos PDF e HTML em uma pasta, extrai as informações iniciais e retorna um DataFrame.

    Args:
        folder_path: O caminho da pasta contendo os arquivos.

    Returns:
        Um DataFrame (cuDF ou pandas) contendo as informações extraídas de cada documento.
    """
    filenames = [f for f in os.listdir(folder_path) if f.endswith(".pdf") or f.endswith(".html")]
    qdoc = len(filenames)
    print(f"Processar {qdoc} documentos de protocolos...")

    results = []

    # Usar tqdm para criar a barra de progresso
    for filename in tqdm(filenames, desc="Processando documentos"):
        file_path = os.path.join(folder_path, filename)

        # Extrair o texto do documento de acordo com o tipo
        if filename.endswith(".pdf"):
            raw_text = extract_raw_text_from_pdf(file_path)
        elif filename.endswith(".html"):
            raw_text = extract_raw_text_from_html(file_path)

        # Extrair informações iniciais
        initial_info = extract_initial_info_spacy(raw_text)
        print(f"Info protocolo extraída: {initial_info.values()}")

        # Adicionar informações à lista de resultados
        results.append(initial_info)

    # Tentar importar cuDF, se não estiver disponível, usar pandas
    try:
        import cudf
        df_lib = cudf
        # Converter a lista de dicionários em um ndarray NumPy (necessário apenas para cuDF)
        data_array = np.array(results, dtype=object)
        df_results = df_lib.DataFrame.from_records(data_array) 
    except ImportError:
        import pandas as pd
        df_lib = pd
        # Criar DataFrame pandas diretamente da lista de dicionários
        df_results = df_lib.DataFrame(results)

    return df_results


# if __name__ == "__main__":

#     results = []  # Inicializar lista para armazenar resultados
#     repo = Repo(search_parent_directories=True)
#     root_folder = repo.working_tree_dir
#     folder_path = os.path.join(root_folder,"_data","in_pdf",filename1)  # ou .html

#     df_results = process_documents_in_folder(folder_path)
#     print(df_results)