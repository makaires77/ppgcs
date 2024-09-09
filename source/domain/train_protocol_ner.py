import os
import re
import spacy
import random
from spacy.training.io import docs_to_json
from spacy.training import Example
from spacy.tokens import DocBin
from tqdm.notebook import tqdm
from doccano_client import DoccanoClient

# Configurações do Doccano (substitua pelos seus dados)
DOCCANO_URL = 'http://localhost:8000'  # URL do seu servidor Doccano
DOCCANO_USERNAME = 'seu_usuario'
DOCCANO_PASSWORD = 'sua_senha'
PROJECT_ID = 1  # ID do projeto no Doccano

def annotate_data_doccano(corpus):
    """
    Realiza anotações usando o Doccano e retorna o corpus anotado.

    Args:
        corpus: A lista de tuplas (texto, anotações) gerada por build_training_corpus.

    Returns:
        O corpus anotado, com as entidades marcadas nas anotações.
    """

    client = DoccanoClient(DOCCANO_URL, DOCCANO_USERNAME, DOCCANO_PASSWORD)

    # Upload dos dados para o Doccano
    for text, _ in tqdm(corpus, desc="Enviando dados para o Doccano"):
        client.post_doc(PROJECT_ID, text)

    # Criar tarefa de anotação no Doccano (ajuste o tipo de tarefa conforme necessário)
    # ... (código para criar a tarefa de anotação no Doccano)

    # Aguardar a conclusão da anotação (você pode implementar um loop com verificação periódica ou usar webhooks)
    # ... 

    # Obter as anotações do Doccano
    annotations = [] 
    for doc in tqdm(client.get_docs(PROJECT_ID), desc="Obtendo anotações do Doccano"):
        annotations.append({"entities": [(span['start_offset'], span['end_offset'], span['label']) for span in doc['annotations']]})

    # Combinar o corpus original com as anotações
    annotated_corpus = [(text, annotations[i]) for i, (text, _) in enumerate(corpus)]

    return annotated_corpus

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


def build_training_corpus(folder_path):
    """
    Monta o corpus de treinamento a partir dos documentos na pasta especificada.

    Args:
        folder_path: O caminho da pasta contendo os documentos.

    Returns:
        Uma lista de tuplas (texto, anotações), onde 'anotações' é um dicionário com a chave 'entities' contendo uma lista de tuplas (start, end, label).
    """
    corpus = []

    for filename in tqdm(os.listdir(folder_path), desc="Montando corpus"):
        if filename.endswith(".pdf") or filename.endswith(".html"):
            file_path = os.path.join(folder_path, filename)
            raw_text = extract_raw_text(file_path)
            corpus.append((raw_text, {"entities": []}))  # Inicialmente sem anotações

    return corpus

def annotate_data(corpus):
    """
    Realiza anotações manuais ou semi-automáticas no corpus de treinamento.

    Args:
        corpus: A lista de tuplas (texto, anotações) gerada por build_training_corpus.

    Returns:
        O corpus anotado, com as entidades marcadas nas anotações.
    """
    # Implemente aqui a lógica para realizar as anotações
    # Você pode usar ferramentas como Prodigy, Doccano ou outras bibliotecas para auxiliar na anotação
    # Ou implementar sua própria interface de anotação
    pass  # Substitua por sua implementação

def train_spacy_model(corpus, model_output_dir, iterations=30):
    """
    Treina um modelo spaCy para reconhecimento de entidades nomeadas em documentos de protocolos.

    Args:
        corpus: O corpus de treinamento anotado.
        model_output_dir: O diretório onde o modelo treinado será salvo.
        iterations: O número de iterações de treinamento.
    """

    # Criar um modelo em branco
    nlp = spacy.blank("pt") 
    ner = nlp.create_pipe("ner")
    nlp.add_pipe("ner")

    # Adicionar os labels das entidades ao modelo
    for _, annotations in corpus:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Desabilitar outros componentes do pipeline durante o treinamento
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):

        # Inicializar os parâmetros do modelo
        optimizer = nlp.begin_training()

        # Loop de treinamento
        for itn in range(iterations):
            random.shuffle(corpus)
            losses = {}
            for text, annotations in tqdm(corpus, desc=f"Iteração {itn+1}"):
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update(
                    [example],
                    drop=0.5,  # Dropout para evitar overfitting
                    sgd=optimizer,
                    losses=losses)
            print(losses)

    # Salvar o modelo treinado
    nlp.to_disk(model_output_dir)

def evaluate_spacy_model(model_path, test_data):
    """
    Avalia o modelo spaCy treinado em um conjunto de dados de teste.

    Args:
        model_path: O caminho para o modelo treinado.
        test_data: O conjunto de dados de teste, no mesmo formato do corpus de treinamento.
    """
    nlp = spacy.load(model_path)
    examples = []
    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    scores = nlp.evaluate(examples)
    print(scores)

if __name__ == "__main__":
    root_folder = os.getcwd()
    data_folder = os.path.join(root_folder, "_data", "in_pdf")  # Ou ajuste para a pasta desejada

    # Montar o corpus de treinamento
    corpus = build_training_corpus(data_folder)

    # Realizar anotações (implemente a função annotate_data)
    annotated_corpus = annotate_data(corpus)

    # Dividir o corpus em treinamento e teste (opcional)
    # ... 

    # Treinar o modelo
    model_output_dir = "modelo_protocolos_ner"
    train_spacy_model(annotated_corpus, model_output_dir)

    # Avaliar o modelo (opcional)
    # evaluate_spacy_model(model_output_dir, test_data)