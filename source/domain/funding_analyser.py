import os
import ast
import time
import cudf
import nltk
import torch
import spacy
import string
import logging
import warnings
import unicodedata
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import contextualSpellCheck

from transformers.tokenization_utils_base import TruncationStrategy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import pipeline, TranslationPipeline
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc, Token
from spacy.language import Language
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from langdetect import detect
from tqdm.notebook import tqdm
from git import Repo
import cudf 
tqdm.pandas()

# Configurar o logging (opcional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning or UserWarning)
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

class FundingEmbeddingGenerator:
    def __init__(self):
        # Carregar o modelo sentence transformer pré-treinado para gerar embeedings
        model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self.model_st = SentenceTransformer(model_name)
        
        # Carregar o modelo na GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"Carregando modelo {model_name} na GPU...")
        else:
            print(f"GPU indisponível, usando apenas CPU...")
        self.model_st.to(self.device)

    def create_embedding_column(self, use_cudf=True):
        """
        Creates the 'texto_para_embedding' column in the df_fomento dataframe by combining selected data and applying preprocessing.

        Args:
            use_cudf: Whether to use cuDF for DataFrame operations (default: True)

        Returns:
            The updated dataframe with the 'texto_para_embedding' column.
        """

        # Informar caminho para arquivo CSV usando raiz do repositório Git como referência
        repo = Repo(search_parent_directories=True)
        root_folder = repo.working_tree_dir
        folder_data_output = os.path.join(root_folder, '_data', 'out_json')
        filename = 'df_fomento_geral.csv'
        pathfilename = os.path.join(folder_data_output, filename)
        pdf = pd.read_csv(pathfilename, header=0)

        def convert_to_dict(text):
            try:
                return ast.literal_eval(text)
            except ValueError:
                return None

        def generate_embedding_text_helper(row, cols_geninfo, cols_details, cols_moreinf):
            # Extrair dados das colunas específicas de interesse
            gen_info_text = ' '.join([str(row[col]) for col in cols_geninfo])
            details_text = ' '.join([str(row['detalhes'][col]) for col in cols_details if col in row['detalhes']])
            more_info_text = ' '.join([str(row['detalhes'][col]) for col in cols_moreinf if col in row['detalhes']])

            # Combinar textos de interesse em string única
            combined_text = f"{gen_info_text} {details_text} {more_info_text}"
            return combined_text

        pdf['detalhes'] = pdf['detalhes'].apply(convert_to_dict)

        # Definir as colunas para a geração dos embeddings
        cols_geninfo = ['financiadora','titulo','palavras-chave']
        cols_details = ['elegibilidade','descricao','valorfinanciado','datalimite']
        cols_moreinf = ['formasolicitacao']

        # Aplciar função de suporte para cada linha do dataframe de editais
        pdf['texto_para_embedding'] = pdf.apply(
            lambda row: generate_embedding_text_helper(
                row,
                cols_geninfo,
                cols_details,
                cols_moreinf
            ),
            axis=1
        )

        # Converter o DataFrame do pandas para cuDF (se o parâmetro de entrada for True)
        df = cudf.from_pandas(pdf) if use_cudf else pdf

        return df

    def generate_embeddings(self, df):
        """
        Generates embeddings for the 'texto_para_embedding' column in the dataframe.

        Args:
            df: The dataframe containing the 'texto_para_embedding' column.

        Returns:
            The embeddings as a numpy array.
        """

        # Obter os textos da coluna 'texto_para_embedding' como uma lista Python
        sentences = df['texto_para_embedding'].to_arrow().to_pylist()
        print(f"\nTotal de sentenças: {len(sentences)}")

        # Gerar os embeddings usando o modelo de sentence transformer
        embeddings = self.model_st.encode(sentences, convert_to_tensor=True, device=self.device)

        return embeddings

    def generate_embeddings_bath(self, df, batch_size=32):  # Adicione o parâmetro batch_size
        """
        Generates embeddings for the 'texto_para_embedding' column in the dataframe, processing in batches.

        Args:
            df: The dataframe containing the 'texto_para_embedding' column.
            batch_size: The batch size for processing (default: 32).

        Returns:
            The embeddings as a numpy array.
        """

        # Obter os textos da coluna 'texto_para_embedding'
        sentences = df['texto_para_embedding'].tolist()

        # Gerar os embeddings em lotes
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            batch_embeddings = self.model_st.encode(batch_sentences, convert_to_tensor=True, device=self.device)
            all_embeddings.extend(batch_embeddings.cpu().numpy())  # Mova os embeddings de volta para a CPU antes de adicioná-los à lista

        return np.array(all_embeddings)

# Classe para pré-processamento com tradução para inglês e correção ortográfica
class ENPreprocessor:
    def __init__(self):
        # Carregar o modelo de tradução e o tokenizador do Hugging Face
        self.model_name = "unicamp-dl/translation-pt-en-t5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_tr = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_tr.to(self.device)

        # Carregar o modelo transformer para o Inglês
        self.nlp_en = spacy.load("en_core_web_trf")

        # Adicionar ao pipeline correção ortográfica
        contextualSpellCheck.add_to_pipe(self.nlp_en)

        # Carregar as stopwords em inglês
        self.stop_words_en = set(stopwords.words('english'))

        # Adicionar as stopwords personalizadas em inglês
        self.stop_words_en.update(["must", "due", "track", "may", "non", "year", "apply", "prepare", "era", "eligibility",
                              "funded value", "deadline", "application form", "description", "name", "address", "phone",
                              "Fax", "e-mail", "email", "contact", "homepage", "home page", "home", "page"])

    def translate_to_en(self, texts):
        try:
            # Traduzir usando o modelo Hugging Face pré-treinado em tradução pt/en
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model_tr.generate(**inputs, max_new_tokens=512)
            translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return translations
        except Exception as e:
            print(f"Erro na tradução: {e}")
            return texts

    def detect_language(self, text):
        try:
            return detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            logging.error(f"Erro ao identificar linguagem")
            return 'unknown'

    def preprocess_text(self, text):
        # Traduzir o texto para inglês (se necessário) em lote
        try:
            # logging.info("Traduzindo texto para o inglês (se necessário)...")
            text_translated = self.translate_to_en([text])[0] if self.detect_language(text) != 'en' else text
        except Exception as e:
            logging.error(f"Erro na tradução: {e}")
            return []

        # Converter para minúsculas e remover pontuação
        # logging.info("Limpando e normalizando o texto...")
        text_translated = text_translated.lower().translate(str.maketrans('', '', string.punctuation))

        # Truncar o texto traduzido se for muito longo
        max_length = 512  # Ajuste conforme necessário
        text_translated = text_translated[:max_length]

        # Aplicar o corretor ortográfico e lematizar em inglês em lote (usando pipe do spaCy)
        # logging.info("Processando o texto com spaCy...")
        with self.nlp_en.disable_pipes('ner'):  # Desabilitar NER para economizar memória da GPU
            # Definir o tamanho do lote para cada envio de entradas para a GPU
            docs = self.nlp_en.pipe([text_translated], batch_size=64)

        for doc in docs:
            words_en = [token.lemma_.lower() if token.text.lower() not in ["institute", "institution", "institutional"] else "institution"
                        for token in doc
                        if token.is_alpha and not token.is_stop and token.lemma_.lower() not in self.stop_words_en]

        return words_en


# Classe para pré-processamento com tradução para português
class BRPreprocessor:
    def __init__(self):
        # Carregar o modelo de tradução e o tokenizador do Hugging Face
        self.model_name = "unicamp-dl/translation-en-pt-t5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_tr = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_tr.to(self.device)

        # Carregar o modelo de linguagem em português do Spacy
        self.nlp_pt = spacy.load("pt_core_news_sm")

        # Adicionar o corretor ortográfico contextual ao pipeline do spaCy (se disponível para português)
        contextualSpellCheck.add_to_pipe(self.nlp_pt)

        # Carregar as stopwords em português
        self.stop_words_pt = set(stopwords.words('portuguese'))

        # Adicionar as stopwords personalizadas em português
        self.stop_words_pt.update(["deve", "devido", "acompanhar", "pode", "não", "ano", "aplicar", "preparar", "era", "elegibilidade",
                              "valorfinanciado", "datalimite", "formuláriodesolicitacao", "descrição", "homepage", "nome",
                              "endereço", "telefone", "fax", "e-mail", "contato", "home page", "casa", "página"])

    def translate_to_pt(self, texts):
        try:
            # Traduzir usando o modelo Hugging Face pré-treinado em tradução en/pt
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            outputs = self.model_tr.generate(**inputs)
            translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return translations
        except Exception as e:
            print(f"Erro na tradução: {e}")
            return texts

    def detect_language(self, text):
        try:
            return detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            logging.error(f"Erro ao identificar linguagem")
            return 'unknown'

    def preprocess_text(self, text):
        # Traduzir o texto para português (se necessário)
        try:
            # logging.info("Traduzindo texto para o português (se necessário)...")
            text_translated = self.translate_to_pt([text])[0] if self.detect_language(text) != 'pt' else text
        except Exception as e:
            logging.error(f"Erro na tradução: {e}")
            return []

        # Converter para minúsculas e remover pontuação
        # logging.info("Limpando e normalizando o texto...")
        text_translated = text_translated.lower().translate(str.maketrans('', '', string.punctuation))

        # Truncar o texto traduzido se for muito longo
        max_length = 512
        text_translated = text_translated[:max_length]

        # Lematizar em português
        # logging.info("Lematizando o texto...")
        doc_pt = self.nlp_pt(text_translated)
        words_pt = [token.lemma_.lower()
                    for token in doc_pt
                    if token.is_alpha and not token.is_stop and token.lemma_.lower() not in self.stop_words_pt
                    and not (token.pos_ == "PROPN" and token.text.lower() not in self.stop_words_pt)]

        return words_pt


# Classe para gerar os gráficos da análise exploratória de dados de editais de fomento
class ExploratoryDataAnalyzer:
    def __init__(self):
        pass

    def analyze_and_visualize(self, all_words, embeddings):
        """
        Performs exploratory data analysis and visualization on the preprocessed text and embeddings.

        Args:
            all_words: A list of lists containing the preprocessed words from the text data.
            embeddings: The embeddings generated from the preprocessed text.
        """

        # 1. Word Frequency Analysis
        word_counts = Counter(word for words in all_words for word in words)
        top_words = word_counts.most_common(20)

        # Plot bar chart of top words
        plt.figure(figsize=(12, 6))
        plt.bar(*zip(*top_words))
        plt.title('Palavras Mais Frequentes (sem Stopwords e com Lematização)')
        plt.xlabel('Palavra')
        plt.ylabel('Frequência')
        plt.xticks(rotation=45)
        plt.show()

        # Create and display word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

        # 2. Visualization of embeddings in 2D using PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=(10, 6))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        plt.title('Visualização dos Embeddings (PCA)')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.show()

        # 3. Visualization of embeddings in 2D using t-SNE
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 6))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        plt.title('Visualização dos Embeddings (t-SNE)')
        plt.xlabel('Dimensão 1')
        plt.ylabel('Dimensão 2')
        plt.show()