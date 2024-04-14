
import re
import os
import json
import nltk
import gensim
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from time import sleep
from PIL import ImageFont
from langdetect import detect
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from sklearn.decomposition import PCA
from transformers import BertModel, BertTokenizer

from translate_en_pt import TranslatorEnPt
from tqdm import tqdm, trange

# Configurar logging para trabalhar com tqdm
tqdm_logger = logging.getLogger('tqdm')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
tqdm_logger.addHandler(handler)
tqdm_logger.setLevel(logging.INFO)

# Baixar os pacotes necessários do NLTK
nltk.download('stopwords')
nltk.download('wordnet')

class LDAExtractor:
    def __init__(self, base_repo_dir, num_topics=5, passes=15, random_state=100, model_name='bert-base-multilingual-cased', lda_model=None, dictionary=None):
        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.folder_utils = os.path.join(base_repo_dir, 'utils')
        self.folder_domain = os.path.join(base_repo_dir, 'source', 'domain')
        self.folder_data_input = os.path.join(base_repo_dir, 'data', 'input')
        self.folder_data_output = os.path.join(base_repo_dir, 'data', 'output')
        self.lda_model = lda_model
        self.dictionary = dictionary

    @staticmethod
    def load_json(file):
        with open(file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_existing_model(self, lda_model_path, dictionary_path):
        """
        Carrega um modelo LDA e um dicionário existentes.
        :param lda_model_path: Caminho para o arquivo do modelo LDA.
        :param dictionary_path: Caminho para o arquivo do dicionário.
        """
        self.lda_model = LdaModel.load(lda_model_path)
        self.dictionary = corpora.Dictionary.load(dictionary_path)

    def preprocess_data(self, documents, language='portuguese'):
        """
        Função para pré-processar os documentos.
        :param documents: Lista de documentos (strings).
        :param language: Idioma para as stopwords (padrão é 'portuguese').
        :return: Lista de documentos pré-processados.
        """
        # Definindo o lematizador e as stopwords
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words(language))

        preprocessed_documents = []
        for i in tqdm(range(len(documents)), desc="Pré-processando"):   
            doc = documents[i]
            
            # Juntar as palavras em uma string se for uma lista
            if isinstance(doc, list):
                doc = ' '.join(doc)

            try:
                detected_language = detect(doc)
                if detected_language == 'en':
                    # tqdm.write("Traduzindo inglês/português")
                    doc = self.translator.translate(doc)
            except:
                continue

            # Remover caracteres especiais
            doc = re.sub(r'\W', ' ', doc)

            # Remover todas as palavras de até dois caracteres
            doc = re.sub(r'\b[a-zA-Z]{1,2}\b', ' ', doc)

            # Remover números
            doc = re.sub(r'\b\d+(\.\d+)?(st|nd|rd|th|º)?\b', ' ', doc)

            # Remover espaços extras
            doc = re.sub(r'\s+', ' ', doc, flags=re.I)

            # Transformando em minúsculas
            doc = doc.lower()

            # Tokenizar e lematizar
            doc = doc.split()
            doc = [lemmatizer.lemmatize(word) for word in doc if word not in stop_words]
            preprocessed_documents.append(doc)

        return preprocessed_documents
   
    def extract_text_from_json(self, filename):
        if filename == None:
            filename = 'output_py_gpu_multithreads.json'
        filepath = os.path.join(self.folder_data_output, filename)
        print(filepath)
        json_data = LDAExtractor.load_json(filepath)
        # print(json_data[0].keys())
        count_empty_title = 0
        count_empty_abstract = 0
        texts = []
        for item in json_data:
            articles = item.get('processed_data').get('articles', [])
            # print(f"Tipo lst_art: {type(articles)}")
            for dic_article in articles:
                # print(f"Tipo dic_art: {type(dic_article)}")
                # print(f"Keys dic_art: {dic_article.keys()}")
                lst_title = dic_article.get('subdict_titulos')
                # print(f"Tipo lst_tit: {type(lst_title)}")
                col_tit = list(lst_title.values())
                # print(f"Tipo col_tit: {type(col_tit)}")
                # print(f"Dado col_tit: {col_tit}")
                title = col_tit[-1]
                if title == '':
                    try:
                        # print(len(lst_title), lst_title)
                        title = lst_title.values()[0]
                        # print(title)
                        texts.append(title)
                    except:
                        try:
                            title=list(dic_article.get('subdict_titulos').values())[0]
                            # print(f"DICT_VALUE tipo: {type(title)} {title}")
                            if title:
                                texts.append(title)
                            else:
                                count_empty_title+=1    
                        except Exception as e:
                            print('Erro:',e)
                            count_empty_title+=1
                else:
                    texts.append(title)    
                abstract = dic_article.get('abstract')
                if abstract:
                    # print(f"Resumo: {abstract}")
                    texts.append(abstract)
                else:
                    count_empty_abstract+=1
        print(f"{len(texts)} textos de títulos e resumos extraídos")
        print(f"{count_empty_title:>0000} títulos vazios | {count_empty_abstract} resumos vazios")
        return texts

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
        outputs = self.model(**inputs)
        # return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
        # Calcula a média das saídas dos tokens para obter um único vetor por documento
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    ## Caso seja necessário reduzir dimensionalidade para aliviar custo computacional
    # def fit_transform(self, json_data_filename):
    #     documents = self.extract_text_from_json(json_data_filename)

    #     # Gerando embeddings para cada documento
    #     embeddings = np.array([self.get_embedding(doc) for doc in tqdm(documents, 
    #                                                                    desc="Gerando embeddings", 
    #                                                                    unit="doc")])

    #     # Verificando a forma dos embeddings
    #     print("Forma dos embeddings:", embeddings.shape)

    #     # Aplicando PCA para reduzir a dimensionalidade
    #     pca = PCA(n_components=50)
    #     reduced_embeddings = pca.fit_transform(embeddings)

    #     print("Criando e treinando o modelo LDA...")
    #     dictionary = corpora.Dictionary([list(map(str, range(50)))])
    #     corpus = [dictionary.doc2bow(vec) for vec in reduced_embeddings]

    #     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=self.num_topics, random_state=self.random_state, passes=self.passes, per_word_topics=True)

    #     return lda_model, dictionary, corpus

    def fit_transform(self, json_data_filename):
        # 1. Extrair e pré-processar texto dos documentos
        documents = self.extract_text_from_json(json_data_filename)
        preprocessed_texts = self.preprocess_data(documents)

        # 2. Criar o dicionário e o corpus para o LDA
        dictionary = corpora.Dictionary(preprocessed_texts)
        corpus = [dictionary.doc2bow(doc) for doc in preprocessed_texts]

        print("Criando e treinando o modelo LDA...")
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=self.num_topics, random_state=self.random_state, passes=self.passes, per_word_topics=True)

        return lda_model, dictionary, corpus

    def train_lda_model(self, json_data_filename=None):
        # 1. Extrair texto dos documentos
        texts = self.extract_text_from_json(json_data_filename)

        # 2. Pré-processar os documentos
        preprocessed_texts = self.preprocess_data(texts)

        # 3. Criar o dicionário e o corpus para o LDA
        dictionary = corpora.Dictionary(preprocessed_texts)
        corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]

        # Salvar o dicionário
        dictionary_save_path = os.path.join(self.folder_data_output, 'dictionary.gensim')
        dictionary.save(dictionary_save_path)
        print(f"Dicionário salvo em: {dictionary_save_path}")

        # Assegure-se de que self.num_topics e self.passes sejam números
        num_topics = self.num_topics
        passes = int(self.passes)

        # Verificar e converter para inteiros se necessário
        if isinstance(num_topics, str):
            num_topics = int(num_topics)

        # Verificar se self.num_topics e self.passes são inteiros
        if not isinstance(self.num_topics, int):
            raise TypeError("num_topics deve ser um inteiro.")
        if not isinstance(passes, int):
            raise TypeError("passes deve ser um inteiro.")

        # Criação do modelo LDA
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, 
                             num_topics=num_topics, 
                             random_state=self.random_state, 
                             passes=passes, 
                             per_word_topics=True)

        print("Modelo LDA treinado com sucesso.")
        return lda_model, dictionary, corpus

    def preprocess_new_documents(self, new_docs):
        """
        Pré-processa novos documentos usando o método de pré-processamento da classe.
        :param new_docs: Lista de documentos (strings) a serem pré-processados.
        :return: Lista de documentos pré-processados no formato bag-of-words.
        """
        # Pré-processar cada documento
        preprocessed_docs = [self.preprocess_data(doc) for doc in new_docs]

        # Converter para o formato bag-of-words usando o dicionário existente
        bow_docs = [self.dictionary.doc2bow(doc) for doc in preprocessed_docs]

        return bow_docs

    def classify_title_to_topic(self, title):
        """
        Classifica um título de artigo no tópico do LDA ao qual ele mais provavelmente pertence.
        :param title: Título do artigo a ser classificado.
        :return: ID do tópico mais provável e sua probabilidade.
        """
        # Pré-processar o título
        preprocessed_title = self.preprocess_data([title])[0]  # Atenção: [title] está entre colchetes

        # Converter o título pré-processado para o formato bag-of-words
        bow_title = self.dictionary.doc2bow(preprocessed_title)

        # Obter a distribuição de tópicos para o título
        topic_distribution = self.lda_model.get_document_topics(bow_title)

        # Encontrar o tópico mais provável
        most_likely_topic, probability = max(topic_distribution, key=lambda x: x[1])

        return most_likely_topic, probability

    def plot_wordcloud(self, lda_model_path=None):
        if lda_model_path is not None:
            self.load_existing_model(lda_model_path)

        # Verificar se o modelo LDA foi carregado
        if self.lda_model is None:
            raise ValueError("O modelo LDA não foi carregado.")

        num_topics = self.num_topics  # Ou o número de tópicos que você deseja visualizar

        for i in range(num_topics):
            topic_words = dict(self.lda_model.show_topic(i, 10))  # 10 palavras mais representativas do tópico
            wordcloud = WordCloud(width=800, height=600, background_color='white').generate_from_frequencies(topic_words)  # Remover font_path

            plt.figure(figsize=(19, 9))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Topic {i+1}')
            plt.show()

    def prepare_wordcloud_data(self, topic_id, num_words=10):
        words, weights = zip(*self.lda_model.show_topic(topic_id, topn=num_words))
        return words, weights

    def plot_wordcloud_plotly(self, topic_id, num_words=10):
        words, weights = self.prepare_wordcloud_data(topic_id, num_words)
        max_weight = max(weights)
        word_sizes = [50 + 150 * (w / max_weight) for w in weights]  # Escalar os tamanhos das palavras

        # Criar um gráfico de dispersão
        trace = go.Scatter(
            x=np.random.rand(len(words)),
            y=np.random.rand(len(words)),
            mode='text',
            text=words,
            marker={'opacity': 0.3},
            textfont={'size': word_sizes, 'color': 'black'}
        )

        layout = go.Layout({
            'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
            'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}
        })

        fig = go.Figure(data=[trace], layout=layout)
        fig.update_layout(title_text=f'Nuvem de Palavras do Tópico {topic_id + 1}')
        fig.show(renderer='notebook')
