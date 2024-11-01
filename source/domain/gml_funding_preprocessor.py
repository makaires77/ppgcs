import torch
import nltk
import spacy
import string
import logging
import contextualSpellCheck
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.corpus import stopwords
from langdetect import detect

# Configurar o logging (opcional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Preprocessor:
    def __init__(self):
        # Inicializar variáveis comuns aqui
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect_language(self, text):
        try:
            return detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            logging.error(f"Erro ao identificar linguagem")
            return 'unknown'

    def translate_to_en(self, texts):
        # Método a ser implementado nas subclasses
        raise NotImplementedError

    def preprocess_text(self, text):
        # Método a ser implementado nas subclasses
        raise NotImplementedError


class ENPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()
        # Carregar o modelo de tradução e o tokenizador do Hugging Face
        self.model_name = "unicamp-dl/translation-pt-en-t5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_tr = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
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


class BRPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()
        # Carregar o modelo de tradução e o tokenizador do Hugging Face
        self.model_name = "unicamp-dl/translation-en-pt-t5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_tr = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
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