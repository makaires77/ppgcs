from transformers import BertModel, BertTokenizer
import torch

class FundingEmbeddings:
    def __init__(self, model_name='bert-base-multilingual-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def preprocess(self, text):
        # Tokenizar o texto, adicionar tokens especiais e converter para tensor do PyTorch
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return inputs

    def generate_embedding(self, text):
        # Processamento do texto
        inputs = self.preprocess(text)
        # Passar o texto tokenizado pelo modelo
        with torch.no_grad():  # Não é necessário calcular gradientes neste contexto
            outputs = self.model(**inputs)
        # Pegar os embeddings do último estado oculto
        embeddings = outputs.last_hidden_state
        # Aqui, usamos a média dos embeddings de token como representação do documento
        embeddings = embeddings.mean(dim=1)
        return embeddings
