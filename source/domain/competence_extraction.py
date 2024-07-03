
import json
import spacy
import torch
import xformers
import numpy as np
import plotly.graph_objects as go
from tqdm.autonotebook import tqdm, trange
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util


class GPUMemoryManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def clear_gpu_cache(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def move_to_cpu(self, tensors):
        for tensor in tensors:
            if tensor is not None and tensor.device.type == "cuda":
                tensor.cpu()

class CompetenceExtraction:
    def __init__(self, curricula_file, model_name="distiluse-base-multilingual-cased-v2"):
        self.curricula_file = curricula_file
        self.nlp_pt = spacy.load("pt_core_news_lg")  # Modelo SpaCy para português
        self.nlp_en = spacy.load("en_core_web_sm")  # Modelo SpaCy para inglês
        self.model = SentenceTransformer(model_name)

    def load_curricula(self):
        with open(self.curricula_file, "r") as f:
            return json.load(f)

    def extract_competences(self, researcher_data):
        competences = []

        # Extrair de áreas de atuação
        for area in researcher_data.get("Áreas", {}).values():
            competences.append(area)

        # Extrair de formações acadêmicas
        for formacao in researcher_data.get("Formação", {}).get("Acadêmica", []):
            competences.append(formacao["Descrição"])

        # Extrair de atuação profissional
        for atuacao in researcher_data.get("Atuação Profissional", []):
            competences.append(atuacao["Descrição"])
            competences.append(atuacao["Outras informações"])

        # Extrair de projetos de pesquisa, extensão e desenvolvimento
        for tipo_projeto in ["ProjetosPesquisa", "ProjetosExtensão", "ProjetosDesenvolvimento"]:
            for projeto in researcher_data.get(tipo_projeto, []):
                competences.append(projeto["titulo_projeto"])
                competences.append(projeto["descricao"])

        # Extrair de produções bibliográficas
        for publicacao in researcher_data.get("Produções", {}).get("Artigos completos publicados em periódicos", []):
            competences.append(publicacao["titulo"])

        # Extrair de resumos publicados em anais de congressos
        for resumo in researcher_data.get("Produções", {}).get("Resumos publicados em anais de congressos", {}).values():
            competences.append(resumo)

        # Extrair de apresentações de trabalho
        for apresentacao in researcher_data.get("Produções", {}).get("Apresentações de Trabalho", {}).values():
            competences.append(apresentacao)

        # Extrair de outras produções bibliográficas
        for producao in researcher_data.get("Produções", {}).get("Outras produções bibliográficas", {}).values():
            competences.append(producao)

        # Extrair de entrevistas e comentários na mídia
        for entrevista in researcher_data.get("Produções", {}).get("Entrevistas, mesas redondas, programas e comentários na mídia", {}).values():
            competences.append(entrevista)

        # Extrair de bancas de concurso público e outras participações
        for tipo_banca, bancas in researcher_data.get("Bancas", {}).items():
            for banca in bancas.values():
                competences.append(banca)

        # Extrair de orientações (se houver)
        orientacoes = researcher_data.get("Orientações", {})
        if isinstance(orientacoes, dict):
            # Se "Orientações" for um dicionário (com tipos de orientação)
            for tipo_orientacao, detalhes in orientacoes.items():
                for detalhe in detalhes:
                    competences.append(detalhe.get("titulo", ""))
                    competences.append(detalhe.get("descricao", ""))
        elif isinstance(orientacoes, list):
            # Se "Orientações" for uma lista (de dicionários de orientações)
            for orientacao in orientacoes:
                competences.append(orientacao.get("titulo", ""))
                competences.append(orientacao.get("descricao", ""))

        return competences

    def preprocess_competences(self, competences):
        processed_competences = []
        for competence in competences:
            if competence:  # Ignorar competências vazias
                doc = self.nlp_en(competence) if competence.isascii() else self.nlp_pt(competence)
                processed_competences.append(" ".join([token.lemma_ for token in doc if not token.is_stop]))
        return processed_competences
       
    def vectorize_competences(self, competences):
        model = SentenceTransformer(self.model_name)  # Carregar o modelo aqui
        model.enable_xformers_memory_efficient_attention()  # Habilitar o xFormers
        competence_vectors = model.encode(competences)
        return competence_vectors

class EmbeddingModelEvaluator:
    def __init__(self, curricula_file, model_names):
        self.curricula_file = curricula_file
        self.model_names = model_names
        self.competence_extractor = CompetenceExtraction(curricula_file)
        self.curricula_data = self.competence_extractor.load_curricula()
        self.gpu_manager = GPUMemoryManager()  # Instanciar o gerenciador de memória da GPU

    def evaluate_models(self, validation_data):
        """
        Avalia os modelos de embedding usando métricas intrínsecas e extrínsecas.

        Args:
            validation_data: Dicionário com pares de competências rotulados como 'similar' ou 'dissimilar'.
        """
        results = {}
        for model_name in self.model_names:
            print(f"\nAvaliando modelo: {model_name}")
            model = SentenceTransformer(modules=[AutoModel.from_pretrained(model_name)])  
            model.enable_xformers_memory_efficient_attention()  # Habilita o xFormers

            # Avaliação intrínseca
            similar_scores, dissimilar_scores = self.evaluate_intrinsic(model, validation_data)
            mean_similar_score = np.mean(similar_scores)
            mean_dissimilar_score = np.mean(dissimilar_scores)

            # Avaliação extrínseca (exemplo: classificação de áreas de pesquisa)
            classifier = LogisticRegression()  # Exemplo de classificador
            X_train, y_train = self.prepare_data_for_classification(model, self.curricula_data)
            classifier.fit(X_train, y_train)
            X_test, y_test = self.prepare_data_for_classification(model, validation_data['similar'] + validation_data['dissimilar'])
            accuracy = classifier.score(X_test, y_test)

            results[model_name] = {
                'similar_score': mean_similar_score,
                'dissimilar_score': mean_dissimilar_score,
                'accuracy': accuracy
            }

            print(f"\nSimilaridade média (competências semelhantes): {mean_similar_score:.4f}")
            print(f"Similaridade média (competências distintas): {mean_dissimilar_score:.4f}")
            print(f"Acurácia na classificação de áreas de pesquisa: {accuracy:.4f}")
            print('-'*125)
            print()

        return results

    def evaluate_intrinsic(self, model, validation_data):
        similar_scores = []
        dissimilar_scores = []

        for label, pairs in validation_data.items():
            for pair in pairs:
                embeddings = model.encode(pair).cpu()
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                if label == 'similar':
                    similar_scores.append(similarity)
                else:
                    dissimilar_scores.append(similarity)

        return similar_scores, dissimilar_scores

    def prepare_data_for_classification(self, model, data):
        X = []
        y = []
        for researcher_data in data:
            competences = self.competence_extractor.extract_competences(researcher_data)
            processed_competences = self.competence_extractor.preprocess_competences(competences)
            embeddings = model.encode(processed_competences)
            mean_embedding = np.mean(embeddings, axis=0)  # Média dos embeddings das competências
            X.append(mean_embedding)
            y.append(researcher_data.get('area_de_pesquisa', 'desconhecido'))  # Substitua 'area_de_pesquisa' pela chave correta
        return X, y

class ModelComparator:
    def __init__(self, evaluation_results):
        self.evaluation_results = evaluation_results

    def get_best_model(self):
        best_model = max(self.evaluation_results, key=self.evaluation_results.get)
        best_score = self.evaluation_results[best_model]
        return best_model, best_score


class PlotlyResultVisualizer:
    def __init__(self, results):
        self.results = results

    def plot_similarity_distributions(self):
        fig = go.Figure()
        for model_name, scores in self.results.items():
            fig.add_trace(go.Histogram(
                x=scores['similar_score'],
                name=f'{model_name} (similar)',
                opacity=0.75,
                histnorm='probability density'
            ))
            fig.add_trace(go.Histogram(
                x=scores['dissimilar_score'],
                name=f'{model_name} (dissimilar)',
                opacity=0.75,
                histnorm='probability density'
            ))

        fig.update_layout(
            barmode='overlay',
            title='Distribuição de Similaridade (Densidade de Probabilidade)',
            xaxis_title='Similaridade',
            yaxis_title='Densidade',
        )
        fig.show()

    def plot_accuracy_comparison(self):
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]

        fig = go.Figure(data=[go.Bar(x=models, y=accuracies)])
        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        fig.update_layout(
            title='Comparação de Acurácia na Classificação',
            xaxis_title='Modelo',
            yaxis_title='Acurácia',
        )
        fig.show()