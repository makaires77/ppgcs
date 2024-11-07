from gml_embeddings_analyser import EmbeddingsMulticriteriaAnalysis
from sentence_transformers import SentenceTransformer

# Definir os nomes de modelo do SentenceTransformer a serem comparados
model_names = [
    'paraphrase-multilingual-MiniLM-L12-v2',
    'all-MiniLM-L6-v2'
]

# Criar uma instância da classe EmbeddingsMulticriteriaAnalysis
analise = EmbeddingsMulticriteriaAnalysis(
    model_names=model_names,
    models= [SentenceTransformer(model_name) for model_name in model_names]
)

# Avaliar clustering (detecta idioma, pré-processa, gera embeddings e mede tempo)
resultados = analise.evaluate_clustering()

# Imprimir os resultados (opcional)
print(resultados)