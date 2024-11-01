import os
from git import Repo
from sentence_transformers import SentenceTransformer

from gml_unsupervised_learning_tools import EmbeedingsMulticriteriaAnalysis
from preprocessor import ENPreprocessor, BRPreprocessor  # Importe as classes de pré-processamento

# Define the model names and the models you want to compare
model_names = [
    'paraphrase-multilingual-MiniLM-L12-v2',
    'all-MiniLM-L6-v2'
    # Add more model names here if needed
]

models = [
    SentenceTransformer(model_name)
    for model_name in model_names
]

# Informar caminho para arquivo CSV usando raiz do repositório Git como referência
repo = Repo(search_parent_directories=True)
root_folder = repo.working_tree_dir
folder_data_output = os.path.join(root_folder, '_data', 'out_json')
filename = 'df_fomento_geral.csv'
pathfilename = os.path.join(folder_data_output, filename)

# Carregar o dataframe
try:
    import cudf
    df_fomento = cudf.read_csv(pathfilename, header=0)
except ImportError:
    print("cuDF não está disponível. Usando Pandas.")
    df_fomento = pd.read_csv(pathfilename, header=0)

# Criar uma instância do pré-processador (escolha ENPreprocessor ou BRPreprocessor)
preprocessor = ENPreprocessor()  # ou preprocessor = BRPreprocessor()

# Criar uma instância da classe EmbeedingsMulticriteriaAnalysis
analise = EmbeedingsMulticriteriaAnalysis(
    data=df_fomento,
    model_names=model_names,
    models=models,
)

# Executar a avaliação de clustering para cada modelo (gera embeddings e mede tempo)
analise.evaluate_clustering()

# Escolher o melhor modelo
melhor_modelo = analise.escolher_melhor_modelo()
print(f"O melhor modelo é: {melhor_modelo}")

# Gerar o relatório de benchmarking (opcional)
analise.generate_report()