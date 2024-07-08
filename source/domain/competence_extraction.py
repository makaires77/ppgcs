import re
import json
import torch
import spacy
import xformers
import numpy as np
import plotly.graph_objects as go
from unidecode import unidecode
from transformers import AutoModel
from tqdm.autonotebook import tqdm, trange
from tqdm import TqdmExperimentalWarning
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sentence_transformers import SentenceTransformer, util

import warnings
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
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
        self.nlp_en = spacy.load("en_core_web_sm")   # Modelo SpaCy para inglês
        self.model = SentenceTransformer(model_name)

    def load_curricula(self):
        with open(self.curricula_file, "r") as f:
            return json.load(f)

    def extrair_info_trabalho(self, texto):
        """
        Extrai título, ano de obtenção e palavras-chave de um texto de trabalho acadêmico.

        Args:
            texto (str): O texto do trabalho acadêmico.

        Returns:
            dict: Um dicionário contendo o título, ano de obtenção e palavras-chave, ou None se não encontrar as informações.
        """
        padrao_titulo = r"[Tt]ítulo\s*:\s*(.*?)\.\s*"
        padrao_ano = r"[Aa]no\s*(?:de\s+)?[Oo]btenção\s*:\s*(\d+)\s*."
        padrao_ano2 = r"[Aa]no\s*(?:de\s+)?[Ff]inalização\s*:\s*(\d+)\s*."
        padrao_ano3 = r"\b(\d{4})\b"
        padrao_palavras_chave_area = r"[Pp]alavras-chave\s*:\s*(.*?)\s*(?::|\.)\s*(.*)"

        titulo = re.search(padrao_titulo, texto)
        try:
            titulo.group(1).strip().title()
            titulo_trabalho = titulo.group(1).strip().title()
        except: 
            titulo_trabalho = texto.split('. ')[0].title()
        ano = re.search(padrao_ano, texto)
        ano2 = re.search(padrao_ano2, texto)
        ano3 = re.search(padrao_ano3, texto)
        try:
            ano_trabalho = int(ano.group(1))
        except:
            try:
                ano_trabalho = int(ano2.group(1))
            except:
                try:
                    ano_trabalho = int(ano3.group(1))
                except:
                    ano_trabalho = '0000'
        palavras_chave_area = re.search(padrao_palavras_chave_area, texto)
        try:
            palavras_trabalho = palavras_chave_area.group(1).strip()
        except:
            palavras_trabalho = ''
        try:
            area_trabalho = palavras_chave_area.group(2).replace(":","").replace('/ ','|').rstrip(' .').strip()
        except:
            area_trabalho = ''
        try:
            tipo_trabalho = texto.split('. ')[0]
        except:
            print(f'Tipo do trabalho não encontrado em: {texto}')
            tipo_trabalho = ''
        try:
            instituicao = texto.split('. ')[1].strip().title()
            # print(f"Restante de dados: {texto.split('. ')[0:]}")
        except:
            print(f'Instituicao do trabalho não encontrada em: {texto}')
            instituicao = ''
        try:
            dic_trabalho = {
                "ano_obtencao": ano_trabalho,
                "titulo": titulo_trabalho,
                "palavras_chave": palavras_trabalho,
                "tipo_trabalho": tipo_trabalho,
                "instituição": instituicao,
                "area_trabalho": area_trabalho,
            }
            string_trabalho=''
            for x in dic_trabalho.values():
                string_trabalho = string_trabalho+' '+str(x)+' |'
            string_trabalho = string_trabalho.rstrip('|').rstrip(' .').strip()

            # if dic_trabalho:
            #     print("Ano de Obtenção:", dic_trabalho["ano_obtencao"])
            #     print("Título trabalho:", dic_trabalho["titulo"])
            #     print(" Palavras-chave:", dic_trabalho["palavras_chave"])
            #     print("  Tipo trabalho:", dic_trabalho["tipo_trabalho"])
            #     print("    Instituição:", dic_trabalho["instituição"])
            #     print("  Área trabalho:", dic_trabalho["area_trabalho"])
            # else:
            #     print("Não foi possível extrair todas as informações do trabalho.")

            return string_trabalho
        except Exception as e:
            print(f'Erro {e}')
            return texto 

    def extract_competences(self, researcher_data):
        competences = []
        
        padrao_titulo = r"[Tt]ítulo\s*:\s*(.*?)\.\s*"
        padrao_ano = r"[Aa]no\s*(?:de\s+)?[Oo]btenção\s*:\s*(\d+)\s*."
        padrao_ano2 = r"[Aa]no\s*(?:de\s+)?[Ff]inalização\s*:\s*(\d+)\s*."
        padrao_ano3 = r"\b(\d{4})\b"
        padrao_palavras_chave_area = r"[Pp]alavras-chave\s*:\s*(.*?)\s*(?::|\.)\s*(.*)"

        def extract(texto):
            titulo = re.search(padrao_titulo, texto)
            
            try:
                info1 = titulo.group(1).strip().title()
                try:
                    info2 = titulo.group(2).strip().title()
                except:
                    info2 = ''
            except: 
                info1 = texto.split('. ')[0].strip().title()
                try:
                    info2 = texto.split('. ')[1].strip().title()
                except:
                    info2 = ''
            ano = re.search(padrao_ano, texto)
            ano2 = re.search(padrao_ano2, texto)
            ano3 = re.search(padrao_ano3, texto)
            # print(ano)
            # print(ano2)
            # print(ano3)
            try:
                ano_trabalho = int(ano.group(1))
            except:
                try:
                    ano_trabalho = int(ano2.group(1))
                except:
                    try:
                        ano_trabalho = int(ano3.group(1))
                    except:
                        ano_trabalho = '----'
            return ano_trabalho, info1, info2

        # Extrair de áreas de atuação
        for area in researcher_data.get("Áreas", {}).values():
            area = area.replace(":","").replace("Subárea ","").replace(".","").replace("/","|").strip()
            competences.append('AtuaçãoPrf: '+area.title())

        # Extrair de formações acadêmicas
        verbose=False
        if verbose:
            print(f"\n{'-'*125}")
        for formacao in researcher_data.get("Formação", {}).get("Acadêmica", []):
            instituicao_formacao = formacao['Descrição'].split('.')[1].strip().title()
            if '(' in instituicao_formacao:
                instituicao_formacao = formacao['Descrição'].split('.')[2].strip().title()
            # print(f"     Instituição: {instituicao_formacao}")
            if verbose:
                print(f" Chaves Formação: {formacao.keys()}")
                print(f"Valores Formação: {formacao.values()}")                
                print(f"Dict   Formações: {formacao}")
            ano_formacao = formacao["Ano"]
            if '-' not in ano_formacao:
                ano_formacao = str(ano_formacao)+' - hoje'
            if 'interr' in ano_formacao:
                ano_interrupcao = formacao["Descrição"].split(':')[-1].strip()
                ano_formacao = f"{str(ano_formacao.split(' ')[0])} - {ano_interrupcao}"
            descr_formacao = formacao["Descrição"].strip().title()
            competences.append(f"FormaçãoAc: {ano_formacao} | {instituicao_formacao} | {descr_formacao}")

        # Extrair de projetos
        for tipo_projeto in ["ProjetosPesquisa", "ProjetosExtensão", "ProjetosDesenvolvimento"]:
            for projeto in researcher_data.get(tipo_projeto, []):
                # print(f' Chaves: {projeto.keys()}')
                # print(f'Valores: {projeto.values()}')
                tipo=None
                if 'Pesquisa' in tipo_projeto:
                    tipo = 'Psq'
                elif 'Extensão' in tipo_projeto:
                    tipo = 'Ext'
                elif 'Desenvolvimento' in tipo_projeto:
                    tipo = 'Dsv'
                descricao_projeto = projeto["descricao"]
                periodo_projeto = projeto["chave"].replace("Atual","hoje")
                titulo_projeto = projeto["titulo_projeto"]
                competences.append(f'Projeto{tipo}: {periodo_projeto} | {titulo_projeto} | {descricao_projeto.title()}')

        # Extrair de produções bibliográficas (artigos, resumos, etc.)
        for tipo_producao, producoes in researcher_data.get("Produções", {}).items():
            if isinstance(producoes, list):  # Artigos completos
                for publicacao in producoes:
                    # print(f'Dados publicação: {publicacao}')
                    if publicacao['fator_impacto_jcr']:
                        competences.append(f"Publicação: {publicacao['ano']} | {float(publicacao['fator_impacto_jcr']):06.2f} | {publicacao['titulo'].title()}")
                    else:
                        competences.append(f"Publicação: {publicacao['ano']} | {'-':5} | {publicacao['titulo'].title()}")
            # elif isinstance(producoes, dict):  # palestra e apresentações em eventos
            #     for item in producoes.values():                  
            #         competences.append(item)

        # Extrair de orientações (se houver)
        orientacoes = researcher_data.get("Orientações", {})
        # print(f'Dicionário orientações: {orientacoes}')
        if isinstance(orientacoes, dict):
            for tipo_orientacao, detalhes in orientacoes.items():
                if verbose:
                    print(tipo_orientacao)
                    if isinstance(detalhes, dict):
                        print([x.detalhes.keys() for x in orientacoes.values()])
                    else:
                        print(f"List  Orientação: {detalhes}")
                if 'conclu' in tipo_orientacao:
                    tipo = 'Con'
                else:
                    tipo = 'And'
                for detalhe in detalhes:
                    doutorados = detalhe.get('Tese de doutorado')
                    if doutorados:
                        for doc in doutorados.values():
                            ano_fim, nome_aluno, titulo_orientacao = extract(doc)
                            competences.append(f'OriDout{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')
                    
                    mestrados = detalhe.get('Dissertação de mestrado')
                    if mestrados:
                        for mes in mestrados.values():
                            ano_fim, nome_aluno, titulo_orientacao = extract(mes)
                            competences.append(f'OriMest{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')
                    
                    especializacoes = detalhe.get('Monografia de conclusão de curso de aperfeiçoamento/especialização')
                    if especializacoes:
                        for esp in especializacoes.values():
                            ano_fim, nome_aluno, titulo_orientacao = extract(esp)
                            competences.append(f'OriEspe{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')
                    
                    graduacoes = detalhe.get('Trabalho de conclusão de curso de graduação')
                    if graduacoes:
                        for grd in graduacoes.values():
                            ano_fim, nome_aluno, titulo_orientacao = extract(grd)
                            competences.append(f'OriGrad{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')
                    
                    iniciacoes = detalhe.get('Iniciação científica')
                    if iniciacoes:
                        for ini in iniciacoes.values():
                            ano_fim, nome_aluno, titulo_orientacao = extract(ini)
                            competences.append(f'OriInic{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')

                    postdocs = detalhe.get('Supervisão de pós-doutorado')
                    if postdocs:
                        for pos in postdocs.values():
                            ano_fim, nome_aluno, titulo_orientacao = extract(pos)
                            competences.append(f'SupPosD{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')

                    postdocs = detalhe.get('Orientações de outra natureza')
                    if postdocs:
                        for pos in postdocs.values():
                            ano_fim, nome_aluno, titulo_orientacao = extract(pos)
                            competences.append(f'OutNatu{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')

                            
        # elif isinstance(orientacoes, list):
        #     print('Lista de orientações')
        #     for orientacao in orientacoes:
        #         print(f'Dados da Orientação: {orientacao}')
        #         titulo_orientacao = orientacao.get("titulo", "")
        #         descricao_orientacao = orientacao.get("descricao", "")
        #         competences.append('Orientação: '+titulo_orientacao.title()+' '+descricao_orientacao.title())

        # Extrair de atuação profissional
        # for atuacao in researcher_data.get("Atuação Profissional", []):
        #     competences.append(atuacao.get("Instituição", ""))  # Adicionando a instituição
        #     competences.append(atuacao.get("Descrição", ""))
        #     competences.append(atuacao.get("Outras informações", ""))
        
        # Extrair de bancas
        # for tipo_banca, bancas in researcher_data.get("Bancas", {}).items():
        #     for banca in bancas.values():
        #         competences.append(banca)

        return competences

    def preprocess_competences(self, competences):
        """
        Pré-processa uma lista de competências, removendo stop words, lematizando e eliminando termos duplicados consecutivos (ignorando maiúsculas e minúsculas).

        Args:
            competences (list): Uma lista de strings representando as competências.

        Returns:
            list: Uma lista de strings contendo as competências pré-processadas.
        """

        processed_competences = []
        for competence in competences:
            if competence:
                doc = self.nlp_en(competence) if competence.isascii() else self.nlp_pt(competence)

                palavras_processadas = []
                eliminar = ['descrição','situação',':']
                ultima_palavra = None
                for token in doc:
                    if not token.is_stop:
                        palavra_atual = token.lemma_.lower().strip()  # Converte para minúsculas
                        if palavra_atual != ultima_palavra  and palavra_atual not in eliminar:
                            palavras_processadas.append(palavra_atual.strip())
                        ultima_palavra = palavra_atual.strip()

                processed_competences.append(" ".join(palavras_processadas))
        return processed_competences
       
    def vectorize_competences(self, competences):
        model = self.model  # Carregar o modelo aqui
        try:
            model.enable_xformers_memory_efficient_attention()  # Habilitar o xFormers
        except:
            pass
        competence_vectors = model.encode(competences)
        return competence_vectors

class EmbeddingModelEvaluator:
    def __init__(self, curricula_file, model_names):
        self.curricula_file = curricula_file
        self.model_names = model_names
        self.competence_extractor = CompetenceExtraction(curricula_file)
        self.curricula_data = self.competence_extractor.load_curricula() # carregar lista de dicionários
        self.gpu_manager = GPUMemoryManager()  # Instanciar o gerenciador de memória da GPU

    def evaluate_intrinsic(self, model, validation_data):  # Remove o parâmetro device
        similar_scores = []
        dissimilar_scores = []

        for label, pairs in validation_data.items():
            for pair in pairs:
                embeddings = model.encode(pair)
                # embeddings = torch.from_numpy(embeddings).cpu()  # Converte para tensor e move para a CPU
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                if label == 'similar':
                    similar_scores.append(similarity)
                else:
                    dissimilar_scores.append(similarity)

        return {
            'mean_similar_score': np.mean(similar_scores),
            'mean_dissimilar_score': np.mean(dissimilar_scores),
            'similar_scores': similar_scores,
            'dissimilar_scores': dissimilar_scores
        }

    def extrair_areas(self, areas_dict):
        lista_grdareas = []
        lista_areas = []
        lista_subareas = []
        # Expressão regular corrigida para extrair as áreas
        pattern = r'Grande área:\s*(.*?)\s*/\s*Área:\s*(.*?)\s*(?:/ Subárea:\s*(.*?)\s*)?\.'

        for _, valor in areas_dict.items():
            match = re.search(pattern, valor)
            if match:
                areas = {
                    'Grande Área': match.group(1).strip() if match.group(1) else None , 
                    'Área': match.group(2).strip() if match.group(2) else None ,
                    'Subárea': match.group(3).strip() if match.group(3) else None  
                }
                lista_grdareas.append(areas.get('Grande Área'))
                lista_areas.append(areas.get('Área'))
                lista_subareas.append(areas.get('Subárea'))

        return {'Grande Áreas': lista_grdareas, 'Áreas': lista_areas, 'Subáreas': lista_subareas}

    def prepare_data_for_classification(self, model, device="gpu"):
        X = []
        y = []
        valid_areas = set()
        all_embeddings = []  # Criando a lista para armazenar os embeddings

        # Primeira passagem para identificar áreas válidas
        for researcher_data in self.curricula_data:
            all_areas_list = self.extrair_areas(researcher_data.get('Áreas', {}))  # Obtém lista de áreas
            # print(f"Lista áreas: {all_areas_list}") # DEBUG
            for area in all_areas_list.get('Áreas'):
                if area and area != 'desconhecido':
                    valid_areas.add(area)

        # Segunda passagem para preparar os dados
        for researcher_data in self.curricula_data:
            competences = self.competence_extractor.extract_competences(researcher_data)
            processed_competences = self.competence_extractor.preprocess_competences(competences)
            areas_list = self.extrair_areas(researcher_data.get('Áreas', {}))  # Obtém lista de áreas

            for area in all_areas_list.get('Áreas'):
                print(f"Área de pesquisa: {area}")
                print(f"Competências extraídas: {competences}")
                print(f"Compet. pré-processadas: {processed_competences}")

                if area in valid_areas and processed_competences:
                    embeddings = model.encode(processed_competences, convert_to_tensor=True, device=device)
                    all_embeddings.extend(embeddings)  # Acumula os embeddings
                    mean_embedding = torch.mean(embeddings, dim=0)  # Calcula a média na GPU
                    X.append(mean_embedding.cpu().numpy())  # Move para CPU e converte para NumPy
                    y.append(area)

        return X, y

    def evaluate_models(self, validation_data, use_cross_validation=True, classifier_name="LogisticRegression"):
        """
        Avalia os modelos de embedding usando métricas intrínsecas e extrínsecas.

        Args:
            validation_data: Dicionário com pares de competências rotulados como 'similar' ou 'dissimilar'.
            use_cross_validation: Se True, usa validação cruzada para avaliação extrínseca.
                                Caso contrário, usa divisão em treinamento e teste.
            classifier_name: Nome do classificador a ser usado na avaliação extrínseca.
                                Opções: "LogisticRegression", "MultinomialNB", "SVC", "RandomForestClassifier".
        """
        # Defina device aqui
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            print("CUDA não disponível, usando CPU.")
        results = {}

        for model_name in self.model_names:
            print(f"\nAvaliando modelo: {model_name}")
            model = SentenceTransformer(model_name, device=device).half() # carrega o modelo ja no dispostivo

            # Avaliação intrínseca
            intrinsic_results = self.evaluate_intrinsic(model, validation_data)
            results[model_name] = intrinsic_results

            # Avaliação extrínseca
            X, y = self.prepare_data_for_classification(model, device)
            if use_cross_validation:
                if len(set(y)) < 2:
                    print(f"Não há classes suficientes para validação cruzada. Pulando modelo {model_name}.")
                    results[model_name].update({'accuracy': None, 'mean_accuracy': None, 'std_accuracy': None})
                else:
                    cross_val_results = self.evaluate_models_cross_validation(model, classifier_name)
                    results[model_name].update(cross_val_results)
                    print(f"Acurácia média (validação cruzada): {cross_val_results['mean_accuracy']:.4f} +/- {cross_val_results['std_accuracy']:.4f}")
            
            # Verifica se há exemplos suficientes para a divisão em treinamento e teste
            elif len(X) >= 2:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Escolha do classificador
                if classifier_name == "LogisticRegression":
                    classifier = LogisticRegression()
                elif classifier_name == "MultinomialNB":
                    classifier = MultinomialNB()
                elif classifier_name == "SVC":
                    classifier = SVC()
                elif classifier_name == "RandomForestClassifier":
                    classifier = RandomForestClassifier()
                else:
                    raise ValueError(f"Classificador inválido: {classifier_name}")

                # Converter os tensores para arrays NumPy e mover para CPU se ainda não estiverem nela
                # X_train = X_train.cpu().numpy()
                # X_test = X_test.cpu().numpy()

                # Treinamento e avaliação do classificador
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)

                # Cálculo das métricas
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                results[model_name].update({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

                print(f"Acurácia: {accuracy:.4f}")
                print(f"Precisão: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-Score: {f1:.4f}")

            else:
                print(f"Não há exemplos suficientes para divisão em treinamento e teste. Pulando modelo {model_name}.")
                results[model_name].update({'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None})

            print('-' * 125)
            print()
        return results


    def evaluate_models_cross_validation(self, model, classifier_name="LogisticRegression", num_folds=5):
        """Avalia os modelos de embedding usando validação cruzada com diferentes classificadores."""
        X, y = self.prepare_data_for_classification(model)

        # Verifica se há classes suficientes para a validação cruzada
        if len(set(y)) < 2:
            print(f"Não há classes suficientes para validação cruzada. Pulando modelo {model}.")
            return {'accuracy': None}  # Ou algum valor padrão para indicar erro

        # Escolha do classificador
        if classifier_name == "LogisticRegression":
            classifier = LogisticRegression()
        elif classifier_name == "MultinomialNB":
            classifier = MultinomialNB()
        elif classifier_name == "SVC":
            classifier = SVC()  # Você pode ajustar os parâmetros do SVM aqui
        elif classifier_name == "RandomForestClassifier":
            classifier = RandomForestClassifier()  # Você pode ajustar os parâmetros da Random Forest aqui
        else:
            raise ValueError(f"Classificador inválido: {classifier_name}")

        scores = cross_val_score(classifier, X, y, cv=num_folds, scoring='accuracy')
        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)

        return {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy
        }

class ModelComparator:
    def __init__(self, evaluation_results):
        self.evaluation_results = evaluation_results

    def get_best_model(self):
        # Filtra apenas modelos com resultados válidos (não None)
        valid_results = {
            model: scores 
            for model, scores in self.evaluation_results.items() 
            if scores.get('accuracy') is not None
        }

        if not valid_results:
            print("Nenhum modelo possui resultados válidos para comparação.")
            return None, None  # ou retorne valores padrão indicando que não há melhor modelo

        # Encontra o melhor modelo com base na acurácia
        best_model = max(valid_results, key=lambda model: valid_results[model]['accuracy'])
        best_score = valid_results[best_model]['accuracy']

        return best_model, best_score

class PlotlyResultVisualizer:
    def __init__(self, results):
        self.results = results

    def plot_similarity_distributions(self):
        fig = go.Figure()
        for model_name, scores in self.results.items():
            fig.add_trace(go.Histogram(
                x=scores['similar_scores'],
                name=f'{model_name} (similar)',
                opacity=0.75,
                histnorm='probability density',
                nbinsx=20 # Adicionado para melhor visualização
            ))
            fig.add_trace(go.Histogram(
                x=scores['dissimilar_scores'],
                name=f'{model_name} (dissimilar)',
                opacity=0.75,
                histnorm='probability density',
                nbinsx=20 # Adicionado para melhor visualização
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