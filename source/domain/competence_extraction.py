import re
import time
import json
import torch
import spacy
import numpy as np
import plotly.graph_objects as go
from unidecode import unidecode
from tqdm import TqdmExperimentalWarning
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
from sklearn.cluster import KMeans  # algoritmo de agrupamento

import warnings
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# from tqdm.autonotebook import tqdm, trange
# import xformers
# from transformers import AutoModel

from gml_memanager import GPUMemoryManager, HardwareEvaluator, ProcessingCapacityEstimator

class CompetenceExtractor:
    def __init__(self, curriculae_path, model_name="distiluse-base-multilingual-cased-v2"):
        """
        Autor: Marcos Aires (Nov 2024)
        Extrai competências de pesquisadores a partir de dados de currículos.

        Args:
            curricula_file (str): Caminho para o arquivo JSON contendo os dados dos currículos.
            model_name (str): Nome do modelo SentenceTransformer a ser usado.
        """
        self.model = SentenceTransformer(model_name)
        self.curriculae_path = curriculae_path
        self.nlp_pt = spacy.load("pt_core_news_lg")  # Modelo SpaCy para português
        self.nlp_en = spacy.load("en_core_web_trf")  # Modelo SpaCy para inglês
       

    def load_curricula(self):
        """
        Carrega os dados dos currículos do arquivo JSON.

        Returns:
            list: Lista de dicionários contendo os dados dos currículos.
        """
        with open(self.curriculae_path, "r") as f:
            return json.load(f)

    # Função para extrair pares de competências versão inicial O(n²)
    # A função extract_competence_pairs inicialmente possuia dois loops aninhados que iteravam sobre todos os pares de pesquisadores, abordagem com complexidade O(n^2) que pode ser ineficiente para um grande número de pesquisadores. Foi otimizada agrupando os pesquisadores por área de atuação e, em seguida, extraindo os pares de competências dentro de cada grupo. Isso reduziu a complexidade e tornou a extração mais rápida.
    # def extract_competence_pairs(self, curricula_data):
    #     similar_pairs = []
    #     dissimilar_pairs = []

    #     # Exemplo: extrair competências da mesma área (similar)
    #     for i in range(len(curricula_data)):
    #         for j in range(i + 1, len(curricula_data)):
    #             researcher1 = curricula_data[i]
    #             researcher2 = curricula_data[j]
    #             if researcher1.get('Áreas') == researcher2.get('Áreas'):  #Ver se área é a mesma
    #                 competences1 = self.extract_competences(researcher1)
    #                 competences2 = self.extract_competences(researcher2)
    #                 for comp1 in competences1:
    #                     for comp2 in competences2:
    #                         similar_pairs.append((comp1, comp2))

    #     # Exemplo: extrair competências de áreas diferentes (dissimilar)
    #     for i in range(len(curricula_data)):
    #         for j in range(i + 1, len(curricula_data)):
    #             researcher1 = curricula_data[i]
    #             researcher2 = curricula_data[j]
    #             if researcher1.get('Áreas') != researcher2.get('Áreas'):  #Ver se área é diferente
    #                 competences1 = self.extract_competences(researcher1)
    #                 competences2 = self.extract_competences(researcher2)
    #                 for comp1 in competences1:
    #                     for comp2 in competences2:
    #                         dissimilar_pairs.append((comp1, comp2))

    #     return {
    #         'similar': similar_pairs,
    #         'dissimilar': dissimilar_pairs
    #     }

    def extract_competence_pairs(self, curricula_data, default_area="Desconhecido"):
        """
        Extrai pares de competências similares e dissimilares dos currículos.

        Args:
            curricula_data (list): Lista de dicionários contendo os dados dos currículos.
            default_area (str): Área padrão para currículos sem área definida.

        Returns:
            dict: Um dicionário contendo duas listas: 'similar' (pares de competências similares)
                  e 'dissimilar' (pares de competências dissimilares).
        """
        similar_pairs = []
        dissimilar_pairs = []
        researchers_by_area = {}

        # Agrupa pesquisadores por área
        for researcher in curricula_data:
            area = researcher.get('Áreas', default_area)  # Usa área padrão se não definida
            if area not in researchers_by_area:
                researchers_by_area[area] = []
            researchers_by_area[area].append(researcher)

        # Extrai pares de competências similares
        for area, researchers in researchers_by_area.items():
            for i in range(len(researchers)):
                for j in range(i + 1, len(researchers)):
                    competences1 = self.extract_competences(researchers[i])
                    competences2 = self.extract_competences(researchers[j])
                    for comp1 in competences1:
                        for comp2 in competences2:
                            similar_pairs.append((comp1, comp2))

        # Extrai pares de competências dissimilares
        areas = list(researchers_by_area.keys())
        for i in range(len(areas)):
            for j in range(i + 1, len(areas)):
                researchers1 = researchers_by_area[areas[i]]
                researchers2 = researchers_by_area[areas[j]]
                for researcher1 in researchers1:
                    for researcher2 in researchers2:
                        competences1 = self.extract_competences(researcher1)
                        competences2 = self.extract_competences(researcher2)
                        for comp1 in competences1:
                            for comp2 in competences2:
                                dissimilar_pairs.append((comp1, comp2))

        return {
            'similar': similar_pairs,
            'dissimilar': dissimilar_pairs
        }

    def extract_info_trabalho(self, texto):
        """
        Extrai informações de um texto de trabalho acadêmico.

        Args:
            texto (str): O texto do trabalho acadêmico.

        Returns:
            str: Uma string contendo as informações extraídas ou o texto original em caso de erro.
        """
        padrao_titulo = r"[Tt]ítulo\s*:\s*(.*?)\.\s*"
        padrao_ano = r"[Aa]no\s*(?:de\s+)?[Oo]btenção\s*:\s*(\d+)\s*."
        padrao_ano2 = r"[Aa]no\s*(?:de\s+)?[Ff]inalização\s*:\s*(\d+)\s*."
        padrao_ano3 = r"\b(\d{4})\b"
        padrao_palavras_chave_area = r"[Pp]alavras-chave\s*:\s*(.*?)\s*(?::|\.)\s*(.*)"

        try:
            titulo = re.search(padrao_titulo, texto)
            titulo_trabalho = titulo.group(1).strip().title() if titulo else ""  # Verificar None
        except AttributeError: 
            print(f"Erro: Não foi possível encontrar o título em: {texto}")
            titulo_trabalho = texto.split('. ')[0].title()

        try:
            ano = re.search(padrao_ano, texto)
            ano_trabalho = int(ano.group(1)) if ano else '0000'  # Verificar se ano é None
        except (AttributeError, IndexError, ValueError):  # ValueError lida com erros de conversão
            try:
                ano2 = re.search(padrao_ano2, texto)
                ano_trabalho = int(ano2.group(1)) if ano2 else '0000'  # Verifica se ano2 é None
            except (AttributeError, IndexError, ValueError):
                try:
                    ano3 = re.search(padrao_ano3, texto)
                    ano_trabalho = int(ano3.group(1)) if ano3 else '0000'  # Verifica se ano3 é None
                except (AttributeError, IndexError, ValueError):
                    print(f"Erro: Não foi possível encontrar o ano em: {texto}")
                    ano_trabalho = '0000'

        try:
            palavras_chave_area = re.search(padrao_palavras_chave_area, texto)
            if palavras_chave_area:  # Verifica se palavras_chave_area é None
                palavras_trabalho = palavras_chave_area.group(1).strip()
                area_trabalho = palavras_chave_area.group(2).replace(":","").replace('/ ','|').rstrip(' .').strip()
            else:
                palavras_trabalho = ''
                area_trabalho = ''
        except (AttributeError, IndexError):
            print(f"Erro: Não foi possível encontrar palavras-chave ou área em: {texto}")
            palavras_trabalho = ''
            area_trabalho = ''

        try:
            tipo_trabalho = texto.split('. ')[0]
            instituicao = texto.split('. ')[1].strip().title()
        except IndexError:
            print(f"Erro: Não foi possível encontrar tipo ou instituição em: {texto}")
            tipo_trabalho = ''
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

            ## DEBUG
            # if dic_trabalho:
            #     print("Ano de Obtenção:", dic_trabalho["ano_obtencao"])
            #     print("Título trabalho:", dic_trabalho["titulo"])
            #     print(" Palavras-chave:", dic_trabalho["palavras_chave"])
            #     print("  Tipo trabalho:", dic_trabalho["tipo_trabalho"])
            #     print("    Instituição:", dic_trabalho["instituição"])
            #     print("  Área trabalho:", dic_trabalho["area_trabalho"])
            # else:
            #     print("Não foi possível extrair todas as informações do trabalho.")

            string_trabalho = ' | '.join([str(x) for x in dic_trabalho.values()])
            return string_trabalho.rstrip(' .').strip()
        except Exception as e:
            print(f'Erro ao extrair informações do trabalho: {e}')
            return texto 

    # def extrair_info_trabalho(self, texto):
    #     """
    #     Extrai título, ano de obtenção e palavras-chave de um texto de trabalho acadêmico.

    #     Args:
    #         texto (str): O texto do trabalho acadêmico.

    #     Returns:
    #         dict: Um dicionário contendo o título, ano de obtenção e palavras-chave, ou None se não encontrar as informações.
    #     """
    #     padrao_titulo = r"[Tt]ítulo\s*:\s*(.*?)\.\s*"
    #     padrao_ano = r"[Aa]no\s*(?:de\s+)?[Oo]btenção\s*:\s*(\d+)\s*."
    #     padrao_ano2 = r"[Aa]no\s*(?:de\s+)?[Ff]inalização\s*:\s*(\d+)\s*."
    #     padrao_ano3 = r"\b(\d{4})\b"
    #     padrao_palavras_chave_area = r"[Pp]alavras-chave\s*:\s*(.*?)\s*(?::|\.)\s*(.*)"

    #     titulo = re.search(padrao_titulo, texto)
    #     try:
    #         titulo.group(1).strip().title() # type: ignore
    #         titulo_trabalho = titulo.group(1).strip().title() # type: ignore
    #     except: 
    #         titulo_trabalho = texto.split('. ')[0].title()
    #     ano = re.search(padrao_ano, texto)
    #     ano2 = re.search(padrao_ano2, texto)
    #     ano3 = re.search(padrao_ano3, texto)
    #     try:
    #         ano_trabalho = int(ano.group(1)) # type: ignore
    #     except:
    #         try:
    #             ano_trabalho = int(ano2.group(1)) # type: ignore
    #         except:
    #             try:
    #                 ano_trabalho = int(ano3.group(1)) # type: ignore
    #             except:
    #                 ano_trabalho = '0000'
    #     palavras_chave_area = re.search(padrao_palavras_chave_area, texto)
    #     try:
    #         palavras_trabalho = palavras_chave_area.group(1).strip() # type: ignore
    #     except:
    #         palavras_trabalho = ''
    #     try:
    #         area_trabalho = palavras_chave_area.group(2).replace(":","").replace('/ ','|').rstrip(' .').strip() # type: ignore
    #     except:
    #         area_trabalho = ''
    #     try:
    #         tipo_trabalho = texto.split('. ')[0]
    #     except:
    #         print(f'Tipo do trabalho não encontrado em: {texto}')
    #         tipo_trabalho = ''
    #     try:
    #         instituicao = texto.split('. ')[1].strip().title()
    #         # print(f"Restante de dados: {texto.split('. ')[0:]}")
    #     except:
    #         print(f'Instituicao do trabalho não encontrada em: {texto}')
    #         instituicao = ''
    #     try:
    #         dic_trabalho = {
    #             "ano_obtencao": ano_trabalho,
    #             "titulo": titulo_trabalho,
    #             "palavras_chave": palavras_trabalho,
    #             "tipo_trabalho": tipo_trabalho,
    #             "instituição": instituicao,
    #             "area_trabalho": area_trabalho,
    #         }
    #         string_trabalho=''
    #         for x in dic_trabalho.values():
    #             string_trabalho = string_trabalho+' '+str(x)+' |'
    #         string_trabalho = string_trabalho.rstrip('|').rstrip(' .').strip()

    #         # if dic_trabalho:
    #         #     print("Ano de Obtenção:", dic_trabalho["ano_obtencao"])
    #         #     print("Título trabalho:", dic_trabalho["titulo"])
    #         #     print(" Palavras-chave:", dic_trabalho["palavras_chave"])
    #         #     print("  Tipo trabalho:", dic_trabalho["tipo_trabalho"])
    #         #     print("    Instituição:", dic_trabalho["instituição"])
    #         #     print("  Área trabalho:", dic_trabalho["area_trabalho"])
    #         # else:
    #         #     print("Não foi possível extrair todas as informações do trabalho.")

    #         return string_trabalho
    #     except Exception as e:
    #         print(f'Erro {e}')
    #         return texto 

    ## A função inicialmente fazia extração de todas seções juntas deixando-a longa. Para facilitar legibilidade e manutenção foi quebrada em funções especialistas para cada seção do currículo
    # def extract_competences(self, researcher_data):
    #     competences = []
        
    #     padrao_titulo = r"[Tt]ítulo\s*:\s*(.*?)\.\s*"
    #     padrao_ano = r"[Aa]no\s*(?:de\s+)?[Oo]btenção\s*:\s*(\d+)\s*."
    #     padrao_ano2 = r"[Aa]no\s*(?:de\s+)?[Ff]inalização\s*:\s*(\d+)\s*."
    #     padrao_ano3 = r"\b(\d{4})\b"
    #     padrao_palavras_chave_area = r"[Pp]alavras-chave\s*:\s*(.*?)\s*(?::|\.)\s*(.*)"

    #     def extract(texto):
    #         titulo = re.search(padrao_titulo, texto)
            
    #         try:
    #             info1 = titulo.group(1).strip().title() # type: ignore
    #             try:
    #                 info2 = titulo.group(2).strip().title() # type: ignore
    #             except:
    #                 info2 = ''
    #         except: 
    #             info1 = texto.split('. ')[0].strip().title()
    #             try:
    #                 info2 = texto.split('. ')[1].strip().title()
    #             except:
    #                 info2 = ''
    #         ano = re.search(padrao_ano, texto)
    #         ano2 = re.search(padrao_ano2, texto)
    #         ano3 = re.search(padrao_ano3, texto)
    #         # print(ano)
    #         # print(ano2)
    #         # print(ano3)
    #         try:
    #             ano_trabalho = int(ano.group(1)) # type: ignore
    #         except:
    #             try:
    #                 ano_trabalho = int(ano2.group(1)) # type: ignore
    #             except:
    #                 try:
    #                     ano_trabalho = int(ano3.group(1)) # type: ignore
    #                 except:
    #                     ano_trabalho = '----'
    #         return ano_trabalho, info1, info2

    #     # Extrair de áreas de atuação
    #     for area in researcher_data.get("Áreas", {}).values():
    #         area = area.replace(":","").replace("Subárea ","").replace(".","").replace("/","|").strip()
    #         competences.append('AtuaçãoPrf: '+area.title())

    #     # Extrair de formações acadêmicas
    #     verbose=False
    #     if verbose:
    #         print(f"\n{'-'*125}")
    #     for formacao in researcher_data.get("Formação", {}).get("Acadêmica", []):
    #         instituicao_formacao = formacao['Descrição'].split('.')[1].strip().title()
    #         if '(' in instituicao_formacao:
    #             instituicao_formacao = formacao['Descrição'].split('.')[2].strip().title()
    #         # print(f"     Instituição: {instituicao_formacao}")
    #         if verbose:
    #             print(f" Chaves Formação: {formacao.keys()}")
    #             print(f"Valores Formação: {formacao.values()}")                
    #             print(f"Dict   Formações: {formacao}")
    #         ano_formacao = formacao["Ano"]
    #         if '-' not in ano_formacao:
    #             ano_formacao = str(ano_formacao)+' - hoje'
    #         if 'interr' in ano_formacao:
    #             ano_interrupcao = formacao["Descrição"].split(':')[-1].strip()
    #             ano_formacao = f"{str(ano_formacao.split(' ')[0])} - {ano_interrupcao}"
    #         descr_formacao = formacao["Descrição"].strip().title()
    #         competences.append(f"FormaçãoAc: {ano_formacao} | {instituicao_formacao} | {descr_formacao}")

    #     # Extrair de projetos
    #     for tipo_projeto in ["ProjetosPesquisa", "ProjetosExtensão", "ProjetosDesenvolvimento"]:
    #         for projeto in researcher_data.get(tipo_projeto, []):
    #             # print(f' Chaves: {projeto.keys()}')
    #             # print(f'Valores: {projeto.values()}')
    #             tipo=None
    #             if 'Pesquisa' in tipo_projeto:
    #                 tipo = 'Psq'
    #             elif 'Extensão' in tipo_projeto:
    #                 tipo = 'Ext'
    #             elif 'Desenvolvimento' in tipo_projeto:
    #                 tipo = 'Dsv'
    #             descricao_projeto = projeto["descricao"]
    #             periodo_projeto = projeto["chave"].replace("Atual","hoje")
    #             titulo_projeto = projeto["titulo_projeto"]
    #             competences.append(f'Projeto{tipo}: {periodo_projeto} | {titulo_projeto} | {descricao_projeto.title()}')

    #     # Extrair de produções bibliográficas (artigos, resumos, etc.)
    #     for tipo_producao, producoes in researcher_data.get("Produções", {}).items():
    #         if isinstance(producoes, list):  # Artigos completos
    #             for publicacao in producoes:
    #                 # print(f'Dados publicação: {publicacao}')
    #                 if publicacao['fator_impacto_jcr']:
    #                     competences.append(f"Publicação: {publicacao['ano']} | {float(publicacao['fator_impacto_jcr']):06.2f} | {publicacao['titulo'].title()}")
    #                 else:
    #                     competences.append(f"Publicação: {publicacao['ano']} | {'000.00':^6} | {publicacao['titulo'].title()}")
    #         # elif isinstance(producoes, dict):  # palestra e apresentações em eventos
    #         #     for item in producoes.values():                  
    #         #         competences.append(item)

    #     # Extrair de orientações (se houver)
    #     orientacoes = researcher_data.get("Orientações", {})
    #     # print(f'Dicionário orientações: {orientacoes}')
    #     if isinstance(orientacoes, dict):
    #         for tipo_orientacao, detalhes in orientacoes.items():
    #             if verbose:
    #                 print(tipo_orientacao)
    #                 if isinstance(detalhes, dict):
    #                     print([x.detalhes.keys() for x in orientacoes.values()])
    #                 else:
    #                     print(f"List  Orientação: {detalhes}")
    #             if 'conclu' in tipo_orientacao:
    #                 tipo = 'Con'
    #             else:
    #                 tipo = 'And'
    #             for detalhe in detalhes:
    #                 doutorados = detalhe.get('Tese de doutorado')
    #                 if doutorados:
    #                     for doc in doutorados.values():
    #                         ano_fim, nome_aluno, titulo_orientacao = extract(doc)
    #                         competences.append(f'OriDout{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')
                    
    #                 mestrados = detalhe.get('Dissertação de mestrado')
    #                 if mestrados:
    #                     for mes in mestrados.values():
    #                         ano_fim, nome_aluno, titulo_orientacao = extract(mes)
    #                         competences.append(f'OriMest{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')
                    
    #                 especializacoes = detalhe.get('Monografia de conclusão de curso de aperfeiçoamento/especialização')
    #                 if especializacoes:
    #                     for esp in especializacoes.values():
    #                         ano_fim, nome_aluno, titulo_orientacao = extract(esp)
    #                         competences.append(f'OriEspe{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')
                    
    #                 graduacoes = detalhe.get('Trabalho de conclusão de curso de graduação')
    #                 if graduacoes:
    #                     for grd in graduacoes.values():
    #                         ano_fim, nome_aluno, titulo_orientacao = extract(grd)
    #                         competences.append(f'OriGrad{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')
                    
    #                 iniciacoes = detalhe.get('Iniciação científica')
    #                 if iniciacoes:
    #                     for ini in iniciacoes.values():
    #                         ano_fim, nome_aluno, titulo_orientacao = extract(ini)
    #                         competences.append(f'OriInic{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')

    #                 postdocs = detalhe.get('Supervisão de pós-doutorado')
    #                 if postdocs:
    #                     for pos in postdocs.values():
    #                         ano_fim, nome_aluno, titulo_orientacao = extract(pos)
    #                         competences.append(f'SupPosD{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')

    #                 postdocs = detalhe.get('Orientações de outra natureza')
    #                 if postdocs:
    #                     for pos in postdocs.values():
    #                         ano_fim, nome_aluno, titulo_orientacao = extract(pos)
    #                         competences.append(f'OutNatu{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')

    #     ## DEBUG                    
    #     # elif isinstance(orientacoes, list):
    #     #     print('Lista de orientações')
    #     #     for orientacao in orientacoes:
    #     #         print(f'Dados da Orientação: {orientacao}')
    #     #         titulo_orientacao = orientacao.get("titulo", "")
    #     #         descricao_orientacao = orientacao.get("descricao", "")
    #     #         competences.append('Orientação: '+titulo_orientacao.title()+' '+descricao_orientacao.title())

    #     # Extrair de atuação profissional
    #     # for atuacao in researcher_data.get("Atuação Profissional", []):
    #     #     competences.append(atuacao.get("Instituição", ""))  # Adicionando a instituição
    #     #     competences.append(atuacao.get("Descrição", ""))
    #     #     competences.append(atuacao.get("Outras informações", ""))
        
    #     # Extrair de bancas
    #     # for tipo_banca, bancas in researcher_data.get("Bancas", {}).items():
    #     #     for banca in bancas.values():
    #     #         competences.append(banca)

    #     return competences

    def extract_competences_from_areas(self, researcher_data):
        """
        Extrai competências das áreas de atuação do pesquisador.

        Args:
            researcher_data (dict): Dicionário contendo os dados do currículo do pesquisador.

        Returns:
            list: Lista de strings representando as competências extraídas das áreas de atuação.
        """
        competences = []
        for area in researcher_data.get("Áreas", {}).values():
            area = area.replace(":","").replace("Subárea ","").replace(".","").replace("/","|").strip()
            competences.append('AtuaçãoPrf: '+area.title())
        return competences

    def extract_competences_from_formacao(self, researcher_data):
        """
        Extrai competências da formação acadêmica do pesquisador.

        Args:
            researcher_data (dict): Dicionário contendo os dados do currículo do pesquisador.

        Returns:
            list: Lista de strings representando as competências extraídas da formação acadêmica.
        """
        competences = []
        for formacao in researcher_data.get("Formação", {}).get("Acadêmica", []):
            try:
                instituicao_formacao = formacao['Descrição'].split('.')[1].strip().title()
                if '(' in instituicao_formacao:
                    instituicao_formacao = formacao['Descrição'].split('.')[2].strip().title()
            except IndexError:
                print(f"Erro: Não foi possível extrair a instituição da formação: {formacao['Descrição']}")
                instituicao_formacao = ""

            ano_formacao = formacao["Ano"]
            if '-' not in ano_formacao:
                ano_formacao = str(ano_formacao)+' - hoje'
            if 'interr' in ano_formacao:
                try:
                    ano_interrupcao = formacao["Descrição"].split(':')[-1].strip()
                    ano_formacao = f"{str(ano_formacao.split(' ')[0])} - {ano_interrupcao}"
                except IndexError:
                    print(f"Erro: Não foi possível extrair o ano de interrupção da formação: {formacao['Descrição']}")
                    ano_formacao = ""

            descr_formacao = formacao["Descrição"].strip().title()
            competences.append(f"FormaçãoAc: {ano_formacao} | {instituicao_formacao} | {descr_formacao}")
        return competences

    def extract_competences_from_projects(self, researcher_data):
        """
        Extrai competências dos projetos do pesquisador.

        Args:
            researcher_data (dict): Dicionário contendo os dados do currículo do pesquisador.

        Returns:
            list: Lista de strings representando as competências extraídas dos projetos.
        """
        competences = []
        for tipo_projeto in ["ProjetosPesquisa", "ProjetosExtensão", "ProjetosDesenvolvimento"]:
            for projeto in researcher_data.get(tipo_projeto, []):
                tipo = None
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
        return competences


    def extract_competences_from_publications(self, researcher_data):
        """
        Extrai competências das publicações do pesquisador.

        Args:
            researcher_data (dict): Dicionário contendo os dados do currículo do pesquisador.

        Returns:
            list: Lista de strings representando as competências extraídas das publicações.
        """
        competences = []
        for tipo_producao, producoes in researcher_data.get("Produções", {}).items():
            if isinstance(producoes, list):  # Artigos completos
                for publicacao in producoes:
                    try:
                        fator_impacto = float(publicacao['fator_impacto_jcr']) if publicacao['fator_impacto_jcr'] else 0.00
                        competences.append(f"Publicação: {publicacao['ano']} | {fator_impacto:06.2f} | {publicacao['titulo'].title()}")
                    except (KeyError, ValueError) as e:
                        print(f"Erro ao extrair dados da publicação: {e}. Dados da publicação: {publicacao}")
        return competences

    def extract_competences_from_orientacoes(self, researcher_data):
        """
        Extrai competências das orientações do pesquisador.

        Args:
            researcher_data (dict): Dicionário contendo os dados do currículo do pesquisador.

        Returns:
            list: Lista de strings representando as competências extraídas das orientações.
        """
        competences = []

        def extract(texto):
            padrao_titulo = r"[Tt]ítulo\s*:\s*(.*?)\.\s*"
            padrao_ano = r"[Aa]no\s*(?:de\s+)?[Oo]btenção\s*:\s*(\d+)\s*."
            padrao_ano2 = r"[Aa]no\s*(?:de\s+)?[Ff]inalização\s*:\s*(\d+)\s*."
            padrao_ano3 = r"\b(\d{4})\b"

            try:
                titulo = re.search(padrao_titulo, texto)
                info1 = titulo.group(1).strip().title() # type: ignore
                try:
                    info2 = titulo.group(2).strip().title() # type: ignore
                except IndexError:
                    info2 = ''
            except AttributeError:
                try:
                    info1 = texto.split('. ')[0].strip().title()
                    info2 = texto.split('. ')[1].strip().title()
                except IndexError:
                    print(f"Erro: Não foi possível extrair informações do texto: {texto}")
                    info1 = ""
                    info2 = ""

            try:
                ano = re.search(padrao_ano, texto)
                ano_trabalho = int(ano.group(1)) # type: ignore
            except (AttributeError, IndexError):
                try:
                    ano2 = re.search(padrao_ano2, texto)
                    ano_trabalho = int(ano2.group(1)) # type: ignore
                except (AttributeError, IndexError):
                    try:
                        ano3 = re.search(padrao_ano3, texto)
                        ano_trabalho = int(ano3.group(1)) # type: ignore
                    except (AttributeError, IndexError):
                        print(f"Erro: Não foi possível extrair o ano do texto: {texto}")
                        ano_trabalho = '----'
            return ano_trabalho, info1, info2

        orientacoes = researcher_data.get("Orientações", {})
        if isinstance(orientacoes, dict):
            for tipo_orientacao, detalhes in orientacoes.items():
                tipo = 'Con' if 'conclu' in tipo_orientacao else 'And'
                for detalhe in detalhes:
                    for tipo_trabalho, trabalhos in detalhe.items():
                        if trabalhos:
                            for trabalho in trabalhos.values():
                                ano_fim, nome_aluno, titulo_orientacao = extract(trabalho)
                                tipo_trabalho_abreviado = ''.join([p[0] for p in tipo_trabalho.split() if p[0].isupper()])
                                competences.append(f'Ori{tipo_trabalho_abreviado}{tipo}: {ano_fim} | {" ".join(unidecode(nome_aluno).title().split())} | {titulo_orientacao.title()}')
        return competences


    def extract_competences(self, researcher_data):
        """
        Extrai competências de um pesquisador a partir de seus dados de currículo.

        Args:
            researcher_data (dict): Dicionário contendo os dados do currículo do pesquisador.

        Returns:
            list: Lista de strings representando as competências extraídas.
        """
        competences = []
        competences.extend(self.extract_competences_from_areas(researcher_data))
        competences.extend(self.extract_competences_from_formacao(researcher_data))
        competences.extend(self.extract_competences_from_projects(researcher_data))
        competences.extend(self.extract_competences_from_publications(researcher_data))
        competences.extend(self.extract_competences_from_orientacoes(researcher_data))
        return competences

    # def preprocess_competences_spacy(self, competences):
    #         """
    #         Pré-processa uma lista de competências, removendo stop words, lematizando e eliminando termos duplicados consecutivos (ignorando maiúsculas e minúsculas).

    #         Args:
    #             competences (list): Uma lista de strings representando as competências.

    #         Returns:
    #             list: Uma lista de strings contendo as competências pré-processadas.
    #         """

    #         processed_competences = []
    #         for competence in competences:
    #             if competence:
    #                 doc = self.nlp_en(competence) if competence.isascii() else self.nlp_pt(competence)

    #                 palavras_processadas = []
    #                 eliminar = ['descrição','situação',':']
    #                 ultima_palavra = None
    #                 for token in doc:
    #                     if token.is_stop or token.lemma_.lower() in eliminar:
    #                         continue
                        
    #                     # Ignora pontuação, espaços em branco e caracteres especiais
    #                     if token.is_punct or token.is_space or not token.is_alpha:
    #                         continue

    #                     palavra_atual = token.lemma_.lower().strip()
    #                     if palavra_atual != ultima_palavra:
    #                         palavras_processadas.append(palavra_atual)
    #                     ultima_palavra = palavra_atual

    #                 processed_competences.append(" ".join(palavras_processadas))
    #         return processed_competences


    def preprocess_competences_spacy(self, competences):
        """
        Pré-processa uma lista de competências usando o spaCy, 
        removendo stop words, lematizando e eliminando termos 
        duplicados consecutivos (ignorando maiúsculas e minúsculas).

        Detecta o idioma (português, espanhol ou alemão), 
        traduz para inglês usando deep_translator se necessário, 
        e formata as competências para facilitar a geração de 
        embeddings com PyTorch.

        Args:
            competences (list): Uma lista de strings representando as competências.

        Returns:
            list: Uma lista de strings contendo as competências pré-processadas.
        """

        processed_competences = []
        for competence in competences:
            if competence:
                # Detectar o idioma com langdetect
                try:
                    from langdetect import detect
                    idioma = detect(competence)
                except:
                    # Se langdetect falhar, usa isascii() como fallback
                    idioma = 'en' if competence.isascii() else 'unknown'

                # Traduzir para inglês se o idioma for português, espanhol ou alemão
                if idioma in ('pt', 'es', 'de'):
                    from deep_translator import GoogleTranslator
                    translator = GoogleTranslator(source=idioma, target='en')
                    try:
                        competence = translator.translate(competence)
                    except Exception as e:
                        print(f"Erro na tradução: {e}")
                        # Opcional: Manter a competência original em caso de erro na tradução
                        continue

                # Carregar o modelo do spaCy em inglês
                nlp = self.nlp_en

                # Processar a competência com o spaCy
                doc = nlp(competence)

                # Remover stop words e lematizar
                palavras_processadas = []
                ultima_palavra = None
                for token in doc:
                    if token.is_stop or not token.is_alpha:
                        continue
                    palavra_atual = token.lemma_.lower()
                    if palavra_atual != ultima_palavra:
                        palavras_processadas.append(palavra_atual)
                    ultima_palavra = palavra_atual

                # Formatar para PyTorch (exemplo)
                processed_competence = {
                    "text": " ".join(palavras_processadas),
                    "language": "en"  # Agora todas as competências estão em inglês
                }
                processed_competences.append(processed_competence)

        return processed_competences


    def preprocess_competences(self, competences):
            """
            Pré-processa uma lista de competências, removendo stop words, 
            lematizando e eliminando termos duplicados consecutivos 
            (ignorando maiúsculas e minúsculas).
            
            Detecta e converte o idioma para inglês, quando necessário, 
            e formata as competências para facilitar a geração de embeddings com PyTorch.

            Args:
                competences (list): Uma lista de strings representando as competências.

            Returns:
                list: Uma lista de strings contendo as competências pré-processadas.
            """

            processed_competences = []
            for competence in competences:
                if competence:
                    # Detectar o idioma, com uma primeira aproximação para detectar o idioma inglês, mas para uma detecção mais precisa, é recomendado usar bibliotecas especializadas.
                    is_english = competence.isascii()

                    # Tokenizar
                    tokens = competence.lower().split()

                    # Remoção de stop words
                    stop_words = set()
                    if is_english:
                        # Usa stop words em inglês
                        stop_words = set(["the", "a", "an", "and", "in", "to", "of", "for", "on", "with", "as", "at", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "will", "would", "should", "can", "could", "may", "might", "must", "ought", "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs", "what", "which", "who", "whom", "whose", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])
                    else:
                        # Usa stop words em português
                        stop_words = set(["de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso", "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você", "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia", "seja", "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse", "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela", "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo", "estou", "está", "estamos", "estão", "estive", "esteve", "estivemos", "estiveram", "estava", "estávamos", "estavam", "estivera", "estivéramos", "esteja", "estejamos", "estejam", "estivesse", "estivéssemos", "estivessem", "estiver", "estivermos", "estiverem", "hei", "há", "havemos", "hão", "houve", "houvemos", "houveram", "houvera", "houvéramos", "haja", "hajamos", "hajam", "houvesse", "houvéssemos", "houvessem", "houver", "houvermos", "houverem", "houverei", "houverá", "houveremos", "houverão", "houveria", "houveríamos", "houveriam", "sou", "somos", "são", "era", "éramos", "eram", "fui", "foi", "fomos", "foram", "fora", "fôramos", "seja", "sejamos", "sejam", "fosse", "fôssemos", "fossem", "for", "formos", "forem", "serei", "será", "seremos", "serão", "seria", "seríamos", "seriam", "tenho", "tem", "temos", "têm", "tinha", "tínhamos", "tinham", "tive", "teve", "tivemos", "tiveram", "tivera", "tivéramos", "tenha", "tenhamos", "tenham", "tivesse", "tivéssemos", "tivessem", "tiver", "tivermos", "tiverem", "terei", "terá", "teremos", "terão", "teria", "teríamos", "teriam"])

                    tokens = [token for token in tokens if token not in stop_words]

                    # Lematização
                    if is_english:
                        # Implementa lematização em inglês (exemplo com nltk)
                        from nltk.stem import WordNetLemmatizer
                        lemmatizer = WordNetLemmatizer()
                        tokens = [lemmatizer.lemmatize(token) for token in tokens]
                    else:
                        # Implementa lematização em português
                        def lematizar(palavra):
                            # Implemente aqui uma lógica simples de lematização, se necessário
                            # Exemplo: remoção de sufixos comuns
                            if palavra.endswith("s"):
                                palavra = palavra[:-1]
                            elif palavra.endswith("es"):
                                palavra = palavra[:-2]
                            elif palavra.endswith("mente"):
                                palavra = palavra[:-5]
                            return palavra

                        tokens = [lematizar(token) for token in tokens]

                    # Eliminação de termos duplicados consecutivos
                    palavras_processadas = []
                    ultima_palavra = None
                    for token in tokens:
                        if token != ultima_palavra:
                            palavras_processadas.append(token)
                        ultima_palavra = token

                    # Formatação para PyTorch (exemplo)
                    processed_competence = {
                        "text": " ".join(palavras_processadas),
                        "language": "en" if is_english else "pt"
                    }
                    processed_competences.append(processed_competence)

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
        self.competence_extractor = CompetenceExtractor(curricula_file)
        self.curricula_data = self.competence_extractor.load_curricula() # carregar lista de dicionários
        self.gpu_manager = GPUMemoryManager()  # Instanciar o gerenciador de memória da GPU

    def benchmark_data_transfer(self, model, sizes, device):
        """Mede o tempo de transferência de dados entre CPU e GPU para um modelo."""
        results = {}
        for size in sizes:
            data_cpu = torch.randn(size, model.get_sentence_embedding_dimension())  # Dados com dimensão do embedding
            data_gpu = torch.randn(size, model.get_sentence_embedding_dimension()).to(device)

            # CPU para GPU
            start_time = time.time()
            data_gpu.copy_(data_cpu)
            torch.cuda.synchronize()
            cpu_to_gpu_time = time.time() - start_time

            # GPU para CPU
            start_time = time.time()
            data_cpu.copy_(data_gpu)
            gpu_to_cpu_time = time.time() - start_time

            results[size] = {
                'cpu_to_gpu': cpu_to_gpu_time,
                'gpu_to_cpu': gpu_to_cpu_time,
            }
        return results

    def benchmark_model(self, model, sentences, device, batch_size=16):  # Reduzido o batch_size
        """Mede o tempo de processamento do modelo (CPU ou GPU) em lotes."""

        # Mover o modelo para o dispositivo desejado
        model.to(device) 

        # Dividir as sentenças em lotes
        batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

        start_time = time.time()
        for batch in batches:
            with torch.no_grad():
                model.encode(batch, convert_to_tensor=True)
        end_time = time.time()

        total_time = end_time - start_time
        num_repetitions = 3
        total_samples = len(sentences) * num_repetitions  # Cálculo do número total de amostras
        return total_time / total_samples # Tempo médio por amostra

    def evaluate_intrinsic(self, model, validation_data):
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
        all_embeddings = []  # Criando a lista para armazenar os embeddings
        MAX_LENGTH = 384

        # Primeira passagem para identificar áreas válidas (CORRIGIDO)
        valid_areas = set()
        for researcher_data in self.curricula_data:
            all_areas_list = self.extrair_areas(researcher_data.get('Áreas', {}))
            for area in all_areas_list.get('Áreas'):  # type: ignore
                if area and area != 'desconhecido':
                    valid_areas.add(area)

        # Segunda passagem para preparar os dados
        for researcher_data in self.curricula_data:
            competences = self.competence_extractor.extract_competences(researcher_data)
            processed_competences = self.competence_extractor.preprocess_competences(competences)
            processed_competences = [comp[:MAX_LENGTH] for comp in processed_competences]
            areas_list = self.extrair_areas(researcher_data.get('Áreas', {}))  # Obtém lista de áreas

            print(f"Área de pesquisa: {area}")
            for area in areas_list.get('Áreas'): # type: ignore
                # print(f"Competências extraídas: {competences}")
                print(f"Compet.pré-processadas: {processed_competences}")

                if area in valid_areas and processed_competences:
                    embeddings = model.encode(processed_competences, convert_to_tensor=True)
                    all_embeddings.extend(embeddings)  # Acumula os embeddings
                    mean_embedding = torch.mean(embeddings, dim=0)  # Calcula a média na GPU
                    X.append(mean_embedding)  # Move para CPU e converte para NumPy
                    y.append(area)

        # Mover o modelo para o dispositivo desejado (se disponível)
        if device == "gpu" and torch.cuda.is_available():
            model.to('cuda')
        elif device == "cpu":
            model.to('cpu')

        return model, X, y # Retornar o modelo junto com os dados

    def prepare_area_classification(self, model, device="gpu"):
        X = []
        y = []  # Substituído por um dicionário de áreas e similaridades
        valid_areas = set()
        all_embeddings = []
        area_embeddings = {}  # Dicionário para armazenar os embeddings das áreas

        if device == "gpu":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
                print("CUDA não disponível, usando CPU.")
        else:
            device = torch.device("cpu")

        # Carregar embeddings das áreas (treinados previamente)
        area_embeddings = np.load("area_embeddings.npy", allow_pickle=True).item()  # Carrega o dicionário de embeddings

        # Primeira passagem para identificar áreas válidas
        for researcher_data in self.curricula_data:
            areas_list = self.extrair_areas(researcher_data.get('Áreas', {}))
            for areas in areas_list:
                if isinstance(areas, dict):
                    area = areas.get('Área')
                elif isinstance(areas, str):
                    area = areas
                else:
                    area = None  # Ou algum valor padrão, se necessário

                if area and area != 'desconhecido':
                    valid_areas.add(area)

        # Segunda passagem para preparar os dados
        for researcher_data in self.curricula_data:
            competences = self.competence_extractor.extract_competences(researcher_data)
            processed_competences = self.competence_extractor.preprocess_competences(competences)
            areas_list = self.extrair_areas(researcher_data.get('Áreas', {}))

            for areas in areas_list:
                if isinstance(areas, dict):
                    area = areas.get('Área')
                elif isinstance(areas, str):
                    area = areas
                else:
                    area = None  # Ou algum valor padrão, se necessário

                if area in valid_areas and processed_competences:
                    embeddings = model.encode(processed_competences, convert_to_tensor=True, device=device)
                    all_embeddings.extend(embeddings)
                    mean_embedding = torch.mean(embeddings, dim=0).cpu().numpy()  # Calcula a média na GPU e move para CPU

                    # Calcular similaridade com as áreas de pesquisa
                    array_areas = np.array(list(area_embeddings.values()))
                    similarities = cosine_similarity(mean_embedding, array_areas)[0]

                    y.append({area: sim for area, sim in zip(area_embeddings.keys(), similarities)})

        # Agrupar áreas de pesquisa
        area_names = list(area_embeddings.keys())
        kmeans = KMeans(n_clusters=5)  # Defina o número de clusters desejado
        kmeans.fit(list(area_embeddings.values()))
        area_clusters = kmeans.labels_

        # Associar pesquisadores aos clusters
        for i, similarities in enumerate(y):
            for area, sim in similarities.items():
                cluster = area_clusters[area_names.index(area)]
                X.append(all_embeddings[i].cpu().numpy()) # Mover para CPU converter para NumPy
                y[i] = cluster  # Substituir o nome da área pelo ID do cluster

        return X, y


    def evaluate_embeddings(self, X, y, metric=cosine_similarity):
        """Avalia a qualidade dos embeddings em relação às áreas de pesquisa."""
        scores = []
        for i, area in enumerate(y):
            # Calcular a similaridade entre o embedding da área e os embeddings de suas competências
            area_idx = [j for j, a in enumerate(y) if a == area]  # Índices das competências da mesma área
            competence_embeddings = [X[j] for j in area_idx]
            similarities = metric([X[i]], competence_embeddings)  # Similaridade entre área e suas competências
            scores.append(np.mean(similarities))  # Média das similaridades

        return np.mean(scores)  # Média geral das similaridades


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
            model = SentenceTransformer(model_name, device=device).half() # type: ignore # carrega o modelo ja no dispostivo

            # Avaliação intrínseca
            intrinsic_results = self.evaluate_intrinsic(model, validation_data)
            results[model_name] = intrinsic_results

            # Avaliação extrínseca
            model, X, y = self.prepare_data_for_classification(model)
            if use_cross_validation:
                if len(set(y)) < 2:
                    print(f"Não há classes suficientes para validação cruzada. Pulando modelo {model}.") # Corrigido para usar 'model'
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
                X_train = [tensor.cpu().numpy() for tensor in X_train]  # Lista de arrays NumPy
                X_train = np.array(X_train)  # Converte para um único array NumPy multidimensional

                # Treinamento e avaliação do classificador
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)

                # Cálculo das métricas
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)  
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)  
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) 

                results[model_name].update({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

                print(f"\nAcurácia: {accuracy:.4f}")
                print(f"Precisão: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"F1-Score: {f1:.4f}")

            else:
                print(f"Não há exemplos suficientes para divisão em treinamento e teste. Pulando modelo {model}.") # Corrigido para usar 'model'
                results[model_name].update({'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None})

            print('-' * 125)
            print()
        return results

    def evaluate_models_cross_validation(self, model, classifier_name="LogisticRegression", num_folds=5):
        """Avalia os modelos de embedding usando validação cruzada com diferentes classificadores."""
        model, X, y = self.prepare_data_for_classification(model)

        # Verifica se há classes suficientes para a validação cruzada
        if len(set(y)) < 2:
            # Lida com o caso em que 'model' pode ser uma string ou um objeto SentenceTransformer
            model_identifier = model if isinstance(model, str) else model.__class__.__name__ 
            print(f"Não há classes suficientes para validação cruzada. Pulando modelo {model_identifier}.")
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

        # Converter X para um array NumPy
        X = np.array(X)
        
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