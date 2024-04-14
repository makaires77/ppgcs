import pdfkit
from weasyprint import HTML
import plotly.offline as pyo
import os, logging, json, html
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import plot
from nltk.corpus import stopwords
from collections import defaultdict
from json_fle_manager import JSONFileManager as jfm
from sklearn.metrics.pairwise import cosine_similarity
import plotly.io as pio

import datetime 
import pandas as pd
from xhtml2pdf import pisa

from jinja2 import Environment, FileSystemLoader


import geopandas as gpd
from shapely.geometry import Point
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt
import mapclassify as mc
import seaborn as sns

pio.renderers.default = "notebook_connected"

class ReportHTML:
    def __init__(self, base_repo_dir):
        self.base_repo_dir = base_repo_dir
        self.folder_utils = os.path.join(base_repo_dir, 'utils')
        self.folder_assets = os.path.join(base_repo_dir, 'assets')
        self.folder_domain = os.path.join(base_repo_dir, 'source', 'domain')
        self.folder_data_input = os.path.join(base_repo_dir, 'data', 'input')
        self.folder_data_output = os.path.join(base_repo_dir, 'data', 'output')            
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
    
    @staticmethod
    def convert_to_html(text):
        html_content = text.replace('\n', '<br>')
        return f"<html><body>{html_content}</body></html>"

    def flatten(self, input_structure):
        out = {}

        # Verificar se a entrada é uma string (presumivelmente JSON) e tentar converter para um dicionário
        if isinstance(input_structure, str):
            try:
                input_structure = json.loads(input_structure)
            except json.JSONDecodeError:
                raise ValueError("A string fornecida não é um JSON válido.")

        # Verificar se a entrada agora é um dicionário
        if not isinstance(input_structure, dict):
            raise ValueError("A entrada deve ser um dicionário ou uma string JSON válida.")

        def flatten_dict(d, parent_key=''):
            for k, v in d.items():
                new_key = parent_key + '_' + k if parent_key else k
                if isinstance(v, dict):
                    if 'descricao' in v:
                        # Usar apenas o código (k) como a chave para descrição
                        out[k] = v['descricao']
                    flatten_dict(v, new_key)

        flatten_dict(input_structure)
        return out

    def get_cod_name(self, codigo, flat_structure):
        # Certificar que o código não tenha um hífen extra no final
        codigo = codigo.strip('-')
        # Buscar o nome correspondente ao código na estrutura achatada
        nome = flat_structure.get(codigo, f"Nome não encontrado para o código fornecido ({codigo})")
        return nome
    
    def atribuir_grande_area_e_area(self, titulos, estrutura_cnpq_json, n_components, palavras_chave_por_area):
        # Converter a string JSON em um dicionário
        try:
            estrutura_cnpq = json.loads(estrutura_cnpq_json)
        except json.JSONDecodeError:
            raise ValueError("A string fornecida não é um JSON válido.")

        if not isinstance(estrutura_cnpq, dict):
            raise ValueError("A estrutura CNPq deve ser um dicionário.")

        # Concatenar as palavras-chave aos títulos
        titulos_com_palavras_chave = titulos[:]
        for area, palavras_chave in palavras_chave_por_area.items():
            titulos_com_palavras_chave.extend(palavras_chave)

        # Modelar os tópicos dos títulos dos artigos e obter nmf, W e vectorizer
        nmf, W, vectorizer = self.modelar_topicos_nmf(titulos_com_palavras_chave, n_components)

        # Vetorizar as descrições das áreas e subáreas do CNPq
        descricoes = []
        for ga in estrutura_cnpq.values():
            for area in ga["areas"].values():
                descricao_area = area["descricao"]
                for subarea in area["subareas"].values():
                    descricao_area += " " + subarea["descricao"]
                descricoes.append(descricao_area)

        X_descricoes = vectorizer.transform(descricoes)

        # Calcular a similaridade entre os tópicos dos artigos e as áreas
        similaridades = cosine_similarity(W.dot(nmf.components_), X_descricoes)

        # Achatando a estrutura do CNPq para facilitar a busca de nomes por código
        flat_structure = self.flatten(estrutura_cnpq)

        # Atribuir cada artigo à área e subárea mais próximas
        atribuicoes = []
        for i in range(len(titulos)):
            indice_mais_proximo = similaridades[i].argmax()
            total_areas = sum(len(ga["areas"]) for ga in estrutura_cnpq.values())

            if indice_mais_proximo >= total_areas:
                raise IndexError("Índice de área calculado está fora do intervalo da estrutura do CNPq.")

            for ga_index, ga in enumerate(estrutura_cnpq.values()):
                if indice_mais_proximo < len(ga["areas"]):
                    grande_area = list(estrutura_cnpq.keys())[ga_index]
                    area = list(ga["areas"].keys())[indice_mais_proximo]
                    break
                else:
                    indice_mais_proximo -= len(ga["areas"])

            nome_grande_area = self.get_cod_name(grande_area, flat_structure)
            nome_area = self.get_cod_name(area, flat_structure)

            atribuicoes.append((titulos[i], grande_area, area, nome_grande_area, nome_area))

        return nmf, W, vectorizer, atribuicoes

    def buscar_area(self, estrutura_cnpq_json, criterio_busca):
        estrutura_cnpq = json.loads(estrutura_cnpq_json)
        for ga_code, ga in estrutura_cnpq.items():
            for a_code, area in ga["areas"].items():
                if criterio_busca.lower() in area['descricao'].lower() or criterio_busca == a_code:
                    return a_code, area['descricao']
        return None, None

    def extrair_palavras_chave_nmf(self, nmf, vectorizer, n_top_words=10):
        feature_names = vectorizer.get_feature_names_out()
        palavras_chave_nmf = {}

        for topic_idx, topic in enumerate(nmf.components_):
            top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
            palavras_chave_nmf[f"Tópico {topic_idx + 1}"] = [feature_names[i] for i in top_features_ind]

        return palavras_chave_nmf

    def salvar_palavras_chave_em_disco(self, palavras_chave, caminho_arquivo):
        try:
            with open(caminho_arquivo, 'w', encoding='utf-8') as arquivo:
                json.dump(palavras_chave, arquivo, ensure_ascii=False, indent=4)
                print(f"Palavras-chave salvas com sucesso em {caminho_arquivo}")
        except IOError as e:
            print(f"Erro ao salvar o arquivo: {e}")

    def solicitar_palavras_chave_areas(self, estrutura_cnpq_json):
        # Converter a string JSON em um dicionário
        try:
            estrutura_cnpq = json.loads(estrutura_cnpq_json)
        except json.JSONDecodeError:
            raise ValueError("A string fornecida não é um JSON válido.")

        if not isinstance(estrutura_cnpq, dict):
            raise ValueError("A estrutura CNPq deve ser um dicionário.")

        palavras_chave_por_area = {}

        for ga_code, ga in estrutura_cnpq.items():
            print(f"\nGrande Área: {ga['descricao']} ({ga_code})")
            for a_code, area in ga["areas"].items():
                print(f"  Área: {area['descricao']} ({a_code})")

                # Solicitar palavras-chave do usuário para esta área
                palavras_chave = input(f"Insira palavras-chave para a Área '{area['descricao']}' (separadas por vírgula): ").split(',')
                palavras_chave = [palavra.strip() for palavra in palavras_chave if palavra.strip()]
                
                if palavras_chave:
                    palavras_chave_por_area[a_code] = palavras_chave

        return palavras_chave_por_area

    def ajustar_atribuicoes_com_areas_atuacao(self, atribuicoes, areas_of_expertise, title_to_researcher, estrutura_cnpq, vectorizer, nmf, W):
        # A estrutura CNPq já é um dicionário, então não é necessário carregá-la de uma string JSON
        flat_structure = self.flatten(estrutura_cnpq)
        atribuicoes_ajustadas = []

        for titulo, cod_grande_area, cod_area, nome_grande_area, nome_area in atribuicoes:
            # Obter o nome do pesquisador para o título atual
            pesquisador = title_to_researcher.get(titulo, "Desconhecido")
            areas_pesquisador = areas_of_expertise.get(pesquisador)

            # Inicialmente, assumimos que as atribuições estão corretas
            atribuicao_ajustada = (titulo, cod_grande_area, cod_area, nome_grande_area, nome_area)

            if areas_pesquisador:
                # Vetorizar as descrições das áreas de atuação do pesquisador
                descricoes_areas_pesquisador = [flat_structure.get(area, '') for area in areas_pesquisador['Área']]
                if descricoes_areas_pesquisador and any(descricao for descricao in descricoes_areas_pesquisador):
                    X_descricoes_pesquisador = vectorizer.transform(descricoes_areas_pesquisador)
                else:
                    # Se não houver descrições válidas, mantenha a atribuição original
                    atribuicoes_ajustadas.append(atribuicao_ajustada)
                    continue

                # Transformar os tópicos do artigo para o espaço de palavras
                topicos_artigo = nmf.transform(vectorizer.transform([titulo]))

                # Transformar os tópicos de volta ao espaço de palavras
                H = nmf.components_
                topic_space = W.dot(H)

                # Calcular a similaridade entre o artigo e as áreas do pesquisador
                similaridades = cosine_similarity(topic_space, X_descricoes_pesquisador)

                # Encontrar a área mais próxima
                indice_mais_proximo = similaridades[0].argmax()
                if indice_mais_proximo < len(areas_pesquisador['Área']):
                    nova_area = areas_pesquisador['Área'][indice_mais_proximo]
                    nome_nova_area = self.get_cod_name(nova_area, flat_structure)
                    atribuicao_ajustada = (titulo, cod_grande_area, nova_area, nome_grande_area, nome_nova_area)

            atribuicoes_ajustadas.append(atribuicao_ajustada)

        return atribuicoes_ajustadas

    def inserir_logotipo(self, imagem, alinhamento="center"):
        """
        Insere um logotipo em um html.

        Args:
            imagem: O caminho para o arquivo .png do logotipo.
            alinhamento: O alinhamento do logotipo no cabecalho de página. Os valores possíveis são "left", "center" e "right".

        Returns:
            O código html do logotipo.
        """

        if alinhamento not in ("left", "center", "right"):
            raise ValueError("O alinhamento deve ser 'left', 'center' ou 'right'.")

        return html.escape(f"<img src='{imagem}' alt='Logotipo' align='{alinhamento}' height='75px'>")

    def generat_title_list_report(self, filepath):
        # Use StringIO to capture print output
        from io import StringIO
        import sys
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # Capture the output
        old_stdout = sys.stdout
        sys.stdout = report_output = StringIO()
        separator = 120
        # Generate the report content
        logo_path = os.path.join(self.folder_assets,'logo_fioce_75h.png')
        print(self.inserir_logotipo(logo_path, alinhamento="left"))
        print("<h1><center><b>Produção em artigos de pesquisa da Fiocruz Ceará</b></center></h1>")
        logging.info("Obtendo os dados do dataset de artigos, aguarde...")
        dataset_articles = jfm.load_json(self.folder_data_output,filepath)
        print(f"<h2><center><b>{len(dataset_articles)} currículos extraídos</b></center></h2>")
        
        title_list = [] # lista de títulos das publicações de cada pesquisador
        title_to_researcher = {}  # Dicionário para mapear título para nome do pesquisador
        areas_of_expertise = {}   # Dicionário para armazenar as áreas de atuação por pesquisador
        
        for curriculo in dataset_articles:
            processed_data = curriculo.get('processed_data')
            researcher_name = processed_data.get('name')
            print(f"<center>{'-'*separator}</center>")
            print(f"<h2><center><b>{processed_data.get('name')}</b></center></h2>")
            print(f"<h4>ÁREAS CNPq:</h4>")
            GAR=[]
            ARE=[]
            SAR=[]
            ESP=[]
            for classificacao in processed_data.get('areas_of_expertise'):
                for i,j in classificacao.items():
                    if i == 'GrandeÁrea':
                        if j:
                            GAR.append(j)
                    elif i == 'Área':
                        if j:
                            ARE.append(j)
                    elif i == 'Subárea':
                        if j:
                            SAR.append(j.strip('.'))
                    elif i == 'Especialidade':
                        if j:
                            ESP.append(j.replace('Especialidade: ','').strip('.'))
            print(f"<pre>        GrandeÁrea: {list(set(GAR))}")
            print(f"             Áreas: {list(set(ARE))}")
            print(f"          Subáreas: {list(set(SAR))}")
            print(f"    Especialidades: {list(set(ESP))}</pre>")
            print(f'\n<h4>ARTIGOS:</h4>')
            
            for dic_art in processed_data.get('articles'):
                valores = list(dic_art.get('subdict_titulos').values())

                title = valores[-1] if valores[-1] != '' else valores[0]
                title_list.append(title)
                title_to_researcher[title] = researcher_name

            # Obter as áreas de atuação do pesquisador
            expert_areas = {
                'GrandeÁrea': list(set(GAR)),
                'Área': list(set(ARE)),
                'Subárea': list(set(SAR)),
                'Especialidade': list(set(ESP))
            }
            areas_of_expertise[processed_data.get('name')] = expert_areas

        # Reset stdout so further print statements go to the console again
        sys.stdout = old_stdout
        # Get the report content as a string
        report_content = report_output.getvalue()
        # Convert the report content to HTML
        html_content = self.convert_to_html(report_content)
        # Save the HTML content to a file
        filename = 'report_fioce_titles.html'
        filepath = os.path.join(self.folder_data_output,filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Close the StringIO object
        report_output.close()
        logging.info("Relatório concluído!")
        logging.info(f"Salvo em: {filepath}")
        
        return title_list, areas_of_expertise, title_to_researcher

    def modelar_topicos_nmf(self, lista_titulos, n_components):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF

        # Combinar stopwords em inglês e português
        stop_words_pt = stopwords.words('portuguese')
        stop_words_en = stopwords.words('english')
        combined_stopwords = stop_words_pt + stop_words_en

        # Pré-processamento e vetorização com TF-IDF
        vectorizer = TfidfVectorizer(stop_words=combined_stopwords)
        X = vectorizer.fit_transform(lista_titulos)

        # Aplicar NMF para modelagem de tópicos
        nmf = NMF(n_components=n_components, random_state=42)
        W = nmf.fit_transform(X)  # W contém os pesos dos tópicos para os documentos

        # Exibir os tópicos e suas principais palavras
        words = vectorizer.get_feature_names_out()
        print(f'\nExtrair palavras-chave por TF-IDF nos títulos de artigo')
        for i, topic in enumerate(nmf.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [words[idx] for idx in top_words_idx]
            print(f"Palavras-chave no Tópico {i+1}: {' '.join(top_words)}")
        print()
        return nmf, W, vectorizer

    def plot_top_words(self, model, feature_names, n_top_words, title):
        fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f'Tópico {topic_idx + 1}', fontdict={'fontsize': 30})
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=20)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=40)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show()

    # # Chamar com o modelo NMF, vetorizador e número de palavras principais a mostrar
    # plot_top_words(nmf, vectorizer.get_feature_names_out(), n_top_words=10, title='Top words for each topic')

    def plot_bubble_chart(self, model, feature_names, n_top_words):
        data = []
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            trace = go.Scatter(
                x=[(topic_idx + 1)] * len(top_features),  # X position for the bubbles
                y=top_features,  # Y position is the top features (words)
                text=top_features,  # Text is the top features (words)
                mode='markers+text',  # Combine markers and text
                marker=dict(
                    size=weights * 100,  # Scale bubble size by weight
                    sizemode='area',
                    sizeref=0.01,
                    sizemin=4,
                    opacity=0.5
                ),
                textposition="bottom center",  # Position the text at the bottom center of the bubble
                name=f'Tópico {topic_idx + 1}'
            )
            data.append(trace)

        layout = go.Layout(
            title='Top words for each topic with NMF',
            xaxis=dict(title='Tópico'),
            yaxis=dict(title='Palavras', automargin=True),  # automargin ajusta automaticamente a margem para se adequar ao texto
            margin=dict(l=150, r=10, b=50, t=30),  # Você pode precisar ajustar esses valores
            hovermode='closest',
            height=2400,  # Aumente a altura da figura
            width=1200,  # Ajuste a largura se necessário
        )

        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            showlegend=False,
            xaxis=dict(tickmode='array', tickvals=list(range(1, model.n_components + 1))),
            yaxis=dict(tickmode='array', tickvals=feature_names)  # Defina explicitamente as marcações do eixo Y se necessário
        )

        # fig.show()
        # graph_div = pyo.plot(fig, include_plotlyjs=False, output_type='div')
        return plot(fig, include_plotlyjs='cdn', output_type='div')

    def criar_relatorio_classificacao_cnpq(self, atribuicoes, feature_names, estrutura_cnpq_flat, nmf):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        if not all(isinstance(item, tuple) and len(item) == 5 for item in atribuicoes):
            raise ValueError("Cada item em 'atribuicoes' deve ser uma tupla com cinco elementos.")

        organizacao = defaultdict(lambda: defaultdict(list))
        for titulo, cod_grande_area, cod_area, nome_grande_area, nome_area in atribuicoes:
            if titulo.strip():
                organizacao[cod_grande_area][cod_area].append(titulo)

        logo_fioce_path = os.path.join(self.folder_assets, 'logo_fioce_75h.png')
        logo_cnpq_path = os.path.join(self.folder_assets, 'logo_cnpq_75h.png')

        # Incorporar a figura de Plotly no HTML
        graph_div = self.plot_bubble_chart(nmf, feature_names, 10)
        
        html_content = f"""
            <div style='text-align: center; margin-top: 0px;'>
                <img src='{logo_fioce_path}' alt='Logotipo Fiocruz' style='max-width: auto; height: 75px;
                <h4 style='text-align: center;'>Coordenação de Pesquisa e Coleções Biológicas da Fiocruz Ceará Relatório do estudo para atribuição automatizada das publicações às Áreas do CNPq</h4>
                <img src='{logo_cnpq_path}' alt='Logotipo CNPq' style='max-width: auto; height: 50px; display: inline; margin-left: 0px;'>
            </div>
            {graph_div}
        """

        for cod_ga, areas in organizacao.items():
            ga_name = self.get_cod_name(cod_ga, estrutura_cnpq_flat)
            html_content += f"<h2>Grande Área: {html.escape(ga_name)} ({cod_ga})</h2>"
            for cod_a, titulos in areas.items():
                a_name = self.get_cod_name(cod_a, estrutura_cnpq_flat)
                html_content += f"<h3>Área: {html.escape(a_name)} ({cod_a})</h3><ul>"
                for titulo in titulos:
                    html_content += f"<li>{html.escape(titulo)}</li>"
                html_content += "</ul>"

        html_content += "</body></html>"

        html_file_path = os.path.join(self.folder_data_output, 'report_fioce_titles_cnpq.html')
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        pdf_file_path = os.path.join(self.folder_data_output, 'report_fioce_titles_cnpq.pdf')
        HTML(html_file_path).write_pdf(pdf_file_path)

        logging.info("Relatório concluído!")
        logging.info(f"Salvo em: {pdf_file_path}")

        return html_content

    def create_cnpq_structure(self):
        import re
        import json
        from cnpq_tree import CNPQtree

        cnpq = CNPQtree(self.base_repo_dir)
        caminho = os.path.join(self.folder_data_input,'cnpq_tabela-areas-conhecimento.pdf')
        df_areas = cnpq.extrair_areas(caminho)

        def contar_marcadores(texto):
            padrao = r'\.00'
            ocorrencias = re.findall(padrao, texto)
            return len(ocorrencias)

        cat_grandeareas={}
        cat_subareas={}
        cat_areas={}
        cat_especialidades={}

        for cod,des in zip(df_areas['Codigo'],df_areas['Descricao']):
            k = contar_marcadores(cod)
            if k==3:
                cat_grandeareas[cod]= des
            elif k==2:
                cat_areas[cod]= des
            elif k==1:
                cat_subareas[cod]= des
            elif k==0:
                cat_especialidades[cod]= des
            else:
                print('Erro na separação')
                print(f'{k} {cod}{des}')

        print(f'{len(cat_grandeareas):4} Grandes Áreas')
        print(f'{len(cat_areas):4} Áreas')
        print(f'{len(cat_subareas):4} Subáreas')
        print(f'{len(cat_especialidades):4} Especialidades')
        print()

        # Criar a estrutura hierárquica em json
        estrutura = {}

        # Montando as grandes áreas
        for ga_code, ga_desc in cat_grandeareas.items():
            estrutura[ga_code] = {"descricao": ga_desc, "areas": {}}

            # Montando as áreas
            for a_code, a_desc in cat_areas.items():
                if a_code.startswith(ga_code.split('.')[0]):
                    estrutura[ga_code]["areas"][a_code] = {"descricao": a_desc, "subareas": {}}

                    # Montando as subáreas
                    for sa_code, sa_desc in cat_subareas.items():
                        if sa_code.startswith(a_code.split('.')[0] + '.' + a_code.split('.')[1]):
                            estrutura[ga_code]["areas"][a_code]["subareas"][sa_code] = {"descricao": sa_desc, "especialidades": {}}

                            # Montando as especialidades
                            for e_code, e_desc in cat_especialidades.items():
                                if e_code.startswith(sa_code.split('.')[0] + '.' + sa_code.split('.')[1] + '.' + sa_code.split('.')[2]):
                                    estrutura[ga_code]["areas"][a_code]["subareas"][sa_code]["especialidades"][e_code] = e_desc

        # Convertendo para JSON
        json_estrutura = json.dumps(estrutura, indent=4)

        return json_estrutura


    def criar_relatorio_classificacao_cnpq_old(self, atribuicoes, feature_names, estrutura_cnpq_flat, nmf):
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Verificar se cada item em atribuicoes é uma tupla com cinco elementos
        for item in atribuicoes:
            if not isinstance(item, tuple) or len(item) != 5:
                raise ValueError("Cada item em 'atribuicoes' deve ser uma tupla com cinco elementos (titulos, grande_area, area, nome_grande_area, nome_area)")

        # Organizar os títulos por Grande Área e Área
        organizacao = defaultdict(lambda: defaultdict(list))
        for titulo, cod_grande_area, cod_area, nome_grande_area, nome_area in atribuicoes:
            # Ignora títulos vazios
            if not titulo.strip():
                continue
            organizacao[cod_grande_area][cod_area].append(titulo)

        # Iniciar a string HTML
        logo_fioce_path = os.path.join(self.folder_assets,'logo_fioce_75h.png')
        logo_cnpq_path = os.path.join(self.folder_assets,'logo_cnpq_75h.png')
        html_content = (
            "<div style='display: flex; justify-content: space-between; align-items: center;'>"
            f"<img src={logo_fioce_path} alt='Logotipo Fiocruz' width='300px'; height: auto;>"
            "<h2 style='flex-grow: 1; text-align: center;'>Coordenação de Pesquisa e Coleções Biológicas da Fiocruz Ceará<br/>Relatório do estudo para atribuição automatizada<br/>das publicações às Áreas do CNPq</h2>"
            f"<img src={logo_cnpq_path} alt='Logotipo CNPq' width: auto; height='90px;'>"
            "</div>"
        )
        # Incorporar a figura de Plotly no HTML
        n_top_words = 10

        # Gerar o div do gráfico
        graph_div = self.plot_bubble_chart(nmf, feature_names, n_top_words)

        # Inserir o div do gráfico no conteúdo HTML
        html_content += graph_div  # Adicione o gráfico ao conteúdo HTML

        for cod_ga, areas in organizacao.items():
            ga_name = self.get_cod_name(cod_ga, estrutura_cnpq_flat)
            html_content += f"<h2>Grande Área: {html.escape(ga_name)} ({cod_ga})</h2>"
            for cod_a, titulos in areas.items():
                a_name = self.get_cod_name(cod_a, estrutura_cnpq_flat)
                html_content += f"<h3>Área: {html.escape(a_name)} ({cod_a})</h3><ul>"
                for titulo in titulos:
                    html_content += f"<li>{html.escape(titulo)}</li>"
                html_content += "</ul>"

        html_content += "</body></html>"

        # Salvar o conteúdo HTML em um arquivo
        html_file_path = os.path.join(self.folder_data_output, 'report_fioce_titles_cnpq.html')
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Convertendo o relatório HTML para PDF
        pdf_file_path = os.path.join(self.folder_data_output, 'report_fioce_titles_cnpq.pdf')
        HTML(html_file_path).write_pdf(pdf_file_path)

        logging.info("Relatório concluído!")
        logging.info(f"Salvo em: {pdf_file_path}")

        return html_content

    def generate_fioce_report_html(self, filename):
        from io import StringIO
        import sys

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        old_stdout = sys.stdout
        sys.stdout = report_output = StringIO()
        separator = 120

        # Caminho para os logotipos
        logo_fioce_path = os.path.join(self.folder_assets, 'logo_fioce_75h.png')
        logo_cnpq_path = os.path.join(self.folder_assets, 'logo_cnpq_75h.png')

        # Inserindo logotipo no início do conteúdo HTML
        print(f"<div style='text-align: center;'>")
        print(f"<img src='{logo_fioce_path}' alt='Logotipo Fiocruz' width='auto' height='75'>")
        print(f"<img src='{logo_cnpq_path}' alt='Logotipo CNPq' width='auto' height='75'>")
        print(f"</div>")
        
        print("<h1><center><b>Produção em artigos de pesquisa da Fiocruz Ceará</b></center></h1>")
        logging.info("Obtendo os dados do dataset de artigos, aguarde...")

        dataset_articles = jfm.load_json(self.folder_data_output, filename)
        print(f"<h2><center><b>{len(dataset_articles)} currículos extraídos</b></center></h2>")
        
        for curriculo in dataset_articles:
            processed_data = curriculo.get('processed_data')
            print(f"<center>{'-' * separator}</center>")
            # print(f"<h2><center><b>{processed_data.get('name')}</b></center></h2>")
            # print(f"<h4>ÁREAS CNPq:</h4>")
            GAR=[]
            ARE=[]
            SAR=[]
            ESP=[]
            for classificacao in processed_data.get('areas_of_expertise'):
                for i,j in classificacao.items():
                    if i == 'GrandeÁrea':
                        if j:
                            GAR.append(j)
                    elif i == 'Área':
                        if j:
                            ARE.append(j)
                    elif i == 'Subárea':
                        if j:
                            SAR.append(j.strip('.'))
                    elif i == 'Especialidade':
                        if j:
                            ESP.append(j.replace('Especialidade: ','').strip('.'))
            print(f"<pre>        GrandeÁrea: {list(set(GAR))}")
            print(f"             Áreas: {list(set(ARE))}")
            print(f"          Subáreas: {list(set(SAR))}")
            print(f"    Especialidades: {list(set(ESP))}</pre>")
            print(f'\n<h4>ARTIGOS:</h4>')
            for dic_art in processed_data.get('articles'):
                valores = list(dic_art.get('subdict_titulos').values())
                # ultimo_valor = next(reversed(dic_art.get('subdict_titulos').items()))
                ultimo_valor = valores[-1]
                if ultimo_valor != '':
                    print(f"{valores[-1]}")
                else:
                    print(f"{valores[0]}")
        # Reset stdout so further print statements go to the console again
        sys.stdout = old_stdout
        # Get the report content as a string
        report_content = report_output.getvalue()
        # Convert the report content to HTML
        html_content = self.convert_to_html(report_content)
        # Save the HTML content to a file
        filename = 'report_fioce_research.html'
        filepath = os.path.join(self.folder_data_output,filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Close the StringIO object
        report_output.close()
        logging.info("Relatório concluído!")
        logging.info(f"Salvo em: {filepath}")
    
    def compare_keys(self, n, dataset_articles, dict_obtained):
        error_list=[]
        keys_l1_expected = set({'processed_data', 'monitoring_data'})
        keys_l2_expected = set({'areas_of_expertise', 'profiling_data', 'name', 'id_lattes', 'articles'})
        keys_l3_expected = set({'subdict_years', 'subdict_areas', 'subdict_autores', 'subdict_titulos', 'subdict_jci', 'subdict_doi'})    
        keys_obtained=set(dict_obtained.keys())
        if keys_obtained == keys_l1_expected:
            pass
            # print(f"Dic {n+1:>02}/{len(dataset_articles)}: Chaves de nível 1 conferidas com sucesso!")
            # print(keys_obtained)
        elif keys_obtained == keys_l2_expected:
            pass
            # print(f"Dic {n+1:>02}/{len(dataset_articles)}: Chaves de nível 2 conferidas com sucesso!")
            # print(keys_obtained)
        elif keys_obtained == keys_l3_expected:
            pass
            # print(f"Dic {n+1:>02}/{len(dataset_articles)}: Chaves de nível 3 conferidas com sucesso!")
            # print(keys_obtained)
        else:
            print(f"Erro nas chaves do dicionário {n}/{len(dataset_articles)}")
            error_list.append(n)
        
        # print(f"Dic {n+1:>02}/{len(dataset_articles)}: Chaves de nível 1,2 e 3 conferidas com sucesso!")
            
        return error_list

    def check_keys(self, filename):
        # Carregar dataset JSON
        dataset_articles = jfm.load_json(self.folder_data_output,filename)
        print(f"{len(dataset_articles)} dicionários de currículos carregados do dataset")
        qty_art = 0
        global_error = []
        success = 0
        for n,curriculo in enumerate(dataset_articles):
            # comparar chaves a nível da estrutura de dados
            err1 = self.compare_keys(n, dataset_articles, curriculo)
            if err1:
                global_error.append(err1)
            else:
                success+=1
            # comparar chaves a nível de currículo
            processed_data = curriculo.get('processed_data')
            err2 = self.compare_keys(n, dataset_articles, processed_data)
            if err2:
                global_error.append(err2)
            else:
                success+=1
            # comparar chaves a nível de dados de artigos
            for dic_art in processed_data.get('articles'):
                qty_art+=len(processed_data.get('articles'))
                err3 = self.compare_keys(n, dataset_articles, dic_art)
                if err3:
                    global_error.append(err3)
                else:
                    success+=1
            # time.sleep(0.05)  
            # clear_output(wait=True)
            
        print(f"{len(dataset_articles):>6} Chaves de dicionários de currículos analisados")
        print(f"{success:>6} análise com êxito, {len(global_error):>6} erros identificados")
        print(f"{qty_art:>6} Chaves de dicionários de artigos analisados")
        
        return global_error