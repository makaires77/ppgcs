import csv
import json
import zipfile
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from io import BytesIO, TextIOWrapper
from IPython.display import clear_output
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

"""
Parâmetros de consulta
Diferentes tipos de parâmetro são usados em uma consulta para determinar a tabela e suas dimensões, que podem ser especificados por uma lista de valores
ou através de identificadores especiais (ex: /all, /last, entre outros). 

Estes parâmetros podem ser resumidos da seguinte forma:
    t (table_code) - é o código da tabela referente ao indicador e a pesquisa;
    p (period) - utilizado para especificar o período;
    v (variable) - para especificar as variáveis desejadas;
    n (territorial_level) - especifica os níveis territoriais;
    n/ (ibge_territorial_code) - inserido dentro do nível territorial, especificar o código territorial do IBGE;
    c/ (classification/categories) - especifica as classificações da tabela e suas respectivas categorias.

Para obter a tabela e os códigos, entrar na interface do SIDRA e buscar a pesquisa/indicador de interesse no buscador https://sidra.ibge.gov.br/home/

https://sidra.ibge.gov.br/pesquisa/snipc/ipca/tabelas/brasil/maio-2024
https://apisidra.ibge.gov.br/home/ajuda#Exemplos

Número	Nome	Período (dezembro 1979 a maio 2024)	Território(BR)
1737	IPCA - Série histórica com número-índice, variação mensal e variações acumuladas em 3 meses, em 6 meses, no ano e em 12 meses (a partir de dezembro/1979)
7060	IPCA - Variação mensal, acumulada no ano, acumulada em 12 meses e peso mensal, para o índice geral, grupos, subgrupos, itens e subitens de produtos e serviços (a partir de janeiro/2020)	Período(janeiro 2020 a maio 2024)	Território (BR, CM, RM, MU)
118	    IPCA dessazonalizado - Série histórica com a variação mensal (a partir de janeiro/1998)	janeiro 1998 a maio 2024	BR
7061	IPCA dessazonalizado - Variação mensal, acumulada no ano e peso mensal, para o índice geral, grupos, subgrupos, itens e subitens de produtos e serviços (a partir de janeiro/2020)	janeiro 2020 a maio 2024	BR, CM, RM, MU

Uma tabela Sidra contém no mínimo 3 dimensões básicas (períodos, variáveis e unidades territoriais), além de até 6 classificações, total de 9 dimensões.
Ex: Tabela 1737 - IPCA - Série histórica com número-índice, variação mensal e variações acumuladas em 3 meses, em 6 meses, no ano e em 12 meses (a partir de dezembro/1979)

Variáveis (/v/)
2266	IPCA - Número-índice (base: dezembro de 1993 = 100)
63	    IPCA - Variação mensal
2263	IPCA - Variação acumulada em 3 meses
2264	IPCA - Variação acumulada em 6 meses
69	    IPCA - Variação acumulada no ano
2265	IPCA - Variação acumulada em 12 meses

Nível Territorial (/n1/):
/n1/ Brasil(1)

Exemplo: /t/1612/n2/all/v/all/p/last/c81/2702/f/u
"""

class IPCAData:
    def __init__(self, anos=[2010, 2024]):
        self.anos = anos
        # self.dados_ipca = self.obter_dados_ipca()
        # print(self.dados_ipca)

    def obter_numeros_indice(self):
        """
        Obtém os números de índice mensais do IPCA e retorna em formato JSON.

        Returns:
            str: String JSON com os números de índice mensais do IPCA.
        """

        url_tab_var = "https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/2266/p/all"
        indices_mensais = {}

        try:
            response = requests.get(url_tab_var)
            response.raise_for_status()
            meses=[]
            indices=[]
            # Ler o CSV com TextIOWrapper
            with TextIOWrapper(BytesIO(response.content), encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=";")
                next(reader)  # Pula o cabeçalho

                for row in reader:
                    if len(row) > 0:
                        linha_limpa = row[0].strip()

                        # Remover caracteres problemáticos
                        linha_limpa = linha_limpa.replace("\"\"", '"')

                        if "D3C" in linha_limpa:
                            meses.append(linha_limpa)
                        if "\"V\":" in linha_limpa:
                            indices.append(linha_limpa)

            return meses, indices

            # # Criar DataFrame a partir do dicionário
            # df = pd.DataFrame.from_dict(indices_mensais, orient='index', columns=['Valor'])
            # df.index.name = 'mes_codigo'

            # # Converter índice para datetime
            # df.index = pd.to_datetime(df.index, format="%Y-%m")

            # # Converter o DataFrame para JSON
            # json_data = df.to_json(orient="index", date_format="iso")
            # return json_data

        except requests.exceptions.RequestException as e:
            print(f"Erro na requisição: {e}")
        except Exception as e:
            print(f"Erro ao processar dados: {e}")
        return None

    def obter_relatorios_remuneracao(self):

        return df_remuneracoes

    def obter_dados_ipca(self):
        """
        Exemplo: https://apisidra.ibge.gov.br/values/<id_1>/<val_1>/<id_2>/<val_2>/...

        /V/  Variável(6):
        2266  IPCA - Número-índice (base: dezembro de 1993 = 100) (Número-índice) - casas decimais: padrão = 13, máximo = 13
        63    IPCA - Variação mensal (%) [janeiro 1980 a maio 2024] - casas decimais: padrão = 2, máximo = 2
        2263  IPCA - Variação acumulada em 3 meses (%) [março 1980 a maio 2024] - casas decimais: padrão = 2, máximo = 2
        2264  IPCA - Variação acumulada em 6 meses (%) [junho 1980 a maio 2024] - casas decimais: padrão = 2, máximo = 2
        69    IPCA - Variação acumulada no ano (%) [janeiro 1980 a maio 2024] - casas decimais: padrão = 2, máximo = 2
        2265  IPCA - Variação acumulada em 12 meses (%) [dezembro 1980 a maio 2024] - casas decimais: padrão = 2, máximo = 2

        F – para especificar o formato dos campos apresentados no resultado
        Especifique /f/a para receber os códigos e os nomes dos descritores (valor padrão, caso o parâmetro f não seja especificado).
        Especifique /f/c para receber apenas os códigos dos descritores.
        Especifique /f/n para receber apenas os nomes dos descritores.
        Especifique /f/u para receber o código e o nome das unidades territoriais consultadas, e o nome dos demais descritores.

        D – para especificar com quantas casas decimais serão formatados os valores numéricos
        Especifique /d/s para formatar os valores com o número de casas decimais padrão para cada variável (valor default, caso o parâmetro d não seja especificado).
        Especifique /d/m para formatar os valores com o número de casas decimais máximo disponível para cada variável (maior precisão).
        Especifique /d/0 a /d/9 para formatar os valores com um número fixo de casas decimais, entre 0 e 9.        

        Retorna todos os números índice mensais disponíveis de 1979 até o mais recente
        /t/1737/n1/all/v/2266/p/all
        """

        url_base = "https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/2266/p/"
        url_final = "/f/a"
        todos_dados = []  # Lista para armazenar os dados de cada ano

        for ano in range(self.anos[0], self.anos[-1] + 1):
            for mes in ['01','02','03','04','05','06','07','08','09','10','11','12']:
                url_ano = url_base + str(ano) + mes + url_final
                print(url_ano)
                response = requests.get(url_ano)
                response.raise_for_status()
                try:
                    todos_dados.append(response)
                except:
                    pass
                # try:
                #     with zipfile.ZipFile(BytesIO(response.content)) as z:
                #         with z.open(z.namelist()[0]) as f:
                #             df = pd.read_csv(f, sep=";", decimal=",")

                #     df = df[["Mês (Código)", "Mês", "Variável", "Valor"]]
                #     df.columns = ["mes_codigo", "mes", "variavel", "valor"]
                #     df["mes_codigo"] = pd.to_datetime(df["mes_codigo"], format="%Y%m")
                #     df["mes"] = df["mes"].astype(str)
                #     df["valor"] = pd.to_numeric(df["valor"])

                #     df_pivot = df.pivot(index="mes_codigo", columns="variavel", values="valor")
                #     df_pivot = df_pivot[["IPCA - Variação acumulada em 12 meses", "IPCA - Variação mensal"]]
                #     todos_dados.append(df_pivot)
                # except Exception as e:
                #     print(f"Erro ao processar dados do IPCA para o ano {ano}: {e}")

        # Concatenar todos os DataFrames em um só
        dados_ipca = pd.concat(todos_dados)
        return dados_ipca

    def obter_dados_ipca(self):
        url_base = "https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/2266/p/"
        url_final = "/f/all/d/v2266%201"
        todos_dados = []  # Lista para armazenar os dados de cada ano

        for ano in range(self.anos[0], self.anos[1] + 1):
            url_ano = url_base + str(ano) + url_final
            response = requests.get(url_ano)
            response.raise_for_status()

            try:
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    with z.open(z.namelist()[0]) as f:
                        df = pd.read_csv(f, sep=";", decimal=",")

                df = df[["Mês (Código)", "Mês", "Variável", "Valor"]]
                df.columns = ["mes_codigo", "mes", "variavel", "valor"]
                df["mes_codigo"] = pd.to_datetime(df["mes_codigo"], format="%Y%m")
                df["mes"] = df["mes"].astype(str)
                df["valor"] = pd.to_numeric(df["valor"])

                df_pivot = df.pivot(index="mes_codigo", columns="variavel", values="valor")
                df_pivot = df_pivot[["IPCA - Variação acumulada em 12 meses", "IPCA - Variação mensal"]]
                todos_dados.append(df_pivot)  # Adicionar dados do ano à lista
            except Exception as e:
                print(f"Erro ao processar dados do IPCA para o ano {ano}: {e}")

        # Concatenar todos os DataFrames em um só
        dados_ipca = pd.concat(todos_dados)
        return dados_ipca

class PoderCompra:
    def __init__(self, dados_ipca):
        self.dados_ipca = dados_ipca

    def calcular_perdas(self, data_base="2010-01-01"):
        dados_filtrados = self.dados_ipca[self.dados_ipca.index >= data_base]
        inflacao_acumulada = (1 + dados_filtrados["IPCA - Variação mensal"] / 100).cumprod()
        perdas = 100 * (1 - 1 / inflacao_acumulada)
        return perdas

    def calcular_recomposicao(self, carreiras, data_base="2010-01-01"):
        perdas_poder_compra = self.calcular_perdas(data_base)
        recomposicao = pd.DataFrame(index=perdas_poder_compra.index)
        for nome, ultimo_reajuste in carreiras.items():
            perdas_desde_reajuste = perdas_poder_compra[perdas_poder_compra.index > ultimo_reajuste]
            recomposicao[nome] = perdas_desde_reajuste
        return recomposicao

class PrevisaoInflacao:
    def __init__(self, dados_ipca):
        self.dados_ipca = dados_ipca

    def prever(self, meses_previsao=12, modelo="arima"):
        if modelo == "arima":
            modelo_arima = ARIMA(self.dados_ipca["IPCA - Variação mensal"], order=(1, 1, 1))
            resultado_arima = modelo_arima.fit()
            previsoes = resultado_arima.forecast(steps=meses_previsao)
            mse = mean_squared_error(self.dados_ipca["IPCA - Variação mensal"], resultado_arima.fittedvalues)
            print(f"MSE do modelo ARIMA: {mse}")
        else:
            raise ValueError("Modelo inválido. Escolha 'arima' ou outro modelo.")

        ultimo_mes = self.dados_ipca.index[-1]
        datas_previsao = pd.date_range(ultimo_mes + pd.DateOffset(months=1), periods=meses_previsao, freq="MS")
        previsoes = pd.DataFrame({"IPCA - Variação mensal": previsoes}, index=datas_previsao)
        return previsoes

class Visualizacao:
    @staticmethod
    def plotar_evolucao_poder_compra(dados_ipca, perdas_poder_compra):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dados_ipca.index, y=dados_ipca["IPCA - Variação acumulada em 12 meses"], name="IPCA Acumulado", mode="lines", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=perdas_poder_compra.index, y=perdas_poder_compra, name="Perda Poder de Compra", mode="lines", line=dict(color="red")))
        fig.update_layout(title="Evolução do Poder de Compra e IPCA Acumulado", xaxis_title="Data", yaxis_title="Variação (%)", legend=dict(x=0, y=1), hovermode="x unified")
        fig.show()

    @staticmethod
    def plotar_defasagem_recomposicao(recomposicao_salarial):
        fig = px.line(recomposicao_salarial, title="Defasagem e Recomposição Salarial")
        fig.update_layout(xaxis_title="Data", yaxis_title="Recomposição (%)", legend_title="Carreira")
        fig.show()