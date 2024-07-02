import zipfile
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error



def obter_dados_ipca(anos=[2010, 2024]):
    """
    Baixa dados históricos do IPCA do SIDRA/API do Banco Central.

    Args:
        anos (list): Lista com o ano inicial e final dos dados desejados.

    Returns:
        pd.DataFrame: DataFrame com os dados do IPCA.
    """

    url_base = "https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/2266/p/"
    url_anos = "/".join(str(ano) for ano in range(anos[0], anos[1] + 1))
    url_completa = url_base + url_anos + "/f/all/d/v2266%201"
    print(url_completa)

    response = requests.get(url_completa)
    print(response)
    response.raise_for_status()  # Verificar se a requisição foi bem-sucedida

    try:
        # Extrair dados do arquivo ZIP
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, sep=";", decimal=",")

        # Selecionar colunas relevantes e renomear
        df = df[["Mês (Código)", "Mês", "Variável", "Valor"]]
        df.columns = ["mes_codigo", "mes", "variavel", "valor"]

        # Converter colunas para os tipos corretos
        df["mes_codigo"] = pd.to_datetime(df["mes_codigo"], format="%Y%m")
        df["mes"] = df["mes"].astype(str)
        df["valor"] = pd.to_numeric(df["valor"])

        # Pivotar o DataFrame para ter os meses como índice e as variáveis como colunas
        df_pivot = df.pivot(index="mes_codigo", columns="variavel", values="valor")

        # Reordenar colunas para ter o IPCA acumulado no ano como primeira coluna
        df_pivot = df_pivot[["IPCA - Variação acumulada em 12 meses", "IPCA - Variação mensal"]]

        return df_pivot
    except Exception as e:
        print(f"Erro ao processar dados do IPCA: {e}")
        return None


def calcular_perdas_poder_compra(dados_ipca, data_base):
    """
    Calcula as perdas no poder de compra da moeda ao longo do tempo.

    Args:
        dados_ipca (pd.DataFrame): DataFrame com os dados do IPCA (obtido de ipca_data.py).
        data_base (str ou pd.Timestamp): Data de referência para o cálculo das perdas.

    Returns:
        pd.Series: Série com as perdas acumuladas no poder de compra para cada mês.
    """

    # Filtrar dados a partir da data base
    dados_filtrados = dados_ipca[dados_ipca.index >= data_base]

    # Calcular inflação acumulada desde a data base
    inflacao_acumulada = (1 + dados_filtrados["IPCA - Variação mensal"] / 100).cumprod()

    # Calcular perdas no poder de compra
    perdas = 100 * (1 - 1 / inflacao_acumulada)

    return perdas

def calcular_recomposicao_salarial(dados_ipca, carreiras, data_base):
    """
    Calcula a recomposição salarial necessária para diferentes carreiras de servidores públicos.

    Args:
        dados_ipca (pd.DataFrame): DataFrame com os dados do IPCA (obtido de ipca_data.py).
        carreiras (dict): Dicionário com as informações das carreiras (nome: último reajuste).
        data_base (str ou pd.Timestamp): Data de referência para o cálculo.

    Returns:
        pd.DataFrame: DataFrame com a recomposição salarial para cada carreira e mês.
    """

    perdas_poder_compra = calcular_perdas_poder_compra(dados_ipca, data_base)

    recomposicao = pd.DataFrame(index=perdas_poder_compra.index)
    for nome, ultimo_reajuste in carreiras.items():
        perdas_desde_reajuste = perdas_poder_compra[perdas_poder_compra.index > ultimo_reajuste]
        recomposicao[nome] = perdas_desde_reajuste

    return recomposicao

def analisar_poder_compra(dados_ipca, inflacao_futura=None, carreiras=None, data_base="2010-01-01"):
    """
    Realiza a análise completa do poder de compra, incluindo perdas e recomposição salarial.

    Args:
        dados_ipca (pd.DataFrame): DataFrame com os dados do IPCA (obtido de ipca_data.py).
        inflacao_futura (pd.Series, optional): Série com previsões de inflação futura.
        carreiras (dict, optional): Dicionário com as informações das carreiras.
        data_base (str ou pd.Timestamp, optional): Data de referência para o cálculo.

    Returns:
        tuple: Tupla com dois DataFrames:
            - perdas_poder_compra: Perdas acumuladas no poder de compra.
            - recomposicao_salarial (opcional): Recomposição salarial para cada carreira.
    """

    perdas_poder_compra = calcular_perdas_poder_compra(dados_ipca, data_base)

    if inflacao_futura is not None:
        perdas_futuras = calcular_perdas_poder_compra(inflacao_futura, perdas_poder_compra.index[-1])
        perdas_poder_compra = pd.concat([perdas_poder_compra, perdas_futuras])

    recomposicao_salarial = None
    if carreiras is not None:
        recomposicao_salarial = calcular_recomposicao_salarial(dados_ipca, carreiras, data_base)

    return perdas_poder_compra, recomposicao_salarial


def prever_inflacao(dados_ipca, meses_previsao=12, modelo="arima"):
    """
    Realiza a previsão da inflação futura utilizando diferentes modelos.

    Args:
        dados_ipca (pd.DataFrame): DataFrame com os dados do IPCA (obtido de ipca_data.py).
        meses_previsao (int, optional): Número de meses para prever a inflação.
        modelo (str, optional): Modelo a ser utilizado ("arima" ou "outro_modelo").

    Returns:
        pd.Series: Série com as previsões de inflação para os próximos meses.
    """

    if modelo == "arima":
        # Previsão com ARIMA
        modelo_arima = ARIMA(dados_ipca["IPCA - Variação mensal"], order=(1, 1, 1))
        resultado_arima = modelo_arima.fit()
        previsoes = resultado_arima.forecast(steps=meses_previsao)

        # Avaliação do modelo (opcional)
        mse = mean_squared_error(dados_ipca["IPCA - Variação mensal"], resultado_arima.fittedvalues)
        print(f"MSE do modelo ARIMA: {mse}")

    elif modelo == "outro_modelo":
        # Implementação de outro modelo de previsão (ex: SARIMA, modelos de Machine Learning)
        # ... (código para o outro modelo)
        pass

    else:
        raise ValueError("Modelo inválido. Escolha 'arima' ou 'outro_modelo'.")

    # Criar índice de datas para as previsões
    ultimo_mes = dados_ipca.index[-1]
    datas_previsao = pd.date_range(ultimo_mes + pd.DateOffset(months=1), periods=meses_previsao, freq="MS")

    # Converter previsões para DataFrame
    previsoes = pd.DataFrame({"IPCA - Variação mensal": previsoes}, index=datas_previsao)

    return previsoes


def plotar_evolucao_poder_compra(dados_ipca, perdas_poder_compra):
    """
    Plota a evolução do poder de compra da moeda ao longo do tempo.

    Args:
        dados_ipca (pd.DataFrame): DataFrame com os dados do IPCA (obtido de ipca_data.py).
        perdas_poder_compra (pd.Series): Série com as perdas acumuladas no poder de compra.
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dados_ipca.index,
        y=dados_ipca["IPCA - Variação acumulada em 12 meses"],
        name="IPCA Acumulado",
        mode="lines",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=perdas_poder_compra.index,
        y=perdas_poder_compra,
        name="Perda Poder de Compra",
        mode="lines",
        line=dict(color="red")
    ))

    fig.update_layout(
        title="Evolução do Poder de Compra e IPCA Acumulado",
        xaxis_title="Data",
        yaxis_title="Variação (%)",
        legend=dict(x=0, y=1),
        hovermode="x unified"
    )

    fig.show()

def plotar_defasagem_recomposicao(recomposicao_salarial):
    """
    Plota a defasagem e a recomposição salarial para cada carreira.

    Args:
        recomposicao_salarial (pd.DataFrame): DataFrame com a recomposição salarial para cada carreira.
    """

    fig = px.line(recomposicao_salarial, title="Defasagem e Recomposição Salarial")
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Recomposição (%)",
        legend_title="Carreira"
    )

    fig.show()