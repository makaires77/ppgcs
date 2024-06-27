# alt.renderers.enable('notebook')
import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, hamming_loss, roc_curve, auc

def calcular_curva_roc_auc(y_true, y_proba):
    """Calcula curva ROC e AUC ROC para cada classe em classificação multirrótulo."""
    mlb = MultiLabelBinarizer()
    mlb.fit(y_true)
    y_true_bin = mlb.transform(y_true)

    # Converter y_true_bin e y_proba para arrays NumPy
    y_true_bin = np.array(y_true_bin)
    y_proba = np.array(y_proba)

    roc_auc = {}
    fpr = {}
    tpr = {}

    # Calcular micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Calcular as curvas ROC para cada classe
    for i in range(y_true_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcular macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_true_bin.shape[1])]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(y_true_bin.shape[1]):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= y_true_bin.shape[1]

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc, fpr, tpr


def calcular_metricas(y_true, y_pred):
    """
    Calcula métricas de desempenho para classificação multirrótulo.

    Args:
        y_true: Lista ou array de rótulos verdadeiros (multirrótulos).
        y_pred: Lista ou array de rótulos preditos (multirrótulos).

    Returns:
        Um dicionário com os nomes das métricas como chaves e os valores calculados como valores.
    """

    # Converter multirrótulos para formato binário
    mlb = MultiLabelBinarizer()
    mlb.fit(y_true)
    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    # Micro-average e macro-average de precisão, recall e F1-score
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average='macro')

    # Hamming Loss
    hamming = hamming_loss(y_true_bin, y_pred_bin)

    # Precision@k e Recall@k (para k = 1, 3 e 5) - Cálculo manual
    precision_at_k = {}
    recall_at_k = {}
    for k in [1, 3, 5]:
        precision_at_k[f'Precision@{k}'] = np.mean([np.sum(np.isin(np.argsort(p_scores)[::-1][:k], t_scores)) / k 
                                    for t_scores, p_scores in zip(y_true_bin, y_pred_bin)])
        recall_at_k[f'Recall@{k}'] = np.mean([np.sum(np.isin(np.argsort(p_scores)[::-1][:k], t_scores)) / len(t_scores) 
                                  for t_scores, p_scores in zip(y_true_bin, y_pred_bin)])

    # Acurácia (todos os rótulos preditos corretamente)
    acuracia = np.mean([np.array_equal(t, p) for t, p in zip(y_true_bin, y_pred_bin)])

    # Dicionário com as métricas
    metricas = {
        'Precision (micro)': precision_micro,
        'Recall (micro)': recall_micro,
        'F1-score (micro)': f1_micro,
        'Precision (macro)': precision_macro,
        'Recall (macro)': recall_macro,
        'F1-score (macro)': f1_macro,
        'Hamming Loss': hamming,
        'Acurácia': acuracia
    }

    # Adicionar as métricas precision@k e recall@k individualmente
    metricas.update(precision_at_k)
    metricas.update(recall_at_k)

    return metricas


def plotar_matriz_confusao(y_true, y_pred, classes):
    """Plota a matriz de confusão para classificação multirrótulo."""

    # Converter multirrótulos em formato binário
    y_true_bin = np.array([[1 if c in t else 0 for c in classes] for t in y_true])
    y_pred_bin = np.array([[1 if c in p else 0 for c in classes] for p in y_pred])

    # Calcular a matriz de confusão
    conf_mat = np.dot(y_true_bin.T, y_pred_bin)

    # Plotar a matriz de confusão com largura definida
    plt.figure(figsize=(12, 6))  # Ajuste para largura e altura
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão")
    plt.show()

    # Create a DataFrame from the confusion matrix and classes
    df = pd.DataFrame(conf_mat, index=classes, columns=classes)

    # Melt the DataFrame to create a long format for Altair
    df_melted = df.reset_index().melt(id_vars='index', var_name='Predito', value_name='Count')

    # Create the heatmap using Altair
    chart = alt.Chart(df_melted,title = "Matriz de Confusão").mark_rect().encode(
        x=alt.X('Predito:O', axis=alt.Axis(title='Predito')),
        y=alt.Y('index:O', axis=alt.Axis(title='Verdadeiro')),
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['Predito', 'index', 'Count']
    ).interactive()

    # Show the chart
    # chart.save('matriz_confusao.png')

# def plotar_metricas(metricas):
#     """
#     Plota um gráfico de barras comparando métricas micro e macro.

#     Args:
#         metricas: Dicionário com os nomes das métricas como chaves e os valores calculados como valores.
#     """

#     # Selecionar as métricas desejadas
#     metricas_selecionadas = {k: v for k, v in metricas.items() if k in ['Precision (micro)', 'Recall (micro)', 'F1-score (micro)', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)']}

#     # Criar um DataFrame para o gráfico
#     df_metricas = pd.DataFrame(metricas_selecionadas, index=["Micro", "Macro"]).T.reset_index()
#     df_metricas = pd.melt(df_metricas, id_vars="index", var_name="Tipo", value_name="Valor")

#     # Plotar o gráfico de barras
#     chart = alt.Chart(df_metricas,title = "Comparação de Métricas Micro e Macro").mark_bar().encode(
#         x=alt.X('index:N', axis=alt.Axis(title='Métrica')),
#         y=alt.Y('Valor:Q', axis=alt.Axis(title='Valor')),
#         column=alt.Column('Tipo:N', title=''),
#         color='Tipo:N',
#         tooltip=['index', 'Valor', 'Tipo']
#     ).properties(
#         width=400  # Defina a largura desejada (em pixels)
#     ).interactive()
    
#     # Exibir o gráfico
#     display(chart)

#     # Save the chart
#     chart.save('metricas_micro_macro.json')

# def plotar_curvas_roc(roc_auc, fpr, tpr, classes):
#     """Plota as curvas ROC para cada classe e as médias micro e macro."""
#     plt.figure(figsize=(10, 8))  # Largura de 12 polegadas, altura de 8 polegadas

#     # Create a DataFrame for Altair
#     data = []
#     for i in range(len(classes)):
#         data.extend([{"Classe": classes[i], "FPR": x, "TPR": y} for x, y in zip(fpr[i], tpr[i])])
#     data.extend([{"Classe": "micro-average", "FPR": x, "TPR": y} for x, y in zip(fpr["micro"], tpr["micro"])])
#     data.extend([{"Classe": "macro-average", "FPR": x, "TPR": y} for x, y in zip(fpr["macro"], tpr["macro"])])
#     df = pd.DataFrame(data)

#     # Create the ROC curve chart using Altair
#     chart = alt.Chart(df, title="Curva ROC multi-classe").mark_line(point=True).encode(
#         x=alt.X('FPR:Q', title='Taxa de Falsos Positivos'),
#         y=alt.Y('TPR:Q', title='Taxa de Verdadeiros Positivos'),
#         color='Classe:N',
#         tooltip=['Classe', 'FPR', 'TPR']
#     ).properties(
#         width=400  # Largura em pixels
#         # width=alt.Step(12)  # Largura em polegadas (12 polegadas = 304.8 pixels)
#     )

#     # Add diagonal reference line
#     rule = alt.Chart(pd.DataFrame({'y': [0, 1]})).mark_rule(color='gray').encode(
#         y='y:Q',
#     )

#     # Combine the ROC curve and the reference line
#     final_chart = chart + rule

#     # Show the chart
#     final_chart.show()

def plotar_metricas(metricas):
    """
    Plota um gráfico de barras comparando métricas micro e macro com rótulos de dados.

    Args:
        metricas: Dicionário com os nomes das métricas como chaves e os valores calculados como valores.
    """

    # Selecionar as métricas desejadas
    metricas_selecionadas = {k: v for k, v in metricas.items() if k in ['Precision (micro)', 'Recall (micro)', 'F1-score (micro)', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)']}

    # Criar um DataFrame para o gráfico
    df_metricas = pd.DataFrame(metricas_selecionadas, index=["Micro", "Macro"]).T.reset_index()
    df_metricas = pd.melt(df_metricas, id_vars="index", var_name="Tipo", value_name="Valor")

    # Plotar o gráfico de barras com largura definida
    base = alt.Chart(df_metricas, title="Comparação de Métricas Micro e Macro").encode(
        x=alt.X('index:N', axis=alt.Axis(title='Métrica')),
        y=alt.Y('Valor:Q', axis=alt.Axis(title='Valor')),
        color='Tipo:N',
        tooltip=['index', 'Valor', 'Tipo']
        ).properties(
        width=308  # Defina a largura desejada (em pixels)
    )

    # Desenhar as barras
    barras = base.mark_bar()

    # Adicionar rótulos de dados
    texto = base.mark_text(
        align='center',
        baseline='bottom',
        dy=-5,  # Nudge text above the bars
    ).encode(
        text=alt.Text('Valor:Q', format='.4f')
    )

    # Combinar as barras e os rótulos de dados
    grafico_com_texto = barras + texto

    # Aplica o facetamento na camada combinada com espaçamento ajustado
    grafico_final = grafico_com_texto.facet(
        column=alt.Column('Tipo:N', title=''),
        spacing=200  # Ajuste o espaçamento para controlar a largura das colunas
    ).interactive()
    
    # Exibir o gráfico
    grafico_final.show()

    # Salvar o gráfico em um arquivo JSON
    grafico_final.save('metricas_micro_macro.json')

def plotar_curvas_roc(roc_auc, fpr, tpr, classes):
    """Plota as curvas ROC para cada classe e as médias micro e macro."""
    plt.figure(figsize=(10.8, 8))  # Largura de 10 polegadas, altura de 8 polegadas

    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--')  # Linha diagonal tracejada
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC multi-classe')
    plt.legend(loc="lower right")
    plt.show()

def validar_modelo(y_true, y_pred, y_proba, classes):
    """Valida um modelo de classificação multirrótulo."""

    # Calcular métricas
    metricas = calcular_metricas(y_true, y_pred)

    # Imprimir métricas
    print("Métricas de Desempenho:")
    print(f"Precision (micro): {metricas['Precision (micro)']:.4f}")
    print(f"Recall (micro): {metricas['Recall (micro)']:.4f}")
    print(f"F1-score (micro): {metricas['F1-score (micro)']:.4f}")
    print(f"Precision (macro): {metricas['Precision (macro)']:.4f}")
    print(f"Recall (macro): {metricas['Recall (macro)']:.4f}")
    print(f"F1-score (macro): {metricas['F1-score (macro)']:.4f}")
    print(f"Hamming Loss: {metricas['Hamming Loss']:.4f}")
    print(f"Acurácia: {metricas['Acurácia']:.4f}")

    # Imprimir precision@k e recall@k
    for k in [1, 3, 5]:
        print(f"Precision@{k}: {metricas[f'Precision@{k}']:.4f}")
        print(f"Recall@{k}: {metricas[f'Recall@{k}']:.4f}")

    # Plotar métricas micro e macro
    plotar_metricas(metricas)

    # Plotar matriz de confusão
    plotar_matriz_confusao(y_true, y_pred, classes)

    # Calcular e plotar curva ROC e AUC ROC
    roc_auc, fpr, tpr = calcular_curva_roc_auc(y_true, y_proba)
    plotar_curvas_roc(roc_auc, fpr, tpr, classes)

    # Imprimir AUCs ROC
    print("\nAUC ROC por classe:")
    for i in range(len(classes)):
        print(f"{classes[i]}: {roc_auc[i]:.4f}")
    print(f"AUC ROC (macro): {roc_auc['macro']:.4f}")
    print(f"AUC ROC (micro): {roc_auc['micro']:.4f}")

if __name__ == "__main__":
    # Validar o modelo
    validar_modelo(y_true, y_pred, classes)