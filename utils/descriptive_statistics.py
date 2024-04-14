import os
import json
import pandas as pd
import numpy as np
import scipy.stats as st
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from statsmodels.formula.api import ols
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import FTestAnovaPower

from json_fle_manager import JSONFileManager

class DescriptiveStatistics:
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as file:
            raw_data = json.load(file)
        self.results = self._reorganize_data_experiments(raw_data)
        self.method_times = self._extract_times()

    def _reorganize_data_sections(self, raw_data):
        # Reorganiza os dados para dispor por seções e tipos de métodos
        organized_data_sections = []
        for item in raw_data:
            section = item.get('section_numbers')
            experiment_number = item.get('experiment_numbers')
            execution_time = item.get('execution_times')

            # Criar uma nova entrada para cada registro
            method_key = f"{section} - Experimento {experiment_number}"
            organized_data_sections.append({'execution_times': {method_key: execution_time}})

        return organized_data_sections

    def _reorganize_data_experiments(self, raw_data):
        # Reorganiza os dados para dispor por seções e tipos de métodos
        organized_data_experiments = []
        for item in raw_data:
            experiment_number = item.get('experiment_numbers')
            execution_time = item.get('execution_times')

            # Criar uma nova entrada para cada registro
            method_key = f"{experiment_number}"
            organized_data_experiments.append({'execution_times': {method_key: execution_time}})

        return organized_data_experiments

    # Reestruturação, assumindo que cada 'experiment_numbers' é um método diferente
    @staticmethod
    def restructure_results(section_results):
        if section_results is None:
            # Supondo que JSONFileManager é uma classe definida em algum lugar no seu projeto
            section_results = JSONFileManager.load_json('pilot_results_sections.json')

        # Criar um dicionário para agrupar tempos de execução por número do experimento
        method_times = {}
        for res in section_results:
            method_name = f"Método {res['experiment_numbers']}"
            method_times.setdefault(method_name, []).append(res['execution_times'])

        # Criar uma lista de dicionários no formato esperado pela classe Plotter
        restructured_results = []
        for method, times in method_times.items():
            restructured_results.append({'execution_times': {method: times}})

        return restructured_results

    def _extract_times(self):
        # Extração de tempos do formato organizado por seções
        method_times = {}
        for result in self.results:
            for method, time in result['execution_times'].items():
                method_times.setdefault(method, []).append(time)
        return method_times

    def compute_statistics(self):
        statistics = {}
        print(f'Tipo de estrutura dos resultados:{type(self.results)}')
        for result in self.results:
            for method, time in result['execution_times'].items():
                statistics.setdefault(method, []).append(time)

        return {
            method: {
                'mean': np.mean(times),
                'median': np.median(times),
                'std_dev': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times)
            }
            for method, times in statistics.items()
        }

    def print_statistics(self):
        stats = self.compute_statistics()
        for method, values in stats.items():
            print(f"   Method: {method:>08}")
            for stat, value in values.items():
                print(f"  {stat:>7}: {value:>7.2f}")
            print()

    def calculate_iqr(self):
        iqr_values = {}
        for method, times in self.method_times.items():
            Q1 = np.percentile(times, 25)
            Q3 = np.percentile(times, 75)
            IQR = Q3 - Q1
            iqr_values[method] = IQR
        return iqr_values

    def interpret_iqr(self, iqr_values, low_variation_threshold=10, high_variation_threshold=50):
        interpretations = {}
        print("  Valores muito acima ou abaixo dos quartis (Q1 - 1.5*IQR, Q3 + 1.5*IQR) podem indicar outliers.\n")
        for method, iqr in iqr_values.items():
            interpretation = f"Para o método '{method}':\n  IQR: {iqr:.2f}\n"
            if iqr < low_variation_threshold:
                interpretation += "  Pouca variação nos dados.\n"
            elif iqr < high_variation_threshold:
                interpretation += "  Variação moderada nos dados.\n"
            else:
                interpretation += "  Alta variação nos dados, possíveis outliers.\n"
            interpretations[method] = interpretation
        return interpretations

    def calculate_confidence_interval(self, method_name, confidence_level=0.95):
        times = self.method_times[method_name]
        mean = np.mean(times)
        sem = st.sem(times)  # Erro padrão da média
        margin_of_error = sem * st.t.ppf((1 + confidence_level) / 2, len(times) - 1)
        return mean - margin_of_error, mean + margin_of_error

class NonParametricTests:
    def __init__(self, descriptive_stats):
        self.descriptive_stats = descriptive_stats

    def mann_whitney_test(self, method1, method2):
        data1, data2 = self.descriptive_stats._extract_times()[method1], self.descriptive_stats._extract_times()[method2]
        u_stat, p_value = st.mannwhitneyu(data1, data2)
        return u_stat, p_value

class DataNormalityTest:
    def __init__(self, descriptive_stats):
        self.descriptive_stats = descriptive_stats

    def test_normality(self):
        self.normality_results = {}
        method_times = self.descriptive_stats._extract_times()

        # Realizar teste de Shapiro-Wilk
        for method, times in method_times.items():
            stat, p = st.shapiro(times)
            self.normality_results[method] = {'statistic': stat, 'p_value': p}

        return self.normality_results

    def interpret_normality_results(self, p_threshold=0.05):
        interpretations = {}
        for method, result in self.normality_results.items():
            p_value_display = f"< 0.001" if result['p_value'] < 0.001 else f"{result['p_value']:.4f}"
            # Interpretação baseada no valor-p do teste
            if result['p_value'] < p_threshold:
                interpretation = f"Para o método '{method}', o teste de Shapiro-Wilk resultou em um valor-p de {p_value_display} que é menor que o limiar de significância de {p_threshold}. Isto sugere que os dados não seguem uma distribuição normal, levando à rejeição da hipótese nula de normalidade."
            else:
                interpretation = f"Para o método '{method}', o teste de Shapiro-Wilk resultou em um valor-p de {p_value_display}, que é maior ou igual ao limiar de significância de {p_threshold}. Isto indica que não há evidências suficientes para rejeitar a hipótese nula, sugerindo que os dados podem seguir uma distribuição normal."
            interpretations[method] = interpretation
        return interpretations

class SampleSizeCalculator:
    def __init__(self, descriptive_stats):
        self.descriptive_stats = descriptive_stats

    def calculate_effect_size(self, method1, method2):
        stats = self.descriptive_stats.compute_statistics()
        stats1, stats2 = stats[method1], stats[method2]

        mean1, std_dev1 = stats1['mean'], stats1['std_dev']
        mean2, std_dev2 = stats2['mean'], stats2['std_dev']

        combined_std_dev = np.sqrt(((stats1['count'] - 1) * std_dev1**2 + (stats2['count'] - 1) * std_dev2**2) / (stats1['count'] + stats2['count'] - 2))
        effect_size = (mean1 - mean2) / combined_std_dev

        return effect_size

    def calculate_sample_size(self, method_names, effect_size, power):
        sample_sizes = {}
        stats = self.desc_stats.compute_statistics()

        analysis = TTestIndPower()

        for method in method_names:
            std_dev = stats[method]['std_dev']
            n = analysis.solve_power(effect_size=effect_size, nobs1=None, alpha=0.05, power=power, ratio=1, alternative='two-sided')
            sample_sizes[method] = n

        return sample_sizes

class BasicSampleSizeCalculator:
    def __init__(self, descriptive_stats, alpha=0.05, power=0.8, effect_size=None):
        self.descriptive_stats = descriptive_stats
        self.alpha = alpha
        self.power = power
        self.effect_size = effect_size

    def calculate_sample_size(self, effect_size, power, alpha=0.05):
        analysis = TTestIndPower()
        return analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0)

    def calculate_effect_size(self, method):
        # Calcular o desvio padrão e a média
        times = [res['execution_times'][method] for res in self.results]
        std_dev = np.std(times, ddof=1)
        mean = np.mean(times)

        # Usar um tamanho de efeito padrão se não for fornecido
        if self.effect_size is None:
            # Cohen's d para tamanho de efeito pequeno
            self.effect_size = 0.2

        return self.effect_size * std_dev / mean

class StudentTTest:
    def __init__(self, descriptive_stats):
        self.descriptive_stats = descriptive_stats

    def apply_t_test(self, method1, method2):
        method_times = self.descriptive_stats._extract_times()
        data1, data2 = method_times[method1], method_times[method2]
        t_stat, p_value = st.ttest_ind(data1, data2, equal_var=False)

        return {'t_statistic': t_stat, 'p_value': p_value}

    def filter_data(self, method_name):
        return [res['execution_times'][method_name] for res in self.results if method_name in res['execution_times']]

    def interpret_results(self, method1, method2, t_test_result, alpha=0.05):
        t_stat = t_test_result['t_statistic']
        p_value = t_test_result['p_value']

        interpretation = f"Comparando '{method1}' com '{method2}':\n"
        interpretation += f"  Estatística t: {t_stat:.3f}\n"
        p_value_display = f"< 0.001" if t_test_result['p_value'] < 0.001 else f"{t_test_result['p_value']:.4f}"
        interpretation += f"  Valor-p: {p_value_display}\n"

        if p_value < alpha:
            interpretation += (f"  Com um nível de significância de " + str(alpha) + ", rejeita-se a hipótese nula.\n" +
                               "  Isso indica que há uma diferença estatisticamente significativa entre os tempos de " +
                               "execução dos métodos comparados.")
        else:
            interpretation += (f"  Com um nível de significância de " + str(alpha) + ", conlui-se que não há evidências suficientes para rejeitar a hipótese nula.\n"+
                               "  Isso sugere que não foi possível detectar diferença estatisticamente significativa entre os tempos de execução dos métodos comparados.")

        return interpretation

class MannWhitneyTest:
    def __init__(self, descriptive_stats):
        self.descriptive_stats = descriptive_stats

    def apply_u_test(self, method1, method2):
        method_times = self.descriptive_stats._extract_times()
        data1, data2 = method_times[method1], method_times[method2]
        u_stat, p_value = st.mannwhitneyu(data1, data2, alternative='two-sided')

        return {'u_statistic': u_stat, 'p_value': p_value}

    def filter_data(self, method_name):
        return [res['execution_times'][method_name] for res in self.results if method_name in res['execution_times']]

    def interpret_results(self, method1, method2, u_test_result, alpha=0.05):
        u_stat = u_test_result['u_statistic']
        p_value = u_test_result['p_value']
        
        interpretation = f"Comparando '{method1}' com '{method2}':\n"
        interpretation += f"  Estatística U: {u_stat:.3f}\n"
        p_value_display = f"< 0.001" if u_test_result['p_value'] < 0.001 else f"{u_test_result['p_value']:.4f}"
        interpretation += f"  Valor-p: {p_value_display}\n"

        if p_value < alpha:
            interpretation += ("  Com um nível de significância de " + str(alpha) + ", rejeita-se a hipótese nula.\n" +
                               "  Isso indica que há uma diferença estatisticamente significativa " +
                               "entre os tempos de execução dos métodos comparados.")
        else:
            interpretation += ("  Com um nível de significância de " + str(alpha) + ", não se rejeita a hipótese nula.\n"+
                               "    Isso sugere que não foi possível detectar diferença estatisticamente significativa entre os tempos de execução dos métodos comparados.")
        return interpretation

class ANOVA:
    def __init__(self, descriptive_stats):
        """
        Inicializa a classe com os dados dos experimentos.
        :param data: DataFrame com os dados dos experimentos.
        """
        self.descriptive_stats = descriptive_stats
        self.data = self._prepare_data_for_anova()

    def _prepare_data_for_anova(self):
        # Estrutura de dados esperada: DataFrame com colunas 'Tempo' e 'Método'
        method_times = self.descriptive_stats._extract_times()
        
        # Preparando o DataFrame para a ANOVA
        anova_data = []
        for method, times in method_times.items():
            for time in times:
                anova_data.append({'Método': method, 'Tempo': time})
        
        return pd.DataFrame(anova_data)

    def perform_anova(self):
        """
        Realiza a Análise de Variância (ANOVA) nos dados fornecidos.
        """
        modelo = ols('Tempo ~ Método', data=self.data).fit()
        anova_resultados = sm.stats.anova_lm(modelo, typ=2)
        return anova_resultados

    def perform_post_hoc_tests(self):
        """
        Realiza testes post-hoc se a ANOVA mostrar diferenças significativas.
        """
        tukey = pairwise_tukeyhsd(endog=self.data['Tempo'], groups=self.data['Método'], alpha=0.05)
        return tukey

    def calculate_sample_size(self, effect_size, alpha, power):
        """
        Calcula o tamanho da amostra necessário para o estudo principal.
        :param effect_size: Tamanho do efeito desejado.
        :param alpha: Nível de significância.
        :param power: Poder do teste.
        """
        analysis = FTestAnovaPower()
        # Como estamos lidando com múltiplos grupos, usamos um ajuste para o número de grupos
        num_groups = len(self.data['Método'].unique())
        sample_size_per_group = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, nobs1=None)
        total_sample_size = sample_size_per_group * num_groups
        return total_sample_size

    def interpret_results(self, anova_results, post_hoc_results):
        """
        Interpreta os resultados da ANOVA e dos testes post-hoc, separando em dois grupos:
        um para diferenças significativas e outro para ausência de evidências suficientes.
        :param anova_results: Resultados da ANOVA.
        :param post_hoc_results: Resultados dos testes post-hoc.
        """
        interpretation = []

        # Interpretação dos resultados da ANOVA
        # print("Resultados da ANOVA:")
        # print(anova_results)
        interpretation.append("Resultados da ANOVA:\n")
        interpretation.append(str(anova_results))

        # Verificando se há diferença significativa na ANOVA
        if anova_results['PR(>F)'].iloc[0] < 0.05:
            # print("\nExistem diferenças significativas entre os métodos.")
            interpretation.append("\nExistem diferenças significativas entre os métodos.\n")

        else:
            # print("\nNão há evidências suficientes para afirmar diferenças significativas entre os métodos.")
            interpretation.append("\nNão há evidências suficientes para afirmar diferenças significativas entre os métodos.\n")


        # Interpretação dos resultados do Teste Post-Hoc
        # print("\nResultados do Teste Post-Hoc (Tukey HSD):")
        # print(post_hoc_results.summary())
        interpretation.append("\nResultados do Teste Post-Hoc (Tukey HSD):\n")
        interpretation.append(str(post_hoc_results.summary()) + "\n")

        # Convertendo os resultados do Tukey HSD em DataFrame
        post_hoc_df = pd.DataFrame(data=post_hoc_results._results_table.data[1:], columns=post_hoc_results._results_table.data[0])

        # Agrupando os resultados
        significant_differences = []
        no_significant_evidence = []

        # Classificando os pares de grupos
        for index, row in post_hoc_df.iterrows():
            pair = f"{row['group1']} e {row['group2']}"
            if row['reject']:
                significant_differences.append((pair, row['p-adj']))
            else:
                no_significant_evidence.append((pair, row['p-adj']))

        # Exibindo os resultados
        if significant_differences:
            # print("\nGrupos com Diferenças Estatisticamente Significativas:")
            interpretation.append("\nGrupos com Diferenças Estatisticamente Significativas:")
            for pair, p_adj in significant_differences:
                p_value_display = f"< 0.001" if p_adj < 0.001 else f"{p_adj:.4f}"
                # print(f"  {pair} apresentam diferenças significativas (p-adj = {p_value_display}).")
                interpretation.append(f"  {pair} apresentam diferenças significativas (p-adj = {p_value_display}).")

        if no_significant_evidence:
            # print("\nGrupos sem Evidências Suficientes para Diferenças Estatisticamente Significativas:")
            interpretation.append("\nGrupos sem Evidências Suficientes para Diferenças Estatisticamente Significativas:")
            for pair, p_adj in no_significant_evidence:
                # print(f"  Não há evidências suficientes para afirmar diferenças significativas entre {pair} (p-adj = {p_adj:.4f}).")
                interpretation.append(f"  Não há evidências suficientes para afirmar diferenças significativas entre {pair} (p-adj = {p_adj:.4f}).")

        return '\n'.join(interpretation)

class ExecutionTimeComparator:
    def __init__(self, descriptive_stats):
        self.descriptive_stats = descriptive_stats

    def compare_methods(self, filter_keyword=None):
        method_times = self.descriptive_stats._extract_times()
        print(f"Estrutura de dados resultados: {type(self.results)}")
        for res in self.results:
            print(f"Estrutura de dados item em resultados: {type(res)}")
            print(f"Item em resultado: {res}")
            for method, time in res['execution_times'].items():
                print(f"Método: {method}")
                print(f" Tempo: {time}")
                if filter_keyword and filter_keyword not in method:
                    continue
                method_times.setdefault(method, []).append(time)

        # Calcular médias e medianas
        averages = {method: np.mean(times) for method, times in method_times.items()}
        medians = {method: np.median(times) for method, times in method_times.items()}

        # Comparar os métodos dois a dois
        comparison_results = []
        methods = list(method_times.keys())
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                method1, method2 = methods[i], methods[j]

                # Assegurar que o maior tempo está no numerador
                if averages[method1] < averages[method2]:
                    method1, method2 = method2, method1

                avg_ratio = np.round(averages[method1] / averages[method2] if averages[method2] else float('inf'), 2)
                median_ratio = np.round(medians[method1] / medians[method2] if medians[method2] else float('inf'), 2)
                comparison_results.append({
                    'method1': method1,
                    'method2': method2,
                    'average_ratio': avg_ratio,
                    'median_ratio': median_ratio
                })

        # Ordenar os resultados pela razão da mediana em ordem decrescente
        comparison_results.sort(key=lambda x: x['median_ratio'], reverse=True)

        return comparison_results

class Plotter:
    def __init__(self, section_results, colors):
        self.colors = colors

        # Transformar section_results em results usando a função restructure_results
        self.results = DescriptiveStatistics.restructure_results(section_results)
        # Coletar todos os nomes dos métodos dos resultados
        self.methods = sorted({method for result in self.results for method in result['execution_times']})

        # Supondo que todos os resultados têm a mesma estrutura
        if self.results:
            # Extrair os nomes dos métodos a partir do primeiro item de results
            self.methods = list(self.results[0]['execution_times'].keys())

    def create_experiment_dataframe(self):
        # Carregar dados dos resultados de monitoramento
        json_results_data =  JSONFileManager.load_json(None,'monitoring_results.json')

        # Processar os dados para criar um DataFrame
        data = []
        for section in json_results_data["experiment_results"]:
            for section_name, experiments in section.items():
                if isinstance(experiments, list):
                    for experiment in experiments:
                        if isinstance(experiment, dict):
                            data.append({
                                "section_numbers": section_name,
                                "experiment_numbers": experiment["experiment_number"],
                                "execution_times": experiment["execution_time"]
                            })
                        else:  # Casos onde a lista contém diretamente os tempos de execução
                            data.append({
                                "section_numbers": section_name,
                                "experiment_numbers": None,
                                "execution_times": experiment
                            })
                else:
                    # Caso onde a seção contém diretamente uma lista de tempos de execução
                    avg_time = sum(experiments) / len(experiments)
                    data.append({
                        "section_numbers": section_name,
                        "average_execution_times": avg_time
                    })

        # Filtrar apenas os dados que possuem "experiment_number"
        experiment_data = [d for d in data if d.get("experiment_numbers") is not None]

        # Criar um DataFrame com esses dados
        sections_df = pd.DataFrame(experiment_data)

        return sections_df

    def plot_scatter_experiment_times(self, sections_df):
        # Converter e ordenar experiment_numbers numericamente
        sections_df['experiment_numbers'] = pd.to_numeric(sections_df['experiment_numbers'])
        sections_df = sections_df.sort_values('experiment_numbers')

        # Criar figura
        fig = go.Figure()

        # Plotar pontos para cada experimento
        for i, exp_num in enumerate(sorted(sections_df['experiment_numbers'].unique())):
            color = self.colors[i] if i < len(self.colors) else 'grey'
            df_filtered = sections_df[sections_df['experiment_numbers'] == exp_num]
            fig.add_trace(go.Scatter(
                x=df_filtered['experiment_numbers'],
                y=df_filtered['execution_times'],
                mode='markers',
                marker_color=color,
                name=str(exp_num)
            ))

        # Atualizar o layout do gráfico
        fig.update_layout(
            xaxis_title="Número do Experimento",
            yaxis_title="Tempo de Execução (s)",
            title="Dispersão dos Tempos de Execução por Número de Experimento",
            width=1800,
            height=600,
            font=dict(size=14)
        )

        fig.show(renderer="notebook")

    def plot_scatter_experiment_sections_times(self, sections_df):
        # Mapeamento de cores para cada número de experimento
        color_map = self.colors[i] if i < len(self.colors) else 'grey'

        # Adicionar uma coluna de cores ao DataFrame
        sections_df['color'] = sections_df['experiment_numbers'].map(color_map)

        # Gerar o gráfico de dispersão
        fig = px.scatter(sections_df, x='section_numbers', y='execution_times', color='color',
                         labels={
                             "section_numbers": "Número da Seção",
                             "execution_times": "Tempo de Execução (s)",
                             "experiment_numbers": "Número do Experimento"
                         },
                         title="Dispersão dos Tempos de Execução por Seção e Experimento")

        # Atualizar o layout do gráfico
        fig.update_layout(
            xaxis_title="Número da Seção",
            yaxis_title="Tempo de Execução (s)",
            legend_title="Número do Experimento",
            width=1800,
            height=600,
            font=dict(size=20)
        )

        fig.show(renderer="notebook")

    def plot_bubble_chart_experiment_sections(self, sections_df):
        # Arredondar os tempos de execução para duas casas decimais e calcular a frequência
        sections_df['rounded_execution_times'] = sections_df['execution_times'].round(2)
        frequency = sections_df.groupby(['experiment_numbers', 'rounded_execution_times']).size().reset_index(name='freq')

        # Criar figura
        fig = go.Figure()

        # Plotar as bolhas para cada experimento
        for i, exp_num in enumerate(sorted(frequency['experiment_numbers'].unique())):
            color = self.colors[i] if i < len(self.colors) else 'grey'
            df_filtered = frequency[frequency['experiment_numbers'] == exp_num]
            fig.add_trace(go.Scatter(
                x=df_filtered['rounded_execution_times'],
                y=df_filtered['freq'],
                mode='markers',
                marker=dict(
                    color=color,
                    size=df_filtered['freq'],  # Tamanho da bolha baseado na frequência
                    sizemode='area',  # Tamanho da bolha representa a área
                    sizeref=2.*max(frequency['freq'])/(40.**2),  # Calibrar o tamanho das bolhas
                    sizemin=4  # Tamanho mínimo da bolha
                ),
                name=str(exp_num)
            ))

        # Atualizar o layout do gráfico
        fig.update_layout(
            xaxis_title="Tempo de Execução Arredondado (s)",
            yaxis_title="Frequência",
            title="Gráfico de Bolhas dos Tempos de Execução por Número de Experimento",
            width=1800,
            height=600,
            font=dict(size=14)
        )

        fig.show(renderer="notebook")


    def plot_3d_bubble_chart_experiment_sections(self, sections_df):
        # Arredondar os tempos de execução para duas casas decimais e calcular a frequência
        sections_df['rounded_execution_times'] = sections_df['execution_times'].round(2)
        frequency = sections_df.groupby(['experiment_numbers', 'rounded_execution_times']).size().reset_index(name='freq')

        # Criar figura
        fig = go.Figure()

        # Calcular o tamanho de referência para as bolhas
        max_freq = max(frequency['freq'])
        sizeref = 2. * max_freq / (10. ** 2)

        # Fator para aumentar o tamanho das bolhas
        size_factor = 5

        # Plotar as bolhas para cada experimento
        for i, exp_num in enumerate(sorted(frequency['experiment_numbers'].unique())):
            color = self.colors[i] if i < len(self.colors) else 'grey'
            df_filtered = frequency[frequency['experiment_numbers'] == exp_num]

            fig.add_trace(go.Scatter3d(
                x=df_filtered['experiment_numbers'],
                y=df_filtered['freq'],
                z=df_filtered['rounded_execution_times'],
                mode='markers',
                marker=dict(
                    color=color,
                    size=df_filtered['freq'] * size_factor,  # Aumentar o tamanho das bolhas
                    sizemode='diameter',
                    sizeref=sizeref,
                    sizemin=4
                ),
                name=str(exp_num)
            ))

        # Atualizar o layout do gráfico
        fig.update_layout(
            scene=dict(
                xaxis_title='Número do Experimento',
                yaxis_title='Frequência',
                zaxis_title='Tempo de Execução Arredondado (s)'
            ),
            title="Gráfico de Bolhas 3D dos Tempos de Execução",
            width=1800,
            height=600,
            font=dict(size=14)
        )

        fig.show(renderer="notebook")

    def plot_3d_bubble_chart_experiment_times(self, sections_df):
        # Arredondar os tempos de execução para duas casas decimais e calcular a frequência
        sections_df['rounded_execution_times'] = sections_df['execution_times'].round(2)
        frequency = sections_df.groupby(['experiment_numbers', 'rounded_execution_times']).size().reset_index(name='freq')

        # Criar figura
        fig = go.Figure()

        # Calcular o tamanho de referência para as bolhas
        max_freq = max(frequency['freq'])
        sizeref = 2. * max_freq / (10. ** 2)  # Ajustar o divisor para controlar o tamanho das bolhas

        # Plotar as bolhas para cada experimento
        for i, exp_num in enumerate(sorted(frequency['experiment_numbers'].unique())):
            color = self.colors[i] if i < len(self.colors) else 'grey'
            df_filtered = frequency[frequency['experiment_numbers'] == exp_num]

            fig.add_trace(go.Scatter3d(
                x=df_filtered['experiment_numbers'],  # Número do Experimento no eixo X
                y=df_filtered['freq'],  # Frequência no eixo Y
                z=df_filtered['rounded_execution_times'],  # Tempo de Execução no eixo Z
                mode='markers',
                marker=dict(
                    color=color,
                    size=df_filtered['freq'],  # Tamanho da bolha baseado na frequência
                    sizemode='diameter',
                    sizeref=sizeref,
                    sizemin=4
                ),
                name=str(exp_num)
            ))

        # Atualizar o layout do gráfico
        fig.update_layout(
            scene=dict(
                xaxis_title='Número do Experimento',
                yaxis_title='Frequência',
                zaxis_title='Tempo de Execução Arredondado (s)'
            ),
            title="Gráfico de Bolhas 3D dos Tempos de Execução",
            width=1800,
            height=600,
            font=dict(size=14)
        )

        fig.show(renderer="notebook")

    def plot_boxplots(self, ylims=None):
        """Plota os valores mínimos, máximos e intervalos entre quartis com destaque para mediana em subplots separados com escalas individualizadas
        """
        # Determinar o número de métodos
        num_methods = len(self.results)

        # Criar uma figura com subplots
        fig = make_subplots(rows=1, cols=num_methods, shared_yaxes=False)

        # Adicionar um boxplot para cada método em seu próprio subplot
        for i, result in enumerate(self.results):
            method_name = list(result['execution_times'].keys())[0]
            times = result['execution_times'][method_name]
            color = self.colors[i] if i < len(self.colors) else 'grey'

            fig.add_trace(
                go.Box(y=times, name=method_name, marker_color=color),
                row=1, col=i+1
            )

            # Calcular a mediana e o desvio padrão
            median = np.median(times)
            std_dev = np.std(times)

            # Adicionar anotações de mediana e desvio padrão no subplot correto
            fig.add_annotation(dict(
                x=0,  # Posição x é 0 dentro do subplot para centralizar
                y=median,
                text=f"Mediana: {median:.2f}",
                showarrow=False,
                xref=f"x{i+1}",  # Posicionando relativo ao eixo x do subplot
                yref=f"y{i+1}",
                font=dict(size=14),
                align="center"
            ))
            fig.add_annotation(dict(
                x=0,  # Posição x é 0 dentro do subplot para centralizar
                y=median,
                text=f"DP: {std_dev:.2f}",
                showarrow=False,
                xref=f"x{i+1}",  # Posicionando relativo ao eixo x do subplot
                yref=f"y{i+1}",
                font=dict(size=12),
                align="center",
                yshift=-15  # Ajuste no deslocamento vertical para não sobrepor a anotação de mediana
            ))

        # Atualizar o layout do gráfico
        fig.update_layout(
            title_text="Boxplot dos Tempos de Execução por Método (Subplots Individuais)",
            showlegend=False,
            width=1800,
            height=600,
            font=dict(size=12)
        )
        fig.show(renderer="notebook")

    def plot_boxplot(self, ylims=None):
        """Plota os valores mínimos, máximos e intervalos entre quartis com destaque para mediana em área de plotagem única onde todos os métodos compartilham a mesmo escala
        """
        if ylims is None:
            ylims = {}

        # Criar uma figura para os boxplots
        fig = go.Figure()

        # Extrair todos os tempos de execução para cada método e plotá-los como um boxplot separado
        for i, result in enumerate(self.results):
            for method, times in result['execution_times'].items():
                color = self.colors[i] if i < len(self.colors) else 'grey'
                fig.add_trace(go.Box(y=times, name=method, marker_color=color, boxmean='sd'))

                # Adicionar anotações para a mediana e o desvio padrão
                median = np.median(times)
                std_dev = np.std(times)

                fig.add_annotation(dict(
                    x=method, y=median,
                    text=f"Mediana: {median:.2f}",
                    showarrow=False,
                    xshift=-20 if i % 2 == 0 else 20,  # Alternar o deslocamento para esquerda/direita para evitar sobreposição
                ))

                fig.add_annotation(dict(
                    x=method, y=median + std_dev,
                    text=f"DP: {std_dev:.2f}",
                    showarrow=False,
                    xshift=20 if i % 2 == 0 else -20,  # Alternar o deslocamento para esquerda/direita para evitar sobreposição
                ))

        # Configurar o layout do gráfico
        fig.update_layout(
            yaxis_title='Tempo de Execução em segundos (s)',
            showlegend=False,
            boxmode='group',  # Agrupar os boxplots por método
            title_text="Comparação dos Tempos de Execução por Método",
            width=1800,
            height=600,
            font=dict(size=12)
        )

        # Exibir o gráfico
        fig.show(renderer="notebook")


    def plot_execution_times_comparison(self):
        # Determinar o número de métodos
        num_methods = len(self.results)

        # Criar uma figura para o gráfico de barras
        fig = go.Figure()

        # Adicionar um gráfico de barras para cada método
        for i, result in enumerate(self.results):
            method_name = list(result['execution_times'].keys())[0]
            times = result['execution_times'][method_name]
            color = self.colors[i] if i < len(self.colors) else 'grey'

            # Calculando as estatísticas para anotações
            mean_time = np.mean(times)
            std_dev = np.std(times)

            fig.add_trace(go.Bar(
                x=[method_name],
                y=[mean_time],
                name=method_name,
                marker_color=color,
                error_y=dict(type='data', array=[std_dev], visible=True)  # Representando o desvio padrão
            ))

            # Adicionar anotações de média e desvio padrão
            fig.add_annotation(dict(
                x=method_name,
                y=mean_time,
                text=f"Média: {mean_time:.2f}<br>DP: {std_dev:.2f}",
                showarrow=False,
                font=dict(size=12),
                xshift=-20 if i % 2 == 0 else 20,  # Deslocamento alternado para evitar sobreposição
                yshift=20
            ))

        # Atualizar o layout do gráfico
        fig.update_layout(
            title_text="Comparação dos Tempos Médios de Execução por Método",
            yaxis_title='Tempo Médio de Execução em segundos (s)',
            showlegend=False,
            width=1800,
            height=600,
            font=dict(size=14)
        )

        fig.show(renderer="notebook")

    def plot_histograms_with_confidence_intervals(self, descriptive_stats, methods, confidence_level=0.95, num_bins=None):
        fig = go.Figure()

        for i, method in enumerate(methods):
            times = descriptive_stats.method_times[method]
            ci_low, ci_high = descriptive_stats.calculate_confidence_interval(method, confidence_level)
            color = self.colors[i % len(self.colors)] if i < len(self.colors) else 'grey'

            # Adicionando histograma com número específico de bins
            fig.add_trace(go.Histogram(
                x=times, 
                name=method, 
                opacity=0.6, 
                marker_color=color, 
                nbinsx=num_bins  # controlar o número de bins
            ))

            # Adicionando linhas de intervalo de confiança
            fig.add_shape(type='line',
                          x0=ci_low, y0=0, x1=ci_low, y1=1,
                          xref='x', yref='paper',
                          line=dict(color='black', width=2))
            fig.add_shape(type='line',
                          x0=ci_high, y0=0, x1=ci_high, y1=1,
                          xref='x', yref='paper',
                          line=dict(color='black', width=2))

        # Atualizar layout do gráfico
        fig.update_layout(barmode='overlay', title='Histogramas com Intervalos de Confiança')
        fig.show(renderer="notebook")

class ResultLoader:
    def __init__(self, folder):
        self.folder = folder

    def list_json(self):
        """
        Lista todos os arquivos JSON no diretório especificado.
        """
        json_files = []
        for file in os.listdir(self.folder):
            if file.endswith('.json'):
                json_files.append(file)
        return json_files

    def load_from_json(self, file_name):
        """
        Carrega um arquivo JSON e retorna seu conteúdo.
        Parâmetros:
            file_name (str): O nome do arquivo JSON a ser carregado.
        Retorna:
            dict: O conteúdo do arquivo JSON.
        """
        file_path = os.path.join(self.folder, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Erro ao carregar o arquivo {file_name}: {e}")
            return None