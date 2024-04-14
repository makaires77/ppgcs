import json
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from abc import ABC, abstractmethod

"""
Detalhes:

Classe ExperimentProfiler: 
    Classe abstrata que define a estrutura básica do perfilador, incluindo a inicialização com segmentos padrão (caso não sejam oferecidos nomes personalizados para os segmentos), métodos abstratos para iniciar e parar a contagem de tempo dos segmentos, e métodos para salvar e gerar relatórios dos dados de perfilamento.

Classe TimeProfiler:
    __init__: Inicializa a classe base e define uma lista para armazenar os dados dos segmentos com o tempo e o experimento associado.

    start_profiling_segment: Inicia a contagem de tempo para um novo segmento e finaliza automaticamente o segmento anterior, se houver. Isso está de acordo com a especificação.
    
    stop_profiling_segment: Encerra o segmento atual, calcula a duração e armazena os dados.

    finalize_profiling: Encerra o último segmento ativo no final do fluxo de execução.

    save_profile_data: Salva os dados coletados em formato JSON, incluindo o nome do experimento.

    plot_results: Gera gráficos de barras a partir dos dados salvos, organizando-os por experimento.

Uso no Fluxo de Execução:
    Inicialização dos perfiladores (TimeProfiler e MemoryProfiler).
    Uso do TimeProfiler para iniciar e alternar entre segmentos durante o fluxo de execução.
    Finalização do perfilamento no final do fluxo de execução com finalize_profiling.
"""

## Classe Base Abstrata com definição dinâmica de Segmentos
class ExperimentProfiler(ABC):
    def __init__(self, segments=None):
        if segments is None:
            segments = ['T01_io_prepare', 'T02_loadto_dev', 'T03_processing', 'T04_unload_dev', 'T05_networkcom', 'T06_syncronize', 'T07_postproces']
        self.profile_data = {segment: 0 for segment in segments}
        self.segment_data = []
        self.current_experiment = None
        self.current_segment = None
        self.start_time = None

    @abstractmethod
    def start_profiling_segment(self, segment):
        pass

    @abstractmethod
    def stop_profiling_segment(self, segment):
        pass

    def set_current_experiment(self, experiment_name):
        """Define o experimento atual para o perfilamento."""
        self.current_experiment = experiment_name

    def get_profile_data(self):
        """Retorna os dados coletados pelo profiler."""
        return self.profile_data

    def save_profile_data(self, filename):
        """Salva os dados do profiler em um arquivo JSON."""
        with open(filename, 'w') as f:
            json.dump(self.profile_data, f, indent=4)

    def integrate_with_external_tool(self, tool):
        """Integra com uma ferramenta externa de perfilamento."""
        ##TO-DO: Implementar lógica para coletar dados da ferramenta externa
        pass

    def generate_report(self):
        """Gera um relatório personalizado dos dados de perfilamento."""
        # Implementação padrão
        profile_data = self.get_profile_data()
        # Gerar e retornar o relatório (Em documento PDF, HTML onde será exibido texto e gráfico)
        return "Relatório padrão: " + str(profile_data)

## Implementação Concreta de Tempo
class TimeProfiler(ExperimentProfiler):
    def __init__(self):
        super().__init__()
        self.experiment_segment_data = []

    def start_profiling_segment(self, segment):
        """Inicia o temporizador para um segmento específico."""
        if self.current_segment is not None:
            # Parar o segmento atual antes de iniciar um novo
            self.stop_profiling_segment(self.current_segment)
        
        self.segment_data.append({
            "Experiment": self.current_experiment,
            "Segment": segment,
            "Time": time.time()
        })

        self.current_segment = segment
        self.start_time = time.time()

    def stop_profiling_segment(self, segment):
        """Finaliza o temporizador para o segmento especificado e registra a duração."""
        if self.start_time is None:
            return

        end_time = time.time()
        duration = end_time - self.start_time
        self.profile_data[segment] += duration
        self.experiment_segment_data.append({
            'Experiment': self.current_experiment,
            'Segment': segment,
            'Time': duration
        })
        self.current_segment = None
        self.start_time = None

    def finalize_profiling(self):
        """Finaliza o perfilamento, encerrando o segmento atual e qualquer monitoramento em execução."""
        if self.current_segment is not None:
            self.stop_profiling_segment(self.current_segment)
            self.current_segment = None

    def save_profile_data(self, filename):
        """Salva os dados do profiler em um arquivo JSON."""
        with open(filename, 'w') as file:
            json.dump(self.experiment_segment_data, file, indent=4)


    ## TO-DO: Hooks para customização
    def on_segment_start(self, segment):
        # Hook chamado no início de um segmento
        pass

    def on_segment_stop(self, segment):
        # Hook chamado no final de um segmento
        pass

    ## Visualização de dados
    def prepare_data_for_plotting(self):
        """
        Prepara os dados de profiling para visualização.
        Retorna uma lista de dicionários com os segmentos e os tempos correspondentes.
        """
        data = []
        for segment, time_spent in self.profile_data.items():
            data.append({'Segment': segment, 'Time': time_spent})
        return data

    def plot_results(self, file_path):
        """Plota os resultados do perfilamento a partir do arquivo de dados."""
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Organizar os dados por experimento
        data_by_experiment = {}
        for item in data:
            experiment = item['Experiment']
            if experiment not in data_by_experiment:
                data_by_experiment[experiment] = []
            data_by_experiment[experiment].append(item)

        # Criar subplots para cada experimento
        fig = make_subplots(rows=len(data_by_experiment), cols=1, subplot_titles=list(data_by_experiment.keys()))

        for i, (experiment, experiment_data) in enumerate(data_by_experiment.items(), start=1):
            for segment in experiment_data:
                fig.add_trace(
                    go.Bar(x=[segment['Segment']], y=[segment['Time']], name=segment['Segment']),
                    row=i, col=1
                )

        fig.update_layout(
            height=400*len(data_by_experiment),
            barmode='stack',
            title_text='Profiling Results by Experiment',
            xaxis_title='Experiment',
            yaxis_title='Time (s)'
        )

        fig.show(renderer='notebook')

    def plot_radar(self, file_path):
        with open(file_path, 'r') as file:
            profile_data = json.load(file)

        df_profile = pd.DataFrame(profile_data)

        # Agrupar e somar os tempos por Segmento e Experimento
        grouped = df_profile.groupby(['Segment', 'Experiment'])['Time'].sum().reset_index()

        # Pivotar os dados
        pivot_df = grouped.pivot(index='Segment', columns='Experiment', values='Time')

        # Função para determinar a unidade de tempo e o fator de escala
        def determine_scale_and_unit(values):
            max_value = max(values)
            if max_value < 120:  # Menos de 2 minutos
                return 1, 'seconds'  # Escala em segundos
            elif max_value < 7200:  # Menos de 2 horas
                return 1/60, 'minutes'  # Escala em minutos
            else:
                return 1/3600, 'hours'  # Escala em horas

        # Preparar os dados para o gráfico de radar
        labels = pivot_df.index
        num_vars = len(labels)

        # Calcular o ângulo de cada eixo no gráfico de radar
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Completar o círculo

        # Inicializar o gráfico de radar
        fig = go.Figure()

        # Determinar a escala e unidade
        scale, unit = determine_scale_and_unit(pivot_df.to_numpy().flatten())

        # Adicionar os dados dos experimentos
        for column in pivot_df.columns:
            values = pivot_df[column].tolist()
            values_scaled = [v * scale for v in values] + [values[0] * scale]
            fig.add_trace(go.Scatterpolar(
                r=values_scaled,
                theta=labels,
                fill='toself',
                name=column
            ))

        # Atualizar o layout do gráfico
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(pivot_df.max()) * scale]
                )),
            showlegend=True,
            title_text=f"Time Spent in Each Segment (in {unit})"
        )

        fig.show(renderer='notebook')

    def generate_report(self):
        """Gera um relatório detalhado dos tempos de execução."""
        profile_data = self.get_profile_data()
        report = "Relatório de Tempo:\n"
        for segment, time in profile_data.items():
            report += f"{segment}: {time:.2f} segundos\n"
        # # TO-DO: adicionar visualizações gráficas ou análises mais profundas
        return report


class MemoryProfiler(ExperimentProfiler):
    def start_profiling_segment(self, segment):
        if self.current_segment is not None:
            self.stop_profiling_segment(self.current_segment)
        
        self.current_segment = segment
        self.start_time = time.time()
        # TO-DO: adicionar lógica para medir o uso de memória

    def stop_profiling_segment(self, segment):
        if segment != self.current_segment:
            print(f"Warning: Trying to stop a segment ({segment}) that is not currently running.")
            return

        if self.start_time is not None:
            end_time = time.time()
            # TO-DO: adicionar lógica para calcular o uso de memória
            self.profile_data[segment] += end_time - self.start_time
            self.current_segment = None
            self.start_time = None


# Exemplo:
# # Criando instâncias dos perfiladores
# # Iniciando perfilamento de memória
# memory_profiler = MemoryProfiler()

# # Iniciando e parando segmentos de tempo
# time_profiler = TimeProfiler()
# time_profiler.start_profiling_segment('T01_io_prepare')
# # ...executar alguma operação...
# time_profiler.start_profiling_segment('T02_loadto_dev')
# # ...executar alguma operação...

# # No final do fluxo de execução do código
# time_profiler.finalize_profiling()