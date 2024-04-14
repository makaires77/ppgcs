from dataset_articles_generator_py import DatasetArticlesGenerator
from experiment_profiler import ExperimentProfiler
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
# from scalene import scalene_profiler
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import subprocess
import threading
import pstats
import socket
import psutil
import json
import time
import tqdm
import sys
import os
import re

class ExperimentMonitor:
    def __init__(self, base_repo_dir, profiler: ExperimentProfiler):
        self.profiler = profiler
        # Definindo os caminhos baseando-se no diretório do repositório
        self.base_repo_dir = base_repo_dir
        self.folder_utils = os.path.join(base_repo_dir, 'utils')
        self.folder_domain = os.path.join(base_repo_dir, 'source', 'domain')
        self.folder_data_input = os.path.join(base_repo_dir, 'data', 'input')
        self.folder_data_output = os.path.join(base_repo_dir, 'data', 'output')
        self.monitoring = False
        self.cpu_usage_list = []
        self.gpu_usage_list = []
        self.thread_count_list = []
        self.process_count_list = []
        self.times = []
        self.experiment_list = []
        self.results = []

        # Lista de funções de experimentos a serem executadas
        self.experiment_functions = [
            self.process_singlethread_cpu_py,
            self.process_singlethread_gpu_py,
            self.process_multithreads_cpu_py,
            self.process_multithreads_gpu_py,
            self.process_multithreads_cpu_go,
            self.process_multithreads_cpu_go_optim
        ]

    ## Checagem da presença de GPU
    def is_gpu_available(self):
        """
        Verifica se uma GPU (NVIDIA ou AMD) está disponível para execução de código.
        Retorna True se uma GPU for encontrada e estiver disponível.
        """
        if self._check_gpu_with_nvidia_smi():
            print("GPU Nvidia detectada por checagem com SMI")
            return True
        elif self._check_gpu_with_pytorch():
            print("GPU Nvidia detectada por checagem com PyTorch")
            return True
        elif self._check_gpu_with_amd():
            print("GPU AMD detectada")
            return True
        else:
            return False

    def _check_gpu_with_amd(self):
        """
        Verifica a disponibilidade de GPUs AMD.
        Esta verificação é mais básica e pode não ser totalmente confiável.
        """
        try:
            result = subprocess.run(['clinfo'], capture_output=True, text=True)
            return "Device Type: GPU" in result.stdout
        except FileNotFoundError:
            # clinfo não está instalado ou não está no PATH
            return False        

    def is_gpu_available(self):
        """
        Verifica se uma GPU está disponível para execução de código.
        Retorna True se uma GPU for encontrada e estiver disponível.
        """
        if self._check_gpu_with_nvidia_smi():
            return True
        elif self._check_gpu_with_pytorch():
            return True
        else:
            return False

    def _check_gpu_with_nvidia_smi(self):
        """
        Verifica a disponibilidade de GPUs NVIDIA usando o comando nvidia-smi.
        """
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return "No devices were found" not in result.stdout
        except FileNotFoundError:
            # nvidia-smi não está instalado ou não está no PATH
            return False

    def _check_gpu_with_pytorch(self):
        """
        Verifica a disponibilidade de GPUs usando PyTorch.
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            print("PyTorch não está instalado, ou caminho não está configurado corretamente")
            return False

    ## Manipulação básica dos arquivos JSON
    def list_json(self, folder=None):
        """
        Lista todos os arquivos JSON no diretório especificado na pastas de dados.
        Retorna:
            list[str]: Lista de nomes de arquivos JSON em ordem alfabética.
        """
        if folder == None:
            folder = self.folder_data_input
        json_files = [file for file in os.listdir(folder) if file.endswith('.json')]
        return sorted(json_files)

    def load_from_json(self, folder, filename):
        """
        Carrega um arquivo JSON e retorna seu conteúdo.
        Parâmetros:
            folder (str): O diretório onde o arquivo JSON está localizado.
            filename (str): O nome do arquivo JSON a ser carregado.
        Retorna:
            dict: O conteúdo do arquivo JSON, ou None se ocorrer um erro.
        """
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            print(f"Arquivo não encontrado: {file_path}")
            return None        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Erro ao carregar o arquivo {filename}: {e}")
            return None

    ## Chamada para cada um dos experimentos a medir
    def process_singlethread_cpu_py(self, extracted_data_list, filepath):
        dataset_generator = DatasetArticlesGenerator(self.base_repo_dir, 
                                                     self.profiler)
        start_time = time.time()
        dataset_generator.process_dicts(extracted_data_list, 
                                        os.path.join(self.folder_data_output, 
                                                     "output_py_cpu_singlethread.json"))
        return time.time() - start_time

    def process_singlethread_gpu_py(self, extracted_data_list, filepath):
        dataset_generator = DatasetArticlesGenerator(self.base_repo_dir, 
                                                     self.profiler)
        start_time = time.time()
        dataset_generator.process_dicts_with_gpu(extracted_data_list,
                                                 os.path.join(self.folder_data_output,"output_py_gpu_singlethread.json"))
        return time.time() - start_time

    def process_multithreads_cpu_py(self, extracted_data_list, filepath):
        self.profiler.set_current_experiment("Multithreads_CPU_PY")
        dataset_generator = DatasetArticlesGenerator(self.base_repo_dir, 
                                                     self.profiler)
        start_time = time.time()
        dataset_generator.process_dicts_multithread(extracted_data_list,
                                                    os.path.join(self.folder_data_output,"output_py_cpu_multithreads.json"))
        return time.time() - start_time

    def process_multithreads_gpu_py(self, extracted_data_list, filepath):
        self.profiler.set_current_experiment("Multithreads_GPU_PY")
        dataset_generator = DatasetArticlesGenerator(self.base_repo_dir, 
                                                     self.profiler)
        start_time = time.time()
        dataset_generator.process_dicts_multithreads_with_gpu(extracted_data_list,
                                                              os.path.join(self.folder_data_output,"output_py_gpu_multithreads.json"))
        return time.time() - start_time

    def process_multithreads_cpu_go(self, extracted_data_list, filepath):
        self.profiler.set_current_experiment("Multithreads_CPU_GO")
        dataset_generator = DatasetArticlesGenerator(self.base_repo_dir, 
                                                     self.profiler)
        start_time = time.time()
        data_go, mtgo_time = dataset_generator.process_data_multithreads_with_go(extracted_data_list)
        return time.time() - start_time

    def process_multithreads_cpu_go_optim(self, extracted_data_list, filepath):
        self.profiler.set_current_experiment("MthreadsSema_CPU_GO")
        dataset_generator = DatasetArticlesGenerator(self.base_repo_dir, 
                                                     self.profiler)
        start_time = time.time()
        data_go, mtgo_time = dataset_generator.process_data_multithreads_with_go_optim(extracted_data_list)
        return time.time() - start_time

    ## Monitorar e coletar dados durante a execução de cada experimento
    def get_process_count(self):
        return len(psutil.pids())
    
    def get_active_processes(self):
        return set(psutil.pids())
        
    def monitor_resources(self, interval=1):
        while self.monitoring:
            current_time = time.time()
            cpu_usage = psutil.cpu_percent(interval=None)
            gpu_usage = self.get_gpu_usage()
            thread_count = threading.active_count()
            process_count = len(psutil.pids())

            # Armazenr dados coletados
            self.times.append(current_time)
            self.cpu_usage_list.append(cpu_usage)
            self.gpu_usage_list.append(gpu_usage)
            self.thread_count_list.append(thread_count)
            self.process_count_list.append(process_count)
            self.experiment_list.append(self.current_experiment)

            time.sleep(interval)

    def get_gpu_usage(self):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'], capture_output=True, text=True)
            return int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
        except FileNotFoundError:
            return 0

    def start_monitoring(self, current_experiment):
        """
        Inicia o monitoramento em um thread separado.
        """
        self.current_experiment = current_experiment
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_resources)
        self.monitor_thread.start()


    def stop_monitoring(self):
        """
        Para o monitoramento e aguarda o término do thread.
        """        
        self.monitoring = False
        self.monitor_thread.join()        

    def save_monitoring_data(self, file_path):
        try:
            data_to_save = {
                "monitoring_data": {
                    "times": self.times,
                    "cpu_usage": self.cpu_usage_list,
                    "gpu_usage": self.gpu_usage_list,
                    "thread_count": self.thread_count_list,
                    "process_count": self.process_count_list,
                    "experiments": self.experiment_list
                },
                # "experiment_results": self.results
            }
            with open(file_path, 'w') as file:
                json.dump(data_to_save, file, indent=4)
        except Exception as e:
            print(f"Erro ao salvar dados de monitoramento: {e}")

    def save_final_results(self, resultfilename='monitoring_results.json'):
        try:
            final_data = {
                "monitoring_data": {
                    "times": self.times,
                    "cpu_usage": self.cpu_usage_list,
                    "gpu_usage": self.gpu_usage_list,
                    "thread_count": self.thread_count_list,
                    "process_count": self.process_count_list,
                    "experiments": self.experiment_list
                },
                "experiment_results": self.results
            }
            final_file_path = os.path.join(self.folder_data_output, resultfilename)
            with open(final_file_path, 'w') as file:
                json.dump(final_data, file, indent=4)
            print(f"\nResultados finais salvos em: {final_file_path}")
        except Exception as e:
            print(f"\nErro ao salvar dados de monitoramento no arquivo geral: {e}")

    def monitor_resources(self, interval=1):
        # Armazenar os processos ativos no início do monitoramento
        unique_processes_at_start = self.get_active_processes()

        while self.monitoring:
            current_time = time.time()
            cpu_usage = psutil.cpu_percent(interval=None)
            gpu_usage = self.get_gpu_usage()
            thread_count = threading.active_count()

            # Atualizar a lista de processos ativos
            current_processes = self.get_active_processes()

            # Identificar novos processos que foram iniciados após o início do monitoramento
            new_processes = current_processes - unique_processes_at_start
            relevant_process_count = len(new_processes)

            # Armazenar dados coletados
            self.times.append(current_time)
            self.cpu_usage_list.append(cpu_usage)
            self.gpu_usage_list.append(gpu_usage)
            self.thread_count_list.append(thread_count)
            self.process_count_list.append(relevant_process_count)
            self.experiment_list.append(self.current_experiment)

            time.sleep(interval)

    def run_experiment(self, experiment_function, *args):
        """
        Executa um único experimento com monitoramento de recursos.
        """
        # Identificar o nome do experimento automaticamente
        experiment_name = experiment_function.__name__

        # Iniciar monitoramento
        self.start_monitoring(experiment_name)
        self.profiler.set_current_experiment(experiment_name)

        # Registrar processos antes do experimento
        processes_before = self.get_active_processes()

        # Executar o experimento
        start_time = time.time()
        result = experiment_function(*args)
        execution_time = time.time() - start_time

        # Registrar processos após o experimento
        processes_after = self.get_active_processes()

        # Parar monitoramento
        self.stop_monitoring()

        # Identificar processos exclusivos do experimento
        unique_processes = processes_after - processes_before

        filepath = os.path.join(self.folder_data_output,'profile_data_all.json')

        return {'result': result, 'execution_time': execution_time, 'unique_processes': unique_processes}

    def run_selected_experiments(self, extracted_data_list, experiment_indices, n):
        for section in range(n):
            section_results = []
            for exp_num in experiment_indices:
                exp_index = exp_num - 1
                if exp_index < 0 or exp_index >= len(self.experiment_functions):
                    print(f"Índice de experimento inválido: {exp_num}")
                    continue

                experiment_function = self.experiment_functions[exp_index]
                print(f"\nIniciando Seção {section + 1}/{n}, Experimento {exp_num}")

                # Definir o caminho do arquivo para resultados do experimento
                filename = f"output_exp{section}_{exp_num}.json"
                filepath = os.path.join(self.folder_data_output, filename)

                self.start_monitoring(f"Seção {section + 1}/{n} - Experimento {exp_num}")
                exec_time = experiment_function(extracted_data_list, filepath)
                self.stop_monitoring()
                
                section_results.append(exec_time)

            self.results.append({
                f'Seção {section + 1}/{n}': section_results
            })

        # Salvar os resultados finais após todas as seções
        self.save_final_results('monitoring_selected_results.json')
        filepath = os.path.join(self.folder_data_output,'profile_data_selected.json')
        self.profiler.save_profile_data(filepath)

    def run_all_experiments(self, extracted_data_list, n):
        ''' 
        Executa todos os experimentos definidos na lista de experiment_functions.
        Cada experimento é executado n vezes, com os resultados de cada seção sendo armazenados.

        Parâmetros:
            extracted_data_list (List[Dict]): Lista de dados extraídos para processamento.
            n (int): Número de vezes para executar cada experimento.

        '''
        for section in range(n):
            section_results = []  # Armazena resultados da seção atual
            for exp_num, experiment_function in enumerate(self.experiment_functions, start=1):
                print(f"\nIniciando Seção {section + 1}/{n}, Experimento {exp_num}/{len(self.experiment_functions)}")

                # Iniciar o monitoramento
                self.start_monitoring(f"Seção {section + 1}/{n} - Experimento {exp_num}/{len(self.experiment_functions)}")
                process_count_before = self.get_process_count()

                # Definir o caminho do arquivo para resultados do experimento
                filename = f"output_exp{section}_{exp_num}.json"
                filepath = os.path.join(self.folder_data_output, filename)

                # Executar o experimento e medir o tempo de execução
                start_time = time.time()
                experiment_function(extracted_data_list, filepath)
                exec_time = time.time() - start_time

                # Parar o monitoramento e registrar os dados coletados
                self.stop_monitoring()
                process_count_after = self.get_process_count()

                # Armazenar a média da contagem de processos antes e depois do experimento
                avg_process_count = (process_count_before + process_count_after) / 2
                self.process_count_list.append(avg_process_count)

                # Adicionar resultados da execução atual ao resultado da seção
                section_results.append({
                    'experiment_number': exp_num,
                    'execution_time': exec_time,
                    'thread_count': self.thread_count_list[-1],
                    'process_count': self.process_count_list[-1]
                })

            # Adicionar os resultados da seção ao resultado geral
            self.results.append({f'Seção {section + 1}/{n}': section_results})

            # Salvar resultados parciais após cada seção
            # self.save_monitoring_data(f'{self.folder_data_output}/partial_results_sec{section + 1}_{n}.json')
            
        # Salvar os resultados finais após a conclusão de todas as seções
        self.save_final_results()
        filepath = os.path.join(self.folder_data_output,'profile_data_all.json')
        self.profiler.save_profile_data(filepath)

    def analyze_performance(self, profile_file='profile_results.prof'):
        # Ler dados do arquivo de perfilamento
        p = pstats.Stats(profile_file)
        p.sort_stats('time')

        # Imprimir as estatísticas
        p.print_stats()

        # Preparar os dados para o gráfico Plotly
        func_list = []
        time_list = []
        for func, stat in p.stats.items():
            func_list.append(f"{func[2]}:{func[0]}")
            time_list.append(stat[2])

        # Criar o gráfico Plotly
        fig = go.Figure(data=go.Bar(x=func_list, y=time_list))
        fig.update_layout(title='Profile Data', xaxis_title='Function', yaxis_title='Time Spent (seconds)')
        fig.show(renderer='notebook')

    # Plotagem ajustada para usar as variáveis de instância
    def plot_monitoring_data(self, file_path=None):
        # Definir o caminho padrão se nenhum for fornecido
        if file_path is None:
            file_path = os.path.join(self.folder_data_output, 'monitoring_results.json')

        # Carregar dados de monitoramento do arquivo JSON
        with open(file_path, 'r') as file:
            data = json.load(file)

        monitoring_data = data["monitoring_data"]

        # Extrair dados específicos para a plotagem
        times = monitoring_data["times"]
        cpu_usage_list = monitoring_data["cpu_usage"]
        gpu_usage_list = monitoring_data["gpu_usage"]
        thread_count_list = monitoring_data["thread_count"]
        process_count_list = monitoring_data["process_count"]
        experiment_list = monitoring_data["experiments"]

        # Converter tempos em minutos desde o início do primeiro registro
        start_time = times[0]
        elapsed_times = [(t - start_time) / 60 for t in times]
        max_time = max(elapsed_times)

        # Iniciar figura de subplots com eixo secundário
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Adicionar a carga da CPU e da GPU
        fig.add_trace(go.Scatter(x=elapsed_times, y=cpu_usage_list, name="Carga da CPU (%)", mode='lines', line=dict(color='orange')), secondary_y=False)
        fig.add_trace(go.Scatter(x=elapsed_times, y=gpu_usage_list, name="Carga da GPU (%)", mode='lines', line=dict(color='green')), secondary_y=False)

        # Adicionar scatter plot para contagem de threads
        fig.add_trace(go.Scatter(x=elapsed_times, y=thread_count_list, name="Threads disparadas", opacity=0.6, marker_color='lightblue'), secondary_y=True)

        # Adicionar scatter plot para a contagem de processos
        fig.add_trace(go.Scatter(x=elapsed_times, y=process_count_list, name="Processos disparados", opacity=0.6, marker_color='purple'), secondary_y=True)

        # Mapear todos os experimentos possíveis para cores
        all_experiments = set(experiment_list)
        experiment_colors = {exp: "hsl({}, 100%, 70%)".format(i * (360 / len(all_experiments))) for i, exp in enumerate(all_experiments)}

        # Mapear os experimentos para suas abreviações e adicionar retângulos
        experiment_labels = {exp: f"S{exp.split(' ')[1][0]}E{exp.split(' ')[4].split('/')[0]}" for exp in all_experiments}
        for experiment in all_experiments:
            exp_times = [t for t, e in zip(elapsed_times, experiment_list) if e == experiment]
            if exp_times:
                start_exp = min(exp_times)
                end_exp = max(exp_times)
                fig.add_shape(type="rect", x0=start_exp, y0=0.9 * 100, x1=end_exp, y1=100, line=dict(color=experiment_colors[experiment]), fillcolor=experiment_colors[experiment], opacity=0.5)
                center_of_experiment = (start_exp + end_exp) / 2
                fig.add_annotation(x=center_of_experiment, y=95, text=experiment_labels[experiment], showarrow=False, font=dict(size=10, color="black"), xanchor="center", yanchor="middle")

        # Atualizar o layout do gráfico
        fig.update_layout(title_text="Monitoramento de CPU, GPU, Threads Ativas e Experimentos", 
                          xaxis_title="Tempo desde o início (minutos)", 
                          xaxis=dict(rangeslider=dict(visible=False)), 
                          yaxis_title="Carga (%)", 
                          legend_title="Legenda", 
                          template="plotly_white")

        # Definir o máximo dos eixos Y
        fig.update_yaxes(title_text="Carga (%)", secondary_y=False, range=[0, 100])
        fig.update_yaxes(title_text="Quantidade de Threads/Processos", secondary_y=True)

        # Exibir o gráfico
        fig.show(renderer='notebook')

    # Função de plotagem destacando threads em barras
    def plot_monitoring_threads_bars(self, filepath=None):
        if filepath == None:
            filepath='/home/mak/gml_classifier-1/data/output/monitoring_selected_results.json'

        with open(filepath, 'r') as file:
            data = json.load(file)

        monitoring_data = data["monitoring_data"]
        experiment_results = data["experiment_results"]

        # Extrair dados específicos para a plotagem
        times = monitoring_data["times"]
        cpu_load = monitoring_data["cpu_usage"]
        print(f"{len(cpu_load)} contagens de carga de CPU, entre {min(cpu_load)} mínimo e {max(cpu_load)} no máximo")
        gpu_load = monitoring_data["gpu_usage"]
        print(f"{len(gpu_load)} contagens de carga de GPU, entre {min(gpu_load)} mínimo e {max(gpu_load)} no máximo")
        thread_count = monitoring_data["thread_count"]
        print(f"{len(thread_count)} contagem de trhreads disparadas entre mínimo de {min(thread_count)} e máximo de {max(thread_count)}")
        process_count = monitoring_data["process_count"]
        print(f"{len(process_count)} contagem de processos disparados entre mínimo de {min(process_count)} e máximo de {max(process_count)}")
        experiment_list = monitoring_data["experiments"]
        print(f"Experimentos: {set(experiment_list)}")

        # Converter tempos em minutos desde o início do primeiro experimento
        start_time = times[0]
        elapsed_times = [(t - start_time) for t in times]  # Convertido para minutos
        max_time = max(elapsed_times)
        print(f"{len(times)} tomadas de tempos durante {np.round(times[-1]-times[0],0)} segundos")
        
        # Iniciar figura de subplots com eixo secundário
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Mapear todos os experimentos possíveis para cores
        all_experiments = set(experiment_list)
        experiment_colors = {exp: "hsl({}, 100%, 70%)".format(i * (360 / len(all_experiments))) for i, exp in enumerate(all_experiments)}

        # Adicionar retângulos para a duração de cada experimento
        experiment_labels = {exp: f"S{exp.split(' ')[1][0]}E{exp.split(' ')[4].split('/')[0]}" for exp in all_experiments}
        for experiment in all_experiments:
            exp_times = [t for t, e in zip(elapsed_times, experiment_list) if e == experiment]
            if exp_times:
                start_exp = min(exp_times)
                end_exp = max(exp_times)
                fig.add_shape(type="rect", x0=start_exp, y0=0.9 * 100, x1=end_exp, y1=100, line=dict(color=experiment_colors[experiment]), fillcolor=experiment_colors[experiment], opacity=0.5)
                center_of_experiment = (start_exp + end_exp) / 2
                fig.add_annotation(x=center_of_experiment, y=95, text=experiment_labels[experiment], showarrow=False, font=dict(size=10, color="black"), xanchor="center", yanchor="middle")

        # Adicionar barras de contagem de threads
        fig.add_trace(go.Bar(
            x=elapsed_times*3,
            y=thread_count,
            name="Threads disparadas",
            opacity=0.25,
            marker_color='yellow'
        ), secondary_y=True)

        # # Adicionar scatter plot para contagem de threads
        # fig.add_trace(go.Scatter(x=elapsed_times, y=thread_count, name="Threads disparadas", opacity=0.6, marker_color='lightblue'), secondary_y=True)

        # Adicionar scatter plot para a contagem de processos
        fig.add_trace(go.Scatter(x=elapsed_times, y=process_count, name="Processos disparados", opacity=0.6, marker_color='purple'), secondary_y=True)

        # Adicionar a carga da CPU e da GPU
        fig.add_trace(go.Scatter(
            x=elapsed_times*3,
            y=cpu_load,
            name="Carga da CPU (%)",
            mode='lines',
            line=dict(color='orange')
        ), secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=elapsed_times*3,
            y=gpu_load,
            name="Carga da GPU (%)",
            mode='lines',
            line=dict(color='green')
        ), secondary_y=False)

        # Adicionar linhas verticais para separar as seções de experimentos
        last_experiment = None
        for i, (time_point, experiment) in enumerate(zip(elapsed_times, experiment_list)):
            if experiment != last_experiment:
                fig.add_shape(
                    type="line",
                    x0=time_point, y0=0, x1=time_point, y1=100,
                    line=dict(color="black", width=2, dash="dash"),
                )
                last_experiment = experiment

        # Atualizar o layout do gráfico
        fig.update_layout(
            title_text="Monitoramento de CPU, GPU, Threads Ativas e Experimentos",
            xaxis_title="Tempo desde o início (segundos)",
            xaxis_range=[0, max_time],
            yaxis_title="Carga (%)",
            legend_title="Legenda",
            template="plotly_white"
        )
        
        # Definir o máximo do eixo principal Y para 100%
        fig.update_yaxes(
            title_text="Carga (%)",
            secondary_y=False,
            range=[0, 100]
        )
        
        # Definir o máximo do eixo secundário Y baseado no valor máximo de threads
        fig.update_yaxes(
            title_text="Quantidade de Threads",
            secondary_y=True,
            range=[0, max(thread_count) * 1.2]
        )
        fig.show(renderer="notebook")
        return fig

    def plot_radar_profile(self, file_path):
        import pandas as pd
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


    def plot_stacked_segment(self, file_path):
        # Supondo que os dados do arquivo JSON estejam no formato correto para carregamento
        with open(file_path, 'r') as file:
            profile_data = json.load(file)

        # Criação de um DataFrame a partir dos dados
        df_profile = pd.DataFrame(profile_data)

        # Agrupar e somar os tempos por Segmento e Experimento
        grouped = df_profile.groupby(['Segment', 'Experiment'])['Time'].sum().reset_index()

        # Pivotar os dados
        pivot_df = grouped.pivot(index='Segment', columns='Experiment', values='Time')

        # Calcular a soma acumulada para cada coluna
        cumulative_sum = pivot_df.cumsum()

        # Calcular o total de cada coluna (experimento)
        total_por_experimento = pivot_df.sum()

        # Converter valores em porcentagens por tipo
        percent_type = pivot_df.div(total_por_experimento) * 100

        # Calcular a porcentagem de cada segmento em relação ao total da coluna empilhada
        percent_df = pivot_df.div(cumulative_sum.max()) * 100

        # Adicionar as barras e os rótulos ao gráfico
        fig = go.Figure()
        for coluna in pivot_df.columns:
            fig.add_trace(
                go.Bar(
                    x=pivot_df.index, 
                    y=pivot_df[coluna], 
                    name=coluna,
                    text=percent_df[coluna].apply(lambda x: '{:.1f}%'.format(x)),
                    textposition='auto'
                )
            )

        # Atualizar layout, se necessário
        fig.update_layout(
            # barmode='',
            title_text="Composição dos Tempos por Experimento em Percentuais"
        )

        # Exibir o gráfico
        fig.show(renderer='notebook')


    def plot_stacked_experiment_percent(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Create a DataFrame
        df_profile = pd.DataFrame(data)
        grouped = df_profile.groupby(['Segment', 'Experiment'])['Time'].sum().reset_index()

        # Calculate the sum of time for each experiment
        total_time_per_experiment = df_profile.groupby('Experiment')['Time'].sum()

        # Calculate the percentage of each segment's time relative to the experiment's total time
        grouped['Percentage'] = grouped.apply(lambda row: (row['Time'] / total_time_per_experiment[row['Experiment']]) * 100, axis=1)

        # Pivot the data to get experiments as columns and segments as indices
        pivot_df = grouped.pivot(index='Experiment', columns='Segment', values='Percentage')

        # Prepare data for stacked bar chart
        fig = go.Figure()

        for experiment in pivot_df.columns:
            fig.add_trace(
                go.Bar(
                    name=experiment,
                    x=pivot_df.index,
                    y=pivot_df[experiment],
                    text=pivot_df[experiment].apply('{:.1f}%'.format),
                    textposition='inside'
                )
            )

        # Update layout for stacked bar chart
        fig.update_layout(
            barmode='stack',
            title='Stacked Bar Chart of Time Percentage per Experiment',
            xaxis_title='Segment',
            yaxis_title='Percentage of Total Time per Experiment',
        )

        # Exibir o gráfico
        fig.show(renderer='notebook')

    def determine_time_unit_and_scale(self, values):
        max_value = max(values)
        if max_value < 60:
            return 'Seconds', 1  # Segundos
        elif max_value < 3600:
            return 'Minutes', 1/60  # Minutos
        else:
            return 'Hours', 1/3600  # Horas
       
    def plot_stacked_experiment(self, file_path):
        with open(file_path, 'r') as file:
            profile_data = json.load(file)

        df_profile = pd.DataFrame(profile_data)
        grouped = df_profile.groupby(['Segment', 'Experiment'])['Time'].sum().reset_index()
        pivot_df = grouped.pivot(index='Experiment', columns='Segment', values='Time')

        time_unit, scale = self.determine_time_unit_and_scale(pivot_df.to_numpy().flatten())

        # Convertendo o DataFrame para a escala desejada
        pivot_df_scaled = pivot_df * scale

        fig = go.Figure()
        for segment in pivot_df_scaled.columns:
            fig.add_trace(go.Bar(
                x=pivot_df_scaled.index,
                y=pivot_df_scaled[segment],
                name=segment,
                text=(pivot_df[segment]/pivot_df.sum(axis=1)*100).apply(lambda x: f"{x:.1f}%"),textposition='inside',
                )
            )

        fig.update_layout(
            barmode='stack',
            title="Stacked Bar Chart of Experiment Times",
            yaxis=dict(
                title=f"Time ({time_unit})",
                type='linear',
                tickmode='array',
                tickvals=[i for i in range(0, int(max(pivot_df_scaled.max())) + 1)],
                ticktext=[f"{i}{time_unit[0]}" for i in range(0, int(max(pivot_df_scaled.max())) + 1)]
            ),
            xaxis=dict(title="Experiments"),
        )

        # Exibir o gráfico
        fig.show(renderer='notebook')


