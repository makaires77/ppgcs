import re
import time
import json
import torch
import spacy
import pynvml
import psutil
import cpuinfo
import xformers
import numpy as np
import plotly.graph_objects as go
from unidecode import unidecode
from transformers import AutoModel
# from tqdm.autonotebook import tqdm, trange
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
from sklearn.cluster import KMeans  # ou outro algoritmo de agrupamento

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

class HardwareEvaluator:
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()  # Verifica a disponibilidade da CUDA no construtor
        self.gpu_properties = torch.cuda.get_device_properties(0) if self.gpu_available else None
        self.cpu_info = cpuinfo.get_cpu_info()
        self.cpu_freq = psutil.cpu_freq()
        self.ram_info = psutil.virtual_memory()

    def print_hardware_info(self):
        print("Informações de Hardware:")
        if self.gpu_available:
            print(f"  GPU: {self.gpu_properties.name}")
            print(f"    Tensor Cores: {self.get_tensor_cores()}")
            print(f"    Núcleos CUDA: {(self.gpu_properties.multi_processor_count * 128) - self.get_tensor_cores()//128}")
            print(f"    Compute Capability: {self.gpu_properties.major}.{self.gpu_properties.minor}")
            print(f"    Frequência Máxima: {self.get_gpu_clock_rate()} GHz")
            print(f"    Memória Total: {self.gpu_properties.total_memory / 1024**3:.2f} GB")
        else:
            print("  GPU: Não disponível")

        print(f"  CPU: {self.cpu_info['brand_raw']}")
        print(f"    Núcleos Físicos: {psutil.cpu_count(logical=False)}")
        print(f"    Núcleos Lógicos: {psutil.cpu_count(logical=True)}")
        print(f"    Frequência Máxima: {self.cpu_freq.max:.2f} GHz")
        print(f"    RAM Total: {self.ram_info.total / 1024**3:.2f} GB")

    def get_tensor_cores(self):
        """Obtém o número de Tensor Cores da GPU.

        Returns:
            int: Número de Tensor Cores ou 0 se a informação não estiver disponível.
        """
        if self.gpu_available:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                # print(gpu_name)

                # Dicionário com informações de Tensor Cores para GPUs Nvidia modernas (CORRIGIDO)
                tensor_cores_dict = {
                    "Quadro RTX 8000": 4608,  # 1152 SMs, cada um com 4 Tensor Cores de segunda geração
                    "Quadro RTX 6000": 3840,  # 960 SMs, cada um com 4 Tensor Cores de segunda geração
                    "Quadro RTX 5000": 3072,  # 768 SMs, cada um com 4 Tensor Cores de segunda geração
                    "Quadro RTX 4000": 2304,  # 576 SMs, cada um com 4 Tensor Cores de segunda geração
                    "RTX 4090": 512,          # 128 SMs, cada um com 4 Tensor Cores de quarta geração
                    "A100": 432,              # 108 SMs, cada um com 4 Tensor Cores de terceira geração
                    "RTX 4080 SUPER": 304,    #  76 SMs, cada um com 4 Tensor Cores de quarta geração
                    "RTX 4080 12GB": 240,     #  60 SMs, cada um com 4 Tensor Cores de quarta geração
                    "RTX 4070 Ti": 80,        #  20 SMs, cada um com 4 Tensor Cores de quarta geração
                    "RTX 4070": 64,           #  16 SMs, cada um com 4 Tensor Cores de quarta geração
                    "RTX 3090 Ti": 108,       #  84 SMs, cada um com 4 Tensor Cores de terceira geração e 28 SMs, cada um com 4 Tensor Cores de segunda geração
                    "RTX 3090": 112,          #  82 SMs, cada um com 4 Tensor Cores de terceira geração e 30 SMs, cada um com 4 Tensor Cores de segunda geração
                    "RTX 3080 Ti": 80,        #  60 SMs, cada um com 4 Tensor Cores de terceira geração e 20 SMs, cada um com 4 Tensor Cores de segunda geração
                    "RTX 3080 12GB": 68,      #  58 SMs, cada um com 4 Tensor Cores de terceira geração e 10 SMs, cada um com 4 Tensor Cores de segunda geração
                    "RTX 3080": 80,           #  68 SMs, cada um com 4 Tensor Cores de terceira geração e 12 SMs, cada um com 4 Tensor Cores de segunda geração
                    "RTX 3070 Ti": 64,        #  48 SMs, cada um com 4 Tensor Cores de terceira geração e 16 SMs, cada um com 4 Tensor Cores de segunda geração
                    "RTX 3070": 46,           #  46 SMs, cada um com 4 Tensor Cores de segunda geração
                    "RTX 3060 Ti": 38,        #  38 SMs, cada um com 4 Tensor Cores de segunda geração
                    "RTX 3060": 28,           #  28 SMs, cada um com 4 Tensor Cores de segunda geração
                    "RTX A5000": 84,          #  66 SMs, cada um com 4 Tensor Cores de terceira geração e 18 SMs, cada um com 4 Tensor Cores de segunda geração
                    "RTX A4000": 64,          #  48 SMs, cada um com 4 Tensor Cores de terceira geração e 16 SMs, cada um com 4 Tensor Cores de segunda geração
                    "RTX A2000": 30,          #  26 SMs, cada um com 4 Tensor Cores de segunda geração e 4 SMs, cada um com 4 Tensor Cores de primeira geração
                }

                for model, cores in tensor_cores_dict.items():
                    if model in gpu_name:
                        return cores

                print(f"Aviso: Número de Tensor Cores desconhecido para a GPU {gpu_name}.")
                return 0

            except pynvml.NVMLError as error:
                print(f"Erro ao obter informações da GPU: {error}")
                return 0
            finally:
                pynvml.nvmlShutdown()
        else:
            return 0

    def get_cpu_ipc(self):
        """Obtém o valor de IPC (Instruções Por Ciclo) da CPU.

        Returns:
            float: Valor de IPC estimado.
        """
        try:
            info = cpuinfo.get_cpu_info()
            cpu_name = info['brand_raw']

            # Estimativa para processadores Intel
            if 'Intel' in cpu_name:
                if 'Core i9' in cpu_name:
                    return 2.5
                elif 'Core i7' in cpu_name:
                    return 2.0
                elif 'Core i5' in cpu_name:
                    return 1.5
                else:
                    return 1.0

            # Estimativa para processadores AMD Ryzen (aprimorada)
            if 'Ryzen' in cpu_name:
                if 'Threadripper' in cpu_name:
                    return 2.2  # Estimativa para Threadripper
                elif 'Zen 2' in cpu_name:
                    return 1.8  # Estimativa para Zen 2
                elif 'Zen 3' in cpu_name:
                    return 2.0  # Estimativa para Zen 3
                elif 'Ryzen 9' in cpu_name:
                    return 2.1
                elif 'Ryzen 7' in cpu_name:
                    return 1.9
                elif 'Ryzen 5' in cpu_name:
                    return 1.7
                else:
                    return 1.5

            else:
                return 1.0  # Valor padrão para outras CPUs

        except:
            return 1.0  # Valor padrão em caso de erro

    def get_cpu_parallel_capacity(self):
        """Retorna o número de núcleos lógicos da CPU, que representam a capacidade teórica de processamento paralelo."""
        return psutil.cpu_count(logical=True)

    def get_gpu_performance(self, operation_type="FLOPS"):
        """Obtém o desempenho da GPU para o tipo de operação especificado.

        Args:
            operation_type (str): Tipo de operação ("FLOPS", "INT8", etc.).

        Returns:
            float: Desempenho da GPU em operações por segundo para o tipo especificado.
        """
        if self.gpu_available:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assumindo uma única GPU
                #info = pynvml.nvmlDeviceGetPerformanceState(handle)  # Removido

                if operation_type == "FLOPS":
                    return self.get_gpu_max_flops()
                elif operation_type == "INT8":
                    # Estimar o desempenho para INT8 (implementar a lógica)
                    pass
                else:
                    raise ValueError("Tipo de operação não suportado")

            except pynvml.NVMLError as error:
                print(f"Erro ao obter informações da GPU: {error}")
                return 0
            finally:
                pynvml.nvmlShutdown()
        else:
            return 0

    ## SEM CONSIDERAR OS TENSOR CORES
    # def get_gpu_max_flops(self):
    #     """Obtém o desempenho máximo da GPU em FLOPS.

    #     Returns:
    #         float: Desempenho máximo da GPU em FLOPS.
    #     """
    #     if torch.cuda.is_available():
    #         pynvml.nvmlInit()
    #         num_gpus = torch.cuda.device_count()
    #         max_flops = 0
    #         for i in range(num_gpus):
    #             handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    #             props = torch.cuda.get_device_properties(i)
    #             clock_rate = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM) # Clock de boost máximo
    #             max_flops += props.multi_processor_count * clock_rate * 2  # FLOPS FP32
    #             # Adicionar estimativa de FLOPS para Tensor Cores (se aplicável)
    #         pynvml.nvmlShutdown()
    #         return max_flops
    #     else:
    #         return 0

    def get_gpu_max_flops(self):
        """Obtém o desempenho máximo da GPU em FLOPS, incluindo Tensor Cores.

        Returns:
            float: Desempenho máximo da GPU em FLOPS.
        """
        if torch.cuda.is_available():
            pynvml.nvmlInit()
            num_gpus = torch.cuda.device_count()
            max_flops = 0
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                props = torch.cuda.get_device_properties(i)
                clock_rate = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)

                # FLOPS FP32
                max_flops += props.multi_processor_count * clock_rate * 2

                # FLOPS Tensor Cores (FP16) - Estimativa para GPUs Nvidia
                if props.major >= 7 and props.minor >= 0:  # Verifica se a GPU tem Tensor Cores
                    max_flops += props.multi_processor_count * clock_rate * 8  # Estimativa para FP16

            pynvml.nvmlShutdown()
            return max_flops
        else:
            return 0

    def estimate_cpu_gpu_overhead(self, model, batch_size, input_data):
        """Estima o overhead de comunicação entre CPU e GPU.

        Args:
            model: Modelo a ser executado.
            batch_size: Tamanho do lote.
            input_data: Dados de entrada para o modelo.

        Returns:
            float: Tempo de overhead estimado em segundos.
        """
        if self.gpu_available:
            # Medir o tempo de execução com e sem transferência de dados para a GPU
            with torch.no_grad():
                # Tempo com transferência de dados
                start_time = time.time()
                input_data = input_data.to('cuda')
                model.to('cuda')
                model(input_data)
                torch.cuda.synchronize()  # Garante que todas as operações na GPU terminaram
                end_time = time.time()
                time_with_transfer = end_time - start_time

                # Tempo sem transferência de dados (execução na CPU)
                start_time = time.time()
                model.to('cpu')
                model(input_data.to('cpu'))
                end_time = time.time()
                time_without_transfer = end_time - start_time

            return time_with_transfer - time_without_transfer
        else:
            return 0

    def measure_real_execution_time(self, model, batch_size, input_data, num_runs=10):
        """Mede o tempo de execução real de um modelo na GPU.

        Args:
            model: Modelo a ser executado.
            batch_size: Tamanho do lote.
            input_data: Dados de entrada para o modelo.
            num_runs: Número de execuções para calcular a média.

        Returns:
            float: Tempo médio de execução em segundos.
        """
        if self.gpu_available:
            model.to('cuda')
            input_data = input_data.to('cuda')
            with torch.no_grad():
                total_time = 0
                for _ in range(num_runs):
                    start_time = time.time()
                    model(input_data)
                    torch.cuda.synchronize()  # Garante que todas as operações na GPU terminaram
                    end_time = time.time()
                    total_time += end_time - start_time
            return total_time / num_runs
        else:
            return 0

    def get_gpu_parallel_capacity(self):
        """Retorna o número de núcleos CUDA da GPU, que representam a capacidade teórica de processamento paralelo."""
        if self.gpu_available:
            return self.gpu_properties.multi_processor_count * self.gpu_properties.max_threads_per_multi_processor
        else:
            return 0  # GPU não disponível

    def get_gpu_clock_rate(self):
        """Retorna o clock rate da GPU em MHz."""
        if self.gpu_available:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Obtém o handle da primeira GPU
            clock_rate = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)  # Clock rate dos núcleos CUDA
            pynvml.nvmlShutdown()
            return clock_rate
        else:
            return 0  # GPU não disponível

    def check_pytorch_gpu_compatibility(self):
        """Verifica a compatibilidade entre PyTorch e GPU e recomenda uma versão compatível do PyTorch."""
        if not self.gpu_available:
            print("CUDA não está disponível. Não é possível verificar a compatibilidade com a GPU.")
            return

        # Obtém as informações da GPU
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_compute_capability = f"{pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]}.{pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]}"
        pynvml.nvmlShutdown()

        # Dicionário de compatibilidade PyTorch-GPU (atualizado)
        compatibility_dict = {
            "3.0": "0.3.0",  # Versão mínima do PyTorch para cada compute capability
            "3.5": "0.4.0",
            "3.7": "1.0.0",
            "5.0": "1.2.0",
            "5.2": "1.3.0",
            "6.0": "1.4.0",
            "6.1": "1.5.0",
            "7.0": "1.6.0",
            "7.5": "1.7.0",
            "8.0": "1.8.0",
            "8.6": "1.9.0",
            "8.9": "1.12.0",  # Incluindo compute capability 8.9 (Ada Lovelace)
            "9.0": "1.13.0",  # Incluindo compute capability 9.0 (Hopper)
            "10.0": "2.0.0", # Incluindo compute capability 10.0 (Blackwell)
        }

        # Verifica a compatibilidade
        if gpu_compute_capability not in compatibility_dict:
            print(f"Aviso: A versão do PyTorch ({torch.__version__}) pode não ter sido testada com a sua GPU ({self.gpu_properties.name}, compute capability {gpu_compute_capability}).")
        else:
            min_pytorch_version = compatibility_dict[gpu_compute_capability]
            if torch.__version__ < min_pytorch_version:
                print(f"Aviso: A versão do PyTorch ({torch.__version__}) pode não ser totalmente compatível com a sua GPU ({self.gpu_properties.name}, compute capability {gpu_compute_capability}).")
                print(f"Sugestão: Instale o PyTorch versão {min_pytorch_version} ou superior.")
            else:
                print("PyTorch e GPU são compatíveis.")

    def check_gpu_memory_health(self):
        """Verifica a saúde da memória da GPU, pulando setores com erro."""
        if not self.gpu_available:
            print("CUDA não está disponível. Não é possível verificar a saúde da memória da GPU.")
            return

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Verifica se a GPU suporta ECC
            ecc_mode = pynvml.nvmlDeviceGetEccMode(handle)
            if ecc_mode == pynvml.NVML_FEATURE_DISABLED:
                print("Aviso: A GPU não possui ECC (Error Correcting Code) habilitado. A detecção de erros de memória pode ser limitada.")

            # Obtém o tamanho da memória da GPU
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = memory_info.total

            # Listas para armazenar resultados
            addresses = []
            error_counts = []

            # Verifica cada setor da memória
            with tqdm(total=total_memory, desc="Verificando memória da GPU", unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for i in range(0, total_memory, 4096):  # Verifica em blocos de 4 KB
                    try:
                        pynvml.nvmlDeviceValidateMemory(handle, i, 4096)  # Verifica o bloco de memória
                        error_counts.append(0)  # Nenhum erro
                    except pynvml.NVMLError as e:
                        if e.value == pynvml.NVML_ERROR_CORRUPTED_MEM:
                            print(f"Erro de memória detectado no endereço {hex(i)}")
                            error_counts.append(1)  # Um erro
                        else:
                            print(f"Erro ao verificar a memória: {e}")
                            error_counts.append(-1)  # Erro desconhecido
                    addresses.append(i)
                    pbar.update(4096)

            # Plota os resultados
            plt.figure(figsize=(12, 6))
            plt.plot(addresses, error_counts, marker='o', linestyle='-', color='blue')
            plt.xlabel("Endereço de Memória (bytes)")
            plt.ylabel("Contagem de Erros")
            plt.title("Verificação da Memória da GPU")
            plt.grid(axis='y', linestyle='--')
            plt.show()

            # Exibe resultados da verificação
            if any(error_counts):  # Verifica se há algum erro
                num_errors = sum(1 for count in error_counts if count > 0)
                error_addresses = [hex(addr) for addr, count in zip(addresses, error_counts) if count > 0]
                print(f"Foram encontrados {num_errors} erros de memória nos seguintes endereços: {', '.join(error_addresses)}")
            else:
                print("Nenhum erro de memória encontrado na GPU.")

        except pynvml.NVMLError as e:
            print(f"Erro ao verificar a saúde da memória da GPU: {e}")

        finally:
            pynvml.nvmlShutdown()

class ProcessingCapacityEstimator:
    def __init__(self, hardware_evaluator):
        self.hardware = hardware_evaluator

    def estimate_cpu_throughput(self, num_samples, instructions_per_sample):
        """Estima o throughput da CPU em operações por segundo.

        Args:
            num_samples: Número de amostras a serem processadas.
            instructions_per_sample: Número médio de instruções por amostra.

        Returns:
            Throughput estimado em operações por segundo.
        """
        clock_speed_ghz = psutil.cpu_freq().current / 1e9  # Frequência em GHz
        ipc = self.hardware.get_cpu_ipc()  # Instruções por ciclo (IPC)
        return num_samples * instructions_per_sample / (clock_speed_ghz * ipc)

    def estimate_cpu_parallel_throughput(self, num_samples, instructions_per_sample, thread_overhead=0.1):
        """Estima o throughput da CPU em operações paralelas por segundo.

        Args:
            num_samples: Número de amostras a serem processadas.
            instructions_per_sample: Número médio de instruções por amostra.
            thread_overhead: Overhead estimado por thread (0 a 1).

        Returns:
            Throughput estimado em operações por segundo.
        """
        num_cores = psutil.cpu_count(logical=True)
        effective_cores = num_cores * (1 - thread_overhead)
        single_thread_throughput = self.estimate_cpu_throughput(num_samples, instructions_per_sample)
        return single_thread_throughput * effective_cores

    def estimate_gpu_parallel_throughput(self, num_operations, operation_type="FLOPS"):
        """Estima o throughput da GPU em operações paralelas por segundo.

        Args:
            num_operations: Número de operações a serem realizadas.
            operation_type: Tipo de operação ("FLOPS", "INT8", etc.).

        Returns:
            Throughput estimado em operações por segundo.
        """
        if self.hardware.gpu_available:
            gpu_performance = self.hardware.get_gpu_performance(operation_type)  # Obter desempenho específico para o tipo de operação
            return gpu_performance * num_operations
        else:
            return 0

    def estimate_gpu_throughput(self, model, batch_size, input_data):
        """Estima o throughput da GPU em inferências por segundo.

        Args:
            model: Modelo a ser executado.
            batch_size: Tamanho do lote.
            input_data: Dados de entrada para o modelo.

        Returns:
            Throughput estimado em inferências por segundo.
        """
        if self.hardware.gpu_available:
            # Medir o tempo de execução real para uma inferência com o modelo, batch_size e input_data
            start_time = time.time()
            model.predict(input_data)
            end_time = time.time()
            inference_time = end_time - start_time
            return batch_size / inference_time 
        else:
            return 0

    def interpret_processing_capacity(self, model, sentences, model_sizes=[1024**2 * x for x in [100, 200, 500, 1000]]):
        """Interpreta a capacidade de processamento e estima o tamanho máximo do modelo.

        Args:
            model_sizes (list): Lista de tamanhos de modelo em bytes para os quais a estimativa será feita.

        """
        print("\nInterpretação da Capacidade de Processamento:")

        if self.hardware.gpu_available:
            print("GPU:")
            for model_size in model_sizes:
                max_batch_size = self.hardware.gpu_properties.total_memory // model_size
                print(f"  - Modelo de {model_size / 1024**2:.0f} MB: Lote máximo de {max_batch_size} amostras")
        else:
            print("  GPU: Não disponível")

        print("CPU:")
        cpu_time = self.hardware.benchmark_model(model, sentences, 'cpu')  # Benchmark na CPU
        for model_size in model_sizes:
            estimated_time = model_size / self.estimate_cpu_parallel_throughput(model_size)
            print(f"  - Modelo de {model_size / 1024**2:.0f} MB: Tempo estimado de processamento: {estimated_time:.4f} segundos por amostra (benchmark: {cpu_time:.4f} s/amostra)")

        if self.hardware.gpu_available:
            print("GPU:")
            gpu_time = self.hardware.benchmark_model(model, sentences, 'cuda')  # Benchmark na GPU
            for model_size in model_sizes:
                estimated_time = model_size / self.estimate_gpu_parallel_throughput(model_size)
                print(f"  - Modelo de {model_size / 1024**2:.0f} MB: Tempo estimado de processamento: {estimated_time:.4f} segundos por amostra (benchmark: {gpu_time:.4f} s/amostra)")

        print("\nRecomendações:")
        if self.hardware.gpu_available:
            print("  - Utilize a GPU para acelerar o processamento, se possível.")
            print("  - Ajuste o tamanho do lote de acordo com a memória disponível da GPU e o tamanho do modelo.")
        else:
            print("  - Considere usar uma máquina com GPU para acelerar o processamento.")
            print("  - Otimize o código para melhor desempenho na CPU.")

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

    def benchmark_model(self, model, sentences, device, batch_size=32):
        """Mede o tempo de processamento do modelo (CPU ou GPU) em lotes."""
        
        # Dividir as sentenças em lotes
        batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

        start_time = time.time()
        for batch in batches:
            with torch.no_grad():
                model.encode(batch, convert_to_tensor=True, device=device)
        end_time = time.time()

        total_time = end_time - start_time
        total_samples = len(sentences) * num_repetitions  # Corrigido o cálculo do número total de amostras
        return total_time / total_samples  # Tempo médio por amostra

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
        MAX_LENGTH = 128

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
            processed_competences = [comp[:MAX_LENGTH] for comp in processed_competences]  # Limitar o comprimento das frases
            areas_list = self.extrair_areas(researcher_data.get('Áreas', {}))  # Obtém lista de áreas

            for area in all_areas_list.get('Áreas'):
                print(f"Área de pesquisa: {area}")
                # print(f"Competências extraídas: {competences}")
                print(f"Compet.pré-processadas: {processed_competences}")

                if area in valid_areas and processed_competences:
                    embeddings = model.encode(processed_competences, convert_to_tensor=True)
                    all_embeddings.extend(embeddings)  # Acumula os embeddings
                    mean_embedding = torch.mean(embeddings, dim=0)  # Calcula a média na GPU
                    X.append(mean_embedding)  # Move para CPU e converte para NumPy
                    y.append(area)

        return X, y

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
            areas_list = extrair_areas(researcher_data.get('Áreas', {}))
            for areas in areas_list:
                area = areas.get('Área')
                if area and area != 'desconhecido':
                    valid_areas.add(area)

        # Segunda passagem para preparar os dados
        for researcher_data in self.curricula_data:
            competences = self.competence_extractor.extract_competences(researcher_data)
            processed_competences = self.competence_extractor.preprocess_competences(competences)
            areas_list = extrair_areas(researcher_data.get('Áreas', {}))

            for areas in areas_list:
                area = areas.get('Área')

                if area in valid_areas and processed_competences:
                    embeddings = model.encode(processed_competences, convert_to_tensor=True, device=device)
                    all_embeddings.extend(embeddings)
                    mean_embedding = torch.mean(embeddings, dim=0).cpu().numpy()  # Calcula a média na GPU e move para CPU

                    # Calcular similaridade com as áreas de pesquisa
                    similarities = cosine_similarity([mean_embedding], list(area_embeddings.values()))[0]
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
                X.append(all_embeddings[i].cpu().numpy())  # Move o embedding para CPU e converte para NumPy
                y[i] = cluster  # Substitui o nome da área pelo ID do cluster

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
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)  # Adicionando zero_division=0
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)  # Adicionando zero_division=0
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)  # Adicionando zero_division=0

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