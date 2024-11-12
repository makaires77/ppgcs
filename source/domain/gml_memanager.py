import time
import torch
import pynvml
import psutil
import cpuinfo
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

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
        self.gpu_properties = torch.cuda.get_device_properties(0)
        self.cpu_info = cpuinfo.get_cpu_info()
        self.cpu_freq = psutil.cpu_freq()
        self.ram_info = psutil.virtual_memory()

    def print_hardware_info(self):
        print("Informações de Hardware:")
        try:
            if self.gpu_available:
                print(f"  GPU: {self.gpu_properties.name}")
                print(f"    Tensor Cores: {self.get_tensor_cores()}")
                print(f"    Núcleos CUDA: {(self.gpu_properties.multi_processor_count * 128) - self.get_tensor_cores()//128}")
                print(f"    Compute Capability: {self.gpu_properties.major}.{self.gpu_properties.minor}")
                print(f"    Frequência Máxima: {self.get_gpu_clock_rate()} GHz")
                print(f"    Memória Total: {self.gpu_properties.total_memory / 1024**3:.2f} GB")
            else:
                print("  GPU: Não disponível")
        except:
            print("  GPU: Não configurada corretamente, ou inexistente")
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

    # def get_gpu_parallel_capacity(self):
    #     """Retorna o número de núcleos CUDA da GPU, que representam a capacidade teórica de processamento paralelo."""
    #     if self.gpu_available:
    #         return self.gpu_properties.multi_processor_count * self.gpu_properties.max_threads_per_multi_processor
    #     else:
    #         return 0  # GPU não disponível

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
                        pynvml.nvmlDeviceValidateInforom(handle)
                        error_counts.append(0)  # Nenhum erro
                    except Exception as e:
                        print(f"Erro ao verificar a memória: {e}")
                        print(f"Bloco sequencial 4k com erro: {i}/{range(0, total_memory, 4096)}")
                        error_counts.append(-1)  # Erro desconhecido
                    addresses.append(i)
                    pbar.update(4096)

            import matplotlib.pyplot as plt

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

        # Verificação de tipo para garantir que 'model' seja um objeto SentenceTransformer
        if not isinstance(model, SentenceTransformer):
            raise TypeError("O argumento 'model' deve ser um objeto SentenceTransformer.")

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
            number_of_sample = 10
            instruct_per_sample = 1000
            estimated_time = model_size / self.estimate_cpu_parallel_throughput(number_of_sample, instruct_per_sample)
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