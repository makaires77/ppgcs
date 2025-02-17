import os
import sys
import glob
import psutil
import platform
import selenium
import subprocess
import numpy as np
from pathlib import Path
from string import Formatter
from datetime import timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

class EnvironmentSetup:
    def __init__(self):
        # Inicialização sem definir o root_path
        self.root_path = None

    def set_root_path(self, folder_name):
        """Define o caminho raiz baseado no nome da pasta fornecido."""
        self.root_path = Path.home() / folder_name
        self.ensure_directories()

    def find_repo_root(self, path='.', depth=10):
        """
        Busca o arquivo .git e retorna string com a pasta raiz do repositório.
        """
        # Prevent infinite recursion by limiting depth
        if depth < 0:
            return None
        path = Path(path).absolute()
        if (path / '.git').is_dir():
            return path
        return self.find_repo_root(str(path.parent), depth-1)
    
    def ensure_directories(self):
        """Garante a existência das pastas necessárias."""
        if self.root_path is None:
            print("Root path não definido.")
            return
        
        if not self.root_path.exists():
            self.root_path.mkdir(parents=True, exist_ok=True)
        
        subfolders = ['in_zip', 'in_xls', 'in_pdf', 'in_csv', 'in_json', 'out_fig', 'out_json']
        for folder in subfolders:
            data_folder = '_data'
            (self.root_path / data_folder / folder).mkdir(parents=True, exist_ok=True)
        print('Todas as pastas necessárias foram garantidas.')

    def strfdelta(self, tdelta, fmt='{H:02}h {M:02}m {S:02}s', inputtype='timedelta'):
        if inputtype == 'timedelta':
            remainder = int(tdelta.total_seconds())
        else:
            conversion_factors = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400, 'w': 604800}
            remainder = int(tdelta) * conversion_factors[inputtype]
        f = Formatter()
        desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
        possible_fields = ('W', 'D', 'H', 'M', 'S')
        constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
        values = {}
        for field in possible_fields:
            if field in desired_fields and field in constants:
                values[field], remainder = divmod(remainder, constants[field])
        return f.format(fmt, **values)

    def tempo(self, start, end):
        t = end - start
        tempo = timedelta(
            weeks=t // (3600 * 24 * 7),
            days=t // (3600 * 24) % 7,
            hours=t // 3600 % 24,
            minutes=t // 60 % 60,
            seconds=t % 60
        )
        fmt = '{H:02}:{M:02}:{S:02}'
        return self.strfdelta(tempo, fmt=fmt, inputtype='timedelta')

    def check_path(self):
        try:
            path_output = subprocess.check_output("echo $PATH", shell=True, text=True).strip()
            return path_output
        except Exception as e:
            print("Erro ao obter PATH:", e)
        return ""

    def check_nvcc(self):
        os_type = platform.system()
        if os_type == "Linux":
            nvcc_path = "/usr/local/cuda/bin/nvcc"
            if not Path(nvcc_path).exists():
                print("NVCC not found in the expected location for Linux.")
                return
        elif os_type == "Windows":
            cuda_paths = glob.glob('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin/nvcc.exe')
            nvcc_path = cuda_paths[0] if cuda_paths else None
            if not nvcc_path:
                print("NVCC not found in the default installation paths for Windows.")
                return
        else:
            print("Unsupported Operating System.")
            return
        # Try to retrieve the NVCC version using the found path
        try:
            nvcc_output = subprocess.check_output([nvcc_path, '-V'], stderr=subprocess.STDOUT, text=True)
            print(nvcc_output)
        except subprocess.CalledProcessError as e:
            print(f"Failed to execute NVCC: {e.output}")
        except OSError as e:
            print(f"NVCC not found: {e.strerror}")

    def try_amb(self):
        pyVer = sys.version
        pipVer = subprocess.run(['pip', '--version'], check=True, stdout=subprocess.PIPE, text=True).stdout
        try:
            conda_env = os.environ['CONDA_DEFAULT_ENV']
        except KeyError:
            conda_env = 'Não disponível'
        print('\nVERSÕES DAS PRINCIPAIS BIBLIOTECAS INSTALADAS NO ENVIROMENT')
        print('Ambiente Conda ativo:', conda_env)
        print('Interpretador em uso:', sys.executable)
        print(' Python:', pyVer, '\n    Pip:', pipVer)

    def get_cpu_info_windows(self):
        try:
            return subprocess.check_output("wmic cpu get Name", shell=True, text=True).split('\n')[1].strip()
        except Exception:
            return "Informação não disponível"

    def get_cpu_info_unix(self):
        try:
            return subprocess.check_output("lscpu", shell=True, text=True)
        except:
            try:
                return subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True, text=True).strip()
            except:
                return "Informação não disponível"

    def get_processor_info_linux(self):
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    return line.split(":")[1].strip()
            return "Modelo de processador não encontrado."
        except FileNotFoundError:
            return "Informação não disponível"

    def try_cpu(self):
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        cpu_times_percent = psutil.cpu_times_percent(interval=1)
        if platform.system() == "Windows":
            cpu_model = self.get_cpu_info_windows()
        else:
            cpu_model = self.get_cpu_info_unix()
        cpu_brand = platform.processor()
        cpu_architecture = platform.architecture()[0]
        cpu_machine_type = platform.machine()
        ram = psutil.virtual_memory()
        total_ram = ram.total / (1024 ** 3)  # Em GB
        used_ram = ram.used / (1024 ** 3)  # Em GB
        disk = psutil.disk_usage('/')
        total_disk = disk.total / (1024 ** 3)  # Em GB
        used_disk = disk.used / (1024 ** 3)  # Em GB
        free_disk = total_disk - used_disk
        used_disk_percent = (used_disk / total_disk) * 100
        free_disk_percent = 100 - used_disk_percent
        print(f"Processador em uso: {cpu_model}")
        print(f"Arquitetura modelo: {cpu_brand}")
        print(f"Arquitetura em uso: {cpu_architecture}")
        print(f"Frequência das CPU: {np.round(cpu_freq.current, 2)} MHz")
        print(f"  Qte CPUs físicas: {cpu_count_physical}")
        print(f"  Qte CPUs lógicas: {cpu_count_logical}")
        print(f"Carga total na CPU: {cpu_percent}%")
        print(f"Ocupação atual CPU: user={cpu_times_percent.user}%, system={cpu_times_percent.system}%, idle={cpu_times_percent.idle}%")
        print(f"\nEspaço Total em disco: {total_disk:.2f} GB")
        print(f"Espaço em disco usado: {used_disk:.2f} GB {used_disk_percent:.1f}%")
        print(f"Espaço em disco livre: {free_disk:.2f} GB {free_disk_percent:.1f}%")
        print(f"\nCapacidade memórias RAM: {total_ram:.2f} GB")
        print(f"Utilização atual da RAM: {used_ram:.2f} GB")

    def try_gpu(self):
        print('\nVERSÕES DOS DRIVERS CUDA, PYTORCH E GPU')
        self.check_nvcc()
        try:
            import torch
            print('    PyTorch:', torch.__version__)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print('Dispositivo:', device)
            print('Disponível:', device, torch.cuda.is_available(), ' | Inicializado:', torch.cuda.is_initialized(), '| Capacidade:', torch.cuda.get_device_capability(device=None))
            print('Nome GPU:', torch.cuda.get_device_name(0), ' | Quantidade:', torch.cuda.device_count())
        except Exception as e:
            print('  ERRO!! Ao configurar a GPU:', e, '\n')

    def try_browser(self):
        print('VERSÕES DO BROWSER E DO CHROMEDRIVER INSTALADAS')
        try:
            if platform.system() == "Windows":
                driver_path = self.find_repo_root(os.getcwd()) / 'chromedriver/chromedriver.exe' # type: ignore
            else:
                driver_path = self.find_repo_root(os.getcwd()) / 'chromedriver/chromedriver' # type: ignore
            print(driver_path)
            service = Service(driver_path)
            driver = webdriver.Chrome(service=service)
            str1 = driver.capabilities['browserVersion']
            str2 = driver.capabilities['chrome']['chromedriverVersion'].split(' ')[0]
            print(f'Versão do GoogleChrome: {str1}')
            print(f'Versão do chromedriver: {str2}')
            driver.quit()
            if str1[0:3] != str2[0:3]:
                print(f"Versões principais Chrome {str1} e Chromedriver {str2} distintas!")
                print(f"Verificar necessidade de atualização!\n")
                # TO-DO Run script de forced update aqui!
                
                # print(f'  Baixar versão atualizada do Chromedriver em:')
                # print(f'  https://googlechromelabs.github.io/chrome-for-testing/#stable')
                # print(f'     Ex:. Chromedriver Versão 119 para Windows:')
                # print(f'	   https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/119.0.6045.105/win64/chromedriver-win64.zip')
                # print(f'     Ex:. Chromedriver Versão 119 para Linux:')
                # print(f'       https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/119.0.6045.105/linux64/chromedriver-linux64.zip')
        except Exception as e:
            print(e)

    def try_chromedriver(self):
        try:
            import os
            os.listdir(self.root_path)
            print(self.root_path)
            return self.root_path
        except Exception as e:
            print(e)
            return self.root_path

    def try_folders(self, drives, pastas, pastasraiz):
        caminho = None
        for drive in drives:
            for pasta in pastas:
                for pastaraiz in pastasraiz:
                    caminho_testado = os.path.join(drive, pasta, pastaraiz)
                    chromedriver_path = os.path.join(caminho_testado, 'chromedriver', 'chromedriver.exe' if os.name == 'nt' else 'chromedriver')
                    if os.path.isfile(chromedriver_path):
                        print(f"Listing files in: {caminho_testado}")
                        print(os.listdir(caminho_testado))
                        caminho = os.path.join(caminho_testado, '')
                        return caminho
        if caminho is None:
            caminho = './home/mak/fioce/'
            raise FileNotFoundError("Chromedriver could not be located in the specified directories.")
        return caminho


    def preparar_pastas(self):
        subfolders = {'in_zip': 'in_zip', 'in_xls': 'in_xls','in_pdf': 'in_pdf',  'in_csv': 'in_csv', 'in_json': 'in_json', 'out_fig': 'out_fig', 'out_json': 'out_json'}
        
        for key, folder_name in subfolders.items():
            root_folder = self.find_repo_root()
            data_folder = os.path.join(str(root_folder),'_data')
            folder_path = self.root_path / data_folder # type: ignore
            if folder_path.exists():
                print(f"Pasta para {key} já existe!")
            else:
                folder_path.mkdir()
                print(f"Pasta para {key} criada com sucesso!")
        
        paths = {key: str(folder_path) for key, folder_path in subfolders.items()}
        
        print('\nCaminho da pasta raiz:', str(self.root_path))
        for key, path in paths.items():
            print(f'Caminho para {key}:', path)

# Exemplo de uso fora da classe:
if __name__ == "__main__":
    folder_name = input("Digite o nome da pasta principal: ")  # Permite ao usuário definir o nome da pasta
    preparer = EnvironmentSetup()
    preparer.try_cpu()
    preparer.try_gpu()
    preparer.try_amb()
    preparer.try_browser()
    preparer.set_root_path(folder_name)
    # preparer.preparar_pastas()