import os
import glob
import getpass
import platform
import subprocess
import requests
from bs4 import BeautifulSoup
from zipfile import ZipFile
from pathlib import Path
import shutil
import stat

class ChromeDriverManager:
    def __init__(self):
        self.repo_root = self.find_repo_root()
        if not self.repo_root:
            print("Não foi possível localizar a raiz do repositório.")
            print("Antes de iniciar, instale o Git e inicialize o repositório.")
            return

    def get_platform_code(self):
        os_platform = platform.system().lower()
        if os_platform == 'linux':
            return 'linux64'
        elif os_platform == 'darwin':
            if platform.machine() == 'arm64':
                return 'mac-arm64'
            else:
                return 'mac-x64'
        elif os_platform == 'windows':
            is_64bits = platform.machine().endswith('64')
            return 'win64' if is_64bits else 'win32'
        return None

    def find_download_url(self, html, file, platform_code):
        soup = BeautifulSoup(html, 'html.parser')
        rows = soup.find_all('tr', class_='status-ok')
        for row in rows:
            platformfile = row.text.split('http')[0]
            if platformfile == file + platform_code:
                if 'chromedriver' in row.text.lower():
                    if platform_code in row.find_all('th')[1].text.lower():
                        url_cell = row.find('td').find('code')
                        if url_cell:
                            return url_cell.text.strip()
        return None

    def download_file(self, url, dest_folder):
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder, exist_ok=True)
        local_filename = url.split('/')[-1]
        full_path = os.path.join(dest_folder, local_filename)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(full_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return full_path

    def unzip_file(self, zip_path, extract_to):
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)  # Remove arquivo zip após extração

    def get_chrome_version(self):
        try:
            if platform.system() == "Linux":  # Adaptando para WSL
                # # Encontra o caminho completo do Chrome no WSL
                # chrome_path = subprocess.run(["which", "google-chrome-stable"], 
                #                             capture_output=True, text=True).stdout.strip()
                # print(f"Caminho do Chrome detectado: {chrome_path}") # Adicione esta linha para depuração

                # command = f"{chrome_path} --version"
                # Usa o comando 'which' para encontrar o caminho do Chrome no WSL
                chrome_path_output = subprocess.run(["which", "google-chrome-stable"], 
                                                    capture_output=True, text=True)
                chrome_path_output.check_returncode()  # Verifica se o comando foi executado com sucesso

                chrome_path = chrome_path_output.stdout.strip()
                print(f"Caminho do Chrome detectado: {chrome_path}") 

                # Usa o caminho encontrado para obter a versão
                command = f"{chrome_path} --version" 
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                result.check_returncode()

                if result.stdout:
                    version_parts = result.stdout.strip().split()
                    if version_parts:
                        version = [part for part in version_parts if '.' in part][0].split('.')[0]
                        return version

                raise subprocess.CalledProcessError(returncode=1, cmd=command)             
            elif platform.system() == "Windows":
                command = "reg query \"HKEY_CURRENT_USER\\Software\\Google\\Chrome\\BLBeacon\" /v version"
            elif platform.system() == "Darwin":
                command = "/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --version"
            else:  # Assuming Linux for else
                command = "google-chrome --version"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            result.check_returncode()  # Raises CalledProcessError for non-zero exit codes
            
            # Verifica se a saída é não vazia e contém a versão
            if result.stdout:
                version_parts = result.stdout.strip().split()
                if version_parts:
                    # Assume que a versão é o último elemento após split e contém pontos.
                    version = [part for part in version_parts if '.' in part][0].split('.')[0]
                    return version
            # Se não encontrar uma versão válida, trata como não instalado
            raise subprocess.CalledProcessError(returncode=1, cmd=command)
        except subprocess.CalledProcessError:
            print("Google Chrome not found. Attempting to install...")
            self.install_google_chrome()
            return None

    def get_chromedriver_version(self):
        """Tenta encontrar e retornar a versão do Chromedriver. Caso não encontre, dispara a atualização."""
        if platform.system() == "Windows":
            chromedriver_path = os.path.join(self.repo_root, 'chromedriver', 'chromedriver.exe')
        elif platform.system() == "Darwin":
            chromedriver_path = os.path.join(self.repo_root, 'chromedriver', 'chromedriver')
        else:  # Assume Linux and other Unix-like systems
            chromedriver_path = os.path.join(self.repo_root, 'chromedriver', 'chromedriver')

        # Verifica se o executável do Chromedriver existe
        if not Path(chromedriver_path).is_file():
            print("Chromedriver não encontrado. Iniciando o processo de atualização...")
            return None

        # Se o arquivo existir, procede com a obtenção da versão
        command = f'"{chromedriver_path}" --version'
        try:
            version = subprocess.run(command, shell=True, capture_output=True, text=True).stdout.strip().split()[1]
            return version.split('.')[0]
        except subprocess.CalledProcessError as e:
            print(f"Erro ao tentar obter a versão do Chromedriver: {e}")
            return None

    def verify_version(self, executable, filepath):
        command = f'{filepath} --version'
        encoding = 'utf-8' if platform.system() == 'Windows' else 'system-default-encoding'
        version = subprocess.run(command, shell=True, capture_output=True, encoding=encoding, errors='replace')
        if version.stdout.split():
            complete = version.stdout.strip()
            principal = complete.split('.')[0].split(' ')[-1]
            print(f'Executável {executable} versão: {principal} em {filepath}')
        elif executable == 'GoogleChrome':
            version = self.get_chrome_version()
            print(f'Executável {executable} versão: {version} em {filepath}')
            principal = version
        else:
            print(f'Não foi possível extrair a versão de {executable} no caminho {filepath}')
            principal = None
        return principal

    def find_repo_root(self, path='.', depth=10):
        if depth < 0:
            return None
        path = Path(path).absolute()
        if (path / '.git').is_dir():
            return path
        return self.find_repo_root(path.parent, depth-1)

    def install_google_chrome(self):
            os_platform = platform.system().lower()
            try:
                if os_platform == 'windows':
                    print("Instalando o Google Chrome no Windows...")
                    subprocess.run(["choco", "install", "googlechrome"], check=True)
                elif os_platform == 'darwin':
                    print("Instalando o Google Chrome no macOS...")
                    subprocess.run(["brew", "install", "--cask", "google-chrome"], check=True)
                else:
                    print("Atualizando a lista de pacotes e instalando o Google Chrome no Linux...")
                    chrome_pkg = "google-chrome-stable_current_amd64.deb"
                    wget_command = ["wget", f"https://dl.google.com/linux/direct/{chrome_pkg}"]
                    update_command = ["sudo", "apt-get", "update"]
                    install_command = ["sudo", "apt-get", "install", "-y", f"./{chrome_pkg}"]
                    fix_command = ["sudo", "apt", "--fix-broken", "install", "-y"]
                    sudo_password = getpass.getpass("Digite sua senha de sudo: ")

                    # Executando comandos com tratamento de erros e logs
                    print("Baixando o pacote do Chrome...")
                    subprocess.run(wget_command, check=True)

                    print("Atualizando a lista de pacotes...")
                    try:
                        # Inclui a confirmação 'Y\n' no input para o comando 'sudo apt-get update'
                        update_process = subprocess.run(update_command, input=sudo_password + '\nY\n', text=True, capture_output=True, timeout=60)  

                        if update_process.returncode != 0:
                            # Tratamento de erro aprimorado para a atualização da lista de pacotes
                            error_message = f"Erro ao atualizar a lista de pacotes: {update_process.stderr.decode()}"
                            raise subprocess.CalledProcessError(returncode=update_process.returncode, 
                                                            cmd=update_command, 
                                                            stderr=error_message)

                        install_process = subprocess.run(install_command, input=sudo_password, text=True, capture_output=True)
                        if install_process.returncode != 0:
                            raise subprocess.CalledProcessError(returncode=install_process.returncode,
                                                            cmd=install_command,
                                                            stderr=install_process.stderr)
                    except subprocess.CalledProcessError as e:
                        print(f"Erro ao instalar o Google Chrome: {e}")
                        if e.stderr:
                            print(f"Detalhes do erro: {e.stderr}") 
                        fix_process = subprocess.run(fix_command, input=sudo_password, text=True, capture_output=True)
                        if fix_process.returncode != 0:
                            raise subprocess.CalledProcessError(returncode=fix_process.returncode,
                                                            cmd=fix_command,
                                                            stderr=fix_process.stderr)
                        # Tenta reinstalar após corrigir as dependências
                        print("Reinstalando o Chrome...")
                        subprocess.run(install_command, input=sudo_password, text=True, check=True)

                    except subprocess.TimeoutExpired:
                        print("Erro: Tempo limite excedido durante a atualização da lista de pacotes.")

                    finally:
                        print("Removendo o pacote baixado...")
                        if os.path.exists(chrome_pkg):
                            os.remove(chrome_pkg)
                        self.remove_deb_files("google-chrome-stable_current_amd64.deb")

            except subprocess.CalledProcessError as e:
                print(f"Erro ao instalar o Google Chrome: {e}")
                if e.stderr:
                    print(f"Detalhes do erro: {e.stderr}")  

    def remove_deb_files(self, deb_file_name):
        # Remove os arquivos .deb e suas versões numeradas na pasta corrente
        for deb_file in glob.glob(deb_file_name + "*"):
            try:
                os.remove(deb_file)
                print(f"Removed {deb_file}")
            except Exception as e:
                print(f"Error removing {deb_file}: {e}")

    def update_chromedriver(self):
        print("Iniciando o processo de atualização do Chromedriver...")
        
        # Obtém a URL de download do Chromedriver compatível
        page_url = 'https://googlechromelabs.github.io/chrome-for-testing/'
        platform_code = self.get_platform_code()
        if not platform_code:
            print(f"Plataforma {platform_code} não é suportada por esta aplicação.")
            print(f"Usar Linux, Windows ou WSL.")
            return

        response = requests.get(page_url)
        if response.status_code != 200:
            print("Erro ao acessar a página de download do Chromedriver.")
            print("Verifique a conexão com a internet.")
            return

        chromedriver_url = self.find_download_url(response.text, 'chromedriver', platform_code)

        if not chromedriver_url:
            print(f"Não foram encontrados downloads necessários para a plataforma atual {platform_code}.")
            print("Para rodar esta aplicação deve-se utilizar Linux ou usar WSL2 no ambiente Windows.")
            return

        if chromedriver_url:
            print(f"\nLinks para versões estáveis mais recentes detectados:")
            print(f"Chromedriver: {chromedriver_url}")
            google_chrome_url = self.find_download_url(response.text,'chrome',platform_code)
            
            # Define a pasta temporária e a pasta destino
            dest_folder = Path(self.repo_root) / 'chromedriver'
            temp_extract_folder = Path(self.repo_root) / "temp_chromedriver"
            temp_extract_folder.mkdir(parents=True, exist_ok=True)

            # Download do Chromedriver para a pasta temporária
            print(f"\nFazendo download do Chromedriver...")
            chromedriver_zip_path = self.download_file(chromedriver_url, temp_extract_folder)

            # Extrai o Chromedriver
            print("Descompactando o Chromedriver...")
            self.unzip_file(chromedriver_zip_path, temp_extract_folder)
            
            # Procura o arquivo do Chromedriver na pasta temporária
            chromedriver_filename = "chromedriver.exe" if os.name == 'nt' else "chromedriver"
            found_files = list(temp_extract_folder.glob(f"**/{chromedriver_filename}"))
            if found_files:
                chromedriver_path = found_files[0]  # Assume que o primeiro arquivo encontrado é o desejado

                # Movendo o executável para a pasta destino
                print("Movendo o executável para a pasta destino...")
                dest_chromedriver_path = shutil.move(str(chromedriver_path), str(dest_folder / chromedriver_filename))

                # Adiciona permissões de execução ao Chromedriver (necessário em Unix-like systems)
                if os.name != 'nt':  # Se não for Windows, modifica as permissões
                    os.chmod(dest_chromedriver_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
                print("Chromedriver atualizado com sucesso.")

                # Apagando os arquivos temporários
                print("Apagando os arquivos temporários...")
                shutil.rmtree(temp_extract_folder)

                # Exibe a versão do Chromedriver descompactado
                version_command = f"{dest_chromedriver_path} --version"
                version = subprocess.run(version_command, shell=True, capture_output=True, text=True)
                print(f"\nVersão do Chromedriver atualizado: {version.stdout.strip()}")
            else:
                print(f"Não foi possível encontrar o Chromedriver em {temp_extract_folder}")
                # Limpa a pasta temporária mesmo assim
                shutil.rmtree(temp_extract_folder)
        else:
            print("Não foi possível encontrar uma versão compatível do Chromedriver para download.")

    def main(self):
        gcversion = self.get_chrome_version()
        cdversion = self.get_chromedriver_version()
        if gcversion != cdversion:
            print(f"Versões {gcversion} Chrome e {cdversion} Chromedriver estão incompatíveis")
            print("Atualizando o Chrome...")  # Adicione esta linha para indicar que a atualização do Chrome está sendo iniciada
            self.install_google_chrome()

            # Verifica a versão do Chrome após a atualização
            new_gcversion = self.get_chrome_version()
            if new_gcversion != gcversion:
                print(f"Chrome atualizado com sucesso para a versão {new_gcversion}")
            else:
                print("Falha ao atualizar o Chrome. Verifique os logs para mais detalhes.")
        else:
            print(f"Versões {gcversion} Chrome e {cdversion} Chromedriver estão compatíveis")