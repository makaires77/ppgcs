import subprocess
import pkg_resources
import sys
from getpass import getpass
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver

class EnvironmentValidator:
    def __init__(self):
        self.required_packages = [
            "requests", "numpy", "pandas", "networkx", "seaborn", "torch", "matplotlib", "h5py", "psutil", "nltk", "PyPDF2", "neo4j", "tqdm", "flask", "urllib3", "py2neo", "pyjarowinkler", "IPython", "selenium", "webdriver_manager", "bs4", "PIL", "sklearn",
        ]
    
    def check_packages(self):
        missing_packages = []
        for package in self.required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print("Missing packages:")
            for pkg in missing_packages:
                print(f"- {pkg}")
            sys.exit(1)
        else:
            print("All required Python packages are installed.")
    
    def check_chrome_and_chromedriver(self):
        try:
            # As configurações do WebDriver Manager e Selenium vão aqui
            options = Options()
            options.headless = True
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            print("Google Chrome e Chromedriver estão instalados e compatíveis.")
            driver.quit()
        except:
            print("Google Chrome não detectado...")
            try:
                self.install_google_chrome_linux()
                print("Instalação concluída, testando execução...")
                try:
                    options = Options()
                    options.headless = True
                    # Tente instalar o Chromedriver e iniciar o Chrome
                    service = Service(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=options)
                    print("Google Chrome e Chromedriver estão instalados e compatíveis.")
                    driver.quit()
                except Exception as e:
                    print(f"Ocorreu um erro ao verificar o Google Chrome e Chromedriver: {e}")
            except Exception as e:
                    print(f"Ocorreu um erro ao instalar o Google Chrome: {e}")
            # Verifica se está rodando em um Jupyter notebook
            if 'ipykernel' in sys.modules:
                # Ambiente de notebook; evita usar sys.exit()
                print("Detectado ambiente de notebook Jupyter/IPython. Não encerrando o kernel.")
            else:
                # Não está em um notebook; é seguro usar sys.exit()
                sys.exit(1)

    def install_google_chrome_linux(self):
        print("Iniciando a instalação do Google Chrome para Linux...")
        sudo_password = getpass("Senha do sudo necessária para instalação do Google Chrome: ")

        # Comando para instalação do Google Chrome
        commands = """
        wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
        sudo -S dpkg -i google-chrome-stable_current_amd64.deb
        sudo apt-get -f install -y
        rm google-chrome-stable_current_amd64.deb
        """
        
        # Executar cada comando separadamente para melhor feedback
        for command in commands.strip().split('\n'):
            print(f"Executando: {command}")
            process = subprocess.Popen(['sudo', '-S'] + command.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, bufsize=1)
            
            # Enviar senha do sudo
            process.stdin.write(sudo_password + '\n')
            process.stdin.flush()

            # Ler a saída em tempo real
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # Verificar por erros
            stderr = process.stderr.read()
            if stderr:
                print("Erro:", stderr.strip())
            if process.returncode != 0:
                print(f"Comando falhou com código {process.returncode}")
                break
        
        print("Google Chrome instalado com sucesso.")

    # def install_google_chrome_linux(self):
    #     print("Iniciando a instalação do Google Chrome para Linux...")
    #     # Solicita a senha do sudo ao usuário
    #     sudo_password = getpass("Senha do sudo necessária para instalação do Google Chrome: ")
        
    #     install_script = """
    #     wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
    #     sudo -S dpkg -i google-chrome-stable_current_amd64.deb
    #     sudo apt-get -f install -y
    #     rm google-chrome-stable_current_amd64.deb
    #     """
    #     try:
    #         # Executa o script de instalação passando a senha do sudo
    #         process = subprocess.Popen('sudo -S -p "" bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
    #         stdout, stderr = process.communicate(input=sudo_password + '\n' + install_script)
    #         if process.returncode == 0:
    #             print("Google Chrome instalado com sucesso.")
    #         else:
    #             print(f"Falha ao instalar o Google Chrome: {stderr}")
    #     except subprocess.CalledProcessError as e:
    #         print(f"Falha ao instalar o Google Chrome: {e}")
    #         if 'ipykernel' in sys.modules:
    #             print("Detectado ambiente de notebook Jupyter/IPython. Não encerrando o kernel.")
    #         else:
    #             sys.exit(1)

    def run_checks(self):
        print("\nChecking for required Python packages...")
        self.check_packages()
        
        print("\nChecking for Google Chrome and Chromedriver...")
        self.check_chrome_and_chromedriver()

if __name__ == "__main__":
    checker = EnvironmentValidator()
    checker.run_checks()