import torch


def check_pytorch():
    try:
        import torch
        print("PyTorch version:", torch.__version__)
    except ImportError:
        print("PyTorch is not installed.")

    try:
        import torch_geometric
        print("PyTorch Geometric version:", torch_geometric.__version__)
    except ImportError:
        print("PyTorch Geometric is not installed.")


def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
        print("GPU Device Name:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU")

    return device


def try_amb():
    ## Visualizar versões dos principais componentes
    import os
    import pip
    import sys
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # !pip3 install shutup
    # import shutup; shutup.please()
    
    pyVer      = sys.version
    pipVer     = pip.__version__
    
    print('\nVERSÕES DAS PRINCIPAIS BIBLIOTECAS INSTALADAS NO ENVIROMENT')
    print('Interpretador em uso:', sys.executable)
    print('    Ambiente ativado:',os.environ['CONDA_DEFAULT_ENV'])
    print('     Python: '+pyVer, '\n        Pip:', pipVer,'\n'
         )

def try_gpu():
    print('\nVERSÕES DO PYTORCH E GPU DISPONÍVEIS')
    try:
        import torch
        print('    PyTorch:',torch.__version__)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Dispositivo:',device)
        print('Disponível :',device,torch.cuda.is_available(),' | Inicializado:',torch.cuda.is_initialized(),'| Capacidade:',torch.cuda.get_device_capability(device=None))
        print('Nome GPU   :',torch.cuda.get_device_name(0),'         | Quantidade:',torch.cuda.device_count(),'\n')
    except Exception as e:
        print('Erro ao configurar a GPU:',e,'\n')
        
def try_browser():
    print('\nVERSÕES O BROWSER E DO CHROMEDRIVER INSTALADAS')
    from selenium import webdriver

    try:
        driver = webdriver.Chrome()
        str1 = driver.capabilities['browserVersion']
        str2 = driver.capabilities['chrome']['chromedriverVersion'].split(' ')[0]
        print(f'     Versão do browser: {str1}')
        print(f'Versão do chromedriver: {str2}')
        driver.quit()

        if str1[0:3] != str2[0:3]: 
            print("Versões incompatíveis, atualizar chromedriver!")
    except Exception as e:
        print(e)