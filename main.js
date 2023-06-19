// Função para consultar e exibir as informações das listas existentes
function consultarListas() {
    fetch('https://api.github.com/repos/makaires77/ppgcs/contents/_data/in_csv')
      .then(response => response.json())
      .then(data => {
        const listaSelect = document.getElementById('lista');
  
        // Limpar as opções existentes
        listaSelect.innerHTML = '';
  
        // Adicionar a opção padrão
        const optionDefault = document.createElement('option');
        optionDefault.value = '';
        optionDefault.textContent = 'Selecione uma lista existente';
        listaSelect.appendChild(optionDefault);
  
        // Preencher as opções com os nomes dos arquivos CSV
        data.forEach(file => {
          if (file.name.endsWith('.csv')) {
            const option = document.createElement('option');
            option.value = file.name;
            option.textContent = file.name;
            listaSelect.appendChild(option);
          }
        });
      })
      .catch(error => {
        console.error('Erro ao consultar as listas:', error);
      });
  }
  
  // Evento ao carregar a página
  document.addEventListener('DOMContentLoaded', () => {
    // Consultar as listas existentes
    consultarListas();
  
    // Evento ao submeter o formulário
    document.getElementById('scrapForm').addEventListener('submit', event => {
      event.preventDefault();
  
      // Obter os valores do formulário
      const periodo = document.getElementById('periodo').value;
      const lista = document.getElementById('lista').value;
      const nome = document.getElementById('nome').value;
      const programa = document.getElementById('programa').value;
      const instituicao = document.getElementById('instituicao').value;
  
      // Chamar a função de scrap com os valores
      scrapData(periodo, lista, nome, programa, instituicao);
    });
  });
  
  /* Aqui está uma análise detalhada das principais partes do código:

  A função consultarListas() é responsável por fazer uma requisição à API do GitHub para obter a lista de arquivos no diretório _data/in_csv do repositório. Em seguida, ele preenche o elemento <select> com as opções correspondentes aos arquivos CSV encontrados.
  
  No evento DOMContentLoaded, que é acionado quando o DOM é carregado, a função consultarListas() é chamada para buscar as listas existentes e preencher o <select> correspondente.
  
  No evento de submissão do formulário (submit), o código impede o comportamento padrão do formulário (recarregar a página) usando event.preventDefault(). Em seguida, ele obtém os valores dos campos do formulário (periodo, lista, nome, programa, instituicao) e chama a função scrapData() para iniciar o processo de scraping com esses valores. */