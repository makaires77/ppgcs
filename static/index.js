// Função para buscar os nomes dos arquivos da pasta do GitHub
function fetchCSVFiles(team) {
    const url = `https://api.github.com/repositories/577948382/contents/_data/in_csv/${team}`;

    fetch(url)
      .then(response => response.json())
      .then(data => {
        const fileNames = data
          .filter(item => item.type === 'file' && item.name.endsWith('.csv'))
          .map(item => item.name);

        displayFileNames(fileNames);
      })
      .catch(error => console.error('Erro ao buscar arquivos CSV:', error));
}
  
  // Função para exibir os nomes dos arquivos na tabela
  function displayFileNames(fileNames) {
    const tableBody = document.getElementById('table-body');
    tableBody.innerHTML = '';
  
    fileNames.forEach(fileName => {
      const tableRow = document.createElement('tr');
      const fileNameCell = document.createElement('td');
      fileNameCell.textContent = fileName;
      tableRow.appendChild(fileNameCell);
      tableBody.appendChild(tableRow);
    });
  }
  
  // Função para adicionar um nome de arquivo à lista
  function addFileNameToList(fileName) {
    const fileList = document.getElementById('file-list');
    const listItem = document.createElement('li');
    listItem.textContent = fileName;
    fileList.appendChild(listItem);
  }
  
  // Evento de envio do formulário para buscar os arquivos CSV
  document.getElementById('team-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const teamSelect = document.getElementById('team-select');
    const selectedTeam = teamSelect.value;
    fetchCSVFiles(selectedTeam);
  });
  
  // Evento de envio do formulário para adicionar um nome de arquivo à lista
  document.getElementById('generate-list-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const fileNameInput = document.getElementById('file-name');
    const fileName = fileNameInput.value;
    addFileNameToList(fileName);
    fileNameInput.value = '';
  });
  
  // Evento de clique do botão para salvar a lista de nomes de arquivo
  document.getElementById('save-list-btn').addEventListener('click', function() {
    const fileList = document.getElementById('file-list');
    const fileNames = [...fileList.children].map(item => item.textContent);
    const fileContent = fileNames.join('\n');
  
    const blob = new Blob([fileContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
  
    const link = document.createElement('a');
    link.href = url;
    link.download = 'lista_nomes_arquivo.txt';
    link.click();
  });
  