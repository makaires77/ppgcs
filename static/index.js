document.addEventListener('DOMContentLoaded', function () {
  // Elementos do formulário
  const teamFormElement = document.getElementById('team-form');
  const teamSelectElement = document.getElementById('team-select');

  // Função para carregar a lista de equipes
  function loadTeamList() {
    fetch('/static/equipes/')
      .then((response) => response.json())
      .then((data) => {
        teamSelectElement.innerHTML = ''; // Limpar a lista de equipes

        // Criar uma opção para cada equipe
        data.folders.forEach((folder) => {
          const option = document.createElement('option');
          option.value = folder;
          option.textContent = folder;
          teamSelectElement.appendChild(option);
        });
      })
      .catch((error) => {
        console.error('Erro ao carregar a lista de equipes:', error);
      });
  }

  // Função para carregar a lista de arquivos da equipe selecionada
  function loadFileList(equipe) {
    fetch(`/static/equipes/${equipe}`)
      .then((response) => response.json())
      .then((data) => {
        const fileListElement = document.getElementById('table-body');
        fileListElement.innerHTML = ''; // Limpar a lista de arquivos

        // Criar uma linha para cada arquivo
        data.files.forEach((file) => {
          const row = document.createElement('tr');
          const cell = document.createElement('td');
          cell.textContent = file;
          row.appendChild(cell);
          fileListElement.appendChild(row);
        });
      })
      .catch((error) => {
        console.error('Erro ao carregar a lista de arquivos:', error);
      });
  }

  // Adicionar evento de escuta no formulário
  teamFormElement.addEventListener('submit', function (event) {
    event.preventDefault(); // Evitar o envio do formulário
    const equipe = teamSelectElement.value;
    if (equipe) {
      loadFileList(equipe);
    }
  });

  // Carregar a lista de equipes ao carregar a página
  loadTeamList();
});
