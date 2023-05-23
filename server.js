const express = require('express');
const path = require('path');

const app = express();
/* const port = 3000; */
const port = process.env.PORT || 8080;

// Configurar o diretório de arquivos estáticos
app.use(express.static(path.join(__dirname, 'static')));
/* app.use('/assets', express.static(path.join(__dirname, 'static', 'assets'))); */
/* app.use(express.static('static')); */

// Rota padrão que envia o arquivo index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'index.html'));
});

// Rota para a página /dashboard_discentes
app.get('/historico_2017_2020', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'historico_2017_2020.html'));
});

// Rota para a página /dashboard_programa
app.get('/dashboard_programa', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'dashboard_programa.html'));
});

// Rota para a página /dashboard_docentes
app.get('/dashboard_docentes', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'dashboard_docentes.html'));
});

// Rota para a página /dashboard_discentes
app.get('/dashboard_discentes', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'dashboard_discentes.html'));
});

// Inicia o servidor
app.listen(port, () => {
  console.log(`Servidor está executando em http://localhost:${port}`);
});