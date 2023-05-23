const express = require('express');
const path = require('path');

const app = express();
const port = process.env.PORT || 3000;

// Define o diretório de arquivos estáticos
app.use(express.static(path.join(__dirname, 'static')));

// Rota padrão que envia o arquivo index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'index.html'));
});

// Configuração para servir os arquivos CSS
/* app.use('/assets', express.static(path.join(__dirname, 'static', 'assets'))); */
/* app.use(express.static(path.join(__dirname, 'static'))); */
app.use(express.static('static'));

// Inicia o servidor
app.listen(port, () => {
  console.log(`Servidor está executando em http://localhost:${port}`);
});
