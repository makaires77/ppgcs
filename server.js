const express = require('express');
const app = express();
const path = require('path');

// Configurar o diretório estático
app.use('/static', express.static(path.join(__dirname, 'static')));

// Rota para servir a página index.html
app.get('/static/index.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'index.html'));
});

// Outras rotas e configurações do servidor...

// Iniciar o servidor
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Servidor iniciado na porta ${port}`);
});