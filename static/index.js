import express from 'express';
import path from 'path';

const app = express();
const port = 3000;

// Define o diretório de arquivos estáticos
app.use(express.static(path.join(__dirname, 'static')));

// Rota principal
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Rota para criação de equipe
app.post('/create-team', (req, res) => {
  // Implemente o código para criação da equipe
  // ...
});

// Inicia o servidor
app.listen(port, () => {
  console.log(`Servidor iniciado em http://localhost:${port}`);
});
