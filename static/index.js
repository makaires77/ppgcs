import express from 'express';
import path from 'path';
<<<<<<< HEAD
=======
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174

const app = express();
const port = 3000;

<<<<<<< HEAD
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
=======
// Serve os arquivos estáticos a partir do diretório pai de __dirname (assumindo que o diretório pai é a raiz do projeto)
app.use(express.static(path.join(__dirname, '../static')));

// Rota principal
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../static/index.html'));
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
});

// Inicia o servidor
app.listen(port, () => {
  console.log(`Servidor iniciado em http://localhost:${port}`);
<<<<<<< HEAD
});
=======
});
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
