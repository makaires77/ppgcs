const express = require('express');
const multer = require('multer');
const fs = require('fs');
const app = express();
const port = 3000;
const path = require('path');

app.use(express.json());

// Middleware para servir arquivos estáticos
app.use(express.static(path.join(__dirname, 'static')));
app.use(express.json());

// Rotas GET
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'index.html'));
});

app.get('/historico', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'historico.html'));
});

app.get('/dashboard_programa', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'dashboard_programa.html'));
});

app.get('/dashboard_docentes', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'dashboard_docentes.html'));
});

app.get('/dashboard_discentes', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'dashboard_discentes.html'));
});

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    // define o diretório de destino
    const dir = './static/equipes/';
    fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    // define o nome do arquivo
    const teamName = req.body['team-input'];
    cb(null, `${teamName}_${file.originalname}`);
  }
});

const upload = multer({ storage: storage });

app.post('/create-team', (req, res) => {
  // O token de autorização é "TOKEN_SECRETO"
  const auth = req.headers.authorization;
  if (auth !== 'R2023') {
    res.status(403).json({message: 'Invalid token'});
    return;
  }

  upload.any()(req, res, (err) => {
    if (err) {
      res.status(500).json({error: err.message});
      return;
    }

    // Enviando uma resposta para o cliente
    res.status(200).json({message: 'Team created'});
  });
});

// Iniciar o servidor
app.listen(port, () => {
  console.log(`Servidor rodando na porta ${port}`);
});
