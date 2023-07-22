import express from 'express';
import cors from 'cors';
import path from 'path';

const app = express();
app.use(cors());
const port = 3000;

app.use(express.static(path.join(path.resolve(), 'static')));

app.get('/', (req, res) => {
  res.sendFile(path.join(path.resolve(), 'static', 'index.html'));
});

app.listen(port, () => {
  console.log(`Servidor iniciado em http://localhost:${port}`);
});