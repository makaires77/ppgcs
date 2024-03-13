<<<<<<< HEAD
const express = require('express');
const cors = require('cors');
=======
import express from 'express';
import cors from 'cors';
import path from 'path';
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174

const app = express();
app.use(cors());
const port = 3000;

<<<<<<< HEAD
import path from 'path';

=======
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
app.use(express.static(path.join(path.resolve(), 'static')));

app.get('/', (req, res) => {
  res.sendFile(path.join(path.resolve(), 'static', 'index.html'));
});

app.listen(port, () => {
<<<<<<< HEAD
  console.log(`Server is running on port ${port}`);
});
=======
  console.log(`Servidor iniciado em http://localhost:${port}`);
});
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
