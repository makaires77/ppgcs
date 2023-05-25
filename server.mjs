const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
const port = 3000;

import path from 'path';

app.use(express.static(path.join(path.resolve(), 'static')));

app.get('/', (req, res) => {
  res.sendFile(path.join(path.resolve(), 'static', 'index.html'));
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
