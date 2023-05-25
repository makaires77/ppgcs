const app = express();
app.use(express.json());

// Middleware para autenticação
app.use((req, res, next) => {
  const token = req.headers.authorization;
  if (token !== 'TOKEN_SECRETO') {
    res.status(401).send({ message: 'Unauthorized' });
  } else {
    next();
  }
});

app.post('/create-team', (req, res) => {
  const { teamName, researchersFile } = req.body;

  const researchers = [];
  fs.createReadStream(researchersFile)
    .pipe(csv())
    .on('data', (data) => researchers.push(data))
    .on('end', () => {
      amqp.connect('amqp://localhost', (err, conn) => {
        if (err) {
          console.error(`Erro ao conectar ao RabbitMQ: ${err}`);
          res.status(500).send({ message: 'Internal Server Error' });
        } else {
          conn.createChannel((err, ch) => {
            if (err) {
              console.error(`Erro ao criar canal do RabbitMQ: ${err}`);
              res.status(500).send({ message: 'Internal Server Error' });
            } else {
              ch.assertQueue('researchers_queue');
              researchers.forEach((researcher) => {
                ch.sendToQueue('researchers_queue', Buffer.from(JSON.stringify(researcher)));
              });
            }
          });

          const driver = neo4j.driver('bolt://localhost', neo4j.auth.basic('neo4j', 'password'));
          const session = driver.session();

          const writeTxResultPromise = session.writeTransaction(async (txc) => {
            const result = await txc.run(
              'CREATE (n:Team { name: $name }) RETURN n', { name: teamName }
            );
            return result.records;
          });

          writeTxResultPromise
            .then(() => {
              session.close();
              driver.close();
              const microservice = spawn('node', ['microservice.js']);
              microservice.on('error', (err) => {
                console.error(`Erro ao executar microservice: ${err}`);
                res.status(500).send({ message: 'Internal Server Error' });
              });
              res.send({ message: 'Sucesso!' });
            })
            .catch((err) => {
              console.error(`Erro ao executar consulta no Neo4j: ${err}`);
              res.status(500).send({ message: 'Internal Server Error' });
            });
        }
      });
    })
    .on('error', (err) => {
      console.error(`Erro ao ler o arquivo: ${err}`);
      res.status(500).send({ message: 'Internal Server Error' });
    });
});

app.listen(3000, () => console.log('Listening on port 3000'));