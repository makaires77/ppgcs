import fs from 'fs';
import csv from 'csv-parser';
import amqp from 'amqplib/callback_api';
import neo4j from 'neo4j-driver';
import express from 'express';

const app = express();
app.use(express.json());

// Middleware para autenticação
app.use((req, res, next) => {
  const token = req.headers.authorization;
  if (token !== 'R2023') {
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
      const rabbitMQUrl = 'amqp://localhost';

      amqp.connect(rabbitMQUrl, (err, conn) => {
        if (err) {
          console.error(`Erro ao conectar ao RabbitMQ: ${err}`);
          res.status(500).send({ message: 'Internal Server Error' });
          return;
        }

        conn.createChannel((err, ch) => {
          if (err) {
            console.error(`Erro ao criar canal do RabbitMQ: ${err}`);
            res.status(500).send({ message: 'Internal Server Error' });
            return;
          }

          ch.assertQueue('researchers_queue');
          for (const researcher of researchers) {
            const researcherJson = JSON.stringify(researcher);
            ch.sendToQueue('researchers_queue', Buffer.from(researcherJson));
          }

          ch.close();
          conn.close();

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
              res.send({ message: 'Sucesso!' });
            })
            .catch((err) => {
              console.error(`Erro ao executar consulta no Neo4j: ${err}`);
              res.status(500).send({ message: 'Internal Server Error' });
            });
        });
      });
    })
    .on('error', (err) => {
      console.error(`Erro ao ler o arquivo: ${err}`);
      res.status(500).send({ message: 'Internal Server Error' });
    });
});

export default app;
