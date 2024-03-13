<<<<<<< HEAD
=======
// pkg\infrastructure\mongo\write_lattes.go
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
package mongo

import (
	"context"
	"log"
<<<<<<< HEAD
	"time"
=======
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

<<<<<<< HEAD
	"github.com/makaires77/ppgcs/pkg/domain/scrap_lattes"
)

// MongoWriter é uma implementação do escritor Lattes usando MongoDB.
=======
	"github.com/makaires77/ppgcs/pkg/domain/researcher"
)

// MongoWriter is an implementation of Lattes writer using MongoDB.
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
type MongoWriter struct {
	client     *mongo.Client
	database   *mongo.Database
	collection *mongo.Collection
}

<<<<<<< HEAD
// NewMongoWriter cria uma nova instância de MongoWriter.
=======
// NewMongoWriter creates a new instance of MongoWriter.
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
func NewMongoWriter(connectionString, databaseName, collectionName string) (*MongoWriter, error) {
	clientOptions := options.Client().ApplyURI(connectionString)

	client, err := mongo.Connect(context.Background(), clientOptions)
	if err != nil {
		return nil, err
	}

	database := client.Database(databaseName)
	collection := database.Collection(collectionName)

	return &MongoWriter{
		client:     client,
		database:   database,
		collection: collection,
	}, nil
}

<<<<<<< HEAD
// WritePesquisador escreve os dados do pesquisador no MongoDB.
func (w *MongoWriter) WritePesquisador(pesquisador *scrap_lattes.Pesquisador) error {
	_, err := w.collection.InsertOne(context.Background(), pesquisador)
	if err != nil {
		log.Printf("Erro ao escrever os dados do pesquisador no MongoDB: %s\n", err)
		return err
	}

	log.Println("Dados do pesquisador escritos com sucesso no MongoDB!")
=======
// WriteResearcher writes researcher data to MongoDB.
func (w *MongoWriter) WriteResearcher(researcher *researcher.Researcher) error {
	_, err := w.collection.InsertOne(context.Background(), researcher)
	if err != nil {
		log.Printf("Error writing researcher data to MongoDB: %s\n", err)
		return err
	}

	log.Println("Researcher data written successfully to MongoDB!")
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174

	return nil
}

<<<<<<< HEAD
// Close fecha a conexão com o MongoDB.
func (w *MongoWriter) Close() {
	err := w.client.Disconnect(context.Background())
	if err != nil {
		log.Printf("Erro ao fechar a conexão com o MongoDB: %s\n", err)
=======
// Close closes the connection to MongoDB.
func (w *MongoWriter) Close() {
	err := w.client.Disconnect(context.Background())
	if err != nil {
		log.Printf("Error closing connection to MongoDB: %s\n", err)
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
	}
}

// Exemplo de uso

<<<<<<< HEAD
func main() {
	connectionString := "mongodb://localhost:27017"
	databaseName := "mydatabase"
	collectionName := "pesquisadores"

	writer, err := NewMongoWriter(connectionString, databaseName, collectionName)
	if err != nil {
		log.Fatal(err)
	}
	defer writer.Close()

	pesquisador := &scrap_lattes.Pesquisador{
		// Preencher os dados do pesquisador
	}

	err = writer.WritePesquisador(pesquisador)
	if err != nil {
		log.Fatal(err)
	}

	log.Println("Dados do pesquisador escritos com sucesso!")

	// Aguardar alguns segundos para exibir os logs antes de encerrar o programa
	time.Sleep(3 * time.Second)
}
=======
// func main() {
// 	connectionString := "mongodb://localhost:27017"
// 	databaseName := "mydatabase"
// 	collectionName := "researchers"

// 	writer, err := NewMongoWriter(connectionString, databaseName, collectionName)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	defer writer.Close()

// 	researcher := &researcher.Researcher{
// 		// Preencher os dados do pesquisador
// 	}

// 	err = writer.WriteResearcher(researcher)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	log.Println("Researcher data written successfully!")

// 	// Aguardar alguns segundos para exibir os logs antes de encerrar o programa
// 	time.Sleep(3 * time.Second)
// }
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
