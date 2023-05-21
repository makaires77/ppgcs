package mongo

import (
	"context"
	"log"
	"time"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	"github.com/makaires77/ppgcs/pkg/domain/scrap_lattes"
)

// MongoWriter é uma implementação do escritor Lattes usando MongoDB.
type MongoWriter struct {
	client     *mongo.Client
	database   *mongo.Database
	collection *mongo.Collection
}

// NewMongoWriter cria uma nova instância de MongoWriter.
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

// WritePesquisador escreve os dados do pesquisador no MongoDB.
func (w *MongoWriter) WritePesquisador(pesquisador *scrap_lattes.Pesquisador) error {
	_, err := w.collection.InsertOne(context.Background(), pesquisador)
	if err != nil {
		log.Printf("Erro ao escrever os dados do pesquisador no MongoDB: %s\n", err)
		return err
	}

	log.Println("Dados do pesquisador escritos com sucesso no MongoDB!")

	return nil
}

// Close fecha a conexão com o MongoDB.
func (w *MongoWriter) Close() {
	err := w.client.Disconnect(context.Background())
	if err != nil {
		log.Printf("Erro ao fechar a conexão com o MongoDB: %s\n", err)
	}
}

// Exemplo de uso

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
