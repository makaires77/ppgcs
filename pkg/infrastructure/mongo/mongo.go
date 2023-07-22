package mongo

import (
	"context"
	"log"

	"github.com/makaires77/ppgcs/pkg/domain/scrap_lattes"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type MongoWriter struct {
	client *mongo.Client
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

// Your existing methods for MongoWriter
