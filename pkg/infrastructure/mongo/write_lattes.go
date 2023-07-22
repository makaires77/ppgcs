// pkg\infrastructure\mongo\write_lattes.go
package mongo

import (
	"context"
	"log"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	"github.com/makaires77/ppgcs/pkg/domain/researcher"
)

// MongoWriter is an implementation of Lattes writer using MongoDB.
type MongoWriter struct {
	client     *mongo.Client
	database   *mongo.Database
	collection *mongo.Collection
}

// NewMongoWriter creates a new instance of MongoWriter.
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

// WriteResearcher writes researcher data to MongoDB.
func (w *MongoWriter) WriteResearcher(researcher *researcher.Researcher) error {
	_, err := w.collection.InsertOne(context.Background(), researcher)
	if err != nil {
		log.Printf("Error writing researcher data to MongoDB: %s\n", err)
		return err
	}

	log.Println("Researcher data written successfully to MongoDB!")

	return nil
}

// Close closes the connection to MongoDB.
func (w *MongoWriter) Close() {
	err := w.client.Disconnect(context.Background())
	if err != nil {
		log.Printf("Error closing connection to MongoDB: %s\n", err)
	}
}

// Exemplo de uso

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
