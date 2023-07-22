// pkg\infrastructure\mongo\mongo.go
package mongo

import (
	"context"
	"log"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	"github.com/makaires77/ppgcs/pkg/domain/researcher"
)

// MongoDriver is an implementation of Lattes writer using MongoDB.
type MongoDriver struct {
	client     *mongo.Client
	database   *mongo.Database
	collection *mongo.Collection
}

// NewMongoDriver creates a new instance of MongoDriver.
func NewMongoDriver(connectionString, databaseName, collectionName string) (*MongoDriver, error) {
	clientOptions := options.Client().ApplyURI(connectionString)

	client, err := mongo.Connect(context.Background(), clientOptions)
	if err != nil {
		return nil, err
	}

	database := client.Database(databaseName)
	collection := database.Collection(collectionName)

	return &MongoDriver{
		client:     client,
		database:   database,
		collection: collection,
	}, nil
}

// WriteResearcher writes researcher data to MongoDB.
func (w *MongoDriver) WriteResearcher(researcher *researcher.Researcher) error {
	_, err := w.collection.InsertOne(context.Background(), researcher)
	if err != nil {
		log.Printf("Error writing researcher data to MongoDB: %s\n", err)
		return err
	}

	log.Println("Researcher data written successfully to MongoDB!")

	return nil
}

// Close closes the connection to MongoDB.
func (w *MongoDriver) Close() {
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

// 	driver, err := NewMongoDriver(connectionString, databaseName, collectionName)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	defer driver.Close()

// 	researcher := &researcher.Researcher{
// 		// Preencher os dados do pesquisador
// 	}

// 	err = driver.WriteResearcher(researcher)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	log.Println("Researcher data written successfully!")

// 	// Aguardar alguns segundos para exibir os logs antes de encerrar o programa
// 	time.Sleep(3 * time.Second)
// }
