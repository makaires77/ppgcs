// pkg\infrastructure\mongo\mongo.go
package mongo

import (
	"context"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type MongoWriter struct {
	client *mongo.Client
}

func NewMongoWriter(connectionString, databaseName, collectionName string) (*MongoWriter, error) {
	clientOptions := options.Client().ApplyURI(connectionString)

	client, err := mongo.Connect(context.Background(), clientOptions)
	if err != nil {
		return nil, err
	}

	// Here, you can use these as local variables instead of storing them in the MongoWriter struct
	database := client.Database(databaseName)
	collection := database.Collection(collectionName)

	return &MongoWriter{
		client: client,
	}, nil
}
