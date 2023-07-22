// pkg/repository/mongodb_repository.go
package repository

import (
	"github.com/makaires77/ppgcs/pkg/domain/researcher"
	"go.mongodb.org/mongo-driver/mongo"
)

type MongoDBRepository struct {
	client *mongo.Client
}

func NewMongoDBRepository(client *mongo.Client) *MongoDBRepository {
	return &MongoDBRepository{
		client: client,
	}
}

func (r *MongoDBRepository) Save(researcher *researcher.Researcher) error {
	// Implement your MongoDB save logic here
}
