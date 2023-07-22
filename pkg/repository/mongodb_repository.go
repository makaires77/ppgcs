// pkg/repository/mongodb_repository.go
package repository

import (
	"log"

	"github.com/makaires77/ppgcs/pkg/domain/researcher"
	"github.com/makaires77/ppgcs/pkg/infrastructure/mongo"
)

type MongoDBRepository struct {
	writer *mongo.MongoWriter
}

func NewMongoDBRepository(writer *mongo.MongoWriter) *MongoDBRepository {
	return &MongoDBRepository{
		writer: writer,
	}
}

func (r *MongoDBRepository) Save(researcher *researcher.Researcher) error {
	err := r.writer.WritePesquisador(researcher)
	if err != nil {
		log.Printf("Error while saving researcher: %s\n", err)
		return err
	}

	return nil
}
