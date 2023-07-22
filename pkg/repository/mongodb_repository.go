// pkg/repository/mongodb_repository.go
package repository

import (
	"github.com/makaires77/ppgcs/pkg/domain/researcher"
	"github.com/makaires77/ppgcs/pkg/infrastructure/mongo"
)

type MongoDBRepository struct {
	writer *mongo.MongoDriver // Use a estrutura correta
}

func NewMongoDBRepository(writer *mongo.MongoDriver) *MongoDBRepository {
	return &MongoDBRepository{
		writer: writer,
	}
}

func (r *MongoDBRepository) Save(researcher *researcher.Researcher) error {
	err := r.writer.WriteResearcher(researcher) // Use o método correto
	if err != nil {
		return err
	}
	return nil
}
