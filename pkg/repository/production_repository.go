// repository/production_repository.go

package repository

import "github.com/makaires77/ppgcs/pkg/domain/publication"

// ProductionRepository é uma interface que define as operações relacionadas a produções.
type ProductionRepository interface {
	GetByID(id string) (*publication.Publication, error)
	Save(production *publication.Publication) error
	Update(production *publication.Publication) error
	DeleteByID(id string) error
	ListAll() ([]*publication.Publication, error)
}
