// pkg/repository/dgraph_repository.go
package repository

import (
	"github.com/dgraph-io/dgo"
	"github.com/makaires77/ppgcs/pkg/domain/researcher"
)

type DGraphRepository struct {
	client *dgo.Dgraph
}

func NewDGraphRepository(client *dgo.Dgraph) *DGraphRepository {
	return &DGraphRepository{
		client: client,
	}
}

func (r *DGraphRepository) Save(researcher *researcher.Researcher) error {
	// Implement your Dgraph save logic here
}
