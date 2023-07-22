// pkg/repository/neo4j_repository.go
package repository

import (
	"github.com/makaires77/ppgcs/pkg/domain/researcher"
	"github.com/neo4j/neo4j-go-driver/neo4j"
)

type Neo4JRepository struct {
	driver neo4j.Driver
}

func NewNeo4jRepository(driver neo4j.Driver) *Neo4JRepository {
	return &Neo4JRepository{
		driver: driver,
	}
}

func (r *Neo4JRepository) Save(researcher *researcher.Researcher) error {
	// Implement your Neo4j save logic here
}
