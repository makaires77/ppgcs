// Em pkg/repository/neo4j_repository.go
package repository

import (
	"github.com/makaires77/ppgcs/pkg/domain/researcher"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
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
	session := r.driver.NewSession(neo4j.SessionConfig{
		AccessMode: neo4j.AccessModeWrite,
	})
	defer session.Close()

	_, err := session.WriteTransaction(func(transaction neo4j.Transaction) (interface{}, error) {
		result, err := transaction.Run(
			"CREATE (a:Researcher {name: $name, id: $id}) RETURN a",
			map[string]interface{}{
				"name": researcher.Nome,
				"id":   researcher.IDLattes,
			},
		)
		if err != nil {
			return nil, err
		}

		_, err = result.Single()
		if err != nil {
			return nil, err
		}

		return nil, nil
	})

	return err
}
