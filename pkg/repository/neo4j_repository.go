// Em pkg/repository/neo4j_repository.go
package repository

import (
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

type Neo4JRepository struct {
	driver neo4j.Driver
}

// ... Código existente ...

// DeleteByID exclui uma publicação pelo seu ID.
func (r *Neo4JRepository) DeleteByID(id string) error {
	session := r.driver.NewSession(neo4j.SessionConfig{
		AccessMode: neo4j.AccessModeWrite,
	})
	defer session.Close()

	_, err := session.WriteTransaction(func(transaction neo4j.Transaction) (interface{}, error) {
		result, err := transaction.Run(
			"MATCH (a:Publication) WHERE a.id = $id DELETE a",
			map[string]interface{}{
				"id": id,
			},
		)
		if err != nil {
			return nil, err
		}

		_, err = result.Consume()
		if err != nil {
			return nil, err
		}

		return nil, nil
	})

	return err
}
