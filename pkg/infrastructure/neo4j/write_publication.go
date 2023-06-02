package neo4j

import (
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

type Neo4jPublicationRepository struct {
	Driver neo4j.Driver
}

func NewNeo4jPublicationRepository(driver neo4j.Driver) *Neo4jPublicationRepository {
	return &Neo4jPublicationRepository{
		Driver: driver,
	}
}
