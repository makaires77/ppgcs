// pkg/repository/researcher_repository.go
package repository

import (
	"context"
	"log"
	"time"

	"github.com/dgraph-io/dgo"
	"github.com/makaires77/ppgcs/pkg/domain/researcher"
	"go.mongodb.org/mongo-driver/mongo"
)

// ResearcherRepository é uma interface que define os métodos para interagir com os diferentes bancos de dados.
type ResearcherRepository interface {
	Save(researcher *researcher.Researcher) error
}

// MongoDBResearcherRepository é a implementação do repositório para o MongoDB.
type MongoDBResearcherRepository struct {
	collection *mongo.Collection
}

// NewMongoDBResearcherRepository cria uma nova instância de MongoDBResearcherRepository.
func NewMongoDBResearcherRepository(client *mongo.Client, databaseName, collectionName string) *MongoDBResearcherRepository {
	database := client.Database(databaseName)
	collection := database.Collection(collectionName)

	return &MongoDBResearcherRepository{
		collection: collection,
	}
}

// Save salva os dados do pesquisador no MongoDB.
func (r *MongoDBResearcherRepository) Save(researcher *researcher.Researcher) error {
	startTime := time.Now()

	_, err := r.collection.InsertOne(context.Background(), researcher)
	if err != nil {
		log.Printf("Erro ao escrever os dados do pesquisador no MongoDB: %s\n", err)
		return err
	}

	elapsedTime := time.Since(startTime)
	log.Printf("Tempo de transação no MongoDB: %s", elapsedTime)

	return nil
}

// DgraphResearcherRepository é a implementação do repositório para o Dgraph.
type DgraphResearcherRepository struct {
	dgraphClient *dgo.Dgraph
}

// NewDgraphResearcherRepository cria uma nova instância de DgraphResearcherRepository.
func NewDgraphResearcherRepository(dgraphClient *dgo.Dgraph) *DgraphResearcherRepository {
	return &DgraphResearcherRepository{
		dgraphClient: dgraphClient,
	}
}

// Save salva os dados do pesquisador no Dgraph.
func (r *DgraphResearcherRepository) Save(researcher *researcher.Researcher) error {
	startTime := time.Now()

	// Implemente a lógica de persistência no Dgraph aqui.
	// Por exemplo:
	// p := &Person{
	// 	Uid:       "_:" + researcher.ID,
	// 	Name:      researcher.Name,
	// 	Age:       researcher.Age,
	// 	CreatedAt: researcher.CreatedAt.Format(time.RFC3339Nano),
	// }
	// pb, err := json.Marshal(p)
	// if err != nil {
	// 	return err
	// }
	// _, err = r.dgraphClient.NewTxn().Mutate(context.Background(), &api.Mutation{
	// 	SetJson: pb,
	// })

	elapsedTime := time.Since(startTime)
	log.Printf("Tempo de transação no Dgraph: %s", elapsedTime)

	return nil
}

// Neo4jResearcherRepository é a implementação do repositório para o Neo4j
