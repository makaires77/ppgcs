<<<<<<< HEAD
=======
// pkg/repository/researcher_repository.go
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
package repository

import (
	"context"
<<<<<<< HEAD
	"fmt"
	"log"

	"encoding/json"
	"os"
	"time"

	"github.com/dgraph-io/dgo"
	"github.com/dgraph-io/dgo/protos/api"
	"github.com/makaires77/ppgcs/pkg/domain/researcher"
	"github.com/neo4j/neo4j-go-driver/neo4j"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"
	"google.golang.org/grpc"
)

type ResearcherRepository struct {
	mongoClient  *mongo.Client
	dgraphClient *dgo.Dgraph
	neo4jDriver  neo4j.Driver
}

type DatabaseCredentials struct {
	Mongo  DatabaseCredentialsInfo `json:"mongo"`
	Dgraph DatabaseCredentialsInfo `json:"dgraph"`
	Neo4j  DatabaseCredentialsInfo `json:"neo4j"`
}

type DatabaseCredentialsInfo struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

func NewResearcherRepository() (*ResearcherRepository, error) {
	// Carregar as credenciais do arquivo
	credentials, err := loadCredentials()
	if err != nil {
		return nil, fmt.Errorf("failed to load credentials: %v", err)
	}

	// Conectar ao MongoDB
	mongoClient, err := connectToMongoDB(credentials.Mongo)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to MongoDB: %v", err)
	}

	// Conectar ao Dgraph
	dgraphClient, err := connectToDgraph(credentials.Dgraph)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Dgraph: %v", err)
	}

	// Conectar ao Neo4j
	neo4jDriver, err := connectToNeo4j(credentials.Neo4j)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Neo4j: %v", err)
	}

	return &ResearcherRepository{
		mongoClient:  mongoClient,
		dgraphClient: dgraphClient,
		neo4jDriver:  neo4jDriver,
	}, nil
}

// Carregar as credenciais do arquivo
func loadCredentials() (DatabaseCredentials, error) {
	file, err := os.Open("config/credentials.json")
	if err != nil {
		return DatabaseCredentials{}, fmt.Errorf("failed to open credentials file: %v", err)
	}
	defer file.Close()

	var credentials DatabaseCredentials
	err = json.NewDecoder(file).Decode(&credentials)
	if err != nil {
		return DatabaseCredentials{}, fmt.Errorf("failed to decode credentials file: %v", err)
	}

	return credentials, nil
}

// Conectar ao MongoDB
func connectToMongoDB(credentials DatabaseCredentialsInfo) (*mongo.Client, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Configurar as opções de conexão com o MongoDB
	mongoOptions := options.Client().ApplyURI("mongodb://" + credentials.Username + ":" + credentials.Password + "@localhost:27017")

	// Conectar ao MongoDB
	client, err := mongo.Connect(ctx, mongoOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to MongoDB: %v", err)
	}

	// Verificar a conexão com o MongoDB
	err = client.Ping(ctx, readpref.Primary())
	if err != nil {
		return nil, fmt.Errorf("failed to ping MongoDB: %v", err)
	}

	return client, nil
}

// Conectar ao Dgraph
func connectToDgraph(credentials DatabaseCredentialsInfo) (*dgo.Dgraph, error) {
	dgraphClient, err := grpc.Dial("localhost:9080", grpc.WithInsecure())
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Dgraph: %v", err)
	}

	return dgo.NewDgraphClient(
		api.NewDgraphClient(dgraphClient),
	), nil
}

// Conectar ao Neo4j
func connectToNeo4j(credentials DatabaseCredentialsInfo) (neo4j.Driver, error) {
	driver, err := neo4j.NewDriver(
		"bolt://localhost:7687",
		neo4j.BasicAuth(credentials.Username, credentials.Password, ""),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Neo4j: %v", err)
	}

	return driver, nil
}

// Implemente aqui as demais funções do repositório ResearcherRepository de acordo com as regras de persistência de cada banco de dados

// Exemplo de função de persistência no MongoDB
func (r *ResearcherRepository) SaveMongoDB(researcher *researcher.Researcher) error {
	startTime := time.Now()

	// Implemente a lógica de persistência no MongoDB
	// Por exemplo:
	// collection := r.mongoClient.Database("mydb").Collection("researchers")
	// _, err := collection.InsertOne(context.Background(), researcher)

	// Exemplo de registro de tempo de transação
	elapsedTime := time.Since(startTime)
	log.Printf("Tempo de transação no MongoDB: %s", elapsedTime)

	return nil
}

// Exemplo de função de persistência no Dgraph
func (r *ResearcherRepository) SaveDgraph(researcher *researcher.Researcher) error {
	startTime := time.Now()

	// Implemente a lógica de persistência no Dgraph
=======
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
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
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

<<<<<<< HEAD
	// Exemplo de registro de tempo de transação
=======
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
	elapsedTime := time.Since(startTime)
	log.Printf("Tempo de transação no Dgraph: %s", elapsedTime)

	return nil
}

<<<<<<< HEAD
// Exemplo de função de persistência no Neo4j
func (r *ResearcherRepository) SaveNeo4j(researcher *researcher.Researcher) error {
	startTime := time.Now()

	// Implemente a lógica de persistência no Neo4j
	// Por exemplo:
	// session := r.neo4jDriver.NewSession(neo4j.SessionConfig{})
	// defer session.Close()
	// _, err := session.Run("CREATE (p:Person {name: $name, age: $age})", map[string]interface{}{
	// 	"name": researcher.Name,
	// 	"age":  researcher.Age,
	// })

	// Exemplo de registro de tempo de transação
	elapsedTime := time.Since(startTime)
	log.Printf("Tempo de transação no Neo4j: %s", elapsedTime)

	return nil
}
=======
// Neo4jResearcherRepository é a implementação do repositório para o Neo4j
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
