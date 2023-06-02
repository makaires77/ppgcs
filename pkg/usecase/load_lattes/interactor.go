package load_lattes

import (
	"log"

	"github.com/makaires77/ppgcs/pkg/domain/scrap_lattes"
	"github.com/makaires77/ppgcs/pkg/infrastructure/mongo"
	"github.com/makaires77/ppgcs/pkg/infrastructure/neo4j"
)

// Interactor é o responsável por carregar os dados do Lattes.
type Interactor struct {
	mongoWriter *mongo.MongoWriter
	neo4jWriter *neo4j.Neo4jWriteLattes
}

// NewInteractor cria uma nova instância de Interactor.
func NewInteractor(mongoWriter *mongo.MongoWriter, neo4jWriter *neo4j.Neo4jWriteLattes) *Interactor {
	return &Interactor{
		mongoWriter: mongoWriter,
		neo4jWriter: neo4jWriter,
	}
}

// LoadPesquisador carrega os dados do pesquisador a partir do Lattes e os armazena no banco de dados.
func (i *Interactor) LoadPesquisador(pesquisadorID string) error {
	// 1. Scrap dos dados do pesquisador a partir do Lattes
	pesquisador, err := scrap_lattes.ScrapPesquisador(pesquisadorID)
	if err != nil {
		log.Printf("Erro ao realizar o scrap dos dados do pesquisador: %s\n", err)
		return err
	}

	// 2. Armazenar os dados no MongoDB
	err = i.mongoWriter.WritePesquisador(pesquisador)
	if err != nil {
		log.Printf("Erro ao armazenar os dados do pesquisador no MongoDB: %s\n", err)
		return err
	}

	// 3. Armazenar os dados no Neo4j
	err = i.neo4jWriter.WritePesquisador(pesquisador)
	if err != nil {
		log.Printf("Erro ao armazenar os dados do pesquisador no Neo4j: %s\n", err)
		return err
	}

	log.Println("Dados do pesquisador carregados e armazenados com sucesso!")

	return nil
}
