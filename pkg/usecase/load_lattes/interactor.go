<<<<<<< HEAD
=======
// pkg\usecase\load_lattes\interactor.go
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
package load_lattes

import (
	"log"

<<<<<<< HEAD
=======
	"github.com/makaires77/ppgcs/pkg/domain/researcher"
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
	"github.com/makaires77/ppgcs/pkg/domain/scrap_lattes"
	"github.com/makaires77/ppgcs/pkg/infrastructure/mongo"
	"github.com/makaires77/ppgcs/pkg/infrastructure/neo4jclient"
)

// Interactor é o responsável por carregar os dados do Lattes.
type Interactor struct {
<<<<<<< HEAD
	mongoWriter *mongo.MongoWriter
=======
	mongoWriter mongo.MongoWriter
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
	neo4jWriter *neo4jclient.Neo4jWriteLattes
}

// NewInteractor cria uma nova instância de Interactor.
<<<<<<< HEAD
func NewInteractor(mongoWriter *mongo.MongoWriter, neo4jWriter *neo4jclient.Neo4jWriteLattes) *Interactor {
=======
func NewInteractor(mongoWriter mongo.MongoWriter, neo4jWriter *neo4jclient.Neo4jWriteLattes) *Interactor {
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
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

<<<<<<< HEAD
	// 2. Armazenar os dados no MongoDB
	err = i.mongoWriter.WritePesquisador(pesquisador)
=======
	// Converter o tipo *scrap_lattes.Pesquisador para *researcher.Researcher
	researcherData := convertToResearcher(pesquisador)

	// 2. Armazenar os dados no MongoDB
	err = i.mongoWriter.WriteResearcher(researcherData)
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
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
<<<<<<< HEAD
=======

// Função para converter de *scrap_lattes.Pesquisador para *researcher.Researcher
func convertToResearcher(p *scrap_lattes.Pesquisador) *researcher.Researcher {
	// Implemente a lógica para converter os campos e criar uma instância de researcher.Researcher
	// Exemplo:
	return &researcher.Researcher{
		Nome:          p.Nome,
		Titulo:        p.Titulo,
		LinkCurriculo: p.LinkCurriculo,
		// Adicione os demais campos...
	}
}
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
