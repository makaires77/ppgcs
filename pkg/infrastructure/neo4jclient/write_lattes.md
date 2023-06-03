package neo4j

import (
	"fmt"
	"log"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"

	"github.com/makaires77/ppgcs/pkg/domain/scrap_lattes"
)

// Neo4jWriter é uma implementação do escritor Lattes usando Neo4j.
type Neo4jWriter struct {
	driver neo4j.Driver
}

// NewNeo4jWriter cria uma nova instância de Neo4jWriter.
func NewNeo4jWriter(uri, username, password string) (*Neo4jWriter, error) {
	driver, err := neo4j.NewDriver(uri, neo4j.BasicAuth(username, password, ""))
	if err != nil {
		return nil, err
	}

	return &Neo4jWriter{
		driver: driver,
	}, nil
}

// WritePesquisador escreve os dados do pesquisador no Neo4j.
func (w *Neo4jWriter) WritePesquisador(pesquisador *scrap_lattes.Pesquisador) error {
	session := w.driver.NewSession(neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close()

	result, err := session.WriteTransaction(func(tx neo4j.Transaction) (interface{}, error) {
		_, err := tx.Run("CREATE (p:Pesquisador {id: $id, nome: $nome})",
			map[string]interface{}{
				"id":   pesquisador.ID,
				"nome": pesquisador.Nome,
			})
		if err != nil {
			return nil, err
		}

		return nil, nil
	})

	if err != nil {
		log.Printf("Erro ao escrever os dados do pesquisador no Neo4j: %s\n", err)
		return err
	}

	log.Println("Dados do pesquisador escritos com sucesso no Neo4j!")
	fmt.Println(result)

	return nil
}

// Close fecha a conexão com o Neo4j.
func (w *Neo4jWriter) Close() {
	err := w.driver.Close()
	if err != nil {
		log.Printf("Erro ao fechar a conexão com o Neo4j: %s\n", err)
	}
}

// Exemplo de uso

func main() {
	uri := "bolt://localhost:7687"
	username := "neo4j"
	password := "password"

	writer, err := NewNeo4jWriter(uri, username, password)
	if err != nil {
		log.Fatal(err)
	}
	defer writer.Close()

	pesquisador := &scrap_lattes.Pesquisador{
		// Preencher os dados do pesquisador
	}

	err = writer.WritePesquisador(pesquisador)
	if err != nil {
		log.Fatal(err)
	}

	log.Println("Dados do pesquisador escritos com sucesso!")

	// Aguardar alguns segundos para exibir os logs antes de encerrar o programa
	time.Sleep(3 * time.Second)
}
