package json_publication

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"

	"github.com/makaires77/ppgcs/pkg/domain/publication"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// ErroPersonalizado é uma struct que implementa a interface de erro
type ErroPersonalizado struct {
	Erro error
	Msg  string
}

func (e *ErroPersonalizado) Error() string {
	return fmt.Sprintf("%v : %v", e.Erro, e.Msg)
}

func LoadPublicationsFromCSV(filePath string) ([]publication.Publication, error) {
	csvFile, err := os.Open(filePath)
	if err != nil {
		return nil, &ErroPersonalizado{
			Erro: err,
			Msg:  fmt.Sprintf("Erro ao abrir o arquivo CSV: %s", filePath),
		}
	}
	defer csvFile.Close()

	reader := csv.NewReader(csvFile)
	reader.Comma = ';'
	reader.LazyQuotes = true

	var publications []publication.Publication
	for {
		_, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, &ErroPersonalizado{
				Erro: err,
				Msg:  fmt.Sprintf("Erro ao ler o arquivo CSV: %s", filePath),
			}
		}

		// Para fins de exemplo, vamos assumir que cada linha do CSV corresponde a uma publicação
		// E os valores das colunas correspondem aos atributos de uma publicação
		publications = append(publications, publication.Publication{
			// atribua os valores da linha para os atributos do struct Publication aqui
			// ...
		})
	}

	return publications, nil
}

//https://pkg.go.dev/github.com/neo4j/neo4j-go-driver/v5/neo4j#example-DriverWithContext-VerifyAuthenticationDriverLevel
//NewDriverWithContext é o ponto de entrada para o driver neo4j para criar uma instância de um driver. É a primeira função a ser chamada para estabelecer uma conexão com um banco de dados neo4j. Ele requer um URI Bolt e autenticação como parâmetros e também pode tomar função(ões) de configuração opcional(is) como parâmetros variáveis.
//Nenhuma conectividade acontece quando NewDriverWithContext é chamado. Chame DriverWithContext.VerifyConnectivity uma vez que o driver seja criado se você quiser verificar antecipadamente se o URI fornecido e credenciais estão corretos.
//Para se conectar a um banco de dados de instância única, você precisa passar um URI com o esquema 'bolt', 'bolt+s' ou 'bolt+ssc'.
//Para se conectar a um banco de dados de cluster causal, você precisa passar um URI com o esquema 'neo4j', 'neo4j+s' ou 'neo4j+ssc' e sua parte do host definida para ser um dos membros principais do cluster.
//Você pode substituir as opções de configuração padrão fornecendo uma(s) função(ões) de configuração
// driver, err = NewDriverWithContext(uri, BasicAuth(username, password), function (config *Config) {
//	config.MaxConnectionPoolSize = 10})

func PersistPublications(driver neo4j.DriverWithContext, publications []publication.Publication) error {
	sessionConfig := neo4j.SessionConfig{
		AccessMode: neo4j.AccessModeWrite,
	}

	session, err := driver.NewSession(sessionConfig)
	if err != nil {
		return &ErroPersonalizado{
			Erro: err,
			Msg:  "Erro ao criar nova sessão com o driver Neo4j",
		}
	}
	defer session.Close()

	// Início de uma transação
	tx, err := session.BeginTransaction()
	if err != nil {
		return &ErroPersonalizado{
			Erro: err,
			Msg:  "Erro ao iniciar uma transação",
		}
	}

	// Insere as publicações na base de dados
	for _, publication := range publications {
		_, err = tx.Run(
			`CREATE (p:Publication $publication)`,
			map[string]interface{}{
				"publication": publication,
			},
		)

		if err != nil {
			// Faz um rollback na transação se houver um erro
			tx.Rollback()
			return &ErroPersonalizado{
				Erro: err,
				Msg:  "Erro ao inserir uma publicação na base de dados",
			}
		}
	}

	// Commit da transação
	err = tx.Commit()
	if err != nil {
		return &ErroPersonalizado{
			Erro: err,
			Msg:  "Erro ao fazer commit da transação",
		}
	}

	return nil
}
