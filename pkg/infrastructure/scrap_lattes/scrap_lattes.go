package scrap_lattes

import (
	"encoding/csv"
	/* 	"errors"
	 */
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/makaires77/ppgcs/pkg/infrastructure/neo4j"
)

type ScrapLattes struct {
	neo4jClient *neo4j.Neo4jClient
}

func NewScrapLattes(neo4jClient *neo4j.Neo4jClient) *ScrapLattes {
	return &ScrapLattes{
		neo4jClient: neo4jClient,
	}
}

// Funcionalidades:

func (s *ScrapLattes) ProcessarArquivo(filePath string) error {
	// Abrir o arquivo CSV
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	// Criar um novo leitor CSV
	reader := csv.NewReader(file)

	// Iniciar um novo grupo de goroutines
	var wg sync.WaitGroup

	// Criar um canal para enviar registros de linha do CSV
	c := make(chan []string)

	// Iniciar a goroutine para ler o arquivo e enviar linhas no canal
	wg.Add(1)
	go func() {
		defer wg.Done()

		for {
			record, err := reader.Read()

			if err != nil {
				if err == io.EOF {
					break
				}

				fmt.Println("Erro ao ler a linha:", err)
			}

			c <- record
		}

		// Fechar o canal após todas as linhas serem lidas
		close(c)
	}()

	// Iniciar a goroutine para processar registros enviados no canal
	wg.Add(1)
	go func() {
		defer wg.Done()

		for record := range c {
			err := s.ProcessarRegistro(record)

			if err != nil {
				fmt.Println("Erro ao processar registro:", err)
			}
		}
	}()

	// Esperar ambas goroutines finalizarem
	wg.Wait()

	return nil
}

func (s *ScrapLattes) ProcessarRegistro(record []string) error {
	// Extrair informações do registro

	nome := record[0]
	lattes := record[1]
	// ...

	// Exemplo de operação: exibir informações
	fmt.Println("Nome:", nome)
	fmt.Println("Lattes:", lattes)
	fmt.Println("---")

	// Adicione aqui qualquer lógica adicional para processar a linha

	return nil
}
