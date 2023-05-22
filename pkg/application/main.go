package main

import (
	"context"
	"encoding/csv"
	"io"
	"log"
	"os"

	"github.com/makaires77/ppgcs/pkg/application"
	"github.com/makaires77/ppgcs/pkg/domain/researcher"
	dgraphRepo "github.com/makaires77/ppgcs/pkg/infrastructure/dgraph"
	mongoRepo "github.com/makaires77/ppgcs/pkg/infrastructure/mongodb"
	neo4jRepo "github.com/makaires77/ppgcs/pkg/infrastructure/neo4j"
)

const (
	entradaCSV   = "/home/marcos/ppgcs/_data/in_csv/lista_dadosdocentes.csv"
	urlBusca     = "http://buscatextual.cnpq.br/buscatextual/busca.do"
	cssCheckbox  = "#buscarDemais"
	cssInput     = "#textoBusca"
	cssBotao     = "#botaoBuscaFiltros"
	cssResultado = "#resultado li"
)

var ctx context.Context

// função principal
func main() {
	file, err := os.Open(entradaCSV)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	mongoRepository := mongoRepo.NewMongoDBRepository()                                             // Conectar ao MongoDB
	dgraphRepository := dgraphRepo.NewDgraphRepository()                                            // Conectar ao Dgraph
	neo4jRepository := neo4jRepo.NewNeo4jRepository()                                               // Conectar ao Neo4j
	service := application.NewResearcherService(mongoRepository, dgraphRepository, neo4jRepository) // Inicializar o serviço

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		nome := record[0] // Supondo que o nome é o primeiro campo

		dados, err := realizarBusca(nome)
		if err != nil {
			log.Println(err)
			continue
		}

		pesquisador := &researcher.Researcher{
			// preencher campos com base nos dados raspados
		}

		err = service.Save(pesquisador)
		if err != nil {
			log.Println(err)
			continue
		}
	}
}

// Função para realizar o scrap
func realizarBusca(nome string) (map[string]interface{}, error) {
	//TODO: Implementar a lógica de raspagem aqui
	// Use o pacote chromedp para automatizar a interação com o site
}
