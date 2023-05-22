package main

import (
	"context"
	"encoding/csv"
	"io"
	"log"
	"os"

	"github.com/chromedp/cdproto/network"
	"github.com/chromedp/cdproto/page"
	"github.com/chromedp/chromedp"
	"github.com/makaires77/ppgcs/pkg/application"
	"github.com/makaires77/ppgcs/pkg/domain/researcher"

	dgraphRepo "github.com/makaires77/ppgcs/pkg/infrastructure/dgraph"
	mongoRepo "github.com/makaires77/ppgcs/pkg/infrastructure/mongo"
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

type ResearcherData struct {
	Nome        string
	Titulacao   string
	Instituicao string
	Area        string
	// Adicione mais campos conforme necessário
}

// Função para realizar o scrap
func realizarBusca(nome string) (*ResearcherData, error) {
	ctx, cancel := chromedp.NewContext(context.Background(), chromedp.WithLogf(log.Printf))
	defer cancel()

	err := chromedp.Run(ctx,
		network.Enable(),
		page.Enable(),
		chromedp.Navigate(urlBusca),
		chromedp.WaitVisible(cssCheckbox),
		chromedp.Click(cssCheckbox),
		chromedp.WaitVisible(cssInput),
		chromedp.SendKeys(cssInput, nome),
		chromedp.Click(cssBotao),
	)
	if err != nil {
		return nil, err
	}

	// Agora, vamos extrair as informações do pesquisador da página de resultados
	var nome, titulacao, instituicao, area string
	err = chromedp.Run(ctx,
		chromedp.Text(cssResultado, &nome),
		// Faça o mesmo para os outros campos (titulacao, instituicao, area) de acordo com os seletores CSS correspondentes
	)
	if err != nil {
		return nil, err
	}

	data := &ResearcherData{
		Nome:        nome,
		Titulacao:   titulacao,
		Instituicao: instituicao,
		Area:        area,
	}

	return data, nil
}
