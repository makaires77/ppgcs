package main

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"strings"
	"sync"

	"github.com/hbollon/go-edlib"
	"github.com/streadway/amqp"

	"github.com/makaires77/ppgcs/pkg/infrastructure/neo4j"
	"github.com/makaires77/ppgcs/pkg/infrastructure/scrap_lattes"
	"github.com/makaires77/ppgcs/pkg/interfaces/rabbitmq"
	"github.com/makaires77/ppgcs/pkg/usecase/fuzzysearch"
)

type Researcher struct {
	Name string
}

// A chamada para a função FuzzySearchSetThreshold está passando uma lista com apenas um item ([]string{autoresArtigo}) porque o algoritmo de pesquisa fuzzy precisa comparar um string (discente) com uma lista de strings (autoresArtigo). Se autoresArtigo já for uma lista de strings, então você pode simplesmente passá-la diretamente: FuzzySearchSetThreshold(discente, autoresArtigo, 3, 0.7, edlib.Levenshtein).

func fuzz() {
	fuzzyService := fuzzysearch.NewFuzzySearchService()

	// Lista de campos de publication.csv:
	// idLattes, nome, tipo, titulo_do_capitulo, idioma, titulo_do_livro, ano, doi, pais_de_publicacao, isbn, nome_da_editora, numero_da_edicao_revisao, organizadores, paginas, autores,autores-endogeno, autores-endogeno-nome, tags, Hash,tipo_producao, natureza, titulo, nome_do_evento,ano_do_trabalho,pais_do_evento,cidade_do_evento,classificacao, periodico, volume, issn,estrato_qualis, editora, numero_de_paginas, numero_de_volumes

	// Load data from CSV files
	discentesData, _ := fuzzyService.LoadCSVData("_data/powerbi/lista_docentes_colaboradores.csv", 1)
	docentesData, _ := fuzzyService.LoadCSVData("_data/powerbi/publicacoes.csv", 1)
	autoresData, _ := fuzzyService.LoadCSVData("_data/powerbi/publicacoes.csv", 14)

	discentes := discentesData
	docentes := docentesData
	autores := autoresData

	// Create a map to track articles by docente
	docenteArtigos := make(map[string]int)
	for _, docente := range docentes {
		docenteArtigos[docente] += 1
	}

	// Conduct fuzzy search for each docente
	docentePercentage := make(map[string]float64)
	for _, discente := range discentes {
		for i, autoresArtigo := range autores {
			res, err := fuzzyService.FuzzySearchSetThreshold(discente, strings.Split(autoresArtigo, ";"), 3, 0.7, edlib.Levenshtein)
			if err != nil {
				fmt.Println(err)
			} else if len(res) > 0 {
				docentePercentage[docentes[i]] += 1
			}
		}
	}

	// Calculate percentage of articles with similarity for each docente
	for docente, count := range docentePercentage {
		docentePercentage[docente] = (count / float64(docenteArtigos[docente])) * 100
		fmt.Printf("Percentual de artigos com co-autoria de discentes para o docente '%s': %.2f%%\n", docente, docentePercentage[docente])
	}
}

func scrapeResearcherInfo(name string) (string, error) {
	return "Informações do pesquisador: " + name, nil
}

func validateFile(fileName string, file io.Reader) error {
	if file == nil {
		return errors.New("the file is empty")
	}

	if ext := filepath.Ext(fileName); !strings.EqualFold(ext, ".csv") {
		return errors.New("only .csv files are allowed")
	}

	return nil
}

func main() {
	// Crie o cliente Neo4J
	neo4jClient, err := neo4j.Neo4jClient("neo4j://localhost:7687", "username", "password")
	if err != nil {
		log.Fatalf("Failed to create neo4j client: %v", err)
	}

	// Crie uma conexão AMQP
	conn, err := amqp.Dial("amqp://guest:guest@localhost:5672/")
	if err != nil {
		log.Fatalf("Failed to connect to RabbitMQ: %v", err)
	}

	// Crie uma instância de ScrapLattes
	scrapLattes := scrap_lattes.NewScrapLattes(neo4jClient)

	// Crie o consumidor RabbitMQ
	consumer := rabbitmq.NewConsumer(conn, "yourQueueName", scrapLattes)

	consumer.Start()

	http.HandleFunc("/start-scraping", func(w http.ResponseWriter, r *http.Request) {
		file, header, err := r.FormFile("file")
		if err != nil {
			http.Error(w, "Error retrieving file", http.StatusInternalServerError)
			return
		}
		defer file.Close()

		// Valida o arquivo
		if err := validateFile(header.Filename, file); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Crie um novo leitor CSV
		csvReader := csv.NewReader(file)

		// Canal para enviar nomes de pesquisadores para as goroutines
		names := make(chan string)

		// WaitGroup para sincronizar as goroutines
		var wg sync.WaitGroup

		// Crie 4 goroutines que processam os nomes de pesquisadores
		for i := 0; i < 4; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for name := range names {
					// Chama o serviço de scrapping
					info, err := scrapeResearcherInfo(name)
					if err != nil {
						fmt.Printf("Erro ao processar o nome '%s': %s\n", name, err)
						continue
					}
					fmt.Printf("Nome processado com sucesso: %s, Informações: %s\n", name, info)
				}
			}()
		}

		// Envia os nomes de pesquisadores para o canal
		for {
			record, err := csvReader.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				http.Error(w, "Error reading CSV file", http.StatusInternalServerError)
				return
			}

			// Verifica se cada linha do CSV tem um campo
			if len(record) != 1 {
				http.Error(w, "Each line in the CSV file must contain exactly one field", http.StatusBadRequest)
				return
			}

			names <- record[0]
		}

		// Fecha o canal para sinalizar que não há mais nomes a serem processados
		close(names)

		// Aguarda todas as goroutines terminarem
		wg.Wait()

		json.NewEncoder(w).Encode(map[string]string{
			"message": "O processo de scraping foi iniciado",
		})
	})

	http.ListenAndServe(":8080", nil)
}
