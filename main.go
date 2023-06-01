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

	"github.com/makaires77/ppgcs/pkg/fuzzysearch"
	"github.com/makaires77/ppgcs/pkg/infrastructure/neo4j"
	"github.com/makaires77/ppgcs/pkg/interfaces/rabbitmq"
)

// idLattes, nome, tipo, titulo_do_capitulo, idioma, titulo_do_livro, ano, doi, pais_de_publicacao, isbn, nome_da_editora, numero_da_edicao_revisao, organizadores, paginas, autores,autores-endogeno, autores-endogeno-nome, tags, Hash,tipo_producao, natureza, titulo, nome_do_evento,ano_do_trabalho,pais_do_evento,cidade_do_evento,classificacao,periodico,volume,issn,estrato_qualis,editora,numero_de_paginas,numero_de_volumes

func fuzz() {
	fuzzyService := fuzzysearch.NewFuzzySearchService()

	// Load data from CSV files
	discentes, _ := fuzzyService.LoadCSVData("_data/powerbi/lista_docentes_colaboradores.csv", 2)
	docente, _ := fuzzyService.LoadCSVData("_data/powerbi/publicacoes.csv", 1)
	autores, _ := fuzzyService.LoadCSVData("_data/powerbi/publicacoes.csv", 14)

	// Conduct fuzzy search for each docente
	for _, discente := range discentes {
		res, err := fuzzyService.FuzzySearchSetThreshold(discente, autores, 3)
		if err != nil {
			fmt.Println(err)
		} else {
			fmt.Printf("Result for '%s' with '0.82' threshold: %s\n", discente, strings.Join(res, " "))
		}
	}
}

// Estrutura de dados para representar um pesquisador
type Researcher struct {
	Name string
}

// Função stub para representar a chamada ao serviço de scrapping
func scrapeResearcherInfo(name string) (string, error) {
	// Adicione a lógica do serviço de scrapping aqui
	// Por enquanto, apenas retorna um valor fictício
	return "Informações do pesquisador: " + name, nil
}

func validateFile(fileName string, file io.Reader) error {
	// Verifica se o arquivo é vazio
	if file == nil {
		return errors.New("the file is empty")
	}

	// Verifica se o arquivo é CSV
	if ext := filepath.Ext(fileName); !strings.EqualFold(ext, ".csv") {
		return errors.New("only .csv files are allowed")
	}

	return nil
}

func main() {

	neo4jClient, err := neo4j.NewClient()
	if err != nil {
		log.Fatalf("Failed to create neo4j client: %v", err)
	}

	consumer := rabbitmq.NewConsumer(neo4jClient)
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
