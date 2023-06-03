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
	"github.com/makaires77/ppgcs/pkg/infrastructure/neo4jclient"
	"github.com/makaires77/ppgcs/pkg/infrastructure/rabbitmq"
	"github.com/makaires77/ppgcs/pkg/usecase/fuzzysearch"
)

type Researcher struct {
	Name string
}

type ScrapLattes struct {
	neo4jClient *neo4jclient.Neo4jClient
}

func NewScrapLattes(neo4jClient *neo4jclient.Neo4jClient) *ScrapLattes {
	return &ScrapLattes{
		neo4jClient: neo4jClient,
	}
}

func fuzz() {
	fuzzyService := fuzzysearch.NewFuzzySearchService()

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
	// Create a Neo4j connection
	neo4jClient, err := neo4jclient.NewNeo4jClient("neo4j://localhost:7687", "username", "password")
	if err != nil {
		log.Fatalf("Failed to create Neo4j client: %v", err)
	}

	// Create a ScrapLattes instance
	scrapLattes := NewScrapLattes(neo4jClient)

	// Create a RabbitMQ connection
	conn, err := rabbitmq.Dial("amqp://guest:guest@localhost:5672/")
	if err != nil {
		log.Fatalf("Failed to create RabbitMQ connection: %v", err)
	}

	// Create a RabbitMQ consumer
	consumer, err := rabbitmq.NewConsumer(conn, "yourQueueName", scrapLattes)
	if err != nil {
		log.Fatalf("Failed to create RabbitMQ consumer: %v", err)
	}

	// Start the RabbitMQ consumer
	go consumer.Start()

	http.HandleFunc("/start-scraping", func(w http.ResponseWriter, r *http.Request) {
		file, header, err := r.FormFile("file")
		if err != nil {
			http.Error(w, "Error retrieving file", http.StatusInternalServerError)
			return
		}
		defer file.Close()

		// Validate the file
		if err := validateFile(header.Filename, file); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Create a new CSV reader
		csvReader := csv.NewReader(file)

		// Channel to send researcher names to goroutines
		names := make(chan string)

		// WaitGroup to synchronize goroutines
		var wg sync.WaitGroup

		// Create 4 goroutines that process researcher names
		for i := 0; i < 4; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for name := range names {
					// Call the scraping service
					info, err := scrapeResearcherInfo(name)
					if err != nil {
						fmt.Printf("Erro ao processar o nome '%s': %s\n", name, err)
						continue
					}
					fmt.Printf("Nome processado com sucesso: %s, Informações: %s\n", name, info)
				}
			}()
		}

		// Send researcher names to the channel
		for {
			record, err := csvReader.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				http.Error(w, "Error reading CSV file", http.StatusInternalServerError)
				return
			}

			// Check if each line in the CSV has one field
			if len(record) != 1 {
				http.Error(w, "Each line in the CSV file must contain exactly one field", http.StatusBadRequest)
				return
			}

			names <- record[0]
		}

		// Close the channel to signal that there are no more names to process
		close(names)

		// Wait for all goroutines to finish
		wg.Wait()

		json.NewEncoder(w).Encode(map[string]string{
			"message": "O processo de scraping foi iniciado",
		})
	})

	http.ListenAndServe(":8080", nil)
}
