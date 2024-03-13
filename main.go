package main

import (
<<<<<<< HEAD
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
	"github.com/makaires77/ppgcs/pkg/infrastructure/mongo"
	"github.com/makaires77/ppgcs/pkg/infrastructure/neo4jclient"
	"github.com/makaires77/ppgcs/pkg/infrastructure/scrap_lattes"
	"github.com/makaires77/ppgcs/pkg/interfaces/rabbitmq"
	"github.com/makaires77/ppgcs/pkg/usecase/fuzzysearch"
	"github.com/makaires77/ppgcs/pkg/usecase/load_lattes"
	"github.com/streadway/amqp"
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

// ExecuteComparison executa a comparação dos nomes
func ExecuteFuzzComparison() {
	fuzz()
}

func scrapeResearcherInfo(name string) (string, error) {
	return "Informações do pesquisador: " + name, nil
}

func validateFile(fileName string, file io.Reader) error {
	if file == nil {
		return errors.New("o arquivo está vazio")
	}

	if ext := filepath.Ext(fileName); !strings.EqualFold(ext, ".csv") {
		return errors.New("apenas arquivos .csv são permitidos")
	}

	return nil
}

func main() {
	// Defina as informações de conexão com o Neo4j
	uri := "bolt://localhost:7687"
	username := "neo4j"
	password := "password"

	// Create a Neo4j connection
	neo4jClient, err := neo4jclient.NewNeo4jClient(uri, username, password)
	if err != nil {
		log.Fatalf("Falha ao criar conexão com o Neo4j: %v", err)
	}

	// Create a MongoDB connection
	mongoWriter, err := mongo.NewMongoWriter("mongodb://localhost:27017", "username", "password")
=======
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/gorilla/mux"
	"github.com/joho/godotenv"
	"github.com/makaires77/ppgcs/cmd/api/handlers"
	"github.com/makaires77/ppgcs/pkg/application"
	"github.com/makaires77/ppgcs/pkg/infrastructure/mongo"
	"github.com/makaires77/ppgcs/pkg/infrastructure/neo4jclient"
	"github.com/makaires77/ppgcs/pkg/repository"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

var server *http.Server

func main() {
	err := godotenv.Load(".env")
	if err != nil {
		log.Fatalf("Erro ao carregar .env file")
	}

	var (
		uri           = os.Getenv("MONGO_URI")
		database      = os.Getenv("MONGO_DATABASE")
		collection    = os.Getenv("MONGO_COLLECTION")
		neo4j_uri     = os.Getenv("NEO4J_URI")
		neo4j_user    = os.Getenv("NEO4J_USER")
		neo4j_pass    = os.Getenv("NEO4J_PASS")
		serverAddress = os.Getenv("SERVER_ADDRESS")
	)

	// Criar uma nova instância de MongoDriver a partir do mongoWriter
	mongoDriver, err := mongo.NewMongoDriver(uri, database, collection)
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
	if err != nil {
		log.Fatalf("Falha ao criar conexão com o MongoDB: %v", err)
	}

<<<<<<< HEAD
	// Create a ScrapLattes instance
	scrapLattes := scrap_lattes.NewScrapLattes(neo4jClient)

	// Create a RabbitMQ connection
	conn, err := amqp.Dial("amqp://guest:guest@localhost:5672/")
	if err != nil {
		log.Fatalf("Falha ao criar conexão com o RabbitMQ: %v", err)
	}

	// Create a RabbitMQ consumer
	consumer := rabbitmq.NewConsumer(conn, "yourQueueName", scrapLattes)
	if err != nil {
		log.Fatalf("Falha ao criar consumidor RabbitMQ: %v", err)
	}

	// Start the RabbitMQ consumer
	go consumer.Start()

	// Create a Neo4jWriteLattes instance
	neo4jWriteLattes, err := neo4jclient.NewNeo4jWriteLattes(uri, username, password)
	if err != nil {
		log.Fatalf("Falha ao criar instância do Neo4jWriteLattes: %v", err)
	}

	// Create a load_lattes.Interactor instance
	interactor := load_lattes.NewInteractor(mongoWriter, neo4jWriteLattes)

	// Create a RabbitMQ enqueuer
	enqueuer, err := rabbitmq.NewEnqueueLattes(interactor, conn, "yourQueueName")
	if err != nil {
		log.Fatalf("Falha ao criar enfileirador RabbitMQ: %v", err)
	}

	http.HandleFunc("/start-scraping", func(w http.ResponseWriter, r *http.Request) {
		file, header, err := r.FormFile("file")
		if err != nil {
			http.Error(w, "Erro ao obter o arquivo", http.StatusInternalServerError)
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
				http.Error(w, "Erro ao ler o arquivo CSV", http.StatusInternalServerError)
				return
			}

			// Check if each line in the CSV has one field
			if len(record) != 1 {
				http.Error(w, "Cada linha do arquivo CSV deve conter exatamente um campo", http.StatusBadRequest)
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

	http.HandleFunc("/enqueue-pesquisador", func(w http.ResponseWriter, r *http.Request) {
		pesquisadorID := r.FormValue("pesquisadorID")
		if pesquisadorID == "" {
			http.Error(w, "Parâmetro pesquisadorID ausente", http.StatusBadRequest)
			return
		}

		err := enqueuer.EnqueuePesquisador(pesquisadorID)
		if err != nil {
			http.Error(w, fmt.Sprintf("Falha ao enfileirar o pesquisadorID %s: %s", pesquisadorID, err), http.StatusInternalServerError)
			return
		}

		json.NewEncoder(w).Encode(map[string]string{
			"message": fmt.Sprintf("PesquisadorID %s enfileirado com sucesso!", pesquisadorID),
		})
	})

	http.ListenAndServe(":8080", nil)
=======
	ctx := context.Background()

	neo4jDriver, err := neo4j.NewDriver(neo4j_uri, neo4j.BasicAuth(neo4j_user, neo4j_pass, ""), func(config *neo4j.Config) {
		// Defina as configurações adicionais aqui, se necessário.
	})
	if err != nil {
		log.Fatalf("Falha ao criar conexão com o Neo4j: %v", err)
	}
	defer neo4jDriver.Close()

	neo4jSession := neo4jDriver.NewSession(neo4j.SessionConfig{})
	defer neo4jSession.Close()

	researcherRepo := repository.NewMongoDBRepository(mongoDriver)

	neo4jClient, err := neo4jclient.NewNeo4jClient(ctx, neo4j_uri, neo4j_user, neo4j_pass)
	if err != nil {
		log.Fatalf("Failed to create Neo4j client: %v", err)
	}
	productionRepo := repository.NewNeo4jRepository(neo4jClient)

	researcherService := application.NewResearcherService(researcherRepo)
	productionService := application.NewProductionService(productionRepo)

	r := mux.NewRouter()

	r.HandleFunc("/api/researcher/{id}", handlers.GetResearcher(researcherService)).Methods("GET")
	r.HandleFunc("/api/production/{id}", handlers.GetProductionHandler(productionService)).Methods("GET")

	server = &http.Server{
		Addr:         serverAddress,
		WriteTimeout: time.Second * 15,
		ReadTimeout:  time.Second * 15,
		IdleTimeout:  time.Second * 60,
		Handler:      r,
	}

	go func() {
		if err := server.ListenAndServe(); err != nil {
			log.Fatal(err)
		}
	}()

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	<-c

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*15)
	defer cancel()
	server.Shutdown(ctx)
	log.Println("Servidor desligando...")
	os.Exit(0)
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
}
