package main

import (
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
	if err != nil {
		log.Fatalf("Falha ao criar conexão com o MongoDB: %v", err)
	}

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
}
