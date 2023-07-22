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
	"github.com/makaires77/ppgcs/pkg/api/handlers"
	"github.com/makaires77/ppgcs/pkg/application"
	"github.com/makaires77/ppgcs/pkg/infrastructure/mongo"
	"github.com/makaires77/ppgcs/pkg/infrastructure/neo4jclient"
	"github.com/makaires77/ppgcs/pkg/infrastructure/repository"
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

	mongoWriter, err := mongo.NewMongoWriter(uri, database, collection)

	if err != nil {
		log.Fatalf("Falha ao criar conexão com o MongoDB: %v", err)
	}

	neo4jclient, err := neo4jclient.NewNeo4jWriter(neo4j_uri, neo4j_user, neo4j_pass)

	if err != nil {
		log.Fatalf("Falha ao criar conexão com o Neo4j: %v", err)
	}

	researcherRepo := repository.NewMongoDBRepository(mongoWriter.Client())
	productionRepo := repository.NewNeo4jRepository(neo4jclient)

	// ResearcherService
	researcherService := application.NewResearcherService(researcherRepo)
	// ProductionService
	productionService := application.NewProductionService(productionRepo)

	r := mux.NewRouter()

	r.HandleFunc("/api/researcher/{id}", handlers.GetResearcher(researcherService)).Methods("GET")
	r.HandleFunc("/api/production/{id}", handlers.GetProduction(productionService)).Methods("GET")

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
