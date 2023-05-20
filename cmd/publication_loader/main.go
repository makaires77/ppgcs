package main

import (
	"fmt"
	"log"
	"os"

	"github.com/makaires77/ppgcs/pkg/infrastructure/dgraph"
	"github.com/makaires77/ppgcs/pkg/infrastructure/json"
	"github.com/makaires77/ppgcs/pkg/interfaces/http/handlers"
	"github.com/makaires77/ppgcs/pkg/usecase/load_publication"
)

func main() {
	// Load JSON data
	publicationData, err := json.LoadPublicationData("_data/in_json/642.files/642.publication.json")
	if err != nil {
		log.Fatal(err)
	}

	// Create Dgraph client
	client, err := dgraph.NewDgraphClient()
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	// Create repository
	repo := dgraph.NewPublicationRepository(client)

	// Create use case interactor
	interactor := load_publication.NewInteractor(repo)

	// Load publications
	err = interactor.LoadPublications(publicationData)
	if err != nil {
		log.Fatal(err)
	}

	// Start API server
	handler := handlers.NewPublicationHandler(interactor)
	addr := ":8080"
	fmt.Printf("Starting API server at %s\n", addr)
	err = handler.StartServer(addr)
	if err != nil {
		log.Fatal(err)
	}

	os.Exit(0)
}
