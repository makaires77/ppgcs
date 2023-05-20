package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/makaires77/ppgcs/pkg/domain/publication"
	"github.com/makaires77/ppgcs/pkg/interfaces/http/handlers"
	"github.com/makaires77/ppgcs/pkg/usecase/load_publication"
)

func main() {
	// Crie uma instância do repositório do Publication
	publicationRepository := publication.NewInMemoryPublicationRepository()

	// Crie uma instância do interactor do Publication
	publicationInteractor := load_publication.NewPublicationInteractor(publicationRepository)

	// Crie uma instância do handler do Publication
	publicationHandler := handlers.NewPublicationHandler(publicationInteractor)

	// Crie um roteador usando o Gorilla Mux
	router := mux.NewRouter()

	// Configure as rotas do Publication
	router.HandleFunc("/publications", publicationHandler.GetPublications).Methods("GET")

	// Inicie o servidor HTTP
	fmt.Println("API do PPGCS em execução na porta 8080...")
	log.Fatal(http.ListenAndServe(":8080", router))
}
