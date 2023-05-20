package main

import (
	"log"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/makaires77/ppgcs/pkg/domain/publication/repository"
	"github.com/makaires77/ppgcs/pkg/interfaces/http/handlers"
	"github.com/makaires77/ppgcs/pkg/usecase/load_publication"
)

func main() {
	// Configuração do repositório de publicação
	publicationRepo := repository.NewPublicationRepository()

	// Configuração do interactor de publicação
	publicationInteractor := load_publication.NewPublicationInteractor(publicationRepo)

	// Configuração do handler de publicação
	publicationHandler := handlers.NewPublicationHandler(publicationInteractor)

	// Configuração do roteador
	router := mux.NewRouter()
	router.HandleFunc("/publications", publicationHandler.GetAllPublications).Methods("GET")

	// Inicialização do servidor HTTP
	log.Println("Servidor iniciado na porta 8080")
	err := http.ListenAndServe(":8080", router)
	if err != nil {
		log.Fatalf("Falha ao iniciar o servidor HTTP: %v", err)
	}
}
