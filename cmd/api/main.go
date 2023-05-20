package main

import (
	"cmd/api"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/makaires77/ppgcs/internal/publication"
)

func main() {
	repo := repository.NewPublicationRepository()
	service := publication.NewService(repo)
	handler := api.NewHandler(service)

	r := mux.NewRouter()
	r.HandleFunc("/publications", handler.GetPublications).Methods("GET")
	r.HandleFunc("/publications/{id}", handler.GetPublication).Methods("GET")

	http.Handle("/", r)

	fmt.Println("Server running on port 8000")
	err := http.ListenAndServe(":8000", nil)
	if err != nil {
		fmt.Printf("Failed to serve: %v\n", err)
	}
}
