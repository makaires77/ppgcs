package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/makaires77/ppgcs/pkg/usecase/load_publication"
)

type PublicationHandler struct {
	Interactor *load_publication.PublicationInteractor
}

func NewPublicationHandler(interactor *load_publication.PublicationInteractor) *PublicationHandler {
	return &PublicationHandler{
		Interactor: interactor,
	}
}

func (h *PublicationHandler) GetPublications(w http.ResponseWriter, r *http.Request) {
	publications, err := h.Interactor.LoadPublications()
	if err != nil {
		http.Error(w, "Failed to load publications", http.StatusInternalServerError)
		return
	}

	// Serialize the publications to JSON
	data, err := json.Marshal(publications)
	if err != nil {
		http.Error(w, "Failed to serialize publications", http.StatusInternalServerError)
		return
	}

	// Set the Content-Type header and write the response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(data)
}

func SetupPublicationHandler(router *mux.Router, interactor *load_publication.PublicationInteractor) {
	handler := NewPublicationHandler(interactor)
	router.HandleFunc("/publications", handler.GetPublications).Methods("GET")
}
