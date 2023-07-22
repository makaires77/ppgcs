package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/makaires77/ppgcs/pkg/application"
)

func GetProduction(service *application.ProductionService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		params := mux.Vars(r)
		id := params["id"]

		production, err := service.GetProduction(id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Convert the "production" object to JSON
		jsonResponse, err := json.Marshal(production)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Set the content type to application/json
		w.Header().Set("Content-Type", "application/json")

		// Write the JSON response to the ResponseWriter
		w.Write(jsonResponse)
	}
}
