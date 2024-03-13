// cmd\api\handlers\production.go
package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/makaires77/ppgcs/pkg/application"
)

// GetProductionHandler retorna um http.HandlerFunc que obtém os detalhes de uma produção pelo seu ID.
func GetProductionHandler(service *application.ProductionService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		params := mux.Vars(r)
		id := params["id"]

		production, err := service.GetProductionByID(id)
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
