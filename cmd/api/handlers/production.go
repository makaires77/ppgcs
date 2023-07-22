package handlers

import (
	"net/http"

	"github.com/gorilla/mux"
	"github.com/makaires77/ppgcs/pkg/application"
)

func GetProduction(service application.ProductionService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		params := mux.Vars(r)
		id := params["id"]

		production, err := service.GetProduction(id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Aqui você precisa implementar a serialização do objeto "production" para JSON
		// e escrever a resposta JSON no ResponseWriter (w).
		// Dependendo da estrutura do seu objeto "production", você pode precisar
		// criar uma estrutura de dados separada para a resposta JSON.
	}
}
