package handlers

import (
	"net/http"

	"github.com/gorilla/mux"
	"github.com/makaires77/ppgcs/pkg/application"
)

func GetResearcher(service application.ResearcherService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		params := mux.Vars(r)
		id := params["id"]

		researcher, err := service.GetResearcher(id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Aqui você precisa implementar a serialização do objeto "researcher" para JSON
		// e escrever a resposta JSON no ResponseWriter (w).
		// Dependendo da estrutura do seu objeto "researcher", você pode precisar
		// criar uma estrutura de dados separada para a resposta JSON.
	}
}
