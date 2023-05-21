package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

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

	// Verifique se o arquivo lista_nomesdocentes.csv existe
	_, err := os.Stat("_data/in_csv/lista_nomesdocentes.csv")
	if err == nil {
		// Arquivo existe, execute o script de upload
		err = UploadCSV(*publicationInteractor, "_data/in_csv/lista_nomesdocentes.csv")
		if err != nil {
			log.Fatalf("Erro ao executar o script de upload: %v", err)
		}
	} else if os.IsNotExist(err) {
		fmt.Println("Arquivo lista_nomesdocentes.csv não encontrado. Certifique-se de fazer o upload antes de executar a API.")
	} else {
		log.Fatalf("Erro ao verificar a existência do arquivo lista_nomesdocentes.csv: %v", err)
	}

	// Inicie o servidor HTTP
	fmt.Println("API do PPGCS em execução na porta 8080...")
	log.Fatal(http.ListenAndServe(":8080", router))
}

// UploadCSV é responsável por fazer o upload do arquivo CSV de pesquisa
func UploadCSV(interactor load_publication.PublicationInteractor, filePath string) error {
	// Implemente aqui a lógica de upload do arquivo CSV e chamadas para as funções necessárias do interactor
	// Certifique-se de manipular erros adequadamente e retornar um erro se ocorrer algum problema durante o upload

	// Exemplo básico de upload
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("Erro ao abrir o arquivo CSV: %w", err)
	}
	defer file.Close()

	// Aqui você pode usar um pacote CSV para ler os dados do arquivo e passá-los para o interactor

	return nil
}
