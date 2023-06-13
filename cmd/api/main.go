package main

import (
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/gorilla/mux"
)

func main() {
	// Crie um roteador usando o Gorilla Mux
	router := mux.NewRouter()

	// Configure a rota para renderizar a página inicial
	router.HandleFunc("/", homeHandler).Methods("GET")

	// Configure a rota para lidar com o upload do arquivo CSV
	router.HandleFunc("/upload", uploadHandler).Methods("POST")

	// Inicie o servidor HTTP
	fmt.Println("Aplicação web do PPGCS em execução na porta 8080...")
	log.Fatal(http.ListenAndServe(":8080", router))
}

func homeHandler(w http.ResponseWriter, r *http.Request) {
	// Carregue o template HTML
	tmpl, err := template.ParseFiles("templates/index.html")
	if err != nil {
		http.Error(w, "Erro ao carregar o template", http.StatusInternalServerError)
		return
	}

	// Renderize o template
	err = tmpl.Execute(w, nil)
	if err != nil {
		http.Error(w, "Erro ao renderizar o template", http.StatusInternalServerError)
		return
	}
}

func uploadHandler(w http.ResponseWriter, r *http.Request) {
	// Obtenha o arquivo enviado pelo usuário
	file, handler, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Erro ao obter o arquivo", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Verifique se o arquivo é um arquivo CSV
	if handler.Header.Get("Content-Type") != "text/csv" {
		http.Error(w, "Arquivo inválido. Por favor, selecione um arquivo CSV.", http.StatusBadRequest)
		return
	}

	// Crie um diretório temporário para armazenar o arquivo
	tempDir := "temp"
	err = os.MkdirAll(tempDir, 0755)
	if err != nil {
		http.Error(w, "Erro ao criar o diretório temporário", http.StatusInternalServerError)
		return
	}

	// Salve o arquivo no diretório temporário
	filePath := tempDir + "/" + handler.Filename
	out, err := os.Create(filePath)
	if err != nil {
		http.Error(w, "Erro ao salvar o arquivo", http.StatusInternalServerError)
		return
	}
	defer out.Close()

	_, err = io.Copy(out, file)
	if err != nil {
		http.Error(w, "Erro ao salvar o arquivo", http.StatusInternalServerError)
		return
	}

	// Execute o scraping dos dados a partir do arquivo CSV
	/* 	err = scrap_lattes.ScrapeData(filePath)
	   	if err != nil {
	   		http.Error(w, "Erro ao executar o scraping dos dados", http.StatusInternalServerError)
	   		return
	   	}

	   	// Exiba uma mensagem de sucesso
	   	fmt.Fprintln(w, "Scraping dos dados concluído com sucesso!") */
}
