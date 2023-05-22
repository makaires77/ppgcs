package main

import (
	"encoding/json"
	"fmt"
	"html/template"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
)

// Estrutura de dados para representar uma lista de pesquisadores
type ResearcherList struct {
	Name           string
	ResearcherName string
	Program        string
	Institution    string
}

// Dados das listas existentes na pasta /home/marcos/ppgcs/_data/in_csv
var existingLists = []string{}

func fetchExistingLists() {
	// Buscar os arquivos da pasta in_csv no GitHub
	resp, err := http.Get("https://api.github.com/repos/makaires77/ppgcs/contents/_data/in_csv")
	if err != nil {
		log.Fatal("Erro ao buscar arquivos:", err)
	}
	defer resp.Body.Close()

	// Ler a resposta da requisição
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Fatal("Erro ao ler resposta:", err)
	}

	// Converter a resposta para uma lista de nomes de arquivo
	var files []struct {
		Name string `json:"name"`
	}
	err = json.Unmarshal(body, &files)
	if err != nil {
		log.Fatal("Erro ao decodificar resposta:", err)
	}

	// Preencher a lista de nomes de arquivo
	for _, file := range files {
		if strings.HasSuffix(file.Name, ".csv") {
			existingLists = append(existingLists, file.Name)
		}
	}
}

// Função para lidar com a rota principal
func homeHandler(w http.ResponseWriter, r *http.Request) {
	// Carregar o template HTML
	tmpl, err := template.ParseFiles("static/index.html")
	if err != nil {
		log.Fatal(err)
	}

	// Renderizar o template HTML com os dados das listas existentes
	err = tmpl.Execute(w, existingLists)
	if err != nil {
		log.Fatal(err)
	}
}

// Função para lidar com o envio do formulário
func submitHandler(w http.ResponseWriter, r *http.Request) {
	// Obter os valores do formulário
	period := r.FormValue("period")
	selectedList := r.FormValue("list")

	// Verificar se uma lista existente foi selecionada
	if selectedList != "" {
		// Realizar o scrap de dados com base na lista selecionada e no período definido
		// Implemente aqui a lógica de scrap

		// Exemplo de resposta
		message := fmt.Sprintf("Scrap de dados realizado para a lista %s no período %s", selectedList, period)
		w.Write([]byte(message))
	} else {
		// Obter os valores do formulário para criar uma nova lista
		researcherName := r.FormValue("researcher_name")
		program := r.FormValue("program")
		institution := r.FormValue("institution")

		// Criar uma nova lista com base nos valores fornecidos
		newList := ResearcherList{
			Name:           "Nova Lista",
			ResearcherName: researcherName,
			Program:        program,
			Institution:    institution,
		}

		// Realizar o scrap de dados com base na nova lista e no período definido
		// Implemente aqui a lógica de scrap

		// Exemplo de resposta
		message := fmt.Sprintf("Nova lista criada: %+v. Scrap de dados realizado no período %s", newList, period)
		w.Write([]byte(message))
	}
}

var templates = template.Must(template.ParseFiles("static/index.html"))

func main() {
	// Buscar as listas existentes
	fetchExistingLists()

	// Configurar roteamento de URLs
	http.HandleFunc("/", homeHandler)
	http.HandleFunc("/submit", submitHandler)

	// Configurar diretório estático para arquivos CSS e JS do Bootstrap
	fs := http.FileServer(http.Dir("static"))
	http.Handle("/static/", http.StripPrefix("/static/", fs))

	// Iniciar o servidor na porta 8080
	log.Println("Servidor rodando em http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
