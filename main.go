package main

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
)

// Estrutura de dados para representar um pesquisador
type Researcher struct {
	Name string
}

// Função stub para representar a chamada ao serviço de scrapping
func scrapeResearcherInfo(name string) (string, error) {
	// Adicione a lógica do serviço de scrapping aqui
	// Por enquanto, apenas retorna um valor fictício
	return "Informações do pesquisador: " + name, nil
}

func validateFile(fileName string, file io.Reader) error {
	// Verifica se o arquivo é vazio
	if file == nil {
		return errors.New("the file is empty")
	}

	// Verifica se o arquivo é CSV
	if ext := filepath.Ext(fileName); !strings.EqualFold(ext, ".csv") {
		return errors.New("only .csv files are allowed")
	}

	return nil
}

func main() {
	http.HandleFunc("/start-scraping", func(w http.ResponseWriter, r *http.Request) {
		file, header, err := r.FormFile("file")
		if err != nil {
			http.Error(w, "Error retrieving file", http.StatusInternalServerError)
			return
		}
		defer file.Close()

		// Valida o arquivo
		if err := validateFile(header.Filename, file); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Crie um novo leitor CSV
		csvReader := csv.NewReader(file)

		// Canal para enviar nomes de pesquisadores para as goroutines
		names := make(chan string)

		// WaitGroup para sincronizar as goroutines
		var wg sync.WaitGroup

		// Crie 4 goroutines que processam os nomes de pesquisadores
		for i := 0; i < 4; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for name := range names {
					// Chama o serviço de scrapping
					info, err := scrapeResearcherInfo(name)
					if err != nil {
						fmt.Printf("Erro ao processar o nome '%s': %s\n", name, err)
						continue
					}
					fmt.Printf("Nome processado com sucesso: %s, Informações: %s\n", name, info)
				}
			}()
		}

		// Envia os nomes de pesquisadores para o canal
		for {
			record, err := csvReader.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				http.Error(w, "Error reading CSV file", http.StatusInternalServerError)
				return
			}

			// Verifica se cada linha do CSV tem um campo
			if len(record) != 1 {
				http.Error(w, "Each line in the CSV file must contain exactly one field", http.StatusBadRequest)
				return
			}

			names <- record[0]
		}

		// Fecha o canal para sinalizar que não há mais nomes a serem processados
		close(names)

		// Aguarda todas as goroutines terminarem
		wg.Wait()

		json.NewEncoder(w).Encode(map[string]string{
			"message": "O processo de scraping foi iniciado",
		})
	})

	http.ListenAndServe(":8080", nil)
}
