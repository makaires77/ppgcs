package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

	"github.com/makaires77/ppgcs/pkg/usecase/nomecomparador"
)

func main() {
	// Abrir o arquivo CSV com os nomes dos autores
	fileAuthors, err := os.Open("_data/powerbi/publicacoes.csv")
	if err != nil {
		log.Fatalf("Falha ao abrir o arquivo CSV dos autores: %v", err)
	}
	defer fileAuthors.Close()

	// Ler os registros do arquivo CSV dos autores
	readerAuthors := csv.NewReader(fileAuthors)
	recordsAuthors, err := readerAuthors.ReadAll()
	if err != nil {
		log.Fatalf("Falha ao ler o arquivo CSV dos autores: %v", err)
	}

	// Abrir o arquivo CSV com os nomes dos discentes
	fileStudents, err := os.Open("_data/powerbi/lista_orientadores-discentes.csv")
	if err != nil {
		log.Fatalf("Falha ao abrir o arquivo CSV dos discentes: %v", err)
	}
	defer fileStudents.Close()

	// Ler os registros do arquivo CSV dos discentes
	readerStudents := csv.NewReader(fileStudents)
	recordsStudents, err := readerStudents.ReadAll()
	if err != nil {
		log.Fatalf("Falha ao ler o arquivo CSV dos discentes: %v", err)
	}

	// Converter os registros em slices de strings
	authors := make([][]string, len(recordsAuthors))
	students := make([][]string, len(recordsStudents))

	for i, record := range recordsAuthors {
		authors[i] = strings.Split(record[14], ";")
	}

	for i, record := range recordsStudents {
		students[i] = []string{record[1]}
	}

	// Criar um canal para enviar atualizações de progresso
	progress := make(chan string)

	// Criar uma WaitGroup para sincronizar as goroutines
	var wg sync.WaitGroup

	// Adicionar a quantidade de comparações ao WaitGroup
	wg.Add(len(authors) * len(students))

	// Goroutine para monitorar o canal de progresso e exibir informações
	go func() {
		for msg := range progress {
			fmt.Println(msg)
		}
	}()

	// Chamar a função CompareNames
	x := &wg
	go nomecomparador.CompareNames(authors, students, x, progress)

	// Aguardar a conclusão de todas as comparações
	wg.Wait()

	// Fechar o canal de progresso
	close(progress)
}
