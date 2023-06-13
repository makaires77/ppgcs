package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gocarina/gocsv"
	"github.com/makaires77/ppgcs/pkg/repository"
	"github.com/makaires77/ppgcs/pkg/support"
	"github.com/makaires77/ppgcs/pkg/usecase/nomecomparador"
)

type CollaborationData struct {
	TotalArticles        int
	TotalCollaboration   int
	PercentCollaboration float64
}

var mu = &sync.Mutex{}

func sequentialCompareAuthorWithStudentNames(authorNames []string, studentNames []string, docentName string, year string, docenteColaboracao map[string]map[string]*CollaborationData) {
	achado := false

	for _, studentName := range studentNames {
		for _, authorName := range authorNames {
			authorName = support.NormalizeName(authorName)
			authorName = strings.TrimSpace(authorName)

			similarity := nomecomparador.JaccardSimilarity(authorName, studentName)
			if similarity >= 0.9 {
				achado = true
				break
			}
		}

		if achado && docenteColaboracao[docentName][year].TotalCollaboration < docenteColaboracao[docentName][year].TotalArticles {
			mu.Lock()
			docenteColaboracao[docentName][year].TotalCollaboration++
			mu.Unlock()
		}
		achado = false
	}

	mu.Lock()
	docenteColaboracao[docentName][year].TotalArticles++
	mu.Unlock()
}

func (cd *CollaborationData) ToStringSlice() []string {
	return []string{
		strconv.Itoa(cd.TotalArticles),
		strconv.Itoa(cd.TotalCollaboration),
		fmt.Sprintf("%.2f", cd.PercentCollaboration),
	}
}

func generateCSV(docenteColaboracao map[string]map[string]*CollaborationData) {
	file, err := os.Create("../../_data/powerbi/colaboracao_discente.csv")
	if err != nil {
		log.Fatalf("Não foi possível criar o arquivo CSV: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Escrevendo cabeçalho do CSV
	err = writer.Write([]string{"nome_docente", "ano", "total_artigos_periodo", "qte_colaboracao_discente", "percentual_colaboracao_discente"})
	if err != nil {
		log.Fatalf("Não foi possível escrever o cabeçalho do CSV: %v", err)
	}

	// Escrevendo cada linha do CSV
	for docentName, data := range docenteColaboracao {
		for year, collaborationData := range data {
			line := append([]string{docentName, year}, collaborationData.ToStringSlice()...)
			err = writer.Write(line)
			if err != nil {
				log.Fatalf("Não foi possível escrever a linha no CSV: %v", err)
			}
		}
	}

	fmt.Println("\nArquivo CSV criado com sucesso.")
}

func main() {
	startTime := time.Now()

	docenteColaboracao := make(map[string]map[string]*CollaborationData)

	fileAuthors, err := os.Open("../../_data/powerbi/publicacoes.csv")
	if err != nil {
		log.Fatalf("Falha ao abrir o arquivo CSV das publicações: %v", err)
	}
	defer fileAuthors.Close()

	var authorRecords []*repository.Publications
	if err := gocsv.UnmarshalFile(fileAuthors, &authorRecords); err != nil {
		log.Fatalf("Falha ao extrair autores: %v", err)
	}

	fileStudents, err := os.Open("../../_data/powerbi/lista_orientadores-discentes.csv")
	if err != nil {
		log.Fatalf("Falha ao abrir o arquivo CSV dos discentes: %v", err)
	}
	defer fileStudents.Close()

	readerStudents := csv.NewReader(fileStudents)
	readerStudents.Comma = ';'
	readerStudents.LazyQuotes = true

	studentRecords, err := readerStudents.ReadAll()
	if err != nil {
		log.Fatalf("Falha ao extrair discentes: %v", err)
	}

	var studentNames []string
	for _, studentRecord := range studentRecords {
		normalizedStudentName := support.NormalizeName(studentRecord[1])
		studentNames = append(studentNames, normalizedStudentName)
	}

	for _, authorRecord := range authorRecords {
		// Só considerar registros com Tipo_producao igual a "PERIODICO"
		if authorRecord.Tipo_producao != "PERIODICO" {
			continue
		}

		authorNames := strings.Split(authorRecord.Autores, ";")
		docentName := authorRecord.Name
		year := authorRecord.Ano

		if docenteColaboracao[docentName] == nil {
			docenteColaboracao[docentName] = make(map[string]*CollaborationData)
		}

		if docenteColaboracao[docentName][year] == nil {
			docenteColaboracao[docentName][year] = &CollaborationData{}
		}

		sequentialCompareAuthorWithStudentNames(authorNames, studentNames, docentName, year, docenteColaboracao)
	}

	numTotalArticles := len(authorRecords)
	elapsedTime := time.Since(startTime)

	fmt.Printf("\nEstatísticas:\n")
	fmt.Printf("Total de artigos: %d\n", numTotalArticles)
	fmt.Printf("Tempo de execução: %s\n", elapsedTime)

	fmt.Println("\nContagem de colaboração por docente:")
	for docentName, collabData := range docenteColaboracao {
		for year, data := range collabData {
			data.PercentCollaboration = (float64(data.TotalCollaboration) / float64(data.TotalArticles))
			fmt.Printf("Docente: %-40s | Ano: %-4s | Colaboração discente: %.2f%%\n", docentName, year, data.PercentCollaboration)
		}
	}

	generateCSV(docenteColaboracao)

	// support.GenerateLog(authorRecords, studentNames, docenteColaboracao, elapsedTime)

	fmt.Println("Programa finalizado.")
}
