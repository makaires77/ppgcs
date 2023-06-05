package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gocarina/gocsv"
	"github.com/makaires77/ppgcs/pkg/repository"
	"github.com/makaires77/ppgcs/pkg/support"
	"github.com/makaires77/ppgcs/pkg/usecase/nomecomparador"
)

var mu = &sync.Mutex{}

// Esta função verifica se algum nome de autor é similar a algum nome de estudante.
func isAuthorSimilarToStudent(authorNames []string, studentNames []string) bool {
	for _, studentName := range studentNames {
		for _, authorName := range authorNames {
			authorName = support.NormalizeName(authorName)
			authorName = strings.TrimSpace(authorName)

			similarity := nomecomparador.JaccardSimilarity(authorName, studentName)
			if similarity >= 0.9 {
				return true
			}
		}
	}
	return false
}

// Esta função incrementa a contagem de colaboração.
func incrementCollaborationCount(docenteColaboracao map[string]map[string]*repository.CollaborationData, docentName string, year string) {
	// Usando uma variável temporária
	data := docenteColaboracao[docentName][year]
	if data.TotalCollaboration < data.TotalArticles {
		mu.Lock()
		data.TotalCollaboration++
		mu.Unlock()
		// Salvando a variável modificada de volta no mapa
		docenteColaboracao[docentName][year] = data
	}
}

// Esta função incrementa a contagem de artigos.
func incrementArticleCount(docenteColaboracao map[string]map[string]*repository.CollaborationData, docentName string, year string) {
	data := docenteColaboracao[docentName][year]
	mu.Lock()
	data.TotalArticles++
	mu.Unlock()
	docenteColaboracao[docentName][year] = data
}

// E a função original agora se torna:
func sequentialCompareAuthorWithStudentNames(authorNames []string, studentNames []string, docentName string, year string, docenteColaboracao map[string]map[string]*repository.CollaborationData) {
	for _, authorName := range authorNames {
		if isAuthorSimilarToStudent([]string{authorName}, studentNames) {
			incrementCollaborationCount(docenteColaboracao, docentName, year)
		}
	}
	incrementArticleCount(docenteColaboracao, docentName, year)
}

func generateCSV(docenteColaboracao map[string]map[string]*repository.CollaborationData) {
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

	docenteColaboracao := make(map[string]map[string]*repository.CollaborationData)

	// fmt.Printf("Lendo arquivos de publicações, aguarde...\n")

	fileAuthors, err := os.Open("../../_data/powerbi/publicacoes.csv")
	if err != nil {
		log.Fatalf("Falha ao abrir o arquivo CSV das publicações: %v", err)
	}
	defer fileAuthors.Close()

	var authorRecords []*repository.Publications
	if err := gocsv.UnmarshalFile(fileAuthors, &authorRecords); err != nil {
		log.Fatalf("Falha ao extrair autores: %v", err)
	}

	// fmt.Printf("Lendo arquivos de lista de discentes, aguarde...\n")

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
			docenteColaboracao[docentName] = make(map[string]*repository.CollaborationData)
		}

		if docenteColaboracao[docentName][year] == nil {
			docenteColaboracao[docentName][year] = &repository.CollaborationData{}
		}

		sequentialCompareAuthorWithStudentNames(authorNames, studentNames, docentName, year, docenteColaboracao)
	}

	numTotalArticles := len(authorRecords)

	fmt.Println("\nContagem de colaboração por docente:")
	for docentName, collabData := range docenteColaboracao {
		for year, data := range collabData {
			data.CalculatePercentage()
			fmt.Printf("Docente: %-40s | Ano: %-4s | %-2d/%-2d | Colaboração discente: %.2f%%\n", docentName, year, data.TotalCollaboration, data.TotalArticles, data.PercentCollaboration)
		}
	}

	fmt.Printf("\nCriando arquivo CSV com dados processados, aguarde...\n")
	generateCSV(docenteColaboracao)

	// // gerando o log de desempenho
	// docenteColaboracaoMap := make(map[string]map[string]int)
	// for docentName, collabData := range docenteColaboracao {
	// 	docenteColaboracaoMap[docentName] = make(map[string]int)
	// 	for year, data := range collabData {
	// 		docenteColaboracaoMap[docentName][year] = data.TotalArticles
	// 	}
	// }

	// exibição dos resultados e registro do desempenho
	elapsedTime := time.Since(startTime)
	fmt.Printf("\nEstatísticas:\n")
	fmt.Printf("Total de artigos: %d\n", numTotalArticles)
	fmt.Printf("Tempo de execução: %s\n", elapsedTime)

	// support.GenerateLog(authorRecords, studentNames, docenteColaboracaoMap, elapsedTime)

	// support.GeneratePDF(authorRecords, studentNames, docenteColaboracaoMap, elapsedTime)

	fmt.Println("Programa finalizado.")
}
