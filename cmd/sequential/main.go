package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gocarina/gocsv"
	"github.com/makaires77/ppgcs/pkg/repository"
	"github.com/makaires77/ppgcs/pkg/support"
	"github.com/makaires77/ppgcs/pkg/usecase/nomecomparador"
)

func sequentialCompareAuthorWithStudentNames(authorNames []string, studentNames []string, docentName string, docenteColaboracao map[string]int, docenteTotalArtigos map[string]int) {
	// Variável para indicar se houve colaboração para o autor atual
	achado := false

	for _, studentName := range studentNames {
		for _, authorName := range authorNames {
			authorName = support.NormalizeName(authorName)
			authorName = strings.TrimSpace(authorName)

			similarity := nomecomparador.JaccardSimilarity(authorName, studentName)
			if similarity >= 0.9 {
				// Indicar que houve colaboração para o autor atual
				achado = true
				break
			}
		}

		// Se houve colaboração para o autor atual, incrementar a contagem de colaboração para o docente
		// Mas apenas se ainda não foi contado para este artigo
		if achado && docenteColaboracao[docentName] < docenteTotalArtigos[docentName] {
			support.Mu.Lock()
			docenteColaboracao[docentName]++
			support.Mu.Unlock()
		}
		achado = false
	}

	// Incrementar a contagem total de artigos do docente
	support.Mu.Lock()
	docenteTotalArtigos[docentName]++
	support.Mu.Unlock()
}

func generateCSV(docenteColaboracao map[string]int, docenteTotalArtigos map[string]int) {
	file, err := os.Create("../../_data/powerbi/colaboracao_discente.csv")
	if err != nil {
		log.Fatalf("Não foi possível criar o arquivo CSV: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Escrevendo cabeçalho do CSV
	err = writer.Write([]string{"nome_docente", "total_artigos_periodo", "qte_colaboracao_discente", "percentual_colaboracao_discente"})
	if err != nil {
		log.Fatalf("Não foi possível escrever o cabeçalho do CSV: %v", err)
	}

	// Escrevendo cada linha do CSV
	for docentName, count := range docenteColaboracao {
		totalArticles := docenteTotalArtigos[docentName]
		percentual := (float64(count) / float64(totalArticles)) * 100
		line := []string{docentName, strconv.Itoa(totalArticles), strconv.Itoa(count), fmt.Sprintf("%.2f", percentual)}
		err = writer.Write(line)
		if err != nil {
			log.Fatalf("Não foi possível escrever a linha no CSV: %v", err)
		}
	}

	fmt.Println("\nArquivo CSV criado com sucesso.")
}

func main() {
	// Iniciar contagem de tempo
	startTime := time.Now()

	// Mapa para armazenar a contagem de colaboração por docente
	docenteColaboracao := make(map[string]int)
	// Mapa para armazenar o total de artigos de cada docente
	docenteTotalArtigos := make(map[string]int)

	// Abrir o arquivo CSV dos autores
	fileAuthors, err := os.Open("../../_data/powerbi/publicacoes.csv")
	if err != nil {
		log.Fatalf("Falha ao abrir o arquivo CSV das publicações: %v", err)
	}
	defer fileAuthors.Close()

	// Ler os registros do arquivo CSV dos autores
	var authorRecords []*repository.Publications
	if err := gocsv.UnmarshalFile(fileAuthors, &authorRecords); err != nil {
		log.Fatalf("Falha ao extrair autores: %v", err)
	}

	fmt.Printf("Total de registros de autores: %d\n", len(authorRecords))

	// Abrir o arquivo CSV dos discentes
	fileStudents, err := os.Open("../../_data/powerbi/lista_orientadores-discentes.csv")
	if err != nil {
		log.Fatalf("Falha ao abrir o arquivo CSV dos discentes: %v", err)
	}
	defer fileStudents.Close()

	// Ler os registros do arquivo CSV dos discentes
	readerStudents := csv.NewReader(fileStudents)
	readerStudents.Comma = ';'
	readerStudents.LazyQuotes = true

	studentRecords, err := readerStudents.ReadAll()
	if err != nil {
		log.Fatalf("Falha ao extrair discentes: %v", err)
	}

	fmt.Printf("Total de registros de discentes: %d\n", len(studentRecords))

	// Extrair a segunda coluna dos discentes
	var studentNames []string
	for _, studentRecord := range studentRecords {
		// log.Println("Nome a normalizar:", studentRecord[1])
		normalizedStudentName := support.NormalizeName(studentRecord[1])
		// log.Println("Nome  normalizado:", normalizedStudentName)
		studentNames = append(studentNames, normalizedStudentName)
	}

	fmt.Println("Comparando nomes de autores com nomes de discentes...")

	// Iterar sobre cada combinação de autor e discente
	for _, authorRecord := range authorRecords {
		authorNames := strings.Split(authorRecord.Autores, ";")
		docentName := authorRecord.Name

		// Chamar a função sequencialmente
		sequentialCompareAuthorWithStudentNames(authorNames, studentNames, docentName, docenteColaboracao, docenteTotalArtigos)
	}

	// Calcular o total de artigos
	numTotalArticles := len(authorRecords)

	// Tempo total de execução
	elapsedTime := time.Since(startTime)

	// Exibir estatísticas
	fmt.Printf("\nEstatísticas:\n")
	fmt.Printf("Total de artigos: %d\n", numTotalArticles)
	fmt.Printf("Tempo de execução: %s\n", elapsedTime)

	// Exibir o resultado da contagem de colaboração por docente
	fmt.Println("\nContagem de colaboração por docente:")
	for docentName, count := range docenteColaboracao {
		totalArticles := docenteTotalArtigos[docentName]
		percentual := (float64(count) / float64(totalArticles)) * 100
		fmt.Printf("Docente: %-40s | %-3d | Colaboração discente: %.2f%%\n", docentName, count, percentual)
	}

	// Gerar o arquivo CSV
	generateCSV(docenteColaboracao, docenteTotalArtigos)

	// Gerar o arquivo de log
	support.GenerateLog(authorRecords, studentNames, docenteColaboracao, elapsedTime)

	// Gerar o arquivo PDF com todos os dados calculados
	// support.GeneratePDF(authorRecords, studentNames, docenteColaboracao, elapsedTime)

	fmt.Println("Programa finalizado.")
}
