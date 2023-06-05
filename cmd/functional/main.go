package functional

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

func sequentialCompareAuthorWithStudentNames(authorNames []string, studentNames []string) (map[string]int, []string) {
	docenteColaboracao := make(map[string]int)
	progress := make([]string, 0)

	for _, studentName := range studentNames {
		for _, authorName := range authorNames {
			authorName = support.NormalizeName(authorName)
			authorName = strings.TrimSpace(authorName)

			similarity := nomecomparador.JaccardSimilarity(authorName, studentName)
			if similarity > 0.86 {
				msg := fmt.Sprintf("DISCENTE %.2f | %-25s | %-25s | De: %-25s", similarity, authorName, studentName, docentName)
				progress = append(progress, msg)
				docenteColaboracao[docentName]++
				break
			} else {
				msg := fmt.Sprintf("-------- %.2f | %-25s | %-25s | De: %-25s", similarity, authorName, studentName, docentName)
				progress = append(progress, msg)
			}
		}
	}

	return docenteColaboracao, progress
}

func main() {
	startTime := time.Now()

	fileAuthors, err := os.Open("../../_data/powerbi/publicacoes.csv")
	if err != nil {
		log.Fatalf("Falha ao abrir o arquivo CSV das publicações: %v", err)
	}
	defer fileAuthors.Close()

	fmt.Println("Lendo o arquivo CSV dos autores...")

	var authorRecords []*repository.Publications
	if err := gocsv.UnmarshalFile(fileAuthors, &authorRecords); err != nil {
		log.Fatalf("Falha ao extrair autores: %v", err)
	}

	fmt.Printf("Total de registros de autores: %d\n", len(authorRecords))

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
		log.Fatalf("Falha ao ler o arquivo CSV dos discentes: %v", err)
	}
	fmt.Printf("Total de registros de discentes: %d\n", len(studentRecords))

	var studentNames []string
	for _, studentRecord := range studentRecords {
		log.Println("Nome a normalizar:", studentRecord[1])
		normalizedStudentName := support.NormalizeName(studentRecord[1])
		log.Println("Nome normalizado:", normalizedStudentName)
		studentNames = append(studentNames, normalizedStudentName)
	}

	progress := make(chan string)

	fmt.Println("Comparando nomes de autores com nomes de discentes...")

	var wg sync.WaitGroup

	for _, authorRecord := range authorRecords {
		authorNames := strings.Split(authorRecord.Autores, ";")
		docentName := authorRecord.Name

		wg.Add(1)
		go func(authorNames []string, docentName string) {
			defer wg.Done()
			docenteColaboracao, progress := sequentialCompareAuthorWithStudentNames(authorNames, studentNames)
			for _, msg := range progress {
				progress <- msg
			}
			support.Mu.Lock()
			for k, v := range docenteColaboracao {
				docenteColaboracao[k] += v
			}
			support.Mu.Unlock()
		}(authorNames, docentName)
	}

	go func() {
		wg.Wait()
		close(progress)
	}()

	for msg := range progress {
		fmt.Println(msg)
	}

	numTotalArticles := len(authorRecords)
	docenteColaboracao := make(map[string]int)

	for docentName := range docenteColaboracao {
		for _, authorRecord := range authorRecords {
			if authorRecord.Name == docentName {
				authorNames := strings.Split(authorRecord.Autores, ";")
				achado := false

				for _, studentName := range studentNames {
					for _, authorName := range authorNames {
						support.NormalizeName(authorName)
						authorName = strings.TrimSpace(authorName)

						similarity := nomecomparador.JaccardSimilarity(authorName, studentName)
						if similarity > 0.86 {
							achado = true
							break
						}
					}
					if achado {
						break
					}
				}
				if achado {
					docenteColaboracao[docentName]++
				}
			}
		}
	}

	elapsedTime := time.Since(startTime)

	fmt.Println("\nContagem de colaboração por docente:")
	for docentName, count := range docenteColaboracao {
		fmt.Printf("Docente: %s | Colaboração: %d\n", docentName, count)
	}

	fmt.Printf("\nEstatísticas:\n")
	fmt.Printf("Total de artigos: %d\n", numTotalArticles)
	fmt.Printf("Tempo de execução: %s\n", elapsedTime)

	support.GenerateLog(authorRecords, studentNames, docenteColaboracao, elapsedTime)
	support.GeneratePDF(authorRecords, studentNames, docenteColaboracao, elapsedTime)

	fmt.Println("Programa finalizado.")
}
