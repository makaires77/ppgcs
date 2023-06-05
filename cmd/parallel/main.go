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
	"github.com/signintech/gopdf"
)

func parallelCompareAuthorWithStudentNames(authorNames []string, studentNames []string, docentName string, progress chan<- string, wg *sync.WaitGroup) {
	defer wg.Done()

	// Mapa para armazenar a contagem de colaboração por docente
	docenteColaboracao := make(map[string]int)

	// Variável para indicar se houve colaboração para o autor atual
	achado := false

	for _, studentName := range studentNames {
		for _, authorName := range authorNames {
			authorName = support.NormalizeName(authorName)
			authorName = strings.TrimSpace(authorName)

			similarity := nomecomparador.JaccardSimilarity(authorName, studentName)
			if similarity > 0.86 {
				msg := fmt.Sprintf("DISCENTE %.2f | %-25s | %-25s | De: %-25s %s", similarity, authorName, studentName, docentName, authorNames)
				progress <- msg

				// Indicar que houve colaboração para o autor atual
				achado = true
				break
			} else {
				msg := fmt.Sprintf("-------- %.2f | %-25s | %-25s | De: %-25s %s", similarity, authorName, studentName, docentName, authorNames)
				progress <- msg
			}
		}

		// Se houve colaboração para o autor atual, incrementar a contagem de colaboração para o docente
		if achado {
			support.Mu.Lock()
			docenteColaboracao[docentName]++
			support.Mu.Unlock()
			achado = false
		}
	}
}

func main() {
	// Iniciar contagem de tempo
	startTime := time.Now()

	// Abrir o arquivo CSV dos autores
	fileAuthors, err := os.Open("_data/powerbi/publicacoes.csv")
	if err != nil {
		log.Fatalf("Falha ao abrir o arquivo CSV das publicações: %v", err)
	}
	defer fileAuthors.Close()

	fmt.Println("Lendo o arquivo CSV dos autores...")

	// Ler os registros do arquivo CSV dos autores
	var authorRecords []*repository.Publications
	if err := gocsv.UnmarshalFile(fileAuthors, &authorRecords); err != nil {
		log.Fatalf("Falha ao extrair autores: %v", err)
	}

	fmt.Printf("Total de registros de autores: %d\n", len(authorRecords))

	// Abrir o arquivo CSV dos discentes
	fileStudents, err := os.Open("_data/powerbi/lista_orientadores-discentes.csv")
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
		log.Fatalf("Falha ao ler o arquivo CSV dos discentes: %v", err)
	}
	fmt.Printf("Total de registros de discentes: %d\n", len(studentRecords))

	// Extrair a segunda coluna dos discentes
	var studentNames []string
	for _, studentRecord := range studentRecords {
		log.Println("Nome a normalizar:", studentRecord[1])
		normalizedStudentName := support.NormalizeName(studentRecord[1])
		log.Println("Nome a normalizado:", normalizedStudentName)
		studentNames = append(studentNames, normalizedStudentName)
	}

	// Criar um canal para enviar atualizações de progresso
	progress := make(chan string)

	fmt.Println("Comparando nomes de autores com nomes de discentes...")

	// Criar uma WaitGroup para sincronizar as goroutines
	var wg sync.WaitGroup

	// Iterar sobre cada combinação de autor e discente
	for _, authorRecord := range authorRecords {
		authorNames := strings.Split(authorRecord.Autores, ";")
		docentName := authorRecord.Name

		// Adicionar a quantidade de comparações ao WaitGroup
		wg.Add(1)

		// Chamar a função em uma goroutine para paralelizar as comparações
		go parallelCompareAuthorWithStudentNames(authorNames, studentNames, docentName, progress, &wg)
	}

	// Goroutine para monitorar o canal de progresso e exibir informações
	go func() {
		for msg := range progress {
			fmt.Println(msg)
		}
	}()

	// Esperar até que todas as goroutines tenham concluído
	wg.Wait()

	// Calcular o total de artigos
	numTotalArticles := len(authorRecords)

	// Mapa para armazenar a contagem de colaboração por docente
	docenteColaboracao := make(map[string]int)

	// Calcular o percentual de colaboração de cada docente
	for docentName := range docenteColaboracao {
		for _, authorRecord := range authorRecords {
			if authorRecord.Name == docentName {
				authorNames := strings.Split(authorRecord.Autores, ";")
				achado := false

				for _, studentName := range studentNames {
					for _, authorName := range authorNames {
						authorName = support.NormalizeName(authorName)
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

	// Fechar o canal de progresso
	close(progress)

	// Tempo total de execução
	elapsedTime := time.Since(startTime)

	// Exibir o resultado da contagem de colaboração por docente
	fmt.Println("\nContagem de colaboração por docente:")
	for docentName, count := range docenteColaboracao {
		fmt.Printf("Docente: %s | Colaboração: %d\n", docentName, count)
	}

	// Exibir estatísticas
	fmt.Printf("\nEstatísticas:\n")
	fmt.Printf("Total de artigos: %d\n", numTotalArticles)
	fmt.Printf("Tempo de execução: %s\n", elapsedTime)

	// Gerar o arquivo de log
	generateLog(authorRecords, studentNames, docenteColaboracao, elapsedTime)

	// Gerar o arquivo PDF com todos os dados calculados
	generatePDF(authorRecords, studentNames, docenteColaboracao, elapsedTime)

	fmt.Println("Programa finalizado.")
}

func generateLog(authorRecords []*repository.Publications, studentNames []string, docenteColaboracao map[string]int, elapsedTime time.Duration) {
	logFile, err := os.Create("log.txt")
	if err != nil {
		log.Fatalf("Falha ao criar o arquivo de log: %v", err)
	}
	defer logFile.Close()

	logger := log.New(logFile, "", log.LstdFlags)

	// Escrever os dados no arquivo de log
	logger.Println("Dados calculados:")
	logger.Printf("Total de artigos: %d\n", len(authorRecords))
	logger.Printf("Total de discentes: %d\n", len(studentNames))
	logger.Printf("Tempo de execução: %s\n", elapsedTime)

	logger.Println("\nContagem de colaboração por docente:")
	for docentName, count := range docenteColaboracao {
		logger.Printf("Docente: %s | Colaboração: %d\n", docentName, count)
	}
}

func generatePDF(authorRecords []*repository.Publications, studentNames []string, docenteColaboracao map[string]int, elapsedTime time.Duration) {
	// Criar um novo PDF
	pdf := gopdf.GoPdf{}
	pdf.Start(gopdf.Config{PageSize: *gopdf.PageSizeA4})
	pdf.AddPage()

	// Definir a fonte e o tamanho do título
	titleFont := "Arial-Bold"
	titleSize := 16.0

	// Definir a fonte e o tamanho do cabeçalho
	headerFont := "Arial-Bold"
	headerSize := 12.0

	// Definir a fonte e o tamanho do texto normal
	textFont := "Arial"
	textSize := 10.0

	// Definir as margens
	var marginLeft float64
	var marginTop float64

	marginLeft = 20.0
	marginTop = 20.0

	// Adicionar o título
	pdf.SetFont(titleFont, "", titleSize)
	pdf.Cell(&gopdf.Rect{
		W: marginLeft,
		H: marginTop,
	}, "Relatório de Colaboração de Docentes")

	// Adicionar a informação de tempo de execução
	pdf.Br(20)
	pdf.SetFont(headerFont, "", headerSize)
	pdf.Cell(nil, "Tempo de Execução:")
	pdf.Br(15)
	pdf.SetFont(textFont, "", textSize)
	pdf.Cell(nil, elapsedTime.String())

	// Adicionar a contagem de colaboração por docente
	pdf.Br(20)
	pdf.SetFont(headerFont, "", headerSize)
	pdf.Cell(nil, "Contagem de Colaboração por Docente:")
	pdf.Br(15)
	pdf.SetFont(textFont, "", textSize)
	for docentName, count := range docenteColaboracao {
		pdf.Cell(nil, fmt.Sprintf("Docente: %s | Colaboração: %d", docentName, count))
		pdf.Br(10)
	}

	// Adicionar informações adicionais
	pdf.Br(20)
	pdf.SetFont(headerFont, "", headerSize)
	pdf.Cell(nil, "Informações Adicionais:")
	pdf.Br(15)
	pdf.SetFont(textFont, "", textSize)
	pdf.Cell(nil, fmt.Sprintf("Total de autores: %d", len(authorRecords)))
	pdf.Br(15)
	pdf.Cell(nil, fmt.Sprintf("Total de artigos: %d", len(authorRecords)))
	pdf.Br(10)
	pdf.Cell(nil, fmt.Sprintf("Total de discentes: %d", len(studentNames)))

	// Salvar o PDF
	err := pdf.WritePdf("_data/pdf/relatorio.pdf")
	if err != nil {
		log.Fatalf("Falha ao salvar o arquivo PDF: %v", err)
	}
}
