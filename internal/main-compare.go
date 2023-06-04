package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

	"github.com/gocarina/gocsv"
	"github.com/makaires77/ppgcs/pkg/usecase/nomecomparador"
)

type Author struct {
	ID                    string `csv:"idLattes"`
	Name                  string `csv:"nome"`
	Tipo                  string `csv:"tipo"`
	Titulo_cap            string `csv:"titulo_do_capitulo"`
	Idioma                string `csv:"idioma"`
	Titulo_livro          string `csv:"titulo_do_livro"`
	Ano                   string `csv:"ano"`
	Doi                   string `csv:"doi"`
	Pais                  string `csv:"pais_de_publicacao"`
	Isbn                  string `csv:"isbn"`
	Editora_livro         string `csv:"nome_da_editora"`
	Edição_livro          string `csv:"numero_da_edicao_revisao"`
	Organizadores         string `csv:"organizadores"`
	Paginas               string `csv:"paginas"`
	Autores               string `csv:"autores"`
	Autores_endogeno      string `csv:"autores-endogeno"`
	Autores_endogeno_nome string `csv:"autores-endogeno-nome"`
	Tags                  string `csv:"tags"`
	Hash                  string `csv:"Hash"`
	Tipo_producao         string `csv:"tipo_producao"`
	Natureza              string `csv:"natureza"`
	Titulo                string `csv:"titulo"`
	Evento                string `csv:"nome_do_evento"`
	Ano_trabalho          string `csv:"ano_do_trabalho"`
	Pais_evento           string `csv:"pais_do_evento"`
	Cidade_evento         string `csv:"cidade_do_evento"`
	Classificação         string `csv:"classificacao"`
	Periodico             string `csv:"periodico"`
	Volume                string `csv:"volume"`
	Issn                  string `csv:"issn"`
	Estrato_qualis        string `csv:"estrato_qualis"`
	Editora_artigo        string `csv:"editora"`
	Numero_paginas        string `csv:"numero_de_paginas"`
}

type Student struct {
	Name_docente  string `csv:"orientador"`
	Name_discente string `csv:"discente"`
}

// func removeAccents(text string) string {
// 	return unidecode.Unidecode(text)
// }

// func clearScreen() {
// 	if runtime.GOOS == "windows" {
// 		cmd := exec.Command("cmd", "/c", "cls")
// 		cmd.Stdout = os.Stdout
// 		cmd.Run()
// 	} else {
// 		cmd := exec.Command("clear")
// 		cmd.Stdout = os.Stdout
// 		cmd.Run()
// 	}
// }

func main() {
	// Abrir o arquivo CSV dos autores
	fileAuthors, err := os.Open("_data/powerbi/publicacoes.csv")
	if err != nil {
		log.Fatalf("Falha ao abrir o arquivo CSV das publicações: %v", err)
	}
	defer fileAuthors.Close()

	fmt.Println("Lendo o arquivo CSV dos autores...")

	// Ler os registros do arquivo CSV dos autores
	var authorRecords []*Author
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
		studentName := studentRecord[1]
		fmt.Println(studentName)
		studentNames = append(studentNames, studentName)
	}

	// // Imprimir a lista de nomes dos discentes
	// fmt.Println("Lista de discentes:")
	// for _, name := range studentNames {
	// 	fmt.Println(name)
	// }

	// Criar um canal para enviar atualizações de progresso
	progress := make(chan string)

	// Criar uma WaitGroup para sincronizar as goroutines
	var wg sync.WaitGroup

	// Adicionar a quantidade de comparações ao WaitGroup
	totalComparisons := len(authorRecords) * len(studentRecords)
	wg.Add(totalComparisons)

	// Goroutine para monitorar o canal de progresso e exibir informações
	go func() {
		for msg := range progress {
			fmt.Println(msg)
		}
	}()

	fmt.Println("Comparando nomes de autores com nomes de discentes...")

	// Variáveis para contagem
	var totalSimilarities int
	var achado int
	// var remainingComparisons int

	// Chamar a função CompareNames para cada combinação de autor e discente
	for _, authorRecord := range authorRecords {
		achado = 0
		for _, studentRecord := range studentNames {
			authorNames := strings.Split(authorRecord.Autores, ";")
			docentName := strings.Split(authorRecord.Name, ";")
			studentName := studentRecord

			for _, authorName := range authorNames {
				authorName = strings.TrimSpace(authorName)
				fmt.Printf("Comparando %s com %s\n", studentName, authorName)

				similarity := nomecomparador.JaccardSimilarity(authorName, studentName)
				if similarity > 0.85 {
					achado = 1
					msg := fmt.Sprintf("%03d | Similaridade %.2f entre: %-40s | %-40s | Docente: %-40s", totalSimilarities, similarity, authorName, studentName, docentName)
					progress <- msg
					// clearScreen()
				}
			}
			wg.Done()
		}
		totalSimilarities = totalSimilarities + achado
	}

	// Calcular as contagens e o percentual de colaboração
	numArticlesWithSimilarities := totalSimilarities
	numTotalArticles := len(authorRecords)
	numTotalDiscentes := len(studentRecords)
	collaborationPercentage := float64(numArticlesWithSimilarities) / float64(numTotalArticles) * 100

	fmt.Printf("\nResumo da comparação:\n")
	fmt.Printf("Artigos com similaridades: %d\n", numArticlesWithSimilarities)
	fmt.Printf("Total de artigos: %d\n", numTotalArticles)
	fmt.Printf("Total de discentes: %d\n", numTotalDiscentes)
	fmt.Printf("Percentual de colaboração: %.2f%%\n", collaborationPercentage)

	// Aguardar a conclusão de todas as comparações
	wg.Wait()

	// Fechar o canal de progresso
	close(progress)

	fmt.Println("Leitura dos arquivos CSV concluída com sucesso.")
}
