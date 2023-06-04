package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"sync"
	"unicode"

	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"

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

func normalizeName(name string) (string, error) {
	name = removePrepositions(name)
	name = normalizeString(name)

	// Verificar se o nome já contém vírgula
	if strings.Contains(name, ",") {
		// Se o nome já contém vírgula, realizar apenas a remoção de acentuação e preposições
		name = convertToInitials(name)
	} else {
		// Se o nome não contém vírgula, trazer o sobrenome para o início e adicionar iniciais
		name = bringLastNameToFront(name)
		name = convertToInitials(name)
	}

	return name, nil
}

func bringLastNameToFront(name string) string {
	names := strings.Fields(name)
	if len(names) > 1 {
		lastName := names[len(names)-1]
		initials := ""
		for i, n := range names[:len(names)-1] {
			if i == 0 {
				// Manter o primeiro nome completo
				initials += n + " "
			} else {
				// Converter os demais nomes em iniciais
				initials += string(n[0]) + ". "
			}
		}
		name = lastName + ", " + strings.TrimSpace(initials)
	}
	return name
}

func removePrepositions(name string) string {
	prepositions := []string{"de", "da", "do", "das", "dos"}
	for _, prep := range prepositions {
		name = strings.ReplaceAll(name, " "+prep+" ", " ")
	}
	return name
}

func convertToInitials(name string) string {
	names := strings.Fields(name)
	initials := ""
	for i, n := range names {
		// Verificar se é um sobrenome
		if i == 0 || strings.Contains(n, "-") {
			initials += n + " "
		} else {
			initials += string(n[0]) + ". "
		}
	}
	return strings.TrimSpace(initials)
}

func normalizeString(s string) string {
	t := transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)), norm.NFC)
	name, _, _ := transform.String(t, s)
	name = removeAccentRunes(name)
	return name
}

func removeAccentRunes(s string) string {
	reg := regexp.MustCompile("[ÀÁÂÃÄÅàáâãäåÈÉÊËèéêëÌÍÎÏìíîïÒÓÔÕÖØòóôõöøÙÚÛÜùúûüÇç]")
	return reg.ReplaceAllString(s, "")
}

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
		log.Println("Nome a normalizar:", studentRecord[1])
		normalizedStudentName, _ := normalizeName(studentRecord[1])
		log.Println("Nome a normalizado:", normalizedStudentName)
		studentNames = append(studentNames, normalizedStudentName)
	}

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

	// Chamar a função CompareNames para cada combinação de autor e discente
	for _, authorRecord := range authorRecords {
		for _, studentName := range studentNames {
			authorNames := strings.Split(authorRecord.Autores, ";")
			docentName := authorRecord.Name

			for _, authorName := range authorNames {
				normalizedName, _ := normalizeName(authorName)
				authorName = strings.TrimSpace(normalizedName)
				// fmt.Printf("Comparando discente %-25s com autor %-25s\n", studentName, authorName)

				similarity := nomecomparador.JaccardSimilarity(authorName, studentName)
				if similarity > 0.86 {
					achado = 1
					msg := fmt.Sprintf("%03d | %.2f | %-25s | %-25s | Currículo: %-25s", totalSimilarities, similarity, authorName, studentName, docentName)
					progress <- msg
				}
			}
		}
		totalSimilarities = totalSimilarities + achado
		achado = 0
		wg.Done()
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
	fmt.Printf("Percentual de colaboração no programa: %.2f%%\n", collaborationPercentage)

	// Aguardar a conclusão de todas as comparações
	wg.Wait()

	// Fechar o canal de progresso
	close(progress)

	fmt.Println("Leitura dos arquivos CSV concluída com sucesso.")

	// Calcular as contagens e o percentual de colaboração
	numArticlesWithSimilarities = totalSimilarities
	numTotalArticles = len(authorRecords)
	numTotalDiscentes = len(studentRecords)
	collaborationPercentage = float64(numArticlesWithSimilarities) / float64(numTotalArticles) * 100

	fmt.Printf("\nResumo da comparação:\n")
	fmt.Printf("Artigos com similaridades: %d\n", numArticlesWithSimilarities)
	fmt.Printf("Total de artigos: %d\n", numTotalArticles)
	fmt.Printf("Total de discentes: %d\n", numTotalDiscentes)
	fmt.Printf("Percentual de colaboração no programa: %.2f%%\n", collaborationPercentage)

	// Totalização do percentual de colaboração de cada docente
	docenteColaboracao := make(map[string]int)
	for _, authorRecord := range authorRecords {
		for _, studentName := range studentNames {
			authorNames := strings.Split(authorRecord.Autores, ";")

			for _, authorName := range authorNames {
				normalizedName, _ := normalizeName(authorName)
				authorName = strings.TrimSpace(normalizedName)

				similarity := nomecomparador.JaccardSimilarity(authorName, studentName)
				if similarity > 0.86 {
					msg := fmt.Sprintf("%03d | %.2f | %-25s | %-25s | Currículo: %-25s", totalSimilarities, similarity, authorName, studentName, authorRecord.Name)
					progress <- msg

					// Incrementar a contagem de colaboração para o docente
					docenteColaboracao[authorRecord.Name]++
				}
			}
		}
		wg.Done()
	}

	// Imprimir a totalização do percentual de colaboração de cada docente
	fmt.Println("\nTotal de colaboração por docente:")
	for docentName, colaboracao := range docenteColaboracao {
		percentual := float64(colaboracao) / float64(numTotalArticles) * 100
		fmt.Printf("%-25s: %.2f%%\n", docentName, percentual)
	}
}
