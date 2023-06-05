package support

import (
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/makaires77/ppgcs/pkg/repository"
	"github.com/signintech/gopdf"
	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

// Criar um Mutex para proteger a variável totalSimilarities
var Mu sync.Mutex

func NormalizeName(name string) string {
	name = RemovePrepositions(name)
	name = NormalizeString(name)

	// Verificar se o nome já contém vírgula
	if strings.Contains(name, ",") {
		// Se o nome já contém vírgula, realizar apenas a remoção de acentuação e preposições
		name = ConvertToInitials(name)
	} else {
		// Se o nome não contém vírgula, trazer o sobrenome para o início e adicionar iniciais
		name = BringLastNameToFront(name)
		name = ConvertToInitials(name)
	}

	return name
}

func BringLastNameToFront(name string) string {
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

func RemovePrepositions(name string) string {
	prepositions := []string{"de", "da", "do", "das", "dos"}
	for _, prep := range prepositions {
		name = strings.ReplaceAll(name, " "+prep+" ", " ")
	}
	return name
}

func ConvertToInitials(name string) string {
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

func NormalizeString(s string) string {
	t := transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)), norm.NFC)
	name, _, _ := transform.String(t, s)
	name = RemoveAccentRunes(name)
	name = strings.ToUpper(name)
	return name
}

func RemoveAccentRunes(s string) string {
	reg := regexp.MustCompile("[ÀÁÂÃÄÅàáâãäåÈÉÊËèéêëÌÍÎÏìíîïÒÓÔÕÖØòóôõöøÙÚÛÜùúûüÇç]")
	return reg.ReplaceAllString(s, "")
}

func GenerateLog(authorRecords []*repository.Publications, studentNames []string, docenteColaboracao map[string]int, elapsedTime time.Duration) {
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

func GeneratePDF(authorRecords []*repository.Publications, studentNames []string, docenteColaboracao map[string]int, elapsedTime time.Duration) {
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
