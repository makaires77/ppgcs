package csv_publication

import (
	"errors"
	"fmt"
	"os"

	"github.com/makaires77/ppgcs/pkg/domain/publication"
)

type CSVReader struct{}

// NewCSVReader creates a new CSV reader
func NewCSVReader() *CSVReader {
	return &CSVReader{}
}

// Lê os arquivos CSV e retorna um array de Publications e Pesquisadores
func (r *CSVReader) ReadCSV(publicationFilePath string, researcherFilePath string) ([]publication.Publication, []publication.Pesquisador, error) {
	// Abre os arquivos
	pubFile, err := os.Open(publicationFilePath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open publication file: %w", err)
	}
	defer pubFile.Close()

	resFile, err := os.Open(researcherFilePath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open researcher file: %w", err)
	}
	defer resFile.Close()

	// Lê os arquivos CSV
	publications, err := readPublications(pubFile)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read publications: %w", err)
	}

	researchers, err := readResearchers(resFile)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read researchers: %w", err)
	}

	return publications, researchers, nil
}

// Lê o arquivo de Publications
func readPublications(file *os.File) ([]publication.Publication, error) {
	// TODO: Implementar a leitura dos dados do CSV em objetos do tipo Publication
	return nil, errors.New("not implemented")
}

// Lê o arquivo de Pesquisadores
func readResearchers(file *os.File) ([]publication.Pesquisador, error) {
	// TODO: Implementar a leitura dos dados do CSV em objetos do tipo Pesquisador
	return nil, errors.New("not implemented")
}
