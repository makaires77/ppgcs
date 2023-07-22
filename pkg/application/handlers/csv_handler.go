// pkg\application\handlers\csv_handler.go
package handlers

import (
	"encoding/csv"
	"os"

	"github.com/makaires77/ppgcs/pkg/domain/researcher"
)

type CsvHandler struct {
	fileName string
	file     *os.File
	writer   *csv.Writer
}

func NewCsvHandler(fileName string) (*CsvHandler, error) {
	file, err := os.Create(fileName)
	if err != nil {
		return nil, err
	}

	writer := csv.NewWriter(file)

	return &CsvHandler{
		fileName: fileName,
		file:     file,
		writer:   writer,
	}, nil
}

func (h *CsvHandler) WriteResearchers(researchers []researcher.Researcher) error {
	// Escreva o cabeçalho do CSV
	if err := h.writeHeader(); err != nil {
		return err
	}

	for _, r := range researchers {
		record := []string{
			r.Nome,
			r.Titulo,
			r.LinkCurriculo,
			r.IDLattes,
			r.DataUltimaAtualizacao,
			r.Resumo,
			r.NomeCitacoesBibliograficas,
			r.IDLattesLink,
			r.OrcidID,
			// Você pode querer juntar todos os elementos de Formacao em uma única string
			// e etc. para os demais campos que são slices.
		}

		if err := h.writer.Write(record); err != nil {
			return err
		}
	}

	h.writer.Flush()
	return h.writer.Error()
}

func (h *CsvHandler) writeHeader() error {
	header := []string{
		"Nome",
		"Titulo",
		"LinkCurriculo",
		"IDLattes",
		"DataUltimaAtualizacao",
		"Resumo",
		"NomeCitacoesBibliograficas",
		"IDLattesLink",
		"OrcidID",
		"Formacao",
		// e etc. para os demais campos do struct Researcher.
	}

	return h.writer.Write(header)
}

func (h *CsvHandler) Close() error {
	return h.file.Close()
}
