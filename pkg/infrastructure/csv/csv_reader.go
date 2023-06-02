//Este código define uma interface Reader com um método Read, que lê um arquivo CSV e retorna seus conteúdos como um slice de slices de strings. A função NewReader retorna uma nova instância da interface Reader. O método Read da estrutura reader implementa o método Read da interface Reader. Ele abre o arquivo CSV, lê todas as linhas e retorna o conteúdo. Se houver um erro durante a abertura ou leitura do arquivo, ele é tratado e apropriado.

package csv

import (
	"encoding/csv"
	"os"

	"github.com/pkg/errors"
)

type Reader interface {
	Read(file string) ([][]string, error)
}

type reader struct{}

func NewReader() Reader {
	return &reader{}
}

func (*reader) Read(file string) ([][]string, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, errors.Wrap(err, "unable to open csv file")
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, errors.Wrap(err, "unable to parse csv file")
	}

	return lines, nil
}
