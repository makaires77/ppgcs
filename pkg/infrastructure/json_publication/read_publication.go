package json

import (
	"encoding/json"
	"io/ioutil"

	"github.com/makaires77/ppgcs/pkg/domain/publication"
)

type Publication struct {
	Natureza            string              `json:"natureza"`
	Titulo              string              `json:"titulo"`
	Idioma              string              `json:"idioma"`
	Periodico           string              `json:"periodico"`
	Ano                 string              `json:"ano"`
	Volume              string              `json:"volume"`
	ISSN                string              `json:"issn"`
	EstratoQualis       string              `json:"estrato_qualis"`
	PaisDePublicacao    string              `json:"pais_de_publicacao"`
	Paginas             string              `json:"paginas"`
	DOI                 string              `json:"doi"`
	Autores             []string            `json:"autores"`
	AutoresEndogeno     []string            `json:"autores-endogeno"`
	AutoresEndogenoNome []map[string]string `json:"autores-endogeno-nome"`
	Tags                []interface{}       `json:"tags"`
	Hash                string              `json:"hash"`
}

type PublicationReader interface {
	ReadPublications(path string) ([]*Publication, error)
}

type publicationReader struct{}

func NewPublicationReader() PublicationReader {
	return &publicationReader{}
}

func (r *publicationReader) ReadPublications(path string) ([]*Publication, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var publications []*Publication
	err = json.Unmarshal(data, &publications)
	if err != nil {
		return nil, err
	}

	return publications, nil
}

func LoadPublicationData(filepath string) ([]publication.Publication, error) {
	data, err := ioutil.ReadFile(filepath)
	if err != nil {
		return nil, err
	}

	var publications []publication.Publication
	err = json.Unmarshal(data, &publications)
	if err != nil {
		return nil, err
	}

	return publications, nil
}
