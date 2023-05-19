package load_publication

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
)

type Publication struct {
	Natureza            string            `json:"natureza"`
	Titulo              string            `json:"titulo"`
	Idioma              string            `json:"idioma"`
	Periodico           string            `json:"periodico"`
	Ano                 string            `json:"ano"`
	Volume              string            `json:"volume"`
	ISSN                string            `json:"issn"`
	EstratoQualis       string            `json:"estrato_qualis"`
	PaisDePublicacao    string            `json:"pais_de_publicacao"`
	Paginas             string            `json:"paginas"`
	DOI                 string            `json:"doi"`
	Autores             []string          `json:"autores"`
	AutoresEndogeno     []string          `json:"autores_endogeno"`
	AutoresEndogenoNome map[string]string `json:"autores_endogeno_nome"`
	Tags                []string          `json:"tags"`
	Hash                string            `json:"hash"`
}

type PublicationInteractor struct {
	PublicationDataPath string
}

func NewPublicationInteractor(publicationDataPath string) *PublicationInteractor {
	return &PublicationInteractor{
		PublicationDataPath: publicationDataPath,
	}
}

func (i *PublicationInteractor) LoadPublications() ([]Publication, error) {
	data, err := ioutil.ReadFile(i.PublicationDataPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read publication data file: %w", err)
	}

	var publications []Publication
	err = json.Unmarshal(data, &publications)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal publication data: %w", err)
	}

	return publications, nil
}
