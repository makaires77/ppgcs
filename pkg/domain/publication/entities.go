package publication

import (
	"encoding/json"
	"io/ioutil"
	"os"
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
	AutoresEndogeno     []string          `json:"autores-endogeno"`
	AutoresEndogenoNome map[string]string `json:"autores-endogeno-nome"`
	Tags                []string          `json:"tags"`
	Hash                string            `json:"Hash"`
}

func LoadEntitiesFromJSON(jsonFilePath string) ([]Publication, error) {
	// Ler o arquivo JSON
	jsonFile, err := os.Open(jsonFilePath)
	if err != nil {
		return nil, err
	}
	defer jsonFile.Close()

	// Decodificar o JSON
	byteValue, err := ioutil.ReadAll(jsonFile)
	if err != nil {
		return nil, err
	}

	var entities []Publication
	if err := json.Unmarshal(byteValue, &entities); err != nil {
		return nil, err
	}

	return entities, nil
}
