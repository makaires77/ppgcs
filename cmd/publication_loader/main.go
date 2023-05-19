package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

type Publication struct {
	Natureza            string
	Titulo              string
	Idioma              string
	Periodico           string
	Ano                 string
	Volume              string
	ISSN                string
	EstratoQualis       string
	PaisDePublicacao    string
	Paginas             string
	DOI                 string
	Autores             []string
	AutoresEndogeno     []string
	AutoresEndogenoNome []map[string]string
	Tags                []interface{}
	Hash                string
}

type Periodico map[string]map[string][]Publication

func LoadEntitiesFromJSON(path string) (Periodico, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var periodicos Periodico
	err = json.Unmarshal(data, &periodicos)
	if err != nil {
		return nil, err
	}

	return periodicos, nil
}

func main() {
	periodicos, err := LoadEntitiesFromJSON("_data/in_json/642.files/642.publication.json")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	// Print loaded entities to check if everything is fine
	for periodico, anos := range periodicos {
		fmt.Printf("Periodico: %s\n", periodico)
		for ano, publicacoes := range anos {
			fmt.Printf("  Ano: %s\n", ano)
			for _, publicacao := range publicacoes {
				fmt.Printf("    Titulo: %s\n", publicacao.Titulo)
				fmt.Printf("    Autores: %v\n", publicacao.Autores)
				// Print other properties as needed...
			}
		}
	}
}
