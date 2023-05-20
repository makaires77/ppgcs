package main

import (
	"fmt"
	"os"

	json "github.com/makaires77/ppgcs/pkg/infrastructure/json_publication"
)

func main() {
	// Defina os caminhos para os arquivos JSON de publicações
	jsonFilePath1 := "_data/in_json/642.files/642.publication.json"
	jsonFilePath2 := "_data/in_json/642.files/644.publication.json"

	// Carregue as entidades dos arquivos JSON
	periodicos1, err := json.LoadEntitiesFromJSON(jsonFilePath1)
	if err != nil {
		fmt.Printf("Erro ao carregar as entidades do arquivo %s: %v\n", jsonFilePath1, err)
		os.Exit(1)
	}

	periodicos2, err := json.LoadEntitiesFromJSON(jsonFilePath2)
	if err != nil {
		fmt.Printf("Erro ao carregar as entidades do arquivo %s: %v\n", jsonFilePath2, err)
		os.Exit(1)
	}

	// Criar um mapa para rastrear as publicações existentes pelo campo Hash
	publicacoesExist := make(map[string]bool)

	// Percorra as publicações do primeiro arquivo JSON e adicione ao mapa de publicações existentes
	for _, anos := range periodicos1 {
		for _, publicacoes := range anos {
			for _, publicacao := range publicacoes {
				publicacoesExist[publicacao.Hash] = true
			}
		}
	}

	// Percorra as publicações do segundo arquivo JSON e adicione ao mapa de publicações existentes, evitando duplicações
	for _, anos := range periodicos2 {
		for _, publicacoes := range anos {
			for _, publicacao := range publicacoes {
				if !publicacoesExist[publicacao.Hash] {
					publicacoesExist[publicacao.Hash] = true
					// Adicione a publicação aos períodicos existentes
					periodicos1[publicacao.Periodico][publicacao.Ano] = append(periodicos1[publicacao.Periodico][publicacao.Ano], publicacao)
				}
			}
		}
	}

	// Imprima as entidades carregadas, incluindo as novas do segundo arquivo, para verificar se está tudo correto
	for periodico, anos := range periodicos1 {
		fmt.Printf("Periodico: %s\n", periodico)
		for ano, publicacoes := range anos {
			fmt.Printf("  Ano: %s\n", ano)
			for _, publicacao := range publicacoes {
				fmt.Printf("    Titulo: %s\n", publicacao.Titulo)
				fmt.Printf("    Autores: %v\n", publicacao.Autores)
				// Imprima outras propriedades conforme necessário...
			}
		}
	}
}
