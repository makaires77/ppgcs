package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/dgraph-io/dgo/v200"
	"github.com/dgraph-io/dgo/v200/protos/api"
	"google.golang.org/grpc"
)

type PublicationLoader struct {
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

func main() {
	// Defina os caminhos para os arquivos JSON de publicações
	jsonFilePath1 := "_data/in_json/642.files/642.publication.json"
	jsonFilePath2 := "_data/in_json/642.files/644.publication.json"

	// Carregue as entidades dos arquivos JSON
	publications1, err := LoadEntitiesFromJSON(jsonFilePath1)
	if err != nil {
		log.Fatalf("Erro ao carregar as entidades do arquivo %s: %v\n", jsonFilePath1, err)
	}

	publications2, err := LoadEntitiesFromJSON(jsonFilePath2)
	if err != nil {
		log.Fatalf("Erro ao carregar as entidades do arquivo %s: %v\n", jsonFilePath2, err)
	}

	// Criar uma conexão gRPC com o servidor Dgraph
	dgraphConn, err := grpc.Dial("localhost:9080", grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Falha ao conectar ao servidor Dgraph: %v", err)
	}
	defer dgraphConn.Close()

	// Criar um cliente Dgraph
	dgraphClient := dgo.NewDgraphClient(api.NewDgraphClient(dgraphConn))

	// Criar um contexto de execução
	ctx := context.Background()

	// Salvar as publicações do primeiro arquivo JSON
	err = savePublications(ctx, dgraphClient, publications1)
	if err != nil {
		log.Fatalf("Falha ao salvar as publicações do arquivo %s: %v\n", jsonFilePath1, err)
	}

	// Salvar as publicações do segundo arquivo JSON, evitando duplicações
	err = savePublicationsIfNotExists(ctx, dgraphClient, publications2)
	if err != nil {
		log.Fatalf("Falha ao salvar as publicações do arquivo %s: %v\n", jsonFilePath2, err)
	}

	// Resto do código...
}

func LoadEntitiesFromJSON(filePath string) ([]PublicationLoader, error) {
	// Abrir o arquivo JSON
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("falha ao abrir o arquivo JSON: %v", err)
	}
	defer file.Close()

	// Ler o conteúdo do arquivo
	content, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("falha ao ler o conteúdo do arquivo JSON: %v", err)
	}

	// Decodificar o JSON para a estrutura de entidades
	var entities []PublicationLoader
	err = json.Unmarshal(content, &entities)
	if err != nil {
		return nil, fmt.Errorf("falha ao decodificar o JSON: %v", err)
	}

	return entities, nil
}

func savePublications(ctx context.Context, dgraphClient *dgo.Dgraph, publications []PublicationLoader) error {
	for _, pub := range publications {
		pb, err := json.Marshal(pub)
		if err != nil {
			return fmt.Errorf("Falha ao serializar a publicação: %v", err)
		}

		mutation := &api.Mutation{
			SetJson:   pb,
			CommitNow: true,
		}

		_, err = dgraphClient.NewTxn().Mutate(ctx, mutation)
		if err != nil {
			return fmt.Errorf("Falha ao executar a mutação: %v", err)
		}
	}

	return nil
}

func savePublicationsIfNotExists(ctx context.Context, dgraphClient *dgo.Dgraph, publications []PublicationLoader) error {
	existingPublications, err := getExistingPublications(ctx, dgraphClient)
	if err != nil {
		log.Fatalf("Falha ao obter as publicações existentes: %v\n", err)
	}

	// Criar um mapa para rastrear as publicações existentes pelo campo Hash
	publicationsExist := make(map[string]bool)

	// Percorrer as publicações existentes e adicionar ao mapa de publicações
	for _, pub := range existingPublications {
		publicationsExist[pub.Hash] = true
	}

	// Salvar apenas as publicações que não existem no banco de dados
	var newPublications []PublicationLoader
	for _, pub := range publications {
		if !publicationsExist[pub.Hash] {
			newPublications = append(newPublications, pub)
		}
	}

	// Salvar as novas publicações
	err = savePublications(ctx, dgraphClient, newPublications)
	if err != nil {
		return fmt.Errorf("Falha ao salvar as novas publicações: %v", err)
	}

	return nil
}

func getExistingPublications(ctx context.Context, dgraphClient *dgo.Dgraph) ([]PublicationLoader, error) {
	query := `
{
	publications(func: has(publicationID)) {
		title
		authors
		# Outros campos da publicação...
	}
}`

	resp, err := dgraphClient.NewTxn().Query(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("Falha ao executar a consulta: %v", err)
	}

	var result struct {
		Publications []PublicationLoader `json:"publications"`
	}

	err = json.Unmarshal(resp.GetJson(), &result)
	if err != nil {
		return nil, fmt.Errorf("Falha ao converter a resposta: %v", err)
	}

	return result.Publications, nil
}
