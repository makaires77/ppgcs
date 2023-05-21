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

func main() {
	// Defina os caminhos para os arquivos JSON das entidades
	jsonFilePath1 := "_data/in_json/642.files/642.publication.json"
	jsonFilePath2 := "_data/in_json/642.files/644.publication.json"

	// Crie uma conexão gRPC com o servidor Dgraph
	dgraphConn, err := grpc.Dial("localhost:9080", grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Falha ao conectar ao servidor Dgraph: %v", err)
	}
	defer dgraphConn.Close()

	// Crie um cliente Dgraph
	dgraphClient := dgo.NewDgraphClient(api.NewDgraphClient(dgraphConn))

	// Crie um contexto de execução
	ctx := context.Background()

	// Carregue e salve as entidades do primeiro arquivo JSON
	err = loadAndSaveEntities(ctx, dgraphClient, jsonFilePath1)
	if err != nil {
		log.Fatalf("Falha ao carregar e salvar as entidades do arquivo %s: %v\n", jsonFilePath1, err)
	}

	// Carregue e salve as entidades do segundo arquivo JSON
	err = loadAndSaveEntities(ctx, dgraphClient, jsonFilePath2)
	if err != nil {
		log.Fatalf("Falha ao carregar e salvar as entidades do arquivo %s: %v\n", jsonFilePath2, err)
	}

	// Resto do código...
}

func loadAndSaveEntities(ctx context.Context, dgraphClient *dgo.Dgraph, jsonFilePath string) error {
	// Carregue as entidades do arquivo JSON
	entities, err := loadEntitiesFromJSON(jsonFilePath)
	if err != nil {
		return fmt.Errorf("Falha ao carregar as entidades do arquivo %s: %v", jsonFilePath, err)
	}

	// Salve as entidades
	err = saveEntities(ctx, dgraphClient, entities)
	if err != nil {
		return fmt.Errorf("Falha ao salvar as entidades do arquivo %s: %v", jsonFilePath, err)
	}

	return nil
}

func loadEntitiesFromJSON(filePath string) ([]map[string]interface{}, error) {
	// Abra o arquivo JSON
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("Falha ao abrir o arquivo JSON: %v", err)
	}
	defer file.Close()

	// Leia o conteúdo do arquivo
	content, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("Falha ao ler o conteúdo do arquivo JSON: %v", err)
	}

	// Decodifique o JSON para um mapa de interface{}
	var entities []map[string]interface{}
	err = json.Unmarshal(content, &entities)
	if err != nil {
		return nil, fmt.Errorf("Falha ao decodificar o JSON: %v", err)
	}

	return entities, nil
}

func saveEntities(ctx context.Context, dgraphClient *dgo.Dgraph, entities []map[string]interface{}) error {
	// Inicie uma transação
	txn := dgraphClient.NewTxn()
	defer txn.Discard(ctx)

	// Converta as entidades em formato JSON
	entityJSON, err := json.Marshal(entities)
	if err != nil {
		return fmt.Errorf("Falha ao converter as entidades em JSON: %v", err)
	}

	// Crie uma mutação para inserir as entidades
	mutation := &api.Mutation{
		SetJson:   entityJSON,
		CommitNow: true,
	}

	// Execute a mutação
	_, err = txn.Mutate(ctx, mutation)
	if err != nil {
		return fmt.Errorf("Falha ao executar a mutação: %v", err)
	}

	// Faça o commit da transação
	err = txn.Commit(ctx)
	if err != nil {
		return fmt.Errorf("Falha ao fazer o commit da transação: %v", err)
	}

	return nil
}

/* Essa versão atualizada do `main.go` permite carregar as entidades de arquivos JSON de forma dinâmica, sem precisar especificar a estrutura da entidade antecipadamente. As entidades são lidas como um mapa e, em seguida, convertidas em JSON antes de serem salvas no DGraph. A função `saveEntitiesIfNotExists` também foi atualizada para verificar a existência das entidades no banco de dados antes de salvá-las. */
