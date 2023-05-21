package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"

	"github.com/dgraph-io/dgo/v210"
	"github.com/dgraph-io/dgo/v210/protos/api"
	"google.golang.org/grpc"
)

func main() {
	// Conexão com o Dgraph
	conn, err := grpc.Dial("localhost:9080", grpc.WithInsecure())
	if err != nil {
		fmt.Printf("Erro ao conectar com o Dgraph: %v\n", err)
		return
	}
	defer conn.Close()

	dc := api.NewDgraphClient(conn)
	dg := dgo.NewDgraphClient(dc)

	// Diretório de entrada dos arquivos JSON
	jsonDir := "_data/in_json"

	// Ler e persistir entidades de todos os arquivos JSON no diretório
	err = readAndPersistEntitiesFromJSONDir(dg, jsonDir)
	if err != nil {
		fmt.Printf("Erro ao ler e persistir entidades: %v\n", err)
		return
	}

	fmt.Println("Entidades carregadas e salvas com sucesso!")
}

func readAndPersistEntitiesFromJSONDir(dg *dgo.Dgraph, jsonDir string) error {
	files, err := ioutil.ReadDir(jsonDir)
	if err != nil {
		return fmt.Errorf("Erro ao ler o diretório JSON: %v", err)
	}

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		filePath := filepath.Join(jsonDir, file.Name())
		err := readAndPersistEntitiesFromJSONFile(dg, filePath)
		if err != nil {
			fmt.Printf("Erro ao ler e persistir entidades do arquivo %s: %v\n", file.Name(), err)
		}
	}

	return nil
}

func readAndPersistEntitiesFromJSONFile(dg *dgo.Dgraph, filePath string) error {
	file, err := ioutil.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("Erro ao ler o arquivo JSON: %v", err)
	}

	// Decodificar o JSON
	var entities []map[string]interface{}
	err = json.Unmarshal(file, &entities)
	if err != nil {
		return fmt.Errorf("Erro ao decodificar o JSON: %v", err)
	}

	// Carregar e persistir as entidades
	if err := loadAndSaveEntities(dg, entities); err != nil {
		return fmt.Errorf("Falha ao carregar e salvar as entidades: %v", err)
	}

	return nil
}

func loadAndSaveEntities(dg *dgo.Dgraph, entities []map[string]interface{}) error {
	txn := dg.NewTxn()
	defer txn.Discard(context.Background())

	for _, entity := range entities {
		// Salvar a entidade no Dgraph
		jsonBytes, err := json.Marshal(entity)
		if err != nil {
			return fmt.Errorf("Erro ao converter a entidade para JSON: %v", err)
		}

		mutation := &api.Mutation{
			CommitNow: true,
			SetJson:   jsonBytes,
		}

		_, err = txn.Mutate(context.Background(), mutation)
		if err != nil {
			return fmt.Errorf("Erro ao salvar a entidade: %v", err)
		}
	}

	return nil
}
