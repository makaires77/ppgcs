package dgraph

import (
	"context"
	"encoding/json"
	"log"
)

type DgraphReader struct {
	client *DgraphClient
}

func NewDgraphReader(client *DgraphClient) *DgraphReader {
	return &DgraphReader{
		client: client,
	}
}

func (r *DgraphReader) FindByID(ctx context.Context, id string, result interface{}) error {
	txn := r.client.NewTransaction()
	defer txn.Discard(ctx)

	query := `query {
		publication(func: uid(` + id + `)) {
			uid
			natureza
			titulo
			idioma
			periodico
			ano
			volume
			issn
			estrato_qualis
			pais_de_publicacao
			paginas
			doi
			autores
			autores_endogeno
			autores_endogeno_nome
			tags
			hash
		}
	}`

	resp, err := txn.QueryWithVars(ctx, query, nil)
	if err != nil {
		log.Printf("Failed to execute Dgraph query: %v", err)
		return err
	}

	type Response struct {
		Publication []*Publication `json:"publication"`
	}

	var data Response
	if err := json.Unmarshal(resp.Json, &data); err != nil {
		log.Printf("Failed to unmarshal Dgraph response: %v", err)
		return err
	}

	if len(data.Publication) > 0 {
		*result.(*Publication) = *data.Publication[0]
	}

	return nil
}
