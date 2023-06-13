package dgraph

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/dgraph-io/dgo/v200"

	"github.com/makaires77/ppgcs/pkg/domain/publication"
)

type DgraphReader struct {
	client *dgo.Dgraph
}

func NewDgraphReader(client *dgo.Dgraph) *DgraphReader {
	return &DgraphReader{
		client: client,
	}
}

func (r *DgraphReader) GetPublicationsByTitle(ctx context.Context, title string) ([]publication.Publication, error) {
	query := `
	{
		publications(func: eq(title, "%s")) {
			natureza
			titulo
			idioma
			periodico
			ano
			volume
			issn
			estratoQualis
			paisDePublicacao
			paginas
			doi
			autores
			autoresEndogeno
			autoresEndogenoNome
			tags
			hash
		}
	}`

	resp, err := r.client.NewTxn().Query(ctx, fmt.Sprintf(query, title))
	if err != nil {
		return nil, fmt.Errorf("falha ao executar a consulta: %v", err)
	}

	var result struct {
		Publications []publication.Publication `json:"publications"`
	}

	err = json.Unmarshal(resp.GetJson(), &result)
	if err != nil {
		return nil, fmt.Errorf("falha ao converter a resposta: %v", err)
	}

	return result.Publications, nil
}
