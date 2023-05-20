package dgraph

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/dgraph-io/dgo/v200"
	"github.com/dgraph-io/dgo/v200/protos/api"

	"github.com/makaires77/ppgcs/pkg/domain/publication"
)

type DgraphWriter struct {
	client *dgo.Dgraph
}

func NewDgraphWriter(client *dgo.Dgraph) *DgraphWriter {
	return &DgraphWriter{
		client: client,
	}
}

func (w *DgraphWriter) SavePublications(ctx context.Context, publications []publication.Publication) error {
	mutations := make([]*api.Mutation, len(publications))

	for i, pub := range publications {
		pb, err := json.Marshal(pub)
		if err != nil {
			return fmt.Errorf("falha ao serializar a publicação: %v", err)
		}

		mutations[i] = &api.Mutation{
			SetJson:   pb,
			CommitNow: true,
		}
	}

	req := &api.Request{
		Mutations: mutations,
	}

	_, err := w.client.NewTxn().Do(ctx, req)
	if err != nil {
		return fmt.Errorf("falha ao executar as mutações: %v", err)
	}

	return nil
}
