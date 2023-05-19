package dgraph

import (
	"context"
	"log"

	"github.com/dgraph-io/dgo/v200/protos/api"
)

type DgraphWriter struct {
	client *DgraphClient
}

func NewDgraphWriter(client *DgraphClient) *DgraphWriter {
	return &DgraphWriter{
		client: client,
	}
}

func (w *DgraphWriter) CreatePublication(ctx context.Context, publication *Publication) error {
	txn := w.client.NewTransaction()
	defer txn.Discard(ctx)

	mutation := &api.Mutation{
		SetJson: publication,
	}

	assigned, err := txn.Mutate(ctx, mutation)
	if err != nil {
		log.Printf("Failed to execute Dgraph mutation: %v", err)
		return err
	}

	if len(assigned.Uids) == 0 {
		return ErrPublicationNotCreated
	}

	publication.ID = assigned.Uids["blank-0"]

	return nil
}
