package dgraph

import (
	"context"
	"fmt"

	"github.com/dgraph-io/dgo/v200"
	"github.com/dgraph-io/dgo/v200/protos/api"
	"google.golang.org/grpc"
)

type DGraphClient struct {
	client *dgo.Dgraph
}

func NewDGraphClient(address string) (*DGraphClient, error) {
	conn, err := grpc.Dial(address, grpc.WithInsecure())
	if err != nil {
		return nil, fmt.Errorf("failed to connect to DGraph: %v", err)
	}

	client := dgo.NewDgraphClient(api.NewDgraphClient(conn))
	dgraphClient := &DGraphClient{client: client}

	return dgraphClient, nil
}

func (c *DGraphClient) NewTransaction() *dgo.Txn {
	return c.client.NewTxn()
}

func (c *DGraphClient) ExecuteQuery(ctx context.Context, query string) (*api.Response, error) {
	txn := c.NewTransaction()
	defer txn.Discard(ctx)

	resp, err := txn.Query(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %v", err)
	}

	return resp, nil
}

func (c *DGraphClient) ExecuteMutation(ctx context.Context, mu *api.Mutation) (*api.Response, error) {
	txn := c.NewTransaction()
	defer txn.Discard(ctx)

	resp, err := txn.Mutate(ctx, mu)
	if err != nil {
		return nil, fmt.Errorf("failed to execute mutation: %v", err)
	}

	return resp, nil
}

func (c *DGraphClient) AlterSchema(ctx context.Context, schema string) error {
	operation := &api.Operation{Schema: schema}

	err := c.client.Alter(ctx, operation)
	if err != nil {
		return fmt.Errorf("failed to alter schema: %v", err)
	}

	return nil
}
