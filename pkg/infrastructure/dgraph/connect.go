package dgraph

import (
	"context"
	"log"

	"github.com/dgraph-io/dgo/v200"
	"github.com/dgraph-io/dgo/v200/protos/api"
	"google.golang.org/grpc"
)

type DgraphClient struct {
	dg *dgo.Dgraph
}

func NewDgraphClient() (*DgraphClient, error) {
	dialOpts := []grpc.DialOption{
		grpc.WithInsecure(),
	}

	conn, err := grpc.Dial("localhost:9080", dialOpts...)
	if err != nil {
		log.Fatalf("Failed to connect to Dgraph: %v", err)
	}

	dg := dgo.NewDgraphClient(api.NewDgraphClient(conn))

	return &DgraphClient{
		dg: dg,
	}, nil
}

func (c *DgraphClient) Close() {
	// Close the connection to Dgraph
	c.dg.Close()
}

func (c *DgraphClient) NewTransaction() *dgo.Txn {
	return c.dg.NewTxn()
}

func (c *DgraphClient) Commit(ctx context.Context, txn *dgo.Txn) error {
	return txn.Commit(ctx)
}