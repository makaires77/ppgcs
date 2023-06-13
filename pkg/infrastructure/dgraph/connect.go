package dgraph

import (
	"github.com/dgraph-io/dgo/v200"
	"github.com/dgraph-io/dgo/v200/protos/api"
	"google.golang.org/grpc"
)

type DgraphClient struct {
	client *dgo.Dgraph
	conn   *grpc.ClientConn
}

func NewDgraphClient() (*DgraphClient, error) {
	conn, err := grpc.Dial("localhost:9080", grpc.WithInsecure())
	if err != nil {
		return nil, err
	}

	client := dgo.NewDgraphClient(api.NewDgraphClient(conn))

	return &DgraphClient{
		client: client,
		conn:   conn,
	}, nil
}

func (c *DgraphClient) Close() {
	c.conn.Close()
}
