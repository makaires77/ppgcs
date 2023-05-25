package neo4j

import (
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

type Client struct {
	driver neo4j.Driver
}

func NewClient() (*Client, error) {
	driver, err := neo4j.NewDriverWithContext("neo4j://localhost:7687", neo4j.BasicAuth("username", "password", ""))
	if err != nil {
		return nil, err
	}

	return &Client{driver: driver}, nil
}

// Insira aqui as funções para interagir com o banco de dados Neo4j, por exemplo:
// func (c *Client) SaveData(data *YourDataType) error { ... }
