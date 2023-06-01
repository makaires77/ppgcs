package neo4j

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"

	publication "github.com/makaires77/ppgcs/pkg/domain/publication"
)

type Client struct {
	driver neo4j.Driver
}

func NewClient(uri, username, password string) (*Client, error) {
	driver, err := neo4j.NewDriver("neo4j://localhost:7687", neo4j.BasicAuth("username", "password", ""))
	if err != nil {
		return nil, fmt.Errorf("failed to create driver: %w", err)
	}

	return &Client{driver: driver}, nil
}

func (c *Client) SavePublication(ctx context.Context, data *publication.Publication) error {
	session := c.driver.NewSession(neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close()

	result, err := session.WriteTransaction(func(tx neo4j.Transaction) (interface{}, error) {
		result, err := tx.Run(
			`MERGE (p:Publication {hash: $hash}) SET p = $props`,
			map[string]interface{}{
				"hash":  data.Hash,
				"props": toProps(data),
			},
		)
		if err != nil {
			return nil, fmt.Errorf("failed to execute query: %w", err)
		}

		if result.Err() != nil {
			return nil, fmt.Errorf("result error: %w", result.Err())
		}

		return result, nil
	})

	if err != nil {
		return fmt.Errorf("failed to execute write transaction: %w", err)
	}

	if result.Err != nil {
		return fmt.Errorf("result error: %w", result.Err())
	}

	return nil
}

func (c *Client) SavePesquisador(ctx context.Context, data *publication.Pesquisador) error {
	session := c.driver.NewSession(neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close()

	result, err := session.WriteTransaction(func(tx neo4j.Transaction) (interface{}, error) {
		result, err := tx.Run(
			`MERGE (p:Pesquisador {idlattes: $idlattes}) SET p = $props`,
			map[string]interface{}{
				"idlattes": data.IDLattes,
				"props":    toProps(data),
			},
		)
		if err != nil {
			return nil, fmt.Errorf("failed to execute query: %w", err)
		}

		if result.Err() != nil {
			return nil, fmt.Errorf("result error: %w", result.Err())
		}

		return result, nil
	})

	if err != nil {
		return fmt.Errorf("failed to execute write transaction: %w", err)
	}

	if result.Err() != nil {
		return fmt.Errorf("result error: %w", result.Err())
	}

	return nil
}

func toProps(v interface{}) map[string]interface{} {
	props := make(map[string]interface{})
	value := reflect.ValueOf(v).Elem()

	for i := 0; i < value.NumField(); i++ {
		field := value.Type().Field(i)
		fieldValue := value.Field(i)

		if !fieldValue.IsValid() || (fieldValue.Kind() == reflect.Ptr && fieldValue.IsNil()) {
			continue
		}

		jsonTag := field.Tag.Get("json")
		if jsonTag == "" {
			continue
		}

		jsonTagParts := strings.Split(jsonTag, ",")
		props[jsonTagParts[0]] = fieldValue.Interface()
	}

	return props
}
