// pkg\infrastructure\neo4jclient\client.go
package neo4jclient

import (
	"context"
	"fmt"

	publication "github.com/makaires77/ppgcs/pkg/domain/publication"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

type Neo4jClient struct {
	driver neo4j.DriverWithContext
}

func NewNeo4jClient(ctx context.Context, uri, username, password string) (*Neo4jClient, error) {
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(username, password, ""))
	if err != nil {
		return nil, fmt.Errorf("neo4j: could not create driver: %w", err)
	}

	return &Neo4jClient{driver: driver}, nil
}

func (c *Neo4jClient) Close(ctx context.Context) error {
	if c.driver != nil {
		return c.driver.Close(ctx)
	}
	return nil
}

func (c *Neo4jClient) SavePublication(ctx context.Context, p publication.Publication) error {
	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		result, err := tx.Run(ctx,
			`CREATE (p:Publication {Natureza: $natureza, Titulo: $titulo, Idioma: $idioma, Periodico: $periodico, Ano: $ano, Volume: $volume, ISSN: $issn, EstratoQualis: $estrato_qualis, PaisDePublicacao: $pais_de_publicacao, Paginas: $paginas, DOI: $doi, Autores: $autores, AutoresEndogeno: $autores_endogeno, AutoresEndogenoNome: $autores_endogeno_nome, Tags: $tags, Hash: $hash})`,
			map[string]interface{}{
				"natureza":              p.Natureza,
				"titulo":                p.Titulo,
				"idioma":                p.Idioma,
				"periodico":             p.Periodico,
				"ano":                   p.Ano,
				"volume":                p.Volume,
				"issn":                  p.ISSN,
				"estrato_qualis":        p.EstratoQualis,
				"pais_de_publicacao":    p.PaisDePublicacao,
				"paginas":               p.Paginas,
				"doi":                   p.DOI,
				"autores":               p.Autores,
				"autores_endogeno":      p.AutoresEndogeno,
				"autores_endogeno_nome": p.AutoresEndogenoNome,
				"tags":                  p.Tags,
				"hash":                  p.Hash,
			},
		)
		if err != nil {
			return nil, fmt.Errorf("neo4j: could not execute query: %w", err)
		}

		if result.Err() != nil {
			return nil, fmt.Errorf("neo4j: query result error: %w", result.Err())
		}

		return nil, nil
	})

	if err != nil {
		return fmt.Errorf("neo4j: could not execute transaction: %w", err)
	}

	return nil
}
