package neo4jclient

import (
	"context"
	"fmt"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"

	publication "github.com/makaires77/ppgcs/pkg/domain/publication"
)

type Neo4jClient struct {
	driver neo4j.Driver
}

func NewNeo4jClient(ctx context.Context, uri, username, password string) (*Neo4jClient, error) {
	driver, err := neo4j.NewDriver(uri, neo4j.BasicAuth(username, password, ""), func(config *neo4j.Config) {
		config.Encrypted = false
	})
	if err != nil {
		return nil, fmt.Errorf("neo4j: could not create driver: %w", err)
	}

	return &Neo4jClient{driver: driver}, nil
}

func (c *Neo4jClient) Close() error {
	if c.driver != nil {
		return c.driver.Close()
	}
	return nil
}

func (c *Neo4jClient) SavePublication(ctx context.Context, p publication.Publication) error {
	session := neo4j.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(context.Background())

	result, err := session.WriteTransaction(
		func(tx neo4j.Transaction) (interface{}, error) {
			result, err := tx.Run(
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
		},
	)

	if err != nil {
		return fmt.Errorf("neo4j: could not execute transaction: %w", err)
	}

	return nil
}
