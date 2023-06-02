package neo4j

import (
	"context"
	"errors"
	"fmt"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"

	publication "github.com/makaires77/ppgcs/pkg/domain/publication"
)

type Neo4jClient struct {
	driver neo4j.DriverWithContext
}

func NewNeo4jClient(uri, username, password string) (*Neo4jClient, error) {
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(username, password, ""))
	if err != nil {
		return nil, fmt.Errorf("neo4j: could not create driver: %w", err)
	}

	return &Neo4jClient{driver: driver}, nil
}

func (c *Neo4jClient) Close() error {
	return c.driver.Close(context.Background())
}

func (c *Neo4jClient) SavePublication(ctx context.Context, p publication.Publication) error {
	sessionConfig := neo4j.SessionConfig{
		AccessMode: neo4j.AccessModeWrite,
		Bookmarks:  []string{},
	}

	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
	if session != nil {
		return errors.New("unable to establish a new session")
	}
	defer session.Close(context.Background())

	err := session.WriteTransaction(func(tx neo4j.Transaction) (interface{}, error) {
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
			return nil, fmt.Errorf("neo4j: could not run transaction: %w", err)
		}

		if result.Err() != nil {
			return nil, fmt.Errorf("neo4j: result error: %w", result.Err())
		}

		return nil, nil
	})

	if err != nil {
		return fmt.Errorf("neo4j: could not execute write transaction: %w", err)
	}

	return nil
}
