<<<<<<< HEAD
=======
// pkg\infrastructure\neo4jclient\client.go
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
package neo4jclient

import (
	"context"
	"fmt"

<<<<<<< HEAD
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"

	publication "github.com/makaires77/ppgcs/pkg/domain/publication"
=======
	publication "github.com/makaires77/ppgcs/pkg/domain/publication"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
)

type Neo4jClient struct {
	driver neo4j.DriverWithContext
}

<<<<<<< HEAD
func NewNeo4jClient(uri, username, password string) (*Neo4jClient, error) {
=======
func NewNeo4jClient(ctx context.Context, uri, username, password string) (*Neo4jClient, error) {
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(username, password, ""))
	if err != nil {
		return nil, fmt.Errorf("neo4j: could not create driver: %w", err)
	}

	return &Neo4jClient{driver: driver}, nil
}

<<<<<<< HEAD
func (c *Neo4jClient) Close() error {
	if c.driver != nil {
		return c.driver.Close(context.Background())
=======
func (c *Neo4jClient) Close(ctx context.Context) error {
	if c.driver != nil {
		return c.driver.Close(ctx)
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
	}
	return nil
}

func (c *Neo4jClient) SavePublication(ctx context.Context, p publication.Publication) error {
	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
<<<<<<< HEAD
	if session == nil {
		return fmt.Errorf("neo4j: unable to establish a new session")
	}
	defer session.Close(context.Background())

	result, err := session.Run(
		ctx,
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
		return fmt.Errorf("neo4j: could not execute query: %w", err)
	}

	if result.Err() != nil {
		return fmt.Errorf("neo4j: query result error: %w", result.Err())
=======
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
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
	}

	return nil
}
