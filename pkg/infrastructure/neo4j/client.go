package neo4j

import (
	"context"
	"errors"

	"github.com/makaires77/ppgcs/pkg/domain/publication"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

type Client struct {
	driver neo4j.DriverWithContext
}

func NewClient(uri, username, password string) (*Client, error) {
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(username, password, ""), func(config *neo4j.Config) {
		config.MaxConnectionPoolSize = 10
	})
	if err != nil {
		return nil, err
	}

	return &Client{driver: driver}, nil
}

func (c *Client) SavePublication(ctx context.Context, data *publication.Publication) error {
	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
	if session == nil {
		return errors.New("unable to establish a new session")
	}
	defer session.Close(ctx)

	if data == nil {
		return errors.New("data is null")
	}
	if data.Titulo == "" {
		return errors.New("titulo is required")
	}
	if data.Hash == "" {
		return errors.New("hash is required")
	}

	cypher := `
		MERGE (p:Publication {hash: $hash})
		ON CREATE SET
			p.natureza = $natureza, 
			p.titulo = $titulo, 
			p.idioma = $idioma, 
			p.periodico = $periodico,
			p.ano = $ano, 
			p.volume = $volume,
			p.issn = $issn,
			p.estratoQualis = $estratoQualis,
			p.paisDePublicacao = $paisDePublicacao,
			p.paginas = $paginas,
			p.doi = $doi,
			p.autores = $autores,
			p.autoresEndogeno = $autoresEndogeno,
			p.autoresEndogenoNome = $autoresEndogenoNome,
			p.tags = $tags
		RETURN p
	`
	tx, err := session.BeginTransaction(ctx)
	if err != nil {
		return err
	}

	_, err = tx.Run(cypher, map[string]interface{}{
		"natureza":            data.Natureza,
		"titulo":              data.Titulo,
		"idioma":              data.Idioma,
		"periodico":           data.Periodico,
		"ano":                 data.Ano,
		"volume":              data.Volume,
		"issn":                data.ISSN,
		"estratoQualis":       data.EstratoQualis,
		"paisDePublicacao":    data.PaisDePublicacao,
		"paginas":             data.Paginas,
		"doi":                 data.DOI,
		"autores":             data.Autores,
		"autoresEndogeno":     data.AutoresEndogeno,
		"autoresEndogenoNome": data.AutoresEndogenoNome,
		"tags":                data.Tags,
		"hash":                data.Hash,
	})

	if err != nil {
		_ = tx.Rollback(ctx)
		return err
	} else {
		err = tx.Commit(ctx)
		if err != nil {
			return err
		}
	}

	return nil
}

func (c *Client) Close() error {
	return c.driver.Close(context.Background())
}
