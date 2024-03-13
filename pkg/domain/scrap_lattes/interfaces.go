package scrap_lattes

import "context"

// Scraper define a interface para o objeto de scraping.
type Scraper interface {
	ScrapPesquisador(ctx context.Context, url string) (*Pesquisador, error)
}

// Repository define a interface do repositório para salvar os dados do pesquisador.
type Repository interface {
	SavePesquisador(ctx context.Context, pesquisador *Pesquisador) error
}
