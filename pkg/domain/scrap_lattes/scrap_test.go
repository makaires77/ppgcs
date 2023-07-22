package scrap_lattes

import (
	"testing"

	"github.com/makaires77/ppgcs/pkg/infrastructure/scrap_lattes/scraper"
)

func TestScrap(t *testing.T) {
	scraper, err := scraper.NewScrapLattes("http://buscatextual.cnpq.br/buscatextual/busca.do")
	if err != nil {
		t.Fatalf("Failed to create scraper: %v", err)
	}

	err = scraper.Scrap()
	if err != nil {
		t.Fatalf("Failed to scrap: %v", err)
	}
}
