package scrap_lattes

import (
	"testing"

	"github.com/makaires77/ppgcs/pkg/infrastructure/scrap_lattes"
)

func TestProcessarRegistro(t *testing.T) {
	scraper := scrap_lattes.NewScrapLattes()

	// Substitua esses valores de acordo com o que você espera do seu CSV.
	testRecord := []string{"Nome Teste", "Lattes Teste"}

	err := scraper.ProcessarRegistro(testRecord)
	if err != nil {
		t.Errorf("ProcessarRegistro failed with error: %v", err)
	}
}
