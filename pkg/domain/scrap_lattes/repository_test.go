// pkg\domain\scrap_lattes\repository_test.go
package scrap_lattes

import (
	"testing"
)

func TestScrapPesquisador(t *testing.T) {
	// Esta é uma URL inválida, então esperamos que ScrapPesquisador retorne um erro
	url := "http://invalid.url"

	_, err := ScrapPesquisador(url)
	if err == nil {
		t.Errorf("ScrapPesquisador(%q) não retornou um erro, mas esperávamos um devido à URL inválida", url)
	}
}
