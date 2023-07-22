// pkg\domain\scrap_lattes\repository.go
package scrap_lattes

import (
	"fmt"
	"log"

	"github.com/PuerkitoBio/goquery"
)

// extrairFormacaoComplementar extrai a formação complementar do pesquisador do HTML.
func extrairFormacaoComplementar(doc *goquery.Document) []string {
	var formacoes []string

	doc.Find("#content .formacao-academica-complementar").Each(func(i int, s *goquery.Selection) {
		formacao := s.Text()
		formacoes = append(formacoes, formacao)
	})

	return formacoes
}

// extrairPublicacoes extrai as publicações do pesquisador do HTML.
func extrairPublicacoes(doc *goquery.Document) ([]Publicacao, error) {
	var publicacoes []Publicacao

	doc.Find("#content .artigo-completo").Each(func(i int, s *goquery.Selection) {
		titulo := s.Find(".informacao-artigo[data-tipo-ordenacao='titulo']").Text()
		ano := s.Find(".informacao-artigo[data-tipo-ordenacao='ano']").Text()

		var autores []string
		s.Find(".informacao-artigo[data-tipo-ordenacao='autor']").Each(func(i int, s *goquery.Selection) {
			autor := s.Text()
			autores = append(autores, autor)
		})

		doi, exists := s.Find("a.icone-producao.icone-doi").Attr("href")
		if !exists {
			return nil, fmt.Errorf("DOI não encontrado")
		}

		publicacao := Publicacao{
			Titulo:  titulo,
			Ano:     ano,
			Autores: autores,
			DOI:     doi,
		}

		publicacoes = append(publicacoes, publicacao)
	})

	return publicacoes, nil
}

// extrairDadosPesquisador extrai os dados do pesquisador do HTML.
func extrairDadosPesquisador(doc *goquery.Document) (*Pesquisador, error) {
	nome := doc.Find("#nome .nome-completo").Text()
	if nome == "" {
		return nil, fmt.Errorf("nome do pesquisador não encontrado")
	}

	formacaoComplementar := extrairFormacaoComplementar(doc)
	publicacoes, err := extrairPublicacoes(doc)
	if err != nil {
		return nil, err
	}

	pesquisador := &Pesquisador{
		Nome:        nome,
		Formacao:    formacaoComplementar,
		Publicacoes: publicacoes,
	}

	return pesquisador, nil
}

// ScrapPesquisador realiza o scraping das informações do pesquisador a partir da URL do seu currículo Lattes.
func ScrapPesquisador(url string) (*Pesquisador, error) {
	doc, err := goquery.NewDocument(url)
	if err != nil {
		log.Printf("Erro ao acessar a URL: %s\n", err)
		return nil, err
	}

	pesquisador, err := extrairDadosPesquisador(doc)
	if err != nil {
		log.Printf("Erro ao extrair os dados do pesquisador: %s\n", err)
		return nil, err
	}

	return pesquisador, nil
}
