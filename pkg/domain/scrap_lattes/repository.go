package scrap_lattes

import (
	"fmt"
	"log"

	"github.com/PuerkitoBio/goquery"
)

// ExtrairFormacaoComplementar extrai a formação complementar do pesquisador do HTML.
func ExtrairFormacaoComplementar(doc *goquery.Document) []string {
	var formacoes []string

	doc.Find("#content .formacao-academica-complementar").Each(func(i int, s *goquery.Selection) {
		formacao := s.Text()
		formacoes = append(formacoes, formacao)
	})

	return formacoes
}

// ExtrairPublicacoes extrai as publicações do pesquisador do HTML.
func ExtrairPublicacoes(doc *goquery.Document) []Publicacao {
	var publicacoes []Publicacao

	doc.Find("#content .artigo-completo").Each(func(i int, s *goquery.Selection) {
		titulo := s.Find(".informacao-artigo[data-tipo-ordenacao='titulo']").Text()
		ano := s.Find(".informacao-artigo[data-tipo-ordenacao='ano']").Text()

		var autores []string
		s.Find(".informacao-artigo[data-tipo-ordenacao='autor']").Each(func(i int, s *goquery.Selection) {
			autor := s.Text()
			autores = append(autores, autor)
		})

		doi, _ := s.Find("a.icone-producao.icone-doi").Attr("href")

		publicacao := Publicacao{
			Titulo:  titulo,
			Ano:     ano,
			Autores: autores,
			DOI:     doi,
		}

		publicacoes = append(publicacoes, publicacao)
	})

	return publicacoes
}

// ExtrairDadosPesquisador extrai os dados do pesquisador do HTML.
func ExtrairDadosPesquisador(doc *goquery.Document) (*Pesquisador, error) {
	nome := doc.Find("#nome .nome-completo").Text()
	if nome == "" {
		return nil, fmt.Errorf("nome do pesquisador não encontrado")
	}

	formacaoComplementar := ExtrairFormacaoComplementar(doc)
	publicacoes := ExtrairPublicacoes(doc)

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

	pesquisador, err := ExtrairDadosPesquisador(doc)
	if err != nil {
		log.Printf("Erro ao extrair os dados do pesquisador: %s\n", err)
		return nil, err
	}

	return pesquisador, nil
}

// Exemplo de uso
/* func main() {
	url := "http://exemplo.com/pesquisador"

	pesquisador, err := ScrapPesquisador(url)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Nome do pesquisador:", pesquisador.Nome)
	fmt.Println("Formação complementar:")
	for _, formacao := range pesquisador.Formacao {
		fmt.Println(formacao)
	}

	fmt.Println("Publicações:")
	for _, publicacao := range pesquisador.Publicacoes {
		fmt.Println("Título:", publicacao.Titulo)
		fmt.Println("Ano:", publicacao.Ano)
		fmt.Println("Autores:")
		for _, autor := range publicacao.Autores {
			fmt.Println(autor)
		}
		fmt.Println("DOI:", publicacao.DOI)
		fmt.Println("---")
	}
}
*/
