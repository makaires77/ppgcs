package publication

type Publication struct {
	Natureza            string            `json:"natureza"`
	Titulo              string            `json:"titulo"`
	Idioma              string            `json:"idioma"`
	Periodico           string            `json:"periodico"`
	Ano                 string            `json:"ano"`
	Volume              string            `json:"volume"`
	ISSN                string            `json:"issn"`
	EstratoQualis       string            `json:"estrato_qualis"`
	PaisDePublicacao    string            `json:"pais_de_publicacao"`
	Paginas             string            `json:"paginas"`
	DOI                 string            `json:"doi"`
	Autores             []string          `json:"autores"`
	AutoresEndogeno     []string          `json:"autores_endogeno"`
	AutoresEndogenoNome map[string]string `json:"autores_endogeno_nome"`
	Tags                []string          `json:"tags"`
	Hash                string            `json:"hash"`
}
