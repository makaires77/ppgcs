package publication

type Publication struct {
	Natureza            string
	Titulo              string
	Idioma              string
	Periodico           string
	Ano                 string
	Volume              string
	ISSN                string
	EstratoQualis       string
	PaisDePublicacao    string
	Paginas             string
	DOI                 string
	Autores             []string
	AutoresEndogeno     []string
	AutoresEndogenoNome []map[string]string
	Tags                []interface{}
	Hash                string
}
