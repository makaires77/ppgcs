package repository

type Publications struct {
	ID                    string `csv:"idLattes"`
	Name                  string `csv:"nome"`
	Tipo                  string `csv:"tipo"`
	Titulo_cap            string `csv:"titulo_do_capitulo"`
	Idioma                string `csv:"idioma"`
	Titulo_livro          string `csv:"titulo_do_livro"`
	Ano                   string `csv:"ano"`
	Doi                   string `csv:"doi"`
	Pais                  string `csv:"pais_de_publicacao"`
	Isbn                  string `csv:"isbn"`
	Editora_livro         string `csv:"nome_da_editora"`
	Edição_livro          string `csv:"numero_da_edicao_revisao"`
	Organizadores         string `csv:"organizadores"`
	Paginas               string `csv:"paginas"`
	Autores               string `csv:"autores"`
	Autores_endogeno      string `csv:"autores-endogeno"`
	Autores_endogeno_nome string `csv:"autores-endogeno-nome"`
	Tags                  string `csv:"tags"`
	Hash                  string `csv:"Hash"`
	Tipo_producao         string `csv:"tipo_producao"`
	Natureza              string `csv:"natureza"`
	Titulo                string `csv:"titulo"`
	Evento                string `csv:"nome_do_evento"`
	Ano_trabalho          string `csv:"ano_do_trabalho"`
	Pais_evento           string `csv:"pais_do_evento"`
	Cidade_evento         string `csv:"cidade_do_evento"`
	Classificação         string `csv:"classificacao"`
	Periodico             string `csv:"periodico"`
	Volume                string `csv:"volume"`
	Issn                  string `csv:"issn"`
	Estrato_qualis        string `csv:"estrato_qualis"`
	Editora_artigo        string `csv:"editora"`
	Numero_paginas        string `csv:"numero_de_paginas"`
}
