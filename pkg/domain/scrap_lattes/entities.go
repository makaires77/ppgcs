package scrap_lattes

import (
	"time"
)

type Reserarcher struct {
	Nome                       string
	Titulo                     string
	LinkCurriculo              string
	IDLattes                   string
	DataUltimaAtualizacao      string
	Resumo                     string
	NomeCitacoesBibliograficas string
	IDLattesLink               string
	OrcidID                    string
	Formacao                   []string
	Formacoes                  []FormacaoAcademica
	PosDoutorado               []FormacaoPosDoc
	FormacoesComplementares    []FormacaoComplementar
	AtuacoesProfissionais      []AtuacaoProfissional
	LinhasPesquisa             []LinhaPesquisa
	ProjetosPesquisa           []ProjetoPesquisa       `json:"projetos_pesquisa"`
	Publicacoes                []Publicacao            `json:"publicacoes"`
	FormacaoAcademica          []Formacao              `json:"formacao_academica"`
	CursosExtraCurriculares    []Curso                 `json:"cursos_extra_curriculares"`
	ExperienciaProfissional    []Experiencia           `json:"experiencia_profissional"`
	Patentes                   []Patente               `json:"patentes"`
	MembroCorpoEditorial       []MembroCorpoEditorial  `json:"membro_corpo_editorial"`
	RevisorPeriodico           []RevisorPeriodico      `json:"revisor_periodico"`
	RevisorProjetoFomento      []RevisorProjetoFomento `json:"revisor_projeto_fomento"`
	PremiosTitulos             []PremioTitulo          `json:"premios_titulos"`
	/* ProducoesCientificas       []ProducoesCientificas  `json:"producoes_cientificas"` */
}

type Pesquisador struct {
	Nome                       string
	Titulo                     string
	LinkCurriculo              string
	IDLattes                   string
	DataUltimaAtualizacao      string
	Resumo                     string
	NomeCitacoesBibliograficas string
	IDLattesLink               string
	OrcidID                    string
	Formacao                   []string
	Formacoes                  []FormacaoAcademica
	PosDoutorado               []FormacaoPosDoc
	FormacoesComplementares    []FormacaoComplementar
	AtuacoesProfissionais      []AtuacaoProfissional
	LinhasPesquisa             []LinhaPesquisa
	ProjetosPesquisa           []ProjetoPesquisa       `json:"projetos_pesquisa"`
	Publicacoes                []Publicacao            `json:"publicacoes"`
	FormacaoAcademica          []Formacao              `json:"formacao_academica"`
	CursosExtraCurriculares    []Curso                 `json:"cursos_extra_curriculares"`
	ExperienciaProfissional    []Experiencia           `json:"experiencia_profissional"`
	Patentes                   []Patente               `json:"patentes"`
	MembroCorpoEditorial       []MembroCorpoEditorial  `json:"membro_corpo_editorial"`
	RevisorPeriodico           []RevisorPeriodico      `json:"revisor_periodico"`
	RevisorProjetoFomento      []RevisorProjetoFomento `json:"revisor_projeto_fomento"`
	PremiosTitulos             []PremioTitulo          `json:"premios_titulos"`
	/* ProducoesCientificas       []ProducoesCientificas  `json:"producoes_cientificas"` */
}

// Producao é uma interface que define os métodos comuns a todas as produções.
type Producao interface {
	GetTitulo() string
	GetData() time.Time
	// Outros métodos comuns a todas as produções...
}

type FormacaoAcademica struct {
	Periodo        string
	Nivel          string
	TituloTrabalho string
	Instituicao    string
	Orientador     string
	Bolsista       string
	AreaEstudo     string
	PalavrasChave  []string
}

type FormacaoPosDoc struct {
	Periodo        string
	Nivel          string
	TituloTrabalho string
	Instituicao    string
	Orientador     string
	Bolsista       string
	AreaEstudo     string
	PalavrasChave  []string
}

type FormacaoComplementar struct {
	Periodo      string
	TituloCurso  string
	CargaHoraria string
	Instituicao  string
}

type AtuacaoProfissional struct {
	Instituicao          string
	VinculoInstitucional []VinculoInstitucional
	Atividades           []Atividade
}

type VinculoInstitucional struct {
	Periodo      string
	Descricao    string
	CargaHoraria int
}

type Atividade struct {
	Periodo        string
	Descricao      string
	Disciplinas    []string
	LinhasPesquisa []string
}

type LinhaPesquisa struct {
	Nome             string
	Objetivo         string
	GrandeArea       []string
	Area             []string
	SubArea          []string
	SetoresAtividade []string
	PalavrasChave    []string
}

type Endereco struct {
	Rua    string `json:"rua"`
	Bairro string `json:"bairro"`
	Cidade string `json:"cidade"`
	Estado string `json:"estado"`
	Pais   string `json:"pais"`
	CEP    string `json:"cep"`
}

type RedeSocial struct {
	Nome string `json:"nome"`
	URL  string `json:"url"`
}

type ProjetoPesquisa struct {
	Titulo        string     `json:"titulo"`
	Descricao     string     `json:"descricao"`
	DataInicio    *time.Time `json:"data_inicio"`
	DataTermino   *time.Time `json:"data_termino"`
	Financiamento string     `json:"financiamento"`
}

// Publicacao representa uma publicação do pesquisador.
type Publicacao struct {
	Titulo  string     `json:"titulo"`
	Resumo  string     `json:"resumo"`
	Data    *time.Time `json:"data"`
	Revista string     `json:"revista"`
	Ano     string
	Autores []string
	DOI     string
}

type Formacao struct {
	Titulo       string `json:"titulo"`
	Instituicao  string `json:"instituicao"`
	AnoConclusao string `json:"ano_conclusao"`
}

type Curso struct {
	Nome          string     `json:"nome"`
	Instituicao   string     `json:"instituicao"`
	DataConclusao *time.Time `json:"data_conclusao"`
}

type Experiencia struct {
	Posicao string     `json:"posicao"`
	Empresa string     `json:"empresa"`
	Inicio  *time.Time `json:"inicio"`
	Termino *time.Time `json:"termino"`
}

type Patente struct {
	Titulo string `json:"titulo"`
	Resumo string `json:"resumo"`
	Ano    int    `json:"ano"`
}

type MembroCorpoEditorial struct {
	Periodo   Periodo `json:"periodo"`
	Periodico string  `json:"periodico"`
}

type Periodo struct {
	Inicio  *time.Time `json:"inicio"`
	Termino *time.Time `json:"termino"` // pode ser nulo para representar "Atual"
}

type RevisorPeriodico struct {
	Periodo   Periodo `json:"periodo"`
	Periodico string  `json:"periodico"`
}

type RevisorProjetoFomento struct {
	Periodo        Periodo `json:"periodo"`
	AgenciaFomento string  `json:"agencia_fomento"`
}

type PremioTitulo struct {
	Ano       int    `json:"ano"`
	Descricao string `json:"descricao"`
}
