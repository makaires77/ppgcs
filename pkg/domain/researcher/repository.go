package researcher

// ResearcherRepository é uma interface que define as operações do repositório de pesquisadores.
type ResearcherRepository interface {
	Save(*Researcher) error
	// Adicione outras funções do repositório, se necessário
}
