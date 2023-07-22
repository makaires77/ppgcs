// pkg\domain\researcher\repository.go
package researcher

// ResearcherRepository é uma interface que define as operações do repositório de pesquisadores.
type ResearcherRepository interface {
	Save(*Researcher) error
	// Adicione outras funções do repositório, se necessário
}

// Exemplo de implementação do repositório (pseudo-código)
type ExampleResearcherRepository struct {
	// Aqui você pode adicionar campos e dependências necessárias para a implementação do repositório.
	// Por exemplo, uma conexão com o banco de dados, uma lista de pesquisadores em memória, etc.
	researchers []*Researcher
}

// Implementação da função Save para o repositório de exemplo
func (r *ExampleResearcherRepository) Save(researcher *Researcher) error {
	r.researchers = append(r.researchers, researcher)
	return nil
}
