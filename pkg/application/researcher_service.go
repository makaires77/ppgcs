// pkg/application/researcher_service.go
package application

import (
	"github.com/makaires77/ppgcs/pkg/domain/researcher"
)

// ResearcherService é responsável por fornecer métodos relacionados a pesquisadores.
type ResearcherService struct {
	researcherRepository researcher.ResearcherRepository // Corrigir o tipo do campo para pesquisadores.ResearcherRepository
}

// NewResearcherService cria uma nova instância de ResearcherService.
func NewResearcherService(repository researcher.ResearcherRepository) *ResearcherService {
	return &ResearcherService{
		researcherRepository: repository,
	}
}

// GetResearcher recupera um pesquisador pelo ID.
func (s *ResearcherService) GetResearcher(id string) (*researcher.Researcher, error) {
	// Implemente a lógica para recuperar o pesquisador do repositório usando o ID fornecido.
	// Por exemplo:
	// return s.researcherRepository.GetByID(id)
	return nil, nil
}

//Neste arquivo, a estrutura ResearcherService usa o padrão de projeto Strategy, também conhecido como Policy, o que permite selecionar um algoritmo de uma família de algoritmos em tempo de execução. Considera-se cada tipo de repositório como um "algoritmo" para armazenar dados. Temos uma única propriedade repo do tipo repository.ResearcherRepository. Essa é a interface que cada implementação de banco de dados específico precisa satisfazer. A função NewResearcherService recebe um argumento repo que também é do tipo repository.ResearcherRepository. Esta função retorna uma nova instância de ResearcherService com repo definido para o valor passado. A função Save na estrutura ResearcherService apenas chama a função Save na propriedade repo. Isso significa que ela irá chamar a função Save do banco de dados específico que foi passado quando a instância de ResearcherService foi criada. Essa função Save será responsável por implementar a lógica específica de como salvar o objeto Researcher no banco de dados específico.
