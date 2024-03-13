<<<<<<< HEAD
=======
// pkg/application/researcher_service.go
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
package application

import (
	"github.com/makaires77/ppgcs/pkg/domain/researcher"
)

<<<<<<< HEAD
type ResearcherService struct {
	mongoRepo  researcher.ResearcherRepository
	dgraphRepo researcher.ResearcherRepository
	neo4jRepo  researcher.ResearcherRepository
}

func NewResearcherService(mongoRepo, dgraphRepo, neo4jRepo researcher.ResearcherRepository) *ResearcherService {
	return &ResearcherService{
		mongoRepo:  mongoRepo,
		dgraphRepo: dgraphRepo,
		neo4jRepo:  neo4jRepo,
	}
}

/*
A função Save recebe os dados de um pesquisador e os salva nos repositórios correspondentes.
Cada repositório deve implementar sua própria função Save, que aceita um ponteiro para Researcher e retorna um erro.
Essas funções devem ser implementadas de acordo com as regras de persistência de cada banco de dados.
*/
func (s *ResearcherService) Save(researcher *researcher.Researcher) error {
	err := s.mongoRepo.Save(researcher)
	if err != nil {
		return err
	}

	err = s.dgraphRepo.Save(researcher)
	if err != nil {
		return err
	}

	err = s.neo4jRepo.Save(researcher)
	if err != nil {
		return err
	}

	return nil
}
=======
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
>>>>>>> c51253137853d9681efc37ff1382c4b6f7ed1174
