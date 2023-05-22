package application

import (
	"github.com/makaires77/ppgcs/pkg/domain/researcher"
)

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
