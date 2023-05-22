package main

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
Para implementar a função Save, precisamos supor que cada repositório (mongoRepo, dgraphRepo e neo4jRepo) tem sua própria função Save.
Cada uma aceita um Researcher e retorna um error. Essas funções devem ser implementadas de acordo com as regras de persistência de cada banco de dados.
*/
func (s *ResearcherService) Save(r *researcher.Researcher) error {
	err := s.mongoRepo.Save(r)
	if err != nil {
		return err
	}

	err = s.dgraphRepo.Save(r)
	if err != nil {
		return err
	}

	err = s.neo4jRepo.Save(r)
	if err != nil {
		return err
	}

	return nil
}
