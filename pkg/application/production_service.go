package application

import (
	"github.com/makaires77/ppgcs/pkg/domain/publication"
	"github.com/makaires77/ppgcs/pkg/repository"
)

type ProductionService struct {
	productionRepo repository.ProductionRepository
}

func NewProductionService(repo repository.ProductionRepository) *ProductionService {
	return &ProductionService{
		productionRepo: repo,
	}
}

// GetProductionByID busca uma produção pelo seu ID.
func (ps *ProductionService) GetProductionByID(id string) (*publication.Publication, error) {
	production, err := ps.productionRepo.GetByID(id)
	if err != nil {
		return nil, err
	}
	return production, nil
}

// CreateProduction cria uma nova produção.
func (ps *ProductionService) CreateProduction(production *publication.Publication) error {
	err := ps.productionRepo.Save(production)
	if err != nil {
		return err
	}
	return nil
}

// UpdateProduction atualiza uma produção existente.
func (ps *ProductionService) UpdateProduction(production *publication.Publication) error {
	err := ps.productionRepo.Update(production)
	if err != nil {
		return err
	}
	return nil
}

// DeleteProduction exclui uma produção pelo seu ID.
func (ps *ProductionService) DeleteProduction(id string) error {
	err := ps.productionRepo.DeleteByID(id)
	if err != nil {
		return err
	}
	return nil
}

// ListProductions lista todas as produções.
func (ps *ProductionService) ListProductions() ([]*publication.Publication, error) {
	productions, err := ps.productionRepo.ListAll()
	if err != nil {
		return nil, err
	}
	return productions, nil
}

// Aqui você pode adicionar outras funções conforme necessário para o serviço de Produção.
