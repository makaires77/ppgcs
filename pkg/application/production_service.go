// application/production_service.go

package application

import (
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

// Aqui você deve adicionar todas as funções que quer usar no seu serviço de Produção.
