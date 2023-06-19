package load_publication

import (
	"fmt"

	"github.com/makaires77/ppgcs/pkg/domain/publication"
)

type PublicationInteractor struct {
	repository publication.PublicationRepository
}

func NewPublicationInteractor(repository publication.PublicationRepository) *PublicationInteractor {
	return &PublicationInteractor{
		repository: repository,
	}
}

func (i *PublicationInteractor) LoadPublications() ([]*publication.Publication, error) {
	// Chame o método GetAll do repositório para obter todas as publicações
	publications, err := i.repository.GetAll()
	if err != nil {
		return nil, fmt.Errorf("failed to load publications: %v", err)
	}

	return publications, nil
}
