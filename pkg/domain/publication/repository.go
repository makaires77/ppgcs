package publication

import "errors"

var ErrPublicationNotFound = errors.New("publication not found")

type PublicationRepository interface {
	Save(publication *Publication) error
	GetByID(id string) (*Publication, error)
	GetAll() ([]*Publication, error)
	Delete(id string) error
}

type InMemoryPublicationRepository struct {
	publications []*Publication
}

func NewInMemoryPublicationRepository() *InMemoryPublicationRepository {
	return &InMemoryPublicationRepository{
		publications: make([]*Publication, 0),
	}
}

func (r *InMemoryPublicationRepository) Save(publication *Publication) error {
	r.publications = append(r.publications, publication)
	return nil
}

func (r *InMemoryPublicationRepository) GetByID(id string) (*Publication, error) {
	for _, publication := range r.publications {
		if publication.ID == id {
			return publication, nil
		}
	}
	return nil, ErrPublicationNotFound
}

func (r *InMemoryPublicationRepository) GetAll() ([]*Publication, error) {
	return r.publications, nil
}

func (r *InMemoryPublicationRepository) Delete(id string) error {
	for i, publication := range r.publications {
		if publication.Hash == id {
			// Remove the publication from the slice
			r.publications = append(r.publications[:i], r.publications[i+1:]...)
			return nil
		}
	}
	return ErrPublicationNotFound
}
