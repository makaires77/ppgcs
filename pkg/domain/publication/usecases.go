package publication

type PublicationInteractor struct {
	repository PublicationRepository
}

func NewPublicationInteractor(repository PublicationRepository) *PublicationInteractor {
	return &PublicationInteractor{
		repository: repository,
	}
}

func (i *PublicationInteractor) CreatePublication(publication *Publication) error {
	// Implement the business logic for creating a publication
	// e.g., validate the publication data, generate an ID, etc.

	// Save the publication in the repository
	err := i.repository.Save(publication)
	if err != nil {
		return err
	}

	return nil
}

func (i *PublicationInteractor) GetPublicationByID(id string) (*Publication, error) {
	// Retrieve the publication from the repository
	publication, err := i.repository.GetByID(id)
	if err != nil {
		return nil, err
	}

	return publication, nil
}

func (i *PublicationInteractor) GetAllPublications() ([]*Publication, error) {
	// Retrieve all publications from the repository
	publications, err := i.repository.GetAll()
	if err != nil {
		return nil, err
	}

	return publications, nil
}

func (i *PublicationInteractor) DeletePublication(id string) error {
	// Delete the publication from the repository
	err := i.repository.Delete(id)
	if err != nil {
		return err
	}

	return nil
}
