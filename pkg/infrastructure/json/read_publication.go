package json

import (
	"encoding/json"
	"fmt"
	"io/ioutil"

	"github.com/makaires77/ppgcs/pkg/domain/publication"
)

type PublicationJSONReader struct {
	FilePath string
}

func NewPublicationJSONReader(filePath string) *PublicationJSONReader {
	return &PublicationJSONReader{
		FilePath: filePath,
	}
}

func (r *PublicationJSONReader) ReadPublications() ([]*publication.Publication, error) {
	data, err := ioutil.ReadFile(r.FilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read publication JSON file: %w", err)
	}

	var publications []*publication.Publication
	err = json.Unmarshal(data, &publications)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal publication JSON: %w", err)
	}

	return publications, nil
}
