package fuzzysearch

import (
	"encoding/csv"
	"fmt"
	"os"
	"strings"

	"github.com/hbollon/go-edlib"
)

type FuzzySearchService struct{}

func NewFuzzySearchService() *FuzzySearchService {
	return &FuzzySearchService{}
}

// LoadCSVData loads data from a CSV file, splits authors by ';' or ',' if applicable and returns it as a slice of strings.
func (s *FuzzySearchService) LoadCSVData(filePath string, columnIndex int) ([]string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("cannot open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ';'
	reader.LazyQuotes = true

	lines, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("cannot read file: %v", err)
	}

	if len(lines) == 0 {
		reader.Comma = ','
		lines, err = reader.ReadAll()
		if err != nil {
			return nil, fmt.Errorf("cannot read file: %v", err)
		}
	}

	data := make([]string, 0, len(lines))
	for _, line := range lines {
		if len(line) > columnIndex {
			if strings.Contains(line[columnIndex], ";") {
				authors := strings.Split(line[columnIndex], ";")
				data = append(data, authors...)
			} else if strings.Contains(line[columnIndex], ",") {
				authors := strings.Split(line[columnIndex], ",")
				data = append(data, authors...)
			} else {
				data = append(data, line[columnIndex])
			}
		}
	}
	return data, nil
}

// FuzzySearchSetThreshold is a wrapper around the `edlib.FuzzySearchSetThreshold` function.
func (s *FuzzySearchService) FuzzySearchSetThreshold(str string, strList []string, resultQuantity int, minSimilarity float32, algorithm edlib.Algorithm) ([]string, error) {
	res, err := edlib.FuzzySearchSetThreshold(str, strList, resultQuantity, minSimilarity, algorithm)
	if err != nil {
		return nil, fmt.Errorf("fuzzy search failed: %v", err)
	}
	return res, nil
}
