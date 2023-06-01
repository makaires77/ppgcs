package fuzzysearch

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"

	"github.com/hbollon/go-edlib"
)

type FuzzySearchService struct{}

func NewFuzzySearchService() *FuzzySearchService {
	return &FuzzySearchService{}
}

func (s *FuzzySearchService) FuzzySearchSetThreshold(input string, strList []string, maxDistance int) ([]string, error) {
	var similarityThreshold float32 = 0.82
	res, err := edlib.FuzzySearchSetThreshold(input, strList, maxDistance, similarityThreshold, edlib.Levenshtein)
	if err != nil {
		return nil, fmt.Errorf("error executing fuzzy search: %v", err)
	}
	return res, nil
}

func (s *FuzzySearchService) LoadCSVData(filePath string, colIndex int) ([]string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("cannot open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ';'
	reader.LazyQuotes = true

	lines, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("cannot read file: %v", err)
	}

	strList := []string{}
	for _, line := range lines {
		if line[colIndex] != "" {
			//strList = append(strList, strings.Split(line[colIndex], ";"))
			strList = append(strList, line[colIndex])

		}
	}
	return strList, nil
}
