package infrastructure

import (
	"encoding/csv"
	"os"
	"strings"
)

func readCSVFile(filePath string) ([]map[string]string, error) {
	// Open the file
	csvfile, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer csvfile.Close()

	// Parse the file
	r := csv.NewReader(csvfile)

	// Read in all records
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}

	// Prepare container
	var entries = []map[string]string{}

	// Iterate through records
	for _, record := range records {
		entry := map[string]string{}
		for i, value := range record {
			// Use the first row as the map key
			entry[strings.ToLower(records[0][i])] = value
		}
		entries = append(entries, entry)
	}

	return entries, nil
}
