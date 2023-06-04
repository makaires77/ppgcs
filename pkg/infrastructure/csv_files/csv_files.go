// csv_files.go
package csv_files

import (
	"encoding/csv"
	"errors"
	"io"
	"os"
)

func ReadCsvFile(filePath string, columnIndex int) ([]string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	var data []string
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if len(record) > columnIndex {
			data = append(data, record[columnIndex])
		} else {
			return nil, errors.New("Column index out of range")
		}
	}
	return data, nil
}
