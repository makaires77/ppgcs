package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"reflect"
)

func handleErr(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func writeCSV(data []map[string]interface{}, filePath string) {
	if len(data) == 0 {
		log.Printf("No data to write to %s", filePath)
		return
	}

	file, err := os.Create(filePath)
	if err != nil {
		log.Printf("error creating file %s: %v", filePath, err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	var headers []string
	for key := range data[0] {
		headers = append(headers, key)
	}

	err = writer.Write(headers)
	if err != nil {
		log.Printf("error writing headers to csv: %v", err)
		return
	}

	for _, v := range data {
		var row []string
		for _, header := range headers {
			value := v[header]
			switch value := value.(type) {
			case nil:
				row = append(row, "")
			case []interface{}:
				row = append(row, arrayToString(value))
			default:
				row = append(row, fmt.Sprint(value))
			}
		}
		err = writer.Write(row)
		if err != nil {
			log.Printf("error writing row to csv: %v", err)
			return
		}
	}

	log.Printf("Wrote %d records to %s", len(data), filePath)
}

func arrayToString(arr []interface{}) string {
	str := ""
	for _, v := range arr {
		switch v := v.(type) {
		case string:
			str += v + ";"
		case map[string]interface{}:
			for key, value := range v {
				str += key + ":" + fmt.Sprint(value) + ","
			}
			str += ";"
		default:
			str += ";"
		}
	}
	return str
}

func main() {
	basePath, err := os.Getwd()
	if err != nil {
		log.Fatalf("error getting current directory: %v", err)
	}

	jsonFilePath := filepath.Join(basePath, "_data/out_json/merged.json")
	csvOutputDir := filepath.Join(basePath, "_data/out_csv/")

	file, err := os.Open(jsonFilePath)
	if err != nil {
		log.Fatalf("error opening json file: %v", err)
	}
	defer file.Close()

	byteValue, _ := ioutil.ReadAll(file)

	var result []map[string]interface{}
	err = json.Unmarshal(byteValue, &result)
	if err != nil {
		log.Fatalf("error unmarshalling json: %v", err)
	}

	if len(result) == 0 {
		log.Printf("No data in JSON file")
		return
	}

	for _, record := range result {
		for entity, value := range record {
			if reflect.TypeOf(value).Kind() == reflect.Slice {
				values, ok := value.([]map[string]interface{})
				if !ok {
					log.Printf("Cannot convert %s to []map[string]interface{}", entity)
					continue
				}
				writeCSV(values, filepath.Join(csvOutputDir, entity+".csv"))
			} else {
				log.Printf("Value for entity %s is not a slice", entity)
			}
		}
	}
}
