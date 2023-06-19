package main

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"os"
)

func main() {
	// list of files
	files := []string{
		"_data/in_json/642.files/642.publication.json",
		"_data/in_json/644.files/644.publication.json",
		"_data/in_json/642.files/642.advise.json",
		"_data/in_json/644.files/644.advise.json",
		/* 		"_data/in_json/642.files/642patents.json",
		   		"_data/in_json/644.files/644patents.json", */

		// adicione todos os arquivos que deseja unir
	}

	mergedData := make([]map[string]interface{}, 0)

	for _, file := range files {
		jsonFile, err := os.Open(file)
		if err != nil {
			log.Fatalf("Failed to open file %s: %v", file, err)
		}
		defer jsonFile.Close()

		byteValue, _ := ioutil.ReadAll(jsonFile)

		// Let's try to unmarshal the content to a single map first
		var singleObject map[string]interface{}
		singleErr := json.Unmarshal(byteValue, &singleObject)
		if singleErr == nil {
			// The content was a single JSON object, add it to mergedData and move on to the next file
			mergedData = append(mergedData, singleObject)
			continue
		}

		// The content was not a single object, let's assume it was an array of objects
		var multipleObjects []map[string]interface{}
		multipleErr := json.Unmarshal(byteValue, &multipleObjects)
		if multipleErr != nil {
			log.Fatalf("Failed to unmarshal file %s: %v", file, multipleErr)
		}

		mergedData = append(mergedData, multipleObjects...)
	}

	// If you want to print merged data
	for _, data := range mergedData {
		for key, value := range data {
			log.Printf("Key: %s, Value: %v\n", key, value)
		}
	}

	// If you want to write merged data to a new JSON file
	file, _ := json.MarshalIndent(mergedData, "", " ")
	_ = ioutil.WriteFile("_data/out_json/merged.json", file, 0644)
}
