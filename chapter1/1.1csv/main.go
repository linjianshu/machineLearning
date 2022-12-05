package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"path"
)

func main() {
	dir, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	join := path.Join(dir, "machineLearning", "iris.csv")
	f, err := os.Open(join)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1

	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(rawCSVData)
}
