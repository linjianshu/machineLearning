package main

import (
	"fmt"
	"github.com/go-gota/gota/dataframe"
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
	irisFile, err := os.Open(join)
	if err != nil {
		log.Fatal(err)
	}
	defer irisFile.Close()

	//dataframe 数据帧
	irisDF := dataframe.ReadCSV(irisFile)
	fmt.Println(irisDF)
}
