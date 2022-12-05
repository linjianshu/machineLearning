package main

import (
	"bufio"
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

	join := path.Join(dir, "machineLearning", "Advertising.csv")
	open, err := os.Open(join)
	if err != nil {
		log.Fatal(err)
	}

	advertDF := dataframe.ReadCSV(open)

	trainingNum := (4 * advertDF.Nrow()) / 5
	testNum := advertDF.Nrow() / 5
	if trainingNum+testNum < advertDF.Nrow() {
		trainingNum++
	}

	trainingIdx := make([]int, trainingNum)
	testIdx := make([]int, testNum)
	for i := 0; i < trainingNum; i++ {
		trainingIdx[i] = i
	}

	for i := 0; i < testNum; i++ {
		testIdx[i] = i + trainingNum
	}

	trainingDF := advertDF.Subset(trainingIdx)
	testDF := advertDF.Subset(testIdx)

	setMap := map[int]dataframe.DataFrame{
		0: trainingDF,
		1: testDF,
	}

	for i, setName := range []string{"training.csv", "test.csv"} {
		create, err := os.Create(path.Join(path.Dir(join), "chapter4", setName))
		if err != nil {
			log.Fatal(err)
		}

		writer := bufio.NewWriter(create)
		err = setMap[i].WriteCSV(writer)
		if err != nil {
			log.Fatal(err)
		}
	}
}
