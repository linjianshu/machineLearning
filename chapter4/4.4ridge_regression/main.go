package main

import (
	"encoding/csv"
	"fmt"
	"github.com/berkmancenter/ridge"
	"github.com/gonum/matrix/mat64"
	"log"
	"os"
	"path"
	"strconv"
)

func main() {
	dir, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	join := path.Join(dir, "machineLearning", "chapter4", "training.csv")
	open, err := os.Open(join)
	if err != nil {
		log.Fatal(err)
	}

	defer open.Close()

	reader := csv.NewReader(open)
	reader.FieldsPerRecord = 4

	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	featureData := make([]float64, 4*len(rawCSVData))
	yData := make([]float64, len(rawCSVData))

	var featureIdx int
	var yIdx int
	for idx, record := range rawCSVData {
		if idx == 0 {
			continue
		}

		for i, val := range record {
			valParsed, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			if i < 3 {
				if i == 0 {
					featureData[featureIdx] = 1
					featureIdx++
				}

				featureData[featureIdx] = valParsed
				featureIdx++
			}

			if i == 3 {
				yData[yIdx] = valParsed
				yIdx++
			}
		}
	}
	features := mat64.NewDense(len(rawCSVData), 4, featureData)
	y := mat64.NewVector(len(rawCSVData), yData)

	r := ridge.New(features, y, 1.0)
	r.Regress()

	c1 := r.Coefficients.At(0, 0)
	c2 := r.Coefficients.At(1, 0)
	c3 := r.Coefficients.At(2, 0)
	c4 := r.Coefficients.At(3, 0)
	fmt.Printf("\nRegression formula: \n")
	fmt.Printf("y = %0.3f +%0.3f TV + %0.3f Radio + %0.3f Newspaper\n\n", c1, c2, c3, c4)

	//评估

}
