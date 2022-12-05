package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
	"log"
	"math"
	"os"
	"path"
)

func main() {
	dir, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	join := path.Join(dir, "machineLearning", "iris.csv")
	irisData, err := base.ParseCSVToInstances(join, true)
	if err != nil {
		log.Fatal(err)
	}
	rf := ensemble.NewRandomForest(10, 2)

	cv, err := evaluation.GenerateCrossFoldValidationConfusionMatrices(irisData, rf, 5)

	mean, variance := evaluation.GetCrossValidatedMetric(cv, evaluation.GetAccuracy)
	stdev := math.Sqrt(variance)
	fmt.Printf("\nAccuracy\n%.2f (+/- %.2f)\n\n", mean, stdev*2)
}
