package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/trees"
	"log"
	"math"
	"math/rand"
	"os"
	"path"
	"time"
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

	rand.Seed(time.Now().UnixNano())

	tree := trees.NewID3DecisionTree(0.6)
	cv, err := evaluation.GenerateCrossFoldValidationConfusionMatrices(irisData, tree, 5)
	if err != nil {
		log.Fatal(err)
	}

	mean, variance := evaluation.GetCrossValidatedMetric(cv, evaluation.GetAccuracy)
	stdev := math.Sqrt(variance)

	fmt.Printf("\nAccuracy\n %.2f (+/- %.2f)\n\n", mean, stdev*2)
}
