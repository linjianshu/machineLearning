package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
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

	//初始化knn分类器 使用欧几里得 k=2
	knn := knn.NewKnnClassifier("euclidean", "linear", 2)

	//交叉验证
	cv, err := evaluation.GenerateCrossFoldValidationConfusionMatrices(irisData, knn, 5)
	if err != nil {
		log.Fatal(err)
	}

	mean, variance := evaluation.GetCrossValidatedMetric(cv, evaluation.GetAccuracy)
	stdev := math.Sqrt(variance)

	fmt.Printf("\nAccuracy\n%.2f (+/- %.2f)\n\n", mean, stdev*2)
}
