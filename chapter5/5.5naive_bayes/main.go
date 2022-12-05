package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/filters"
	"github.com/sjwhitworth/golearn/naive"
	"log"
	"os"
	"path"
)

func main() {
	dir, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	join := path.Join(dir, "machineLearning", "chapter5", "5.5naive_bayes", "training.csv")
	//TODO 需要修改
	trainingData, err := base.ParseCSVToInstances(join, true)
	if err != nil {
		log.Fatal(err)
	}

	nb := naive.NewBernoulliNBClassifier()
	nb.Fit(convertToBinary(trainingData))

	join = path.Join(dir, "machineLearning", "chapter5", "5.5naive_bayes", "test.csv")
	testData, err := base.ParseCSVToTemplatedInstances(join, true, trainingData)
	if err != nil {
		log.Fatal(err)
	}

	predict, err := nb.Predict(convertToBinary(testData))
	if err != nil {
		log.Fatal(err)
	}

	cm, err := evaluation.GetConfusionMatrix(testData, predict)

	if err != nil {
		log.Fatal(err)
	}

	accuracy := evaluation.GetAccuracy(cm)
	fmt.Printf("\nAccuracy: %0.2f\n\n", accuracy)
}

func convertToBinary(src base.FixedDataGrid) base.FixedDataGrid {
	b := filters.NewBinaryConvertFilter()
	attrs := base.NonClassAttributes(src)
	for _, a := range attrs {
		b.AddAttribute(a)
	}
	b.Train()
	ret := base.NewLazilyFilteredInstances(src, b)
	return ret
}
