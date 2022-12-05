package main

import (
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"github.com/sajari/regression"
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

	join := path.Join(dir, "machineLearning", "chapter4", "training.csv")
	open, err := os.Open(join)
	if err != nil {
		log.Fatal(err)
	}
	defer open.Close()

	trainingDF := dataframe.ReadCSV(open)
	var r regression.Regression
	r.SetObserved("Sales")
	r.SetVar(0, "TV")
	r.SetVar(1, "Radio")

	//训练
	for i := 0; i < trainingDF.Nrow(); i++ {
		r.Train(regression.DataPoint(trainingDF.Col("Sales").Float()[i], []float64{trainingDF.Col("TV").Float()[i], trainingDF.Col("Radio").Float()[i]}))
	}
	r.Run()

	fmt.Printf("Regression Formula: \n%v\n", r.Formula)

	//测试和验证
	var mAE float64
	s := path.Join(dir, "machineLearning", "chapter4", "test.csv")
	file, err := os.Open(s)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	testDF := dataframe.ReadCSV(file)
	observed := testDF.Col("Sales").Float()
	for i, observedVal := range observed {
		tvVal := testDF.Col("TV").Float()[i]
		radioVal := testDF.Col("Radio").Float()[i]
		predictVal, err := r.Predict([]float64{tvVal, radioVal})
		if err != nil {
			log.Fatal(err)
		}
		mAE += math.Abs(observedVal-predictVal) / float64(len(observed))
	}

	fmt.Printf("MAE = %0.2f\n\n", mAE)
}
