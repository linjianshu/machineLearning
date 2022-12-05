package main

import (
	"encoding/csv"
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"github.com/sajari/regression"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"log"
	"math"
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
	trainingData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	//dataframe版本
	//trainingDF := dataframe.ReadCSV(open)

	var r regression.Regression
	r.SetObserved("Sales")
	r.SetVar(0, "TV")

	for i, record := range trainingData {
		if i == 0 {
			continue
		}

		yVal, err := strconv.ParseFloat(record[3], 64)
		if err != nil {
			log.Fatal(err)
		}

		tvVal, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Fatal(err)
		}

		//训练
		r.Train(regression.DataPoint(yVal, []float64{tvVal}))
	}

	//dataframe版本
	//yVal := trainingDF.Col("Sales").Float()
	//tvVal := trainingDF.Col("TV").Float()
	//for i, y := range yVal {
	//	r.Train(regression.DataPoint(y, []float64{tvVal[i]}))
	//}

	r.Run()
	fmt.Printf("\nRegression Formula: \n%v\n\n", r.Formula)

	//评估
	f, err := os.Open(path.Join(path.Dir(join), "test.csv"))
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader = csv.NewReader(f)

	reader.FieldsPerRecord = 4
	testData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	var mAE float64
	for i, record := range testData {
		if i == 0 {
			continue
		}

		yObserved, err := strconv.ParseFloat(record[3], 64)
		if err != nil {
			log.Fatal(err)
		}

		tvVal, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Fatal(err)
		}

		yPredicted, err := r.Predict([]float64{tvVal})
		if err != nil {
			log.Fatal(err)
		}

		mAE += math.Abs(yPredicted-yObserved) / float64(len(testData))
	}

	fmt.Printf("MAE = %0.2f\n\n", mAE)

	//可视化 建立在整体的数据之上
	s := path.Join(dir, "machineLearning", "Advertising.csv")
	file, err := os.Open(s)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	advertDF := dataframe.ReadCSV(file)

	p := plot.New()
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	tvVals := advertDF.Col("TV").Float()
	salesVals := advertDF.Col("Sales").Float()

	lineVals := make(plotter.XYs, advertDF.Nrow())
	scatterVals := make(plotter.XYs, advertDF.Nrow())
	for i := 0; i < len(lineVals); i++ {
		lineVals[i].X = tvVals[i]
		predict, err := r.Predict([]float64{tvVals[i]})
		if err != nil {
			log.Fatal(err)
		}
		lineVals[i].Y = predict

		scatterVals[i].X = tvVals[i]
		scatterVals[i].Y = salesVals[i]
	}

	line, err := plotter.NewLine(lineVals)
	if err != nil {
		log.Fatal(err)
	}
	line.LineStyle.Width = vg.Points(1)
	line.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
	p.Add(line)

	scatter, err := plotter.NewScatter(scatterVals)
	if err != nil {
		log.Fatal(err)
	}
	scatter.GlyphStyle.Radius = vg.Points(3)
	p.Add(scatter)

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "regression_line.png"); err != nil {
		log.Fatal(err)
	}
}
