package main

import (
	"encoding/csv"
	"fmt"
	"gonum.org/v1/gonum/stat"
	"io"
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

	join := path.Join(dir, "machineLearning", "continuous_data.csv")
	irisFile, err := os.Open(join)
	if err != nil {
		log.Fatal(err)
	}
	defer irisFile.Close()

	reader := csv.NewReader(irisFile)

	var observed []float64
	var predicted []float64

	line := 1
	for {
		//读
		record, err := reader.Read()

		//如果读完了就break
		if err == io.EOF {
			break
		}

		//跳过标题
		if line == 1 {
			line++
			continue
		}

		//转
		observedVal, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Printf("Parsing line %d failed, unexpected type\n", line)
			continue
		}

		//转
		predictedVal, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			log.Printf("Parsing line %d failed, unexpected type\n", line)
			continue
		}

		observed = append(observed, observedVal)
		predicted = append(predicted, predictedVal)
		line++
	}

	//平均绝对误差
	var mAE float64
	//均方差
	var mSE float64
	for idx, oVal := range observed {
		mAE += math.Abs(oVal-predicted[idx]) / float64(len(observed))
		mSE += math.Pow(oVal-predicted[idx], 2) / float64(len(observed))
	}

	fmt.Printf("\nMAE = %0.2f\n", mAE)
	fmt.Printf("\nMSE = %0.2f\n\n", mSE)

	//计算R方 决定系数 越高越好
	rSquared := stat.RSquaredFrom(observed, predicted, nil)
	fmt.Printf("\nR^2 = %0.2f\n\n", rSquared)
}
