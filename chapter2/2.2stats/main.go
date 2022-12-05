package main

import (
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"github.com/montanaflynn/stats"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
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

	irisDF := dataframe.ReadCSV(irisFile)

	//集中趋势测量法
	//获取列数据
	sepalLength := irisDF.Col("sepal_length").Float()

	//计算平均值
	meanVal := stat.Mean(sepalLength, nil)

	//计算众数
	modeVal, modeCount := stat.Mode(sepalLength, nil)

	//计算中位数
	medianVal, err := stats.Median(sepalLength)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\nSepal Length Summary Statistics: \n")
	fmt.Printf("Mean value: %0.2f\n", meanVal)
	fmt.Printf("Mode value: %0.2f\n", modeVal)
	fmt.Printf("Mode count: %d\n", int(modeCount))
	fmt.Printf("Median value: %0.2f\n\n", medianVal)

	//离中趋势测量法
	//取列数据
	petalLength := irisDF.Col("petal_length").Float()
	//最小值
	minVal := floats.Min(petalLength)
	//最大值
	maxVal := floats.Max(petalLength)
	//极差
	rangeVal := maxVal - minVal
	//方差
	varianceVal := stat.Variance(petalLength, nil)
	//标准差
	stdDevVal := stat.StdDev(petalLength, nil)

	inds := make([]int, len(petalLength))
	floats.Argsort(petalLength, inds)

	//分位数
	quant25 := stat.Quantile(0.25, stat.Empirical, petalLength, nil)
	quant50 := stat.Quantile(0.50, stat.Empirical, petalLength, nil)
	quant75 := stat.Quantile(0.75, stat.Empirical, petalLength, nil)

	fmt.Printf("\nSepal Length Summary Statistics:\n")
	fmt.Printf("Max value: %0.2f\n", maxVal)
	fmt.Printf("Min value: %0.2f\n", minVal)
	fmt.Printf("Range value: %0.2f\n", rangeVal)
	fmt.Printf("Variance value: %0.2f\n", varianceVal)
	fmt.Printf("Std Dev value: %0.2f\n", stdDevVal)
	fmt.Printf("25 Quantile: %0.2f\n", quant25)
	fmt.Printf("50 Quantile: %0.2f\n", quant50)
	fmt.Printf("75 Quantile: %0.2f\n", quant75)
}
