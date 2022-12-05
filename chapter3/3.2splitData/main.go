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

	join := path.Join(dir, "machineLearning", "diabetes.csv")
	open, err := os.Open(join)
	if err != nil {
		log.Fatal(err)
	}
	defer open.Close()
	diabetesDF := dataframe.ReadCSV(open)

	//80%训练集 20%测试集
	trainingNum := (4 * diabetesDF.Nrow()) / 5
	testNum := diabetesDF.Nrow() / 5

	if trainingNum+testNum < diabetesDF.Nrow() {
		trainingNum++
	}

	//存放下标
	trainingIdx := make([]int, trainingNum)
	testIdx := make([]int, testNum)

	for i := 0; i < trainingNum; i++ {
		trainingIdx[i] = i
	}

	for i := 0; i < testNum; i++ {
		testIdx[i] = i + trainingNum
	}

	//取子集
	trainingDF := diabetesDF.Subset(trainingIdx)
	testDF := diabetesDF.Subset(testIdx)

	setMap := map[int]dataframe.DataFrame{
		0: trainingDF,
		1: testDF,
	}

	//划分训练集和测试集
	for idx, setName := range []string{"training.csv", "test.csv"} {
		f, err := os.Create(path.Join(path.Dir(join), setName))
		if err != nil {
			log.Fatal(err)
		}
		writer := bufio.NewWriter(f)

		//写入csv文件
		if err := setMap[idx].WriteCSV(writer); err != nil {
			log.Fatal(err)
		}
	}
}
