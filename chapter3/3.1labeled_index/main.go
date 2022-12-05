package main

import (
	"encoding/csv"
	"fmt"
	"io"
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

	join := path.Join(dir, "machineLearning", "labeled.csv")
	open, err := os.Open(join)
	if err != nil {
		log.Fatal(err)
	}
	defer open.Close()

	reader := csv.NewReader(open)

	var observed []int
	var predicted []int
	line := 1
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}

		if line == 1 {
			line++
			continue
		}

		observedVal, err := strconv.Atoi(record[0])
		if err != nil {
			log.Printf("Parsing line %d failed, unexpected type\n", line)
			continue
		}

		predictedVal, err := strconv.Atoi(record[1])
		if err != nil {
			log.Printf("Parsing line %d failed, unexpected type\n", line)
			continue
		}

		observed = append(observed, observedVal)
		predicted = append(predicted, predictedVal)
		line++
	}

	//猜测是+ 结果是+
	//猜测是- 结果是-
	//TP+TN
	var truePosNeg int

	for idx, oVal := range observed {
		if oVal == predicted[idx] {
			truePosNeg++
		}
	}

	//准确度 = (TP+TN)/(TP+TN+FP+FN) 预测正确的百分比
	//精确度 = TP/(TP+FP)  阳性预测中实际阳性的比率
	//召回率 = TP/(TP+FN)  阳性预测中被标注为阳性的比率

	//准确度计算
	accuracy := float64(truePosNeg) / float64(len(observed))
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)

	//精确度和召回率计算
	classes := []int{0, 1, 2}

	for _, class := range classes {
		var truePos int
		var falsePos int
		var falseNeg int
		for idx, oVal := range observed {
			switch oVal {
			case class:
				if predicted[idx] == class {
					truePos++
					continue
				}
				falseNeg++

			default:
				if predicted[idx] == class {
					falsePos++
				}
			}
		}

		//精确度计算
		precision := float64(truePos) / float64(truePos+falsePos)

		//召回率计算
		recall := float64(truePos) / float64(truePos+falseNeg)
		fmt.Printf("\nPrecision (class %d) = %0.2f", class, precision)
		fmt.Printf("\nRecall (class %d) = %0.2f", class, recall)
	}
}
