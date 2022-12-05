package main

import (
	"fmt"
	"gonum.org/v1/gonum/integrate"
	"gonum.org/v1/gonum/stat"
)

func main() {
	scores := []float64{0.1, 0.35, 0.4, 0.8}
	classes := []bool{true, false, true, false}

	tpr, fpr, thresh := stat.ROC(nil, scores, classes, nil)
	fmt.Println(thresh)
	auc := integrate.Trapezoidal(fpr, tpr)

	fmt.Printf("true positive rate: %v\n", tpr)
	fmt.Printf("false positive rate: %v\n", fpr)
	fmt.Printf("auc: %v\n", auc)

}
