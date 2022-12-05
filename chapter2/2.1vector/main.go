package main

import (
	"fmt"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func main() {
	myVector := mat.NewVecDense(2, []float64{11.0, 5.2})
	fmt.Println(myVector)

	vectorA := []float64{11.0, 5.2, -1.3}
	vectorB := []float64{-7.2, 4.2, 5.1}

	//dot 向量积
	dotProduct := floats.Dot(vectorA, vectorB)
	fmt.Printf("The dot product of A and B is: %0.2f\n", dotProduct)

	//scale 向量缩放
	floats.Scale(1.5, vectorA)
	fmt.Printf("Scaling A by 1.5 gives: %v\n", vectorA)

	//norm 向量的范数
	normB := floats.Norm(vectorB, 2)
	fmt.Printf("The norm/length of B is: %0.2f\n", normB)

	vectorC := mat.NewVecDense(3, []float64{11.0, 5.2, -1.3})
	vectorD := mat.NewVecDense(3, []float64{-7.2, 4.2, 5.1})

	//向量积
	dot := mat.Dot(vectorC, vectorD)
	fmt.Printf("The dot product of A and B is: %0.2f\n", dot)

	//向量缩放
	vectorC.ScaleVec(1.5, vectorC)
	fmt.Printf("Scaling C by 1.5 gives: %v\n", vectorC)

	//向量范数
	normD := blas64.Nrm2(vectorD.RawVector())
	fmt.Printf("The norm/length of B is: %0.2f\n", normD)
}
