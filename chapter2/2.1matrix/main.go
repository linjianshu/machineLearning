package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"log"
	"math"
)

func main() {
	data := []float64{1.2, -5.7, -2.4, 7.3}

	//init matrix 初始化
	a := mat.NewDense(2, 2, data)

	//format 格式化
	fa := mat.Formatted(a, mat.Prefix(" "))
	fmt.Printf("mat = %v\n\n", fa)

	//取值
	val := a.At(0, 1)
	fmt.Printf("The value of a at (0,1) is: %.2f\n\n", val)

	//取列
	col := mat.Col(nil, 0, a)
	fmt.Printf("The values in the 1st column are: %v\n\n", col)

	//取行
	row := mat.Row(nil, 1, a)
	fmt.Printf("The values in the 2nd row are: %v\n\n", row)

	//设置值
	a.Set(0, 1, 11.2)
	fmt.Println(mat.Formatted(a, mat.Prefix(" ")))

	//设置行
	a.SetRow(0, []float64{14.3, -4.2})
	fmt.Println(mat.Formatted(a, mat.Prefix(" ")))

	//设置列
	a.SetCol(0, []float64{1.7, -0.3})
	fmt.Println(mat.Formatted(a, mat.Prefix(" ")))

	c := mat.NewDense(3, 3, []float64{1, 2, 3, 0, 4, 5, 0, 0, 6})
	d := mat.NewDense(3, 3, []float64{8, 9, 10, 1, 4, 2, 9, 0, 2})
	e := mat.NewDense(3, 2, []float64{3, 2, 1, 4, 0, 8})

	//书中是0 0 nil 参数目前不匹配 需要指定行列
	//f := mat.NewDense(0, 0, nil)

	//初始化res
	f := mat.NewDense(3, 3, nil)
	//c+d
	f.Add(c, d)
	fd := mat.Formatted(f, mat.Prefix("       "))
	fmt.Printf("f = c + d = %0.4v\n\n", fd)

	//初始化res
	g := mat.NewDense(3, 2, nil)
	//c*e
	g.Mul(c, e)
	gd := mat.Formatted(g, mat.Prefix("        "))
	fmt.Printf("g = c * d = %0.4v\n\n", gd)

	//初始化res
	h := mat.NewDense(3, 3, nil)
	//c^5
	h.Pow(c, 5)
	hd := mat.Formatted(h, mat.Prefix("        "))
	fmt.Printf("h = c ^ 5 = %0.4v\n\n", hd)

	//初始化res
	i := mat.NewDense(3, 3, nil)
	sqrt := func(_ int, _ int, v float64) float64 {
		return math.Sqrt(v)
	}

	//c^0.5  apply方法可以对矩阵元素应用任何函数 可以对下标选择性应用函数
	i.Apply(sqrt, c)
	id := mat.Formatted(i, mat.Prefix("        "))
	fmt.Printf("i = sqrt(c) = %0.4v\n\n", id)

	j := mat.NewDense(3, 3, []float64{1, 2, 3, 0, 4, 5, 0, 0, 6})
	//计算转置矩阵
	jd := mat.Formatted(j.T(), mat.Prefix(" "))
	fmt.Printf("j^T = %v\n\n", jd)

	//计算矩阵的行列式
	deta := mat.Det(j)
	fmt.Printf("det(j) = %.2f\n\n", deta)

	//计算逆矩阵
	jInverse := mat.NewDense(3, 3, nil)
	if err := jInverse.Inverse(j); err != nil {
		log.Fatal(err)
	}

	jid := mat.Formatted(jInverse, mat.Prefix(" "))
	fmt.Printf("j^-1 = %v\n\n", jid)

}
