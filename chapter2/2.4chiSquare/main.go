package main

import (
	"fmt"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

// 对某地居民的调查显示,所有居民中有60%没有经常锻炼 , 25%是零星锻炼, 15%经常锻炼.在做了一些建模并提供一些社区服务之后,调查重复了相同的问题.
// 跟踪调查由500名居民完成,结果如下:
// 没有经常锻炼:260
// 零星锻炼:135
// 经常锻炼:105
// 总数:500
// 现在 要确定是否有证据表明居民的反应在统计上有重大的转变
func main() {
	//观测值
	observed := []float64{48, 52}
	//预期值
	expected := []float64{50, 50}

	//计算卡方统计量
	chiSquare := stat.ChiSquare(observed, expected)
	fmt.Println(chiSquare)

	observed1 := []float64{260.0, 135.0, 105.0}
	totalObserved1 := 500.0
	expected1 := []float64{totalObserved1 * 0.60, totalObserved1 * 0.25, totalObserved1 * 0.15}
	chiSquare1 := stat.ChiSquare(observed1, expected1)
	fmt.Printf("\nChi-square: %0.2f\n", chiSquare1)

	//? create a Chi-squared distribution with K degrees of freedom.
	//in this case we have K=3-1=2 , because the degrees of freedom
	//for a Chi-squared distribution is the number of possible
	//categories minus one.
	chiDist := distuv.ChiSquared{
		K:   2.0,
		Src: nil,
	}

	//计算p值
	pValue := chiDist.Prob(chiSquare1)
	fmt.Printf("p-value: %0.4f\n\n", pValue)
}
