package main

import (
	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
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

	//初始化底片
	p := plot.New()

	//标题和y标签设置
	p.Title.Text = "Box plots"
	p.Y.Label.Text = "Values"

	//? create the box for our data
	w := vg.Points(50)

	//根据每个列名遍历
	for idx, colName := range irisDF.Names() {
		if colName != "species" {

			//创建收纳数据的切片 长度是数据列的长度
			v := make(plotter.Values, irisDF.Nrow())

			//拿到数据列的数据 依次赋值
			for i, floatVal := range irisDF.Col(colName).Float() {
				v[i] = floatVal
			}

			//创建箱型图
			boxPlot, err := plotter.NewBoxPlot(w, float64(idx), v)
			if err != nil {
				log.Fatal(err)
			}

			//在底片上添加
			p.Add(boxPlot)
		}
	}

	//设置x标签
	p.NominalX("sepal_length", "sepal_width", "petal_length", "petal_width")

	//保存
	if err := p.Save(6*vg.Inch, 8*vg.Inch, "boxplots.png"); err != nil {
		log.Fatal(err)
	}

}
