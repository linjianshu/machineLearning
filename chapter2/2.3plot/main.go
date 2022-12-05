package main

import (
	"fmt"
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

	//遍历数据帧的列名
	for _, colName := range irisDF.Names() {
		if colName != "species" {
			//初始化收纳数据的切片 长度是数据列的长度
			v := make(plotter.Values, irisDF.Nrow())
			//获取数据列 依次赋值
			for i, floatVal := range irisDF.Col(colName).Float() {
				v[i] = floatVal
			}

			//初始化底片
			p := plot.New()
			//底片标题
			p.Title.Text = fmt.Sprintf("Histogram of a %s", colName)

			//初始化直方图 16格
			h, err := plotter.NewHist(v, 16)
			if err != nil {
				log.Fatal(err)
			}

			//? Normalize the histogram.
			h.Normalize(1)

			//添加进底片
			p.Add(h)

			//保存底片
			if err := p.Save(4*vg.Inch, 4*vg.Inch, colName+"_hist.png"); err != nil {
				log.Fatal(err)
			}
		}
	}
}
