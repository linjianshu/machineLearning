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
	join := path.Join(dir, "machineLearning", "Advertising.csv")
	open, err := os.Open(join)
	if err != nil {
		log.Fatal(err)
	}
	defer open.Close()
	advertDF := dataframe.ReadCSV(open)
	advertSummary := advertDF.Describe()
	fmt.Println(advertSummary)

	for _, colName := range advertDF.Names() {
		v := make(plotter.Values, advertDF.Nrow())
		for i, floatVal := range advertDF.Col(colName).Float() {
			v[i] = floatVal
		}

		p := plot.New()
		p.Title.Text = fmt.Sprintf("Histogram of a %s", colName)
		
		hist, err := plotter.NewHist(v, 16)
		if err != nil {
			log.Fatal(err)
		}
		hist.Normalize(1)
		p.Add(hist)
		if err := p.Save(4*vg.Inch, 4*vg.Inch, colName+"_hist.png"); err != nil {
			log.Fatal(err)
		}
	}
}
