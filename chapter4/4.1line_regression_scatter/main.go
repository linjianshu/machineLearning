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

	join := path.Join(dir, "machineLearning", "Advertising.csv")
	open, err := os.Open(join)
	if err != nil {
		log.Fatal(err)
	}
	defer open.Close()

	advertDF := dataframe.ReadCSV(open)

	yVals := advertDF.Col("Sales").Float()

	for _, colName := range advertDF.Names() {
		pts := make(plotter.XYs, advertDF.Nrow())
		for i, floatVal := range advertDF.Col(colName).Float() {
			pts[i].X = floatVal
			pts[i].Y = yVals[i]
		}

		p := plot.New()
		p.X.Label.Text = colName
		p.Y.Label.Text = "y"
		p.Add(plotter.NewGrid())

		scatter, err := plotter.NewScatter(pts)
		if err != nil {
			log.Fatal(err)
		}

		scatter.GlyphStyle.Radius = vg.Points(3)
		p.Add(scatter)
		if err := p.Save(4*vg.Inch, 4*vg.Inch, colName+"_scatter.png"); err != nil {
			log.Fatal(err)
		}
	}
}
