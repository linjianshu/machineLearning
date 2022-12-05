package main

import (
	"flag"
	"log"
	"os"
	"path/filepath"
)

func main() {
	modeldir := flag.String("dir", "", "")
	imagefile := flag.String("image", "", "")
	flag.Parse()
	if *modeldir == "" || *imagefile == "" {
		flag.Usage()
		return
	}

	modelfile := filepath.Join(*modeldir, "tensorflow_inception_graph.pb")
	labelsfile := filepath.Join(*modeldir, "imagenet_comp_graph_label_strings.txt")

	model, err := os.ReadFile(modelfile)
	if err != nil {
		log.Fatal(err)
	}

	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

}
