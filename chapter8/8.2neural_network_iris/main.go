package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"log"
	"math"
	"math/rand"
	"os"
	"path"
	"strconv"
	"time"
)

func main() {
	dir, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	join := path.Join(dir, "machineLearning", "chapter8", "8.2neural_network_iris", "train.csv")
	open, err := os.Open(join)
	if err != nil {
		log.Fatal(err)
	}

	defer open.Close()

	reader := csv.NewReader(open)
	reader.FieldsPerRecord = 7
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	var inputsIndex int
	var labelsIndex int

	for idx, record := range rawCSVData {
		if idx == 0 {
			continue
		}

		for i, val := range record {
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)

	config := neuralNetConfig{
		inputNeurons:  4,
		outputNeurons: 3,
		hiddenNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.3,
	}

	network := newNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	//评估测试集
	join = path.Join(dir, "machineLearning", "chapter8", "8.2neural_network_iris", "test.csv")
	file, err := os.Open(join)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	newReader := csv.NewReader(file)
	newReader.FieldsPerRecord = 7

	records, err := newReader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	testInputsData := make([]float64, 4*len(records))
	testLabelsData := make([]float64, 3*len(records))
	testInputsIndex := 0
	testLabelsIndex := 0
	for idx, record := range records {
		//跳过表头
		if idx == 0 {
			continue
		}
		for i, val := range record {
			parseFloat, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			if i == 4 || i == 5 || i == 6 {
				testLabelsData[testLabelsIndex] = parseFloat
				testLabelsIndex++
				continue
			}

			testInputsData[testInputsIndex] = parseFloat
			testInputsIndex++
		}
	}

	testInputs := mat.NewDense(len(records), 4, testInputsData)
	testLabels := mat.NewDense(len(records), 3, testLabelsData)

	predicts, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}
	var truePosNeg int
	numPreds, _ := predicts.Dims()
	for i := 0; i < numPreds; i++ {
		//获得标签行
		labelRow := mat.Row(nil, i, testLabels)
		//确定品种
		var species int
		for idx, label := range labelRow {
			if label == 1.0 {
				species = idx
				break
			}
		}

		//要从预测中取最大值
		if predicts.At(i, species) == floats.Max(mat.Row(nil, i, predicts)) {
			truePosNeg++
		}
	}

	accuracy := float64(truePosNeg) / float64(numPreds)
	fmt.Printf("\nAccuracy = % 0.2f\n\n", accuracy)

}

func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {
	if nn.wHidden == nil || nn.wOut == nil || nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("the supplied neural net weights and biases are empty")
	}

	r, _ := x.Dims()
	_, c := nn.wHidden.Dims()
	hiddenLayerInput := mat.NewDense(r, c, nil)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	r, c = hiddenLayerInput.Dims()
	hiddenLayerActivations := mat.NewDense(r, c, nil)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	r, _ = hiddenLayerActivations.Dims()
	_, c = nn.wOut.Dims()
	outputLayerInput := mat.NewDense(r, c, nil)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBout := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBout, outputLayerInput)

	r, _ = x.Dims()
	_, c = nn.wOut.Dims()
	output := mat.NewDense(r, c, nil)
	outputLayerInput.Apply(applySigmoid, outputLayerInput)
	return output, nil
}

type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

type neuralNet struct {
	config neuralNetConfig
	//网络隐藏层的权重
	wHidden *mat.Dense
	//网络隐藏层的偏差
	bHidden *mat.Dense
	//网络输出层的权重
	wOut *mat.Dense
	//网络输出层的偏差
	bOut *mat.Dense
}

// 初始化一个神经网络
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// 基于输入矩阵和输出矩阵 训练神经网络 完成反向传播 并将产生的训练权重和偏差放入接收器nn中
func (nn *neuralNet) train(x, y *mat.Dense) error {
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	//初始化网络隐藏层的权重矩阵 有4个输入节点 3个隐藏层节点 那么就是4*3的矩阵
	wHiddenRaw := make([]float64, nn.config.hiddenNeurons*nn.config.inputNeurons)
	//初始化网络隐藏层的偏差矩阵 就是网络隐藏层的节点个数 4个隐藏层节点 那么就是计算出4个结果 就会有4个偏差
	bHiddenRaw := make([]float64, nn.config.hiddenNeurons)

	//初始化网络输出层的权重矩阵 有2个输出节点 3个隐藏层节点 那么就是2*3的矩阵
	wOutRaw := make([]float64, nn.config.hiddenNeurons*nn.config.outputNeurons)
	//初始化网络输出层的偏差矩阵 有2个输出节点 那么就是计算出2个结果 就会有2个偏差
	bOutRaw := make([]float64, nn.config.outputNeurons)

	for _, param := range [][]float64{wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw} {
		for i := range param {
			//使用随机数发生器初始化矩阵参数
			param[i] = randGen.Float64()
		}
	}

	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, wHiddenRaw)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, bHiddenRaw)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, wOutRaw)
	bOut := mat.NewDense(1, nn.config.outputNeurons, bOutRaw)

	//定义输出矩阵
	r, c := y.Dims()
	output := mat.NewDense(r, c, nil)

	//遍历每个阶段 完成反向传播
	//计算输出的前馈阶段 应用权重 偏差变化
	for i := 0; i < nn.config.numEpochs; i++ {
		//1.complete the feed forward process
		//定义隐藏层输入
		r, _ = x.Dims()
		_, c = wHidden.Dims()
		hiddenLayerInput := mat.NewDense(r, c, nil)
		//隐藏层输入 = 输入层输入矩阵x * 隐藏层权重
		hiddenLayerInput.Mul(x, wHidden)

		//隐藏层的偏差调整 变量v添加到隐藏层的偏差矩阵的第一行的某列中
		addBHidden := func(_, col int, v float64) float64 {
			return v + bHidden.At(0, col)
		}

		//应用进去 隐藏层的输入+=隐藏层的偏差
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		//定义激活矩阵
		r, c = hiddenLayerInput.Dims()
		hiddenLayerActivations := mat.NewDense(r, c, nil)

		//定义激活函数
		applySigmoid := func(_, _ int, v float64) float64 {
			return sigmoid(v)
		}
		//应用激活函数 影响的是隐藏层的输入矩阵
		//激活矩阵 = 隐藏层输入 * 激活函数(引入非线性)
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		//定义输出层的输入
		r, _ = hiddenLayerActivations.Dims()
		_, c = wOut.Dims()

		outputLayerInput := mat.NewDense(r, c, nil)
		//输出层输入 = 隐藏层激活矩阵 * 输出层权重
		outputLayerInput.Mul(hiddenLayerActivations, wOut)

		//输出层的偏差调整 变量v添加到输出层的偏差矩阵的第一行的某列中
		addBout := func(_, col int, v float64) float64 {
			return v + bOut.At(0, col)
		}
		//应用进去 输出层的输入+=输出层的偏差
		outputLayerInput.Apply(addBout, outputLayerInput)

		//应用激活函数 影响的是输出层的输入矩阵
		output.Apply(applySigmoid, outputLayerInput)

		//2.complete the backpropagation
		//计算整体的输出偏差
		r, c = y.Dims()
		netWorkError := mat.NewDense(r, c, nil)
		netWorkError.Sub(y, output)

		//计算输出层的偏差的导数 △y/△x = prime
		r, c = output.Dims()
		slopeOutputLayer := mat.NewDense(r, c, nil)
		applySigmoidPrime := func(_, _ int, v float64) float64 {
			return sigmoidPrime(v)
		}
		slopeOutputLayer.Apply(applySigmoidPrime, output)

		//计算隐藏层输出的偏差的导数
		r, c = hiddenLayerActivations.Dims()
		slopeHiddenLayer := mat.NewDense(r, c, nil)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		//计算输出层的增量
		r, c = slopeOutputLayer.Dims()
		dOutput := mat.NewDense(r, c, nil)
		dOutput.MulElem(netWorkError, slopeOutputLayer)

		//计算隐藏层的偏差
		r, _ = dOutput.Dims()
		_, c = wOut.T().Dims()
		errorAtHiddenLayer := mat.NewDense(r, c, nil)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		//计算隐藏层的增量
		r, c = slopeHiddenLayer.Dims()
		dHiddenLayer := mat.NewDense(r, c, nil)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		//3.利用增量更新网络的权重和偏差
		//adjust the parameters
		r, _ = hiddenLayerActivations.T().Dims()
		_, c = dOutput.Dims()
		wOutAdj := mat.NewDense(r, c, nil)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		//根据学习率调整步长
		wOutAdj.Scale(nn.config.learningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		r, _ = x.T().Dims()
		_, c = dHiddenLayer.Dims()
		wHiddenAdj := mat.NewDense(r, c, nil)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut
	return nil
}

func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()
	var output *mat.Dense
	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)

	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)

	default:
		return nil, errors.New("invalid axis , must be 0 or 1")
	}
	return output, nil
}

func sigmoidPrime(v float64) float64 {
	return v * (1.0 - v)
}
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
