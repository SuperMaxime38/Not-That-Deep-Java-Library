package ntdjl;

import org.ejml.simple.SimpleMatrix;

import ntdjl.utils.ActivationFunction;

public class Layer {
	
	SimpleMatrix weights;
	SimpleMatrix bias;
	ActivationFunction activ;
	
	public Layer(int inputSize, int outputSize, ActivationFunction fn) {
		this.weights = SimpleMatrix.random(outputSize, inputSize);
		this.bias = SimpleMatrix.random(outputSize, 1);
		this.activ = fn;
	}
	
	public SimpleMatrix forward(SimpleMatrix input) {
		SimpleMatrix output = MathAreGoods.calculateLayerValue(this.weights, input, this.bias);
		switch(this.activ) {
		case SIGMOID:
			output = MathAreGoods.sigmoid(output);
		}
		return output;
	}
	
	public SimpleMatrix getWeights() {
		return this.weights;
	}
	
	public SimpleMatrix getBias() {
		return this.bias;
	}
}
