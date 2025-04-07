package ntdjl;

import org.ejml.simple.SimpleMatrix;

public class Layer {
	
	SimpleMatrix weights;
	SimpleMatrix bias;
	
	public Layer(int inputSize, int outputSize) {
		this.weights = SimpleMatrix.random(outputSize, inputSize);
		this.bias = SimpleMatrix.random(outputSize, 1);
	}
	
	public SimpleMatrix forward(SimpleMatrix input) {
		return MathAreGoods.calculateLayerValue(this.weights, input, this.bias);
	}
	
	public SimpleMatrix getWeights() {
		return this.weights;
	}
	
	public SimpleMatrix getBias() {
		return this.bias;
	}
}
