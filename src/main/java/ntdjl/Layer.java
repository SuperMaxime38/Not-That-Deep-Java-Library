package ntdjl;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import ntdjl.utils.ActivationFunction;

public class Layer {
	
	SimpleMatrix weights;
	SimpleMatrix bias;
	ActivationFunction activ;
	
	public Layer(int inputSize, int outputSize, ActivationFunction fn) {
		Random rand = new Random();
        weights = SimpleMatrix.random_DDRM(outputSize, inputSize, -1.0, 1.0, rand);
        bias = SimpleMatrix.random_DDRM(outputSize, 1, -1.0, 1.0, rand);
		this.activ = fn;
		
		System.out.println("weights:\n" + weights);
		System.out.println("bias:\n" + bias);
	}
	
//	public SimpleMatrix forward(SimpleMatrix input) {
//		SimpleMatrix output = MathAreGoods.calculateLayerValue(this.weights, input, this.bias);
//		switch(this.activ) {
//		case SIGMOID:
//			output = MathAreGoods.sigmoid(output);
//			//System.out.println("called sigmoid");
//		}
//		return output;
//	}
	
	public SimpleMatrix forward(SimpleMatrix input) {
	    SimpleMatrix z = weights.mult(input);

	    // Broadcasting du biais
	    SimpleMatrix biasMatrix = new SimpleMatrix(bias.getNumRows(), input.getNumCols());
	    for (int i = 0; i < input.getNumCols(); i++) {
	        biasMatrix.insertIntoThis(0, i, bias);
	    }

	    z = z.plus(biasMatrix);

	    return MathAreGoods.sigmoid2(z);
	}
	
	public void backprop() {
		
	}
	
	public SimpleMatrix getWeights() {
		return this.weights;
	}
	
	public SimpleMatrix getBias() {
		return this.bias;
	}
}
