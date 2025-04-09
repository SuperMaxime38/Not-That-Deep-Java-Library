package ntdjl;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import ntdjl.utils.ActivationFunction;

public class Layer {
	
	SimpleMatrix weights;
	SimpleMatrix bias;
	ActivationFunction activ;
	
	// Pour stocker temporairement l’entrée (z) et la sortie (a)
    public SimpleMatrix lastZ, lastA;
	
	public Layer(int inputSize, int outputSize, ActivationFunction fn) {
		Random rand = new Random();
        weights = SimpleMatrix.random_DDRM(outputSize, inputSize, -1.0, 1.0, rand);
        bias = SimpleMatrix.random_DDRM(outputSize, 1, -1.0, 1.0, rand);
		this.activ = fn;
	}
	
	public SimpleMatrix forward(SimpleMatrix input) {
	    SimpleMatrix z = weights.mult(input);

	    // Broadcasting du biais
	    SimpleMatrix biasMatrix = new SimpleMatrix(bias.getNumRows(), input.getNumCols());
	    for (int i = 0; i < input.getNumCols(); i++) {
	        biasMatrix.insertIntoThis(0, i, bias);
	    }

	    z = z.plus(biasMatrix);

	    this.lastZ = z;
	    
	    applyActivationFunction();
	    
        return this.lastA;
	}
	
	// Applique la rétropropagation sur ce layer
    public SimpleMatrix backward(SimpleMatrix dA, SimpleMatrix A_prev, double learningRate) {
        SimpleMatrix dZ = dA.elementMult(MathsAreGood.sigmoidDerivative(this.lastA));
        SimpleMatrix dW = dZ.mult(A_prev.transpose()).divide(A_prev.getNumCols());
        SimpleMatrix dB = MathsAreGood.meanRows(dZ);
        SimpleMatrix dA_prev = weights.transpose().mult(dZ);

        // Met à jour les poids
        weights = weights.minus(dW.scale(learningRate));
        bias = bias.minus(dB.scale(learningRate));

        return dA_prev;
    }
    
    public void applyActivationFunction() {
		switch (this.activ) {
		case SIGMOID:
			this.lastA = MathsAreGood.sigmoid(this.lastZ);
			break;
		default:
			this.lastA = this.lastZ;
			break; // DEFAULT IS ALWAYS_ACTIVE
		}
	}
	
	
	public SimpleMatrix getWeights() {
		return this.weights;
	}
	
	public SimpleMatrix getBias() {
		return this.bias;
	}
}
