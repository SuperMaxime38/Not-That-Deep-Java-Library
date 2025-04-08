package ntdjl;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import ntdjl.utils.Pair;

public class NN {
	ArrayList<Layer> layers;
	
	public NN() {
		layers = new ArrayList<Layer>();
	}
	
	public void addLayer(Layer layer) {
		layers.add(layer);
	}
	
	public ArrayList<Layer> getLayers() {
		return layers;
	}
	
	public void removeLayer(Layer layer) {
		layers.remove(layer);
	}
	
	public Pair feed_forward(SimpleMatrix input) {
		SimpleMatrix output = input;
		ArrayList<SimpleMatrix> layers_outputs = new ArrayList<SimpleMatrix>();
	    for (Layer layer : layers) {
	        output = layer.forward(output);
	        layers_outputs.add(output);
	    }
	    
	    return new Pair(output, layers_outputs);
	}
	
	public double cost(SimpleMatrix yHat, SimpleMatrix y) {
	    // Différence entre la prédiction et la vérité
	    SimpleMatrix diff = yHat.minus(y);

	    // Élévation au carré
	    SimpleMatrix squared = diff.elementMult(diff);

	    // Moyenne de toutes les valeurs
	    double sum = squared.elementSum();
	    return sum / (y.getNumRows() * y.getNumCols());
	}
	
	public void backpropagate(SimpleMatrix X, SimpleMatrix Y, double learningRate) {
	    Pair result = feed_forward(X);
	    SimpleMatrix yHat = (SimpleMatrix) result.getA();
	    @SuppressWarnings("unchecked")
		ArrayList<SimpleMatrix> activations = (ArrayList<SimpleMatrix>) result.getB();

	    // Calcul du gradient du coût
	    SimpleMatrix dA = yHat.minus(Y); // pour MSE et sigmoid en sortie : dA = (yHat - y)

	    // On démarre depuis la dernière couche
	    for (int i = layers.size() - 1; i >= 0; i--) {
	        SimpleMatrix A_prev = (i == 0) ? X : activations.get(i - 1);
	        dA = layers.get(i).backward(dA, A_prev, learningRate);
	    }
	}
	
	public void train(SimpleMatrix A0, SimpleMatrix Y, int epochs, double learningRate) {
	    for (int epoch = 0; epoch < epochs; epoch++) {
	    	
	        // Forward pass
	        Pair result = this.feed_forward(A0);
	        SimpleMatrix yHat = (SimpleMatrix) result.getA();
//	        @SuppressWarnings("unchecked")
//	        ArrayList<SimpleMatrix> activations = (ArrayList<SimpleMatrix>) result.getB();

	        // Calcul du coût
	        double loss = this.cost(yHat, Y);
	        //System.out.println("Époque " + epoch + " | Coût: " + loss);

	        // Backpropagation (gère aussi la mise à jour des poids)
	        this.backpropagate(A0, Y, learningRate);
	        
	        if (epoch % 100 == 0) {
	            System.out.println("Époque " + epoch + " | Coût: " + loss);
	        }
	    }
	}
}
