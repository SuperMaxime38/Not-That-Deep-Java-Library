package ntdjl;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

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
	
	public SimpleMatrix feed_forward(SimpleMatrix input) {
		SimpleMatrix output = input;
	    for (Layer layer : layers) {
	        output = layer.forward(output);
	        System.out.println("output: "+output);
	    }
	    
	    return output;
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
}
