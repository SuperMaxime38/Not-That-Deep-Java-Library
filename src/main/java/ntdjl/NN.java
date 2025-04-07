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
	    }
	    
	    return output;
	}
}
