package ntdjl;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

import ntdjl.utils.ActivationFunction;
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

	        // Calcul du coût
	        double loss = this.cost(yHat, Y);

	        // Backpropagation (gère aussi la mise à jour des poids)
	        this.backpropagate(A0, Y, learningRate);
	        
	        if (epoch % 100 == 0) {
	            System.out.println("Époque " + epoch + " | Coût: " + loss);
	        }
	    }
	}
	
	public SimpleMatrix predict(SimpleMatrix input) {
	    Pair result = this.feed_forward(input);
	    return (SimpleMatrix) result.getA(); // La sortie finale (yHat)
	}
	
	public int predictClass(SimpleMatrix input) {
	    SimpleMatrix output = this.predict(input);
	    
	    int maxIndex = 0;
	    double maxValue = output.get(0);

	    for (int i = 1; i < output.getNumElements(); i++) {
	        if (output.get(i) > maxValue) {
	            maxValue = output.get(i);
	            maxIndex = i;
	        }
	    }

	    return maxIndex;
	}
	
	public void save(String filename) throws IOException {
	    try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename))) {
	        out.writeInt(layers.size());
	        for (Layer layer : layers) {
	            out.writeObject(layer.activ); // enum
	            out.writeObject(layer.weights.getDDRM());
	            out.writeObject(layer.bias.getDDRM());
	        }
	    }
	}
	
	public void load(String filename) throws IOException, ClassNotFoundException {
	    layers.clear();

	    try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename))) {
	        int numLayers = in.readInt();
	        for (int i = 0; i < numLayers; i++) {
	            ActivationFunction activ = (ActivationFunction) in.readObject();
	            DMatrixRMaj weights = (DMatrixRMaj) in.readObject();
	            DMatrixRMaj bias = (DMatrixRMaj) in.readObject();

	            Layer layer = new Layer(0, 0, activ); // tailles dummy
	            layer.weights = new SimpleMatrix(weights);
	            layer.bias = new SimpleMatrix(bias);
	            layer.activ = activ;

	            layers.add(layer);
	        }
	    }
	}
	
	public NN clone() {
	    NN copy = new NN();
	    for (Layer layer : this.layers) {
	        copy.addLayer(layer.clone());
	    }
	    return copy;
	}
	
	public void mutate(double rate) {
	    for (Layer layer : layers) {
	        layer.mutate(rate);
	    }
	}

}
