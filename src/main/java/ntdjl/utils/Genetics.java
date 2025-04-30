package ntdjl.utils;

import java.util.ArrayList;
import java.util.List;

import ntdjl.Layer;
import ntdjl.NN;

public class Genetics {
	
	public static List<NN> createRandomModels(int[] strcuture, ActivationFunction fn, int amount) {
		List<NN> models = new ArrayList<NN>();
		
		for(int i = 0; i < amount; i++) {
			NN model = new NN();
			
			for(int j = 0; j < strcuture.length - 1; j++) {
				model.addLayer(new Layer(strcuture[j], strcuture[j+1], fn));
			}
			
			models.add(model);
		}
		
		return models;
	}
	
	public static List<NN> cloneModel(NN model, int amount) {
		List<NN> models = new ArrayList<NN>();
		
		for(int i = 0; i < amount; i++) {
			models.add(model.clone());
		}
		
		return models;
	}
}
