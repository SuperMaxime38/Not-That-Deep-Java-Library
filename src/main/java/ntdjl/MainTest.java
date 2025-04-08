package ntdjl;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import ntdjl.utils.ActivationFunction;
import ntdjl.utils.Pair;

public class MainTest {
	
	public static void main(String[] args) {
		int[] n = {2, 100, 50, 1};
		
		NN model = new NN();
		
		model.addLayer(new Layer(n[0], n[1], ActivationFunction.SIGMOID));
		model.addLayer(new Layer(n[1], n[2], ActivationFunction.SIGMOID));
		model.addLayer(new Layer(n[2], n[3], ActivationFunction.SIGMOID));
		
		
		double[][] x = {
			{150, 70},
			{254, 73},
			{312, 68},
			{120, 60},
			{154, 61},
			{212, 65},
			{216, 67},
			{145, 67},
			{184, 64},
			{130, 69}
		};
		
		SimpleMatrix A0 = new SimpleMatrix(x);
		A0 = A0.transpose();

		double[] y = {
			0,
			1,
			1,
			0,
			0,
			1,
			1,
			0,
			1,
			0
		};

		SimpleMatrix Y = new SimpleMatrix(y);
		int m = 10;
		
		Y.reshape(n[3], m);
		
		model.train(A0, Y, 1000, 0.1);
		
		Pair out = model.feed_forward(A0);
		System.out.println("Final res: " + out.getA());
		
		
		
//		int epochs = 10; // training for 1000 iterations
//		double alpha = 0.1; // set learning rate to 0.1
//		ArrayList<Double> costs = new ArrayList<Double>(); // list to store costs
//		
//		for(int e= 0; e<epochs; e++) {
//			Pair forward = model.feed_forward(A0);
//			SimpleMatrix output = (SimpleMatrix) forward.getA();
//			@SuppressWarnings("unchecked")
//			ArrayList<SimpleMatrix> layers_outputs = (ArrayList<SimpleMatrix>) forward.getB();
//			
//			double error = model.cost(output, Y);
//			costs.add(error);
//			
//			
//		}
		
//		Pair forward = model.feed_forward(A0);
//		SimpleMatrix output = (SimpleMatrix) forward.getA();
//		@SuppressWarnings("unchecked")
//		ArrayList<SimpleMatrix> layers_outputs = (ArrayList<SimpleMatrix>) forward.getB();
//		
//		System.out.println("Output: " + output.toString());
//		
//		double cost = model.cost(output, Y);
//		System.out.println("cost: " + cost);
	}
}
