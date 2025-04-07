package ntdjl;

import org.ejml.simple.SimpleMatrix;

import ntdjl.utils.ActivationFunction;

public class MainTest {
	
	public static void main(String[] args) {
		int[] n = {2, 3, 3, 1};
		
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
		
		SimpleMatrix output = model.feed_forward(A0);
		System.out.println("Output: " + output.toString());
	}
}
