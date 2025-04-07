package ntdjl;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class MainTest {
	
	public static void main(String[] args) {
		int[] n = {2, 3, 3, 1};
		
		System.out.println("layer 0: " + n[0]);
		System.out.println("layer 1: " + n[1]);
		System.out.println("layer 2: " + n[2]);
		System.out.println("layer 3: " + n[3]);
		
		Random rdm = new Random();
		
		SimpleMatrix W1 = SimpleMatrix.random(n[1], n[0]);
		SimpleMatrix b1 = SimpleMatrix.random(n[1], 1);
		SimpleMatrix W2 = SimpleMatrix.random(n[2], n[1]);
		SimpleMatrix b2 = SimpleMatrix.random(n[2], 1);
		SimpleMatrix W3 = SimpleMatrix.random(n[3], n[2]);
		SimpleMatrix b3 = SimpleMatrix.random(n[3], 1);
		
		
		System.out.println("W1: " + W1.toString());
		System.out.println("b1: " + b1.toString());
		System.out.println("W2: " + W2.toString());
		System.out.println("b2: " + b2.toString());
		System.out.println("W3: " + W3.toString());
		System.out.println("b3: " + b3.toString());
		
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
		
		System.out.println("A0: " + A0.toString());
		System.out.println("W1: " + W1.toString());
		System.out.println("b1: " + b1.toString());
		
		SimpleMatrix Z1 = W1.mult(A0).plus(b1);
		
		SimpleMatrix A1 = MathAreGoods.sigmoid(Z1);
		
		SimpleMatrix Z2 = W2.mult(A1).plus(b2);
		
		SimpleMatrix A2 = MathAreGoods.sigmoid(Z2);
		
		SimpleMatrix Z3 = W3.mult(A2).plus(b3);
		
		SimpleMatrix A3 = MathAreGoods.sigmoid(Z3);
		
		System.out.println("A3: " + A3.toString());
		
//		double[] test = {
//			-99,
//			1,
//			2,
//			3,
//			99
//		};
//		
//		System.out.println(MathAreGoods.sigmoid(new SimpleMatrix(test)));
	}
}
