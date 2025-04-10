package ntdjl;

import org.ejml.simple.SimpleMatrix;

public class MathsAreGood {
	
	public static SimpleMatrix sigmoid(SimpleMatrix matrix) {
        SimpleMatrix result = new SimpleMatrix(matrix.getNumRows(), matrix.getNumCols());
        for (int i = 0; i < matrix.getNumRows(); i++) {
            for (int j = 0; j < matrix.getNumCols(); j++) {
                double val = matrix.get(i, j);
                result.set(i, j, 1.0 / (1.0 + Math.exp(-val)));
            }
        }
        return result;
    }
	
	public static SimpleMatrix sigmoidDerivative(SimpleMatrix z) {
	    SimpleMatrix s = sigmoid(z);
	    return s.elementMult(s.scale(-1).plus(1.0)); // s * (1 - s)
	}
	
	public static SimpleMatrix meanRows(SimpleMatrix matrix) {
	    SimpleMatrix result = new SimpleMatrix(matrix.getNumRows(), 1);
	    for (int i = 0; i < matrix.getNumRows(); i++) {
	        double sum = 0;
	        for (int j = 0; j < matrix.getNumCols(); j++) {
	            sum += matrix.get(i, j);
	        }
	        result.set(i, 0, sum / matrix.getNumCols());
	    }
	    return result;
	}
	public static SimpleMatrix relu(SimpleMatrix matrix) {
	    SimpleMatrix result = new SimpleMatrix(matrix.getNumRows(), matrix.getNumCols());
	    for (int i = 0; i < matrix.getNumRows(); i++) {
	        for (int j = 0; j < matrix.getNumCols(); j++) {
	            double val = matrix.get(i, j);
	            result.set(i, j, 0.5*(val + Math.abs(val)));
	        }
	    }
	    return result;
	}

	public static SimpleMatrix gelu(SimpleMatrix matrix) {
	    SimpleMatrix result = new SimpleMatrix(matrix.getNumRows(), matrix.getNumCols());
	    for (int i = 0; i < matrix.getNumRows(); i++) {
	        for (int j = 0; j < matrix.getNumCols(); j++) {
	            double val = matrix.get(i, j);

	            double gelu_value = 0.5*val*(1 + Math.tanh(Math.sqrt(2/Math.PI)*(val+0.044715*Math.pow(val, 3))));

	            result.set(i, j, 1.0 / gelu_value);
	        }
	    }
	    return result;
	}

	public static SimpleMatrix elu(SimpleMatrix matrix) {
	    SimpleMatrix result = new SimpleMatrix(matrix.getNumRows(), matrix.getNumCols());
	    for (int i = 0; i < matrix.getNumRows(); i++) {
	        for (int j = 0; j < matrix.getNumCols(); j++) {
	            double val = matrix.get(i, j);

	            double elu_value = val;

	            if(val < 0) {
	                elu_value = Math.exp(val) - 1;
	            }

	            result.set(i, j, elu_value);
	        }
	    }
	    return result;
	}
}
