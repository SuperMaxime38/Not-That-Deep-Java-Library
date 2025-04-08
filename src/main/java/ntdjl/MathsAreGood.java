package ntdjl;

import org.ejml.simple.SimpleMatrix;

public class MathsAreGood {
	

//	public static SimpleMatrix sigmoid(SimpleMatrix input) {
//	    DMatrixRMaj data = input.getDDRM();
//	    DMatrixRMaj resultData = new DMatrixRMaj(data.numRows, data.numCols);
//	
//	    double[] inputData = data.data;
//	    double[] resultArray = resultData.data;
//	    int length = inputData.length;
//	
//	    for (int i = 0; i < length; i++) {
//	        resultArray[i] = 1.0 / (1.0 + Math.exp(-inputData[i]));
//	    }
//	
//	    System.out.println("resultData " + resultData);
//	    return SimpleMatrix.wrap(resultData);
//	}
	
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


}
