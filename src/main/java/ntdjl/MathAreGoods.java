package ntdjl;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

public class MathAreGoods {
	

	public static SimpleMatrix sigmoid(SimpleMatrix input) {
	    DMatrixRMaj data = input.getDDRM();
	    DMatrixRMaj resultData = new DMatrixRMaj(data.numRows, data.numCols);
	
	    double[] inputData = data.data;
	    double[] resultArray = resultData.data;
	    int length = inputData.length;
	
	    for (int i = 0; i < length; i++) {
	        resultArray[i] = 1.0 / (1.0 + Math.exp(-inputData[i]));
	    }
	
	    System.out.println("resultData " + resultData);
	    return SimpleMatrix.wrap(resultData);
	}
	
	public static SimpleMatrix sigmoid2(SimpleMatrix matrix) {
        SimpleMatrix result = new SimpleMatrix(matrix.getNumRows(), matrix.getNumCols());
        for (int i = 0; i < matrix.getNumRows(); i++) {
            for (int j = 0; j < matrix.getNumCols(); j++) {
                double val = matrix.get(i, j);
                result.set(i, j, 1.0 / (1.0 + Math.exp(-val)));
            }
        }
        return result;
    }
	
//	public static SimpleMatrix calculateLayerValue(SimpleMatrix W, SimpleMatrix A_prev, SimpleMatrix b) {
//		System.out.println("a:"+A_prev);
//		System.out.println("w:"+W);
//		System.out.println("b:"+b);
//	    SimpleMatrix Z = W.mult(A_prev);
//
//	    int numRows = Z.getNumRows();
//	    int numCols = Z.getNumCols();
//
//	    for (int col = 0; col < numCols; col++) {
//	        for (int row = 0; row < numRows; row++) {
//	        	//System.out.println("val b"+ b.get(row, 0));
//	            Z.set(row, col, Z.get(row, col) + b.get(row, 0));
//	        System.out.println("z row:"+Z.getRow(row));
//	        }
//	    }
//
//	    return Z;
//	}


}
