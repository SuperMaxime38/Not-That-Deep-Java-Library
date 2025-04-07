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
	
	    return SimpleMatrix.wrap(resultData);
	}
	
	public SimpleMatrix calculateLayerValue(SimpleMatrix W, SimpleMatrix A_prev, SimpleMatrix b) {
	    SimpleMatrix Z = W.mult(A_prev);

	    int numRows = Z.getNumRows();
	    int numCols = Z.getNumCols();

	    for (int col = 0; col < numCols; col++) {
	        for (int row = 0; row < numRows; row++) {
	            Z.set(row, col, Z.get(row, col) + b.get(row, 0));
	        }
	    }

	    return Z;
	}

}
