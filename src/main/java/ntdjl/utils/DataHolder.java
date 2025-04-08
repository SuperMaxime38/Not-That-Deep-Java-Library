package ntdjl.utils;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

public class DataHolder {

	ArrayList<ArrayList<Double>> datas;
	SimpleMatrix mat;
	
	public DataHolder() {
		datas = new ArrayList<ArrayList<Double>>();
	}
	
	public void addData(ArrayList<Double> data) {
		datas.add(data);
	}
	
	public ArrayList<ArrayList<Double>> getDatas() {
		return datas;
	}
	
	public SimpleMatrix getMat() {
		return mat;
	}
	
	public void setMat(SimpleMatrix mat) {
		this.mat = mat;
	}
	
	public void print() {
		for (int i = 0; i < datas.size(); i++) {
			System.out.println(datas.get(i).toString());
		}
	}
	
	public void printMat() {
		System.out.println(mat.toString());
	}
	
	public void printMat(int row, int col) {
		System.out.println(mat.get(row, col));
	}
	
	public SimpleMatrix convertToMatrix() {
		int rows = datas.size();
		int cols = datas.get(0).size();
		
		double[][] table = new double[rows][cols];
		
		for(int x = 0; x < rows; x++) {
			for(int y = 0; y < cols; y++) {
				table[x][y] = datas.get(x).get(y);
			}
		}
		
		mat = new SimpleMatrix(table);
		return mat.transpose();
	}
}
