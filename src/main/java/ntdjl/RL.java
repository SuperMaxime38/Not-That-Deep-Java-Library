package ntdjl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import ntdjl.utils.ActivationFunction;
import ntdjl.utils.Genetics;

public class RL {
	
	List<NN> agents;
	Iterator<NN> it;
	
	HashMap<NN, Integer> scores;
	
	int[] structure;
	ActivationFunction fn;
	
	int topOneClones, topTwoClones, topThreeClones;
	
	int batch;
	
	public RL(int[] structure, ActivationFunction fn, int amount) {
		this.agents = new ArrayList<NN>();
		this.structure = structure;
		this.fn = fn;
		
		this.agents = Genetics.createRandomModels(structure, fn, amount);
		
		this.scores = new HashMap<NN, Integer>();
	
		this.it = this.agents.iterator();

		this.topOneClones = 3;
		this.topTwoClones = 2;
		this.topThreeClones = 1;
		
		this.batch = 0;
	}
	
	public void nextGeneration() {
		NN top1 = null, top2 = null, top3 = null;
		int score1 = -1, score2 = -1, score3 = -1;
		
		for(NN model : scores.keySet()) {
			int score = scores.get(model);
			
			if(score > score1) {
				top1 = model;
				score1 = score;
			} else if(score > score2) {
				top2 = model;
				score2 = score;
			} else if(score > score3) {
				top3 = model;
				score3 = score;
			}
		}

		List<NN> clonesOne = Genetics.cloneModel(top1, this.topOneClones);
		List<NN> clonesTwo = Genetics.cloneModel(top2, this.topTwoClones);
		List<NN> clonesThree = Genetics.cloneModel(top3, this.topThreeClones);
		
		for(int a = 1; a < clonesOne.size(); a++) {
			NN clone = clonesOne.get(a);
			clone.mutate(0.01);
			clonesOne.set(a, clone);
		}
		
		for(int a = 1; a < clonesTwo.size(); a++) {
			NN clone = clonesTwo.get(a);
			clone.mutate(0.01);
			clonesTwo.set(a, clone);
		}
		
		for(int a = 1; a < clonesThree.size(); a++) {
			NN clone = clonesThree.get(a);
			clone.mutate(0.01);
			clonesThree.set(a, clone);
		}
		
		this.agents.clear();
		this.agents.addAll(clonesOne);
		this.agents.addAll(clonesTwo);
		this.agents.addAll(clonesThree);
		
		this.it = this.agents.iterator();
		
		this.batch++;
	}
	
	public List<NN> getAgents() {
		return this.agents;
	}
	
	public NN nextAgent() {
		return it.next();
	}
	
	public int getScore(NN model) {
		if(this.scores.containsKey(model)) return this.scores.get(model);
		return -1;
	}
	
	public void setScore(NN model, int score) {
		scores.put(model, score);
	}
	
	public void setTopClones(int topOne, int topTwo, int topThree) {
		if((topOne+topTwo+topThree) != agents.size()) {
			System.err.println("WARNING: The amount of clones are not the same as the amount of agents");
		}
		
		this.topOneClones = topOne;
		this.topTwoClones = topTwo;
		this.topThreeClones = topThree;
	}
}
