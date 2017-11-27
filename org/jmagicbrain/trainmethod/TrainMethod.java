package org.jmagicbrain.trainmethod;

import org.jmagicbrain.NeuralNetwork;

public abstract class TrainMethod
{
	
	protected NeuralNetwork neuralNetwork;
	protected double[][] trainingSet;
	protected double[][] expectedOutput;
	
	public TrainMethod(double[][] trainingSet, double[][] expectedOutput){
		this.setExpectedOutput(expectedOutput);
		this.setTrainingSet(trainingSet);
	}
	
	public void setTrainingSet(double[][] trainingSet){
		
		this.trainingSet = trainingSet;
	}
	
	public void setExpectedOutput(double[][] expectedOutput){
		
		this.expectedOutput = expectedOutput;
	}
	
	public double[][] getTrainingSet(){
		return this.trainingSet;
	}
	
	public double[][] getExpectedOutput(){
		return this.expectedOutput;
	}
	
	public void setNeuralNetwork(NeuralNetwork neuralNetwork){
		this.neuralNetwork = neuralNetwork;
	}
	
	public abstract void train();

}