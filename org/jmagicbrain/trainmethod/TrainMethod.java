package org.jmagicbrain.trainmethod;

import org.jmagicbrain.NeuralNetwork;
import org.jmagicbrain.functions.ErrorFunction;

public abstract class TrainMethod
{
	
	protected NeuralNetwork neuralNetwork;
	protected double[][] trainingSet;
	protected double[][] expectedOutput;
	
	protected ErrorFunction errorFunction;
	
	public TrainMethod(double[][] trainingSet, double[][] expectedOutput, ErrorFunction errorFunction){
		this.setExpectedOutput(expectedOutput);
		this.setTrainingSet(trainingSet);
		this.errorFunction = errorFunction;
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