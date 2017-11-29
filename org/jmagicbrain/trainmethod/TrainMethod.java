package org.jmagicbrain.trainmethod;

import org.jmagicbrain.NeuralNetwork;
import org.jmagicbrain.functions.ActivationFunction;
import org.jmagicbrain.functions.ErrorFunction;
import java.util.List;

public abstract class TrainMethod
{
	
	protected NeuralNetwork neuralNetwork;
	protected List<List<Double>> trainingSet;
	protected List<List<Double>> expectedOutput;
	
	protected ErrorFunction errorFunction;

	protected final int maxEpochs;
	protected final double maxError;
	
	public TrainMethod(int maxEpochs, double maxError, List<List<Double>> trainingSet, List<List<Double>> expectedOutput, ErrorFunction errorFunction){
		this.setExpectedOutput(expectedOutput);
		this.setTrainingSet(trainingSet);
		this.errorFunction = errorFunction;
		this.maxEpochs = maxEpochs;
		this.maxError = maxError;
	}
	
	public void setTrainingSet(List<List<Double>> trainingSet){
		
		this.trainingSet = trainingSet;
	}
	
	public void setExpectedOutput(List<List<Double>> expectedOutput){
		
		this.expectedOutput = expectedOutput;
	}
	
	public List<List<Double>> getTrainingSet(){
		return this.trainingSet;
	}

	public List<List<Double>> getExpectedOutput(){
		return this.expectedOutput;
	}
	
	public void setNeuralNetwork(NeuralNetwork neuralNetwork){
		this.neuralNetwork = neuralNetwork;
	}
	
	public abstract void train();

}