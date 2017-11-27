package org.jmagicbrain;

import org.jmagicbrain.exceptions.InvalidNeuralNetworkArguments;
import org.jmagicbrain.functions.ActivationFunction;
import org.jmagicbrain.functions.ErrorFunction;
import org.jmagicbrain.initializers.WeightInitializer;

import java.util.LinkedList;

public class NeuralNetwork {

    private double[][] layers;
    private double[][][] weightsMatrix;
    private double numberOfWeights;
    private WeightInitializer initializer;

    private NeuralNetwork(WeightInitializer weightInitializer, ErrorFunction errorFunction, ActivationFunction activationFunction, Object trainingMethod, Integer ... layersArray){
        this.layers = new double[layersArray.length][];
        this.weightsMatrix = new double[layersArray.length - 1][][];

        for(int i = 0; i < layersArray.length; i++){
            this.layers[i] = new double[layersArray[i] + 1];
            layers[i][layers[i].length - 1] = 1;
            if(i > 0){
                weightsMatrix[i-1] = new double[this.layers[i].length - 1][this.layers[i - 1].length];
            }
        }

        double initialLimit = 0;

        for (int i = 0; i < weightsMatrix.length; i++){
            for(int j = 0; j < weightsMatrix[i].length; j++){
                // TODO: Define a initMethod
                initialLimit = Math.sqrt(6.0 /(layersArray[i] + layersArray[i + 1]));
                for(int k = 0; k < weightsMatrix[i][j].length - 1; k++){
                    weightsMatrix[i][j][k] = (Math.random() % ((initialLimit * 2) + 1)) - initialLimit;
                }
            }
        }
    }

    public double[][][] getWeights(){
        return weightsMatrix;
    }

    public class NeuralNetworkBuilder {
        private ErrorFunction errorFunction;
        private ActivationFunction activationFunction;
        private Object trainingMethod;
        private LinkedList<Integer> neuronQueue;
        private WeightInitializer initializer;

        public NeuralNetworkBuilder(){
            neuronQueue = new LinkedList<Integer>();
        }

        public NeuralNetworkBuilder setErrorFunction(ErrorFunction errorFunction) {
            this.errorFunction = errorFunction;
            return this;
        }

        public NeuralNetworkBuilder setActivationFunction(ActivationFunction activationFunction){
            this.activationFunction = activationFunction;
            return this;
        }

        public NeuralNetworkBuilder setTrainingMethod(){
            return this;
        }

        public NeuralNetworkBuilder addLayer(int numberOfNeurons){
            this.neuronQueue.add(numberOfNeurons);
            return this;
        }

        public NeuralNetworkBuilder setWeightInitializer(WeightInitializer initializer){
            this.initializer = initializer;
            return this;
        }

        public NeuralNetwork build() throws InvalidNeuralNetworkArguments{
            Integer[] layers = new Integer[neuronQueue.size()];
            neuronQueue.toArray(layers);
            if(layers.length == 0){
                throw new InvalidNeuralNetworkArguments("There are no layers");
            }
            if(errorFunction == null){
                throw new InvalidNeuralNetworkArguments("Error function not set");
            }
            if(activationFunction == null){
                throw  new InvalidNeuralNetworkArguments("Activation function not set");
            }
            if(trainingMethod == null){
                throw new InvalidNeuralNetworkArguments("Training method not set");
            }
            if(initializer == null){
                throw new InvalidNeuralNetworkArguments("Weight initializer not set");
            }
            return new NeuralNetwork(initializer, errorFunction, activationFunction, trainingMethod, layers);
        }
    }
}
