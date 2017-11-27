package org.jmagicbrain;

import org.jmagicbrain.exceptions.InvalidNeuralNetworkArguments;
import org.jmagicbrain.functions.ActivationFunction;
import org.jmagicbrain.functions.ErrorFunction;

import java.util.LinkedList;

public class NeuralNetwork {

    private double[][] layers;
    private double[][][] weightsMatrix;
    private double numberOfWeights;

    private NeuralNetwork(ErrorFunction errorFunction, ActivationFunction activationFunction, Object trainingMethod, Integer ... layers){
        // TODO: Body
    }

    public double[][][] getWeights(){
        return weightsMatrix;
    }

    public class NeuralNetworkBuilder {
        private ErrorFunction errorFunction;
        private ActivationFunction activationFunction;
        private Object trainingMethod;
        private LinkedList<Integer> neuronQueue;

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
            return new NeuralNetwork(errorFunction, activationFunction, trainingMethod, layers);
        }
    }
}
