package org.jmagicbrain;

import org.jmagicbrain.exceptions.InvalidNeuralNetworkArguments;
import org.jmagicbrain.functions.ActivationFunction;
import org.jmagicbrain.functions.ErrorFunction;
import org.jmagicbrain.initializers.WeightInitializer;

import java.util.LinkedList;

/**
 * Clase principal, crea red neuronal.
 * Para construirse es necesario usar su Builder
 * de la forma: new NeuralNetwork(args).setActivationFunction(args).build();
 */
public class NeuralNetwork {
    private double[][] layers;
    private double[][][] weightsMatrix;
    private double numberOfWeights;
    private WeightInitializer initializer;
    private ErrorFunction errorFunction;
    private ActivationFunction activationFunction;
    private Object trainingMethod;

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

        weightInitializer.initialize(weightsMatrix);
    }

    /**
     * Obtiene los pesos y umbrales de la red neuronal
     * @return los pesos y umbrales en un arreglo de matrices
     */
    public double[][][] getWeights(){
        return weightsMatrix;
    }

    /**
     * Iguala los valores de la capa de entrada para poder evaluar
     * @param inputs los valores a evaluar. NORMALIZADOS
     */
    public void setInputLayer(double ... inputs){
        if (inputs.length != layers[0].length){
            // TODO: Exception
        }
        System.arraycopy(inputs, 0, this.layers[0], 0, this.layers[0].length);
    }

    /**
     * Obtiene el resultado de evaluar la red neuronal
     * @return El 'output layer'
     */
    public double[] getOutputLayer(){
        return this.layers[layers.length - 1];
    }

    /**
     * Evalua la red neuronal
     */
    public void think(){
        double sum = 0;
        // Number of layers - 1
        for(int i = 0; i < weightsMatrix.length; i++){
            // The next layer
            for(int j = 0; j < weightsMatrix[i].length; j++){
                for(int k = 0; k < weightsMatrix[i][j].length; k++){
                    sum += layers[i][k] * weightsMatrix[i][j][k];
                }
                layers[i+1][j] = activationFunction.func(sum);
                sum = 0;
            }
        }
    }
<<<<<<< HEAD
    /**
     * Funcion que devuelve los pesos de la red neuronal
     * @return double[][][] con los valores actuales de los pesos de la red neuronal
     */
    public double[][][] getWeights(){
        return weightsMatrix;
    }

    /**
     * Clase utilizada para hcaer la construcción de la red neuronal
     */
    public class NeuralNetworkBuilder {
=======

    /**
     * Entrena la red neuronal
     */
    public void train(){
        // trainingMethod.train();
    }

    /**
     * Constructor para la red neuronal
     */
    public static class NeuralNetworkBuilder {
>>>>>>> 05debcc38d81b7226bc1f01934eaddc057a0a007
        private ErrorFunction errorFunction;
        private ActivationFunction activationFunction;
        private Object trainingMethod;
        private LinkedList<Integer> neuronQueue;
        private WeightInitializer initializer;

        /**
         * Constructor de NeuralNetworkBuilder
         */
        public NeuralNetworkBuilder(){
            neuronQueue = new LinkedList<Integer>();
        }

        /**
         * Establece la instacia de ErrorFunction
         * @param errorFunction instancia de ErrorFunction
         * @return Referencia a si mismo
         */
        public NeuralNetworkBuilder setErrorFunction(ErrorFunction errorFunction) {
            this.errorFunction = errorFunction;
            return this;
        }

        /**
         * Establece la instancia de ActivationFunction
         * @param activationFunction Instancia de ActivationFunction
         * @return Referencia a si mismo
         */
        public NeuralNetworkBuilder setActivationFunction(ActivationFunction activationFunction){
            this.activationFunction = activationFunction;
            return this;
        }

        /** Establece la instancia de TrainingMethod
         * @param trainingMethod Instancia de TrainingMethod
         * @return Referencia a si mismo
         */
        public NeuralNetworkBuilder setTrainingMethod(Object trainingMethod){
            this.trainingMethod = trainingMethod;
            return this;
        }

        /**
         * Agrega una capapa de 'numberOfNeurons' neuronas
         * @param numberOfNeurons Referencia a si mismo
         * @return
         */
        public NeuralNetworkBuilder addLayer(int numberOfNeurons){
            this.neuronQueue.add(numberOfNeurons);
            return this;
        }

        /**
         * Establence la instancia de WeightInitializer
         * @param initializer Instalcia de WeightInitializer
         * @return Referencia a si mismo
         */
        public NeuralNetworkBuilder setWeightInitializer(WeightInitializer initializer){
            this.initializer = initializer;
            return this;
        }

        /**
         * Crea y retorna una referencia a un objeto
         * @return Referecia al objeto construido
         * @throws InvalidNeuralNetworkArguments
         */
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
