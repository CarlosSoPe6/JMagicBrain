package org.jmagicbrain;

import org.jmagicbrain.exceptions.InvalidNeuralNetworkArguments;
import org.jmagicbrain.functions.ActivationFunction;
import org.jmagicbrain.functions.ErrorFunction;
import org.jmagicbrain.initializers.WeightInitializer;
import org.jmagicbrain.trainmethod.TrainMethod;
import java.util.List;

import java.util.Arrays;
import java.util.LinkedList;

/**
 * Clase principal, crea red neuronal.
 * Para construirse es necesario usar su Builder
 * de la forma: new NeuralNetwork(args).setActivationFunction(args).build();
 */
public class NeuralNetwork {
    private double[][] layers;
    private double[][][] weightsMatrix;

    private int numberOfWeights;

    public ErrorFunction getErrorFunction() {
        return errorFunction;
    }

    public void setErrorFunction(ErrorFunction errorFunction) {
        this.errorFunction = errorFunction;
    }

    private ErrorFunction errorFunction;
    private ActivationFunction activationFunction;
    private TrainMethod trainingMethod;
    private Integer[] layersArray;

    private NeuralNetwork(WeightInitializer weightInitializer, ErrorFunction errorFunction, ActivationFunction activationFunction, TrainMethod trainingMethod, Integer ... layersArray){
        this.layers = new double[layersArray.length][];
        this.weightsMatrix = new double[layersArray.length - 1][][];

        for(int i = 1; i < layersArray.length; i++){
            numberOfWeights += (layersArray[i] * layersArray[i-1]) + layersArray[i];
        }

        for(int i = 0; i < layersArray.length; i++){
            this.layers[i] = new double[layersArray[i] + 1];
            layers[i][layers[i].length - 1] = 1;
            if(i > 0){
                weightsMatrix[i-1] = new double[this.layers[i].length - 1][this.layers[i - 1].length];
            }
        }

        this.layersArray = layersArray;

        weightInitializer.initialize(weightsMatrix);

        this.errorFunction = errorFunction;
        this.trainingMethod = trainingMethod;
        this.activationFunction = activationFunction;

        errorFunction.setNeuralNetwork(this);
        trainingMethod.setNeuralNetwork(this);
    }

    /**
     * Obtiene el número total de pesos
     * @return total de pesos y umbrales
     */
    public int getNumberOfWeights() {
        return numberOfWeights;
    }

    /**
     * Obtiene los pesos y umbrales de la red neuronal
     * @return los pesos y umbrales en un arreglo de matrices
     */
    public double[][][] getWeights(){
        return weightsMatrix;
    }

    /**
     * Establece los pesos de la matriz
     * @param weights arreglo de matrices de pesos
     */
    public void setWeights(double[][][] weights){
        for(int i = 0; i < this.weightsMatrix.length; i++){
            for(int j = 0; j < this.weightsMatrix[i].length; j++){
                System.arraycopy(weightsMatrix[i][j], 0, this.weightsMatrix[i][j], 0, this.weightsMatrix[i][j].length);
            }
        }
    }

    public Integer[] getLayersArray() {
        return layersArray;
    }

    /**
     * Establece los pesos de la matriz
     * @param weights vector de pesos
     */
    public void setWeights(double[] weights){
        int l = 0;
        for (int i = 0; i < this.weightsMatrix.length; i++){
            for(int j = 0; j < this.weightsMatrix[i].length; j++){
                for(int k = 0; k < this.weightsMatrix[i][j].length; k++){
                    this.weightsMatrix[i][j][k] = weights[l];
                    l++;
                }
            }
        }
    }

    /**
     * Iguala los valores de la capa de entrada para poder evaluar
     * @param inputs los valores a evaluar. NORMALIZADOS
     */
    public void setInputLayer(double ... inputs){
        if (inputs.length != layers[0].length){
            // TODO: Exception
        }
        System.arraycopy(inputs, 0, this.layers[0], 0, this.layers[0].length - 1);
    }

    /**
     * Obtiene el resultado de evaluar la red neuronal
     * @return El 'output layer'
     */
    public double[] getOutputLayer(){
        return this.layers[layers.length - 1];
    }

    /**
     * Regresa las capas de la red neuronal
     * @return las capas de la red neuronal
     */
    public double[][] getLayers(){
        return this.layers;
    }

    /**
     * La función de activación de la instancia actual
     * @return La función de activación
     */
    public ActivationFunction getActivationFunction() {
        return activationFunction;
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

    /**
     * Entrena la red neuronal
     */
    public double train(){
        return trainingMethod.train();
    }

    /**
     * Constructor para la red neuronal
     */
    public static class NeuralNetworkBuilder {
        private ErrorFunction errorFunction;
        private ActivationFunction activationFunction;
        private TrainMethod trainingMethod;
        private LinkedList<Integer> neuronList;
        private WeightInitializer initializer;

        /**
         * Constructor de NeuralNetworkBuilder
         */
        public NeuralNetworkBuilder(){
            neuronList = new LinkedList<Integer>();
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
        public NeuralNetworkBuilder setTrainingMethod(TrainMethod trainingMethod){
            this.trainingMethod = trainingMethod;
            return this;
        }

        /**
         * Agrega una capapa de 'numberOfNeurons' neuronas
         * @param numberOfNeurons Referencia a si mismo
         * @return
         */
        public NeuralNetworkBuilder addLayer(int numberOfNeurons){
            this.neuronList.add(numberOfNeurons);
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
         * @throws InvalidNeuralNetworkArguments En caso de tener un argumento inválido
         */
        public NeuralNetwork build() throws InvalidNeuralNetworkArguments{
            Integer[] layers = new Integer[neuronList.size()];
            neuronList.toArray(layers);

            for (Integer layer : layers) {
                if (layer < 1) {
                    throw new InvalidNeuralNetworkArguments("A layer contains  neurons");
                }
            }

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
