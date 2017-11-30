package org.jmagicbrain.trainmethod;

import org.jmagicbrain.NeuralNetwork;
import org.jmagicbrain.functions.ErrorFunction;

import java.util.List;

public class BackPropagation extends TrainMethod{

    private double[][][] previousDeltasMatrix;
    private double[][] sigmas;

    private final double learningRate;
    private final double momentum;

    private BackPropagation(double learningRate, double momentum, int maxEpochs, double maxError, double[][] trainingSet, double[][] expectedOutput, ErrorFunction errorFunction) {
        super(maxEpochs, maxError, trainingSet, expectedOutput, errorFunction);
        this.momentum = momentum;
        this.learningRate = learningRate;
    }
 
    @Override
    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        super.setNeuralNetwork(neuralNetwork);
        this.sigmas = new double[neuralNetwork.getLayersArray().length - 1][];
        this.previousDeltasMatrix = new double[neuralNetwork.getLayersArray().length - 1][][];
        for (int i = 1; i < neuralNetwork.getLayersArray().length; i++){
            this.sigmas[i - 1] = new double[neuralNetwork.getLayersArray()[i]];
            this.previousDeltasMatrix[i-1] = new double[neuralNetwork.getLayers()[i].length - 1][neuralNetwork.getLayers()[i - 1].length];
        }
    }

    @Override
    public void train() {
        double error = 0;
        int currentEpoch = 0;
        double delta = 0;
        double sigma = 0;

        double[] currentTrain;
        double[] expected;
        double[] currentOutput;

        while(currentEpoch < maxEpochs){
            for(int e = 0; e < trainingSet.length; e++){
                currentTrain = trainingSet[e];
                expected = expectedOutput[e];

                neuralNetwork.setInputLayer(currentTrain);
                neuralNetwork.think();
                currentOutput = neuralNetwork.getOutputLayer();

                // Sigma for output, the last row
                for(int j = 0; j < currentOutput.length - 1; j++) {
                    sigmas[sigmas.length - 1][j] =
                        (currentOutput[j] - expectedOutput[e][j]) *
                        neuralNetwork.getActivationFunction().deriv(currentOutput[j]);
                }

                // Others sigmas
                for(int i = sigmas.length - 2; i >= 0; i--){
                    for(int j = 0; j < sigmas[i].length; j++){
                        for(int k = 0; k < sigmas[i + 1].length; k++){
                            sigma += neuralNetwork.getWeights()[i][j][k] * sigmas[i + 1][k];
                        }
                        sigmas[i][j] = sigma * neuralNetwork.getActivationFunction().deriv(neuralNetwork.getLayers()[i + 1][j]);
                        sigma = 0;
                    }
                }

                // Update weights
                for(int i = 0; i < neuralNetwork.getWeights().length; i++){
                    for(int j = 0; j < neuralNetwork.getWeights()[i].length; j++){
                        for(int k = 0; k < neuralNetwork.getWeights()[i][j].length; k++){
                            delta = (learningRate * neuralNetwork.getLayers()[i][k] * sigmas[i][j])
                                    + (momentum * previousDeltasMatrix[i][j][k]);

                            neuralNetwork.getWeights()[i][j][k] -= delta;
                            previousDeltasMatrix[i][j][k] = delta;
                        }
                    }
                }
            }
            currentEpoch++;
        }
    }

    public static class BackPropagationBuilder{
        private double learningRate;
        private double momentum;
        private int maxEpochs;
        private double maxError;
        private double[][] trainingSet;
        private double[][] expectedOutput;
        private ErrorFunction errorFunction;

        /**
         * Establece el learning rate del backpopagation
         * @param learningRate El learning rate
         * @return Referencia a si mismo
         */
        public BackPropagationBuilder setLearningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        /**
         * Establece el 'momentum' delbackpopagation
         * @param momentum El momentum
         * @return Referencia a si mismo
         */
        public BackPropagationBuilder setMomentum(double momentum) {
            this.momentum = momentum;
            return this;
        }

        /**
         * Establece las epocas máximas para el training set
         * @param maxEpochs Máximo de epocas
         * @return Referencia a si mismo
         */
        public BackPropagationBuilder setMaxEpochs(int maxEpochs) {
            this.maxEpochs = maxEpochs;
            return this;
        }

        /**
         * Establece el error máximo, para la condición de salida
         * @param maxError El error máximo
         * @return Referencia a si mismo
         */
        public BackPropagationBuilder setMaxError(double maxError) {
            this.maxError = maxError;
            return this;
        }

        /**
         * Establece el training set a evaluar
         * @param trainingSet El training set
         * @return Referencia a si mismo
         */
        public BackPropagationBuilder setTrainingSet(double[][] trainingSet) {
            this.trainingSet = trainingSet;
            return this;
        }

        /**
         * Establece la salida esperada
         * @param expectedOutput La salida esperada
         * @return Referencia a si mismo
         */
        public BackPropagationBuilder setExpectedOutput(double[][] expectedOutput) {
            this.expectedOutput = expectedOutput;
            return this;
        }

        /**
         * Establece la función de error
         * @param errorFunction La función de error
         * @return Referencia a si mismo
         */
        public BackPropagationBuilder setErrorFunction(ErrorFunction errorFunction) {
            this.errorFunction = errorFunction;
            return this;
        }

        /**
         * Crea una instancia de BackPropagation
         * @return Instancia de BackPropagation
         */
        public BackPropagation build(){
            // TODO: Check params

            return new BackPropagation(
                    learningRate,
                    momentum,
                    maxEpochs,
                    maxError,
                    trainingSet,
                    expectedOutput,
                    errorFunction
            );
        }
    }
}
 