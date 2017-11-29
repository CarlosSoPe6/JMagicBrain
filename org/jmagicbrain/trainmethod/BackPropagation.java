package org.jmagicbrain.trainmethod;

import org.jmagicbrain.NeuralNetwork;
import org.jmagicbrain.functions.ErrorFunction;

public class BackPropagation{
    /*
    private double[][][] previousDeltasMatrix;
    private double[] errorVector;
    private double[][] sigmas;

    private final double learningRate;
    private final double momentum;

    public BackPropagation(double learningRate, double momentum, int maxEpochs, double maxError, double[][] trainingSet, double[][] expectedOutput, ErrorFunction errorFunction) {
        super(maxEpochs, maxError, trainingSet, expectedOutput, errorFunction);
        this.momentum = momentum;
        this.learningRate = learningRate;
    }

    @Override
    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        super.setNeuralNetwork(neuralNetwork);
    }

    @Override
    public void train() {
        int epocs = 0;
        int hits = 0;
        while (epocs < maxEpochs && hits < super.trainingSet.size()) {
            //Probaly doesnt works
            errorFunction.getError();
            double sigma = 0;
            double delta;

            // Sigma for output, the last row
            for (int i = 0; i < sigmas[sigmas.length - 1].length; i++) {
                sigmas[sigmas.length - 1][i] = (super.neuralNetwork.setLayer[sigmas.length][i] - expectedOutput[i]) * dSigmoid(layers[sigmas.length][i]);
            }

            // Others sigmas
            for (int i = sigmas.length - 2; i >= 0; i--) {
                for (int j = 0; j < sigmas[i].length; j++) {
                    for (int k = 0; k < sigmas[i + 1].length; k++) {
                        //System.out.printf("i: %d, j: %d, k: %d \n", i, j, k);
                        // TODO: Check this works properly
                        sigma += weightsMatrix[i][j][k] * sigmas[i + 1][k];
                    }
                    sigmas[i][j] = sigma * dSigmoid(layers[i + 1][j]);
                    sigma = 0;
                }
            }

            // Update weights
            for (int i = 0; i < weightsMatrix.length; i++) {
                for (int j = 0; j < weightsMatrix[i].length; j++) {
                    for (int k = 0; k < weightsMatrix[i][j].length; k++) {
                        delta = (LEARNING_RATE * layers[i][k] * sigmas[i][j])
                                + (MOMENTUM * previousDeltasMatrix[i][j][k]);

                        weightsMatrix[i][j][k] -= delta;
                        previousDeltasMatrix[i][j][k] = delta;
                    }
                }
            }
        }
    }*/
}
