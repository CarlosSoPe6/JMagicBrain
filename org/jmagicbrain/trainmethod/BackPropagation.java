package org.jmagicbrain.trainmethod;

import org.jmagicbrain.NeuralNetwork;
import org.jmagicbrain.functions.ErrorFunction;

import java.util.List;

public class BackPropagation extends TrainMethod{

    private double[][][] previousDeltasMatrix;
    private double[] errorVector;
    private double[][] sigmas;
    private int iterations = 0;

    private final double learningRate;
    private final double momentum;

    public BackPropagation(double learningRate, double momentum, int maxEpochs, double maxError, List<List<Double>> trainingSet, List<List<Double>> expectedOutput, ErrorFunction errorFunction) {
        super(maxEpochs, maxError, trainingSet, expectedOutput, errorFunction);
        this.momentum = momentum;
        this.learningRate = learningRate;
        sigmasInit();
    }

    @Override
    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        super.setNeuralNetwork(neuralNetwork);
    }

    @Override
    public void train() {
        int hits = 0;
        double[][] layers = neuralNetwork.getLayers();
        double[] expected = new double[expectedOutput.size()];
        int ite = 0;
        for(List<Double> l: expectedOutput) {
            for (double d : l) {
                expected[ite] = d;
            }
        }


        //Probably doesn't works
        double sigma = 0;
        double delta;
        errorVector[iterations] = errorFunction.getError(trainingSet, expectedOutput);
        // Sigma for output, the last row
        for (int i = 0; i < sigmas[sigmas.length - 1].length; i++) {
            sigmas[sigmas.length - 1][i] = (layers[sigmas.length][i] - expected[i]) * neuralNetwork.getActivationFunction().deriv(layers[sigmas.length][i]);
        }

        // Others sigmas
        for (int i = sigmas.length - 2; i >= 0; i--) {
            for (int j = 0; j < sigmas[i].length; j++) {
                for (int k = 0; k < sigmas[i + 1].length; k++) {
                    sigma += neuralNetwork.getWeights()[i][j][k] * sigmas[i + 1][k];
                }
                sigmas[i][j] = sigma * neuralNetwork.getActivationFunction().deriv(layers[i + 1][j]);
                sigma = 0;
            }
        }

        // Update weights
        for (int i = 0; i < neuralNetwork.getWeights().length; i++) {
            for (int j = 0; j < neuralNetwork.getWeights()[i].length; j++) {
                for (int k = 0; k < neuralNetwork.getWeights()[i][j].length; k++) {
                    delta = (learningRate * layers[i][k] * sigmas[i][j])
                            + (momentum * previousDeltasMatrix[i][j][k]);

                    neuralNetwork.getWeights()[i][j][k] -= delta;
                    previousDeltasMatrix[i][j][k] = delta;
                }
            }
        }
        iterations ++;
    }


    private void sigmasInit() {
        this.sigmas = new double[neuralNetwork.getLayers().length - 1][];

    }
}
