package org.jmagicbrain.errorfunction;

import org.jmagicbrain.NeuralNetwork;

public class BPErrorFunction extends ErrorFunction {
    /**
     * Inicia la instancia de ErrorFunction
     *
     * @param neuralNetwork Neural network to evaluate
     */
    public BPErrorFunction(NeuralNetwork neuralNetwork) {
        super(neuralNetwork);
    }

    @Override
    public double getError(double[][] trainingSet, double[][] objective) {
        return 0;
    }
}
