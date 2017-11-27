package org.jmagicbrain.errorfuntion;

import org.jmagicbrain.NeuralNetwork;

public abstract class ErrorFunction {

    protected NeuralNetwork neuralNetwork;

    /**
     * Construct the ErrorFunction instance.
     * @param neuralNetwork Neural network to evaluate
     */
    public ErrorFunction(NeuralNetwork neuralNetwork){
        this.neuralNetwork = neuralNetwork;
    }

    /**
     * Gets the error with defined error function
     * @param trainingSet The data to get the values
     * @param objective The objetive data to calculate the error
     * @return The error
     */
    public abstract double getError(double[][] trainingSet, double[][] objective);
}
