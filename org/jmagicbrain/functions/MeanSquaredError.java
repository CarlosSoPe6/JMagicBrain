package org.jmagicbrain.functions;

import org.jmagicbrain.NeuralNetwork;

/**
 * Clase utilizada para calcular el error en Enjambre de Particulas
 */
public class MeanSquaredError extends ErrorFunction {
    /**
     * Inicia la instancia de ErrorFunction
     *
     * @param neuralNetwork Neural network to evaluate
     */
    public MeanSquaredError(NeuralNetwork neuralNetwork) {
        super(neuralNetwork);
    }

    @Override
    public double getError(double[][] trainingSet, double[][] objective) {
        return 0;
    }
}
