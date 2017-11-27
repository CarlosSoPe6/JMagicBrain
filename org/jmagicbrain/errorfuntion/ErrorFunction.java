package org.jmagicbrain.errorfuntion;

import org.jmagicbrain.NeuralNetwork;

/**
 *
 */
public abstract class ErrorFunction {

    /**
     * La red neuronal donde se obtendrá la evaluación
     */
    protected NeuralNetwork neuralNetwork;

    /**
     * Inicia la instancia de ErrorFunction
     * @param neuralNetwork Neural network to evaluate
     */
    public ErrorFunction(NeuralNetwork neuralNetwork){
        this.neuralNetwork = neuralNetwork;
    }

    /**
     * Obtiene el error de la función
     * @param trainingSet Los datos a evaluar
     * @param objective El resultado esperadao para cada caso
     * @return El error
     */
    public abstract double getError(double[][] trainingSet, double[][] objective);
}
