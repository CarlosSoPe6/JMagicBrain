package org.jmagicbrain.functions;

import org.jmagicbrain.NeuralNetwork;

import java.util.List;

/**
 *Clase abstracta padre de las clases que contengan la función para
 * calcular el error
 */
public abstract class ErrorFunction {

    /**
     * La red neuronal donde se obtendrá la evaluación
     */
    protected NeuralNetwork neuralNetwork;

    /**
     * Establece la red neuronal con la cual se evaluará
     * @param neuralNetwork
     */
    public final void setNeuralNetwork(NeuralNetwork neuralNetwork){
        this.neuralNetwork = neuralNetwork;
    }

    /**
     * Obtiene el error de la función
     * @param trainingSet Los datos a evaluar
     * @param objective El resultado esperadao para cada caso
     * @return El error
     */
    public abstract double getError(List<List<Double>> trainingSet, List<List<Double>> objective);
}
