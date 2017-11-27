package org.jmagicbrain;

/**
 * Interface utilzada para asignar la función de activación de la red neuronal
 */
public interface ActivationFunction {
    /**
     * Funcion de activación
     * @param x el parametro a evaluar en la función
     * @return el resultado del valor evauluado en la función
     */
    double func(double x);

    /**
     * La derivada de la funcioión de activación usada para el método de
     * aprendizaje Backpropagation
     * @param x el parametro a evaluar en la función derivada
     * @return el resultado del valor evaluado en la función derivada
     */
    double deriv(double x);
}
