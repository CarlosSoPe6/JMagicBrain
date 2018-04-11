package org.jmagicbrain.initializers;

/**
 * Interface que implementan las clases que inicialicen
 * los pesos de la red neuronal
 */
public interface WeightInitializer {
    public void initialize(double[][][] weights);
}
