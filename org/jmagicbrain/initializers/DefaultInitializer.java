package org.jmagicbrain.initializers;

/**
 *Inicializador por defualt que se usa en la red neuronal
 */
public class DefaultInitializer implements WeightInitializer{
    @Override
    public void initialize(double[][][] weights) {
        double initialLimit = 0;
        for (int i = 0; i < weights.length; i++){
            for(int j = 0; j < weights[i].length; j++){
                // TODO: Get the limits
                //initialLimit = Math.sqrt(6.0 /(weights[i][j].length + weights[i][j + 1].length));
                // System.out.println(initialLimit);
                for(int k = 0; k < weights[i][j].length - 1; k++){
                    weights[i][j][k] = (Math.random() % ((initialLimit * 2) + 1)) - initialLimit;
                }
            }
        }
    }
}
