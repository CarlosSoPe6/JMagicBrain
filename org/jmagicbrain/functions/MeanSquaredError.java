package org.jmagicbrain.functions;

import java.util.List;

/**
 * Clase utilizada para calcular el error en Enjambre de Particulas
 */
public class MeanSquaredError extends ErrorFunction {

    @Override
    public double getError(double[][] trainingSet, double[][] objective) {
        double sumSquaredError = 0.0;
        double[] output;
        for(int i = 0; i < trainingSet.length; i++){

            neuralNetwork.setInputLayer(trainingSet[i]);
            neuralNetwork.think();
            output = neuralNetwork.getOutputLayer();
            for(int j = 0; j < objective[i].length; j++){
                sumSquaredError += ((output[j] - objective[i][j]) * (output[j] - objective[i][j]));
            }
        }
        return sumSquaredError / trainingSet.length;
    }
}
