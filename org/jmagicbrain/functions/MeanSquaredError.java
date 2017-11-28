package org.jmagicbrain.functions;

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
            for(int j = 0; j < output.length; j++){
                sumSquaredError += Math.pow(output[j] - objective[i][j],2.0);
            }
        }
        return sumSquaredError / ((double) trainingSet.length);
    }
}
