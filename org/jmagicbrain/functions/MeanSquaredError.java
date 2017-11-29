package org.jmagicbrain.functions;

import java.util.List;

/**
 * Clase utilizada para calcular el error en Enjambre de Particulas
 */
public class MeanSquaredError extends ErrorFunction {

    @Override
    public double getError(List<List<Double>> trainingSet, List<List<Double>> objective) {
        double sumSquaredError = 0.0;
        double[] output;
        double[] xValues = new double[trainingSet.get(0).size()];
        double[] tValues = new double[objective.get(0).size()];
        for(int i = 0; i < trainingSet.size(); i++){

            for(int j = 0; j < trainingSet.get(i).size(); j++){
                xValues[j] = trainingSet.get(i).get(j);
            }

            neuralNetwork.setInputLayer(xValues);
            neuralNetwork.think();
            output = neuralNetwork.getOutputLayer();
            for(int j = 0; j < objective.get(i).size(); j++){
                tValues[j] = objective.get(i).get(j);
            }
            for(int j = 0; j < objective.get(i).size(); j++){
                sumSquaredError += ((output[j] - tValues[j]) * (output[j] - tValues[j]));
            }
        }

        return sumSquaredError / trainingSet.size();
    }
}
