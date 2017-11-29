package org.jmagicbrain.functions;

import org.jmagicbrain.NeuralNetwork;
import java.util.List;

/**
 * Clase utilizada para calcular el error en Backpropagation
 */
public class BPErrorFunction extends ErrorFunction {

    @Override
    public double getError(List<List<Double>> trainingSet, List<List<Double>> objective) {
        neuralNetwork.setInputLayer(trainingSet.get(0));
        neuralNetwork.think();
        double[] outputVector = neuralNetwork.getOutputLayer();
        double error = 0;
        for(int i = 0; i < outputVector.length; i++){
            error += Math.pow(objective.get(0).get(i) - outputVector[i], 2);
        }
        return  error * 0.5;

    }
}
