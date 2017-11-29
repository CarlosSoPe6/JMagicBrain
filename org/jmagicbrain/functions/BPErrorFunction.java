package org.jmagicbrain.functions;

import org.jmagicbrain.NeuralNetwork;
import java.util.List;

import java.util.List;

/**
 * Clase utilizada para calcular el error en Backpropagation
 */
public class BPErrorFunction extends ErrorFunction {

    @Override
    public double getError(List<List<Double>> trainingSet, List<List<Double>> objective) {
<<<<<<< HEAD
        neuralNetwork.setInputLayer(trainingSet.get(0));
=======
        // neuralNetwork.setInputLayer(trainingSet[0]);
>>>>>>> 0520c15a1665a21303d549d5db5182d30d8f6cb4
        neuralNetwork.think();
        double[] outputVector = neuralNetwork.getOutputLayer();
        double error = 0;
        for(int i = 0; i < outputVector.length; i++){
<<<<<<< HEAD
            error += Math.pow(objective.get(0).get(i) - outputVector[i], 2);
=======
            //error += Math.pow(objective[0][i] - outputVector[i], 2);
>>>>>>> 0520c15a1665a21303d549d5db5182d30d8f6cb4
        }
        return  error * 0.5;

    }
}
