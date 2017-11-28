package org.jmagicbrain.tests;

import org.jmagicbrain.NeuralNetwork;
import org.jmagicbrain.functions.ActivationFunction;
import org.jmagicbrain.functions.ErrorFunction;
import org.jmagicbrain.functions.MeanSquaredError;
import org.jmagicbrain.functions.Sigmoid;
import org.jmagicbrain.initializers.DefaultInitializer;
import org.jmagicbrain.initializers.WeightInitializer;
import org.jmagicbrain.trainmethod.ParticleSwarmOptimization;
import org.jmagicbrain.trainmethod.TrainMethod;

public class WineAnalizer {
    public static void main(String[] args) {
        ActivationFunction activationFunction = new Sigmoid();
        ErrorFunction errorFunction = new MeanSquaredError();
        WeightInitializer weightInitializer = new DefaultInitializer();
        TrainMethod trainMethod = new ParticleSwarmOptimization(
                1.03,
                1.03,
                13,
                10.0,
                -10.0,
            100,
                0.01,
                null,
                null,
                errorFunction
        );

        NeuralNetwork nn = new NeuralNetwork.NeuralNetworkBuilder()
                .setActivationFunction(activationFunction)
                .setErrorFunction(errorFunction)
                .setTrainingMethod(trainMethod)
                .setWeightInitializer(weightInitializer)
                .addLayer(13)
                .addLayer(4)
                .addLayer(3)
                .build();

    }
}
