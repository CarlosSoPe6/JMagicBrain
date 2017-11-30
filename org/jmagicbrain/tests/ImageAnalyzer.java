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
import org.jmagicbrain.utils.Normalizers;
import org.jmagicbrain.utils.Readers;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ImageAnalyzer {
    public static void main(String[] args){

        byte[][] dataMatrix = new byte[10][];
        double[][] dataset;
        double[][] expected = null;
        try {
            dataMatrix[0] = Readers.readFileAsBytes("C:\\Desarrollo\\DataSets\\Numeros\\0.png");
            dataMatrix[1] = Readers.readFileAsBytes("C:\\Desarrollo\\DataSets\\Numeros\\1.png");
            dataMatrix[2] = Readers.readFileAsBytes("C:\\Desarrollo\\DataSets\\Numeros\\2.png");
            dataMatrix[3] = Readers.readFileAsBytes("C:\\Desarrollo\\DataSets\\Numeros\\3.png");
            dataMatrix[4] = Readers.readFileAsBytes("C:\\Desarrollo\\DataSets\\Numeros\\4.png");
            dataMatrix[5] = Readers.readFileAsBytes("C:\\Desarrollo\\DataSets\\Numeros\\5.png");
            dataMatrix[6] = Readers.readFileAsBytes("C:\\Desarrollo\\DataSets\\Numeros\\6.png");
            dataMatrix[7] = Readers.readFileAsBytes("C:\\Desarrollo\\DataSets\\Numeros\\7.png");
            dataMatrix[8] = Readers.readFileAsBytes("C:\\Desarrollo\\DataSets\\Numeros\\8.png");
            dataMatrix[9] = Readers.readFileAsBytes("C:\\Desarrollo\\DataSets\\Numeros\\9.png");

        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }

        dataset = Normalizers.normalizeWithDefaults(dataMatrix);

        ActivationFunction activationFunction = new Sigmoid();
        ErrorFunction errorFunction = new MeanSquaredError();
        WeightInitializer weightInitializer = new DefaultInitializer();
        TrainMethod trainMethod = new ParticleSwarmOptimization.ParticleSwarmOptimizationBuilder()
                .setProbDeath(0.01)
                .setW(0.3911)
                .setCongitiveLocalConstant(0.1099)
                .setSocialGlobalConstant(0.1099)
                .setNumberOfParticles(1000)
                .setMaxX(50)
                .setMinX(-50)
                .setMaxEpochs(1000)
                .setMaxError(0.09)
                .setTrainingSet(dataset)
                .setExpectedOutput(expected)
                .setErrorFunction(errorFunction)
                .build();

        NeuralNetwork neuralNetwork = new NeuralNetwork.NeuralNetworkBuilder()
                .setWeightInitializer(weightInitializer)
                .setErrorFunction(errorFunction)
                .setActivationFunction(activationFunction)
                .setTrainingMethod(trainMethod)
                .addLayer(1024)
                .addLayer(1088)
                .addLayer(10)
                .build();


    }
}
