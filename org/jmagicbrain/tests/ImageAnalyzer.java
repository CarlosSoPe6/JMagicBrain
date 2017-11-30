package org.jmagicbrain.tests;

import org.jmagicbrain.NeuralNetwork;
import org.jmagicbrain.functions.ActivationFunction;
import org.jmagicbrain.functions.ErrorFunction;
import org.jmagicbrain.functions.MeanSquaredError;
import org.jmagicbrain.functions.Sigmoid;
import org.jmagicbrain.initializers.DefaultInitializer;
import org.jmagicbrain.initializers.WeightInitializer;
import org.jmagicbrain.trainmethod.BackPropagation;
import org.jmagicbrain.trainmethod.ParticleSwarmOptimization;
import org.jmagicbrain.trainmethod.TrainMethod;
import org.jmagicbrain.utils.Normalizers;
import org.jmagicbrain.utils.Readers;

import java.io.IOException;
import java.util.Arrays;

public class ImageAnalyzer {
    public static void main(String[] args){

        int[][] dataMatrix = new int[10][];
        double[][] dSet;
        double[][] expected = new double[10][10];
        try {
            dataMatrix[0] = Readers.getBufferedImage("C:\\Desarrollo\\DataSets\\Numeros\\Train\\0.png");
            dataMatrix[1] = Readers.getBufferedImage("C:\\Desarrollo\\DataSets\\Numeros\\Train\\1.png");
            dataMatrix[2] = Readers.getBufferedImage("C:\\Desarrollo\\DataSets\\Numeros\\Train\\2.png");
            dataMatrix[3] = Readers.getBufferedImage("C:\\Desarrollo\\DataSets\\Numeros\\Train\\3.png");
            dataMatrix[4] = Readers.getBufferedImage("C:\\Desarrollo\\DataSets\\Numeros\\Train\\4.png");
            dataMatrix[5] = Readers.getBufferedImage("C:\\Desarrollo\\DataSets\\Numeros\\Train\\5.png");
            dataMatrix[6] = Readers.getBufferedImage("C:\\Desarrollo\\DataSets\\Numeros\\Train\\6.png");
            dataMatrix[7] = Readers.getBufferedImage("C:\\Desarrollo\\DataSets\\Numeros\\Train\\7.png");
            dataMatrix[8] = Readers.getBufferedImage("C:\\Desarrollo\\DataSets\\Numeros\\Train\\8.png");
            dataMatrix[9] = Readers.getBufferedImage("C:\\Desarrollo\\DataSets\\Numeros\\Train\\9.png");

        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }

        dSet = Normalizers.normalizeWithDefaults(dataMatrix);

        for (int i = 0; i < expected.length; i++){
            expected[i][i] = 1.0;
        }

        ActivationFunction activationFunction = new Sigmoid();
        ErrorFunction errorFunction = new MeanSquaredError();
        WeightInitializer weightInitializer = new DefaultInitializer();
        TrainMethod trainMethod = new ParticleSwarmOptimization.ParticleSwarmOptimizationBuilder()
                .setProbDeath(0.009)
                .setW(0.0999989999)
                .setCongitiveLocalConstant(0.08245)
                .setSocialGlobalConstant(0.08245)
                .setNumberOfParticles(25)
                .setMaxX(50)
                .setMinX(-50)
                .setMaxEpochs(1000)
                .setMaxError(0.09)
                .setTrainingSet(dSet)
                .setExpectedOutput(expected)
                .setErrorFunction(errorFunction)
                .build();

        /*TrainMethod trainMethod = new BackPropagation.BackPropagationBuilder()
                .setErrorFunction(errorFunction)
                .setMaxEpochs(10000)
                .setMaxError(0.01)
                .setMomentum(0.705425)
                .setLearningRate(0.4599963)
                .setTrainingSet(dSet)
                .setExpectedOutput(expected)
                .build();*/

        NeuralNetwork neuralNetwork = new NeuralNetwork.NeuralNetworkBuilder()
                .setWeightInitializer(weightInitializer)
                .setErrorFunction(errorFunction)
                .setActivationFunction(activationFunction)
                .setTrainingMethod(trainMethod)
                .addLayer(dSet[0].length)
                .addLayer(102)
                .addLayer(10)
                .build();


        neuralNetwork.train();

        neuralNetwork.setInputLayer(dSet[0]);
        neuralNetwork.think();

        System.out.print(Arrays.toString(neuralNetwork.getOutputLayer()));
    }
}
