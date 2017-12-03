package org.jmagicbrain.tests;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.jmagicbrain.NeuralNetwork;
import org.jmagicbrain.functions.ActivationFunction;
import org.jmagicbrain.functions.ErrorFunction;
import org.jmagicbrain.functions.MeanSquaredError;
import org.jmagicbrain.functions.Sigmoid;
import org.jmagicbrain.initializers.DefaultInitializer;
import org.jmagicbrain.initializers.WeightInitializer;
import org.jmagicbrain.trainmethod.BackPropagation;
import org.jmagicbrain.trainmethod.TrainMethod;
import org.jmagicbrain.utils.NNIO;
import org.jmagicbrain.utils.Normalizers;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

public class JSONTest {
    public static void main(String[] args) {
        double[][][] pesos;
        try {
            pesos = NNIO.importNN("C:\\Users\\Escuela\\OneDrive - ITESO\\Programación\\JAVA\\POO\\Proyecto\\test01.json");
        }
       catch (Exception e) {
            e.printStackTrace();
            pesos = new double[1][1][1];
       }


        double NormalizationMax[] = {14.83, 5.8, 3.23, 30, 162, 3.88, 5.08, 0.66, 3.58, 13, 1.71, 4, 1680};
        double NormalizationMin[] = {11.03, 0.74, 1.36, 10.6, 70, 0.98, 0.34, 0.13, 0.41, 1.28, 0.48, 1.27, 278};

        int dslen = 178;
        Reader in = null;
        double[][] dataset = new double[dslen][13];
        double[][] expected = new double[dslen][];

        try {
            in = new FileReader("C:\\Desarrollo\\hhh.csv");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } finally {
            if(in == null) System.exit(1);
        }

        Iterable<CSVRecord> records = null;
        try {
            records = CSVFormat.EXCEL.parse(in);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if(records == null) System.exit(1);
        }

        ArrayList<CSVRecord> recordArrayList = new ArrayList<>(178);
        CSVRecord record;
        records.iterator();
        Iterator<CSVRecord> iterator = records.iterator();
        while (iterator.hasNext()) {
            record = iterator.next();
            recordArrayList.add(record);
        }

        for(int i = 0; i < dslen; i++){
            CSVRecord e = recordArrayList.get(i);
            for(int j = 0; j < 13; j++){
                dataset[i][j] = (Normalizers.normalizeRange(Double.parseDouble(e.get(j + 1)), NormalizationMin[j], NormalizationMax[j]));
            }
        }

        ActivationFunction activationFunction = new Sigmoid();
        ErrorFunction errorFunction = new MeanSquaredError();
        WeightInitializer weightInitializer = new DefaultInitializer();
        TrainMethod trainMethod = new BackPropagation.BackPropagationBuilder()
                .setErrorFunction(errorFunction)
                .setMaxEpochs(100)
                .setMaxError(0.01)
                .setMomentum(0.6)
                .setLearningRate(0.4)
                .setTrainingSet(dataset)
                .setExpectedOutput(expected)
                .build();

        NeuralNetwork nn = new NeuralNetwork.NeuralNetworkBuilder()
                .setActivationFunction(activationFunction)
                .setErrorFunction(errorFunction)
                .setTrainingMethod(trainMethod)
                .setWeightInitializer(weightInitializer)
                .addLayer(13)
                .addLayer(4)
                .addLayer(3)
                .build();

        nn.setWeights(pesos);

        double[] input = new double[dataset[0].length];

        for (int i = 0; i < input.length; i++){
            input[i] = dataset[0][i];
        }

        nn.setInputLayer(input);
        nn.think();
        System.out.println(Arrays.toString(nn.getOutputLayer()));

        double[][][] d = nn.getWeights();
        for (int i = 0; i < d.length; i++) {
            for(int j = 0; j < d[i].length; j++) {
                for (int k = 0; k < d[i][j].length; k++) {
                    System.out.print(d[i][j][k] + ", ");
                }
                System.out.print("\n");
            }
            System.out.print("\n\n");
        }
    }
}
