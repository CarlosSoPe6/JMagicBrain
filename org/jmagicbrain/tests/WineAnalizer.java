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
import org.jmagicbrain.trainmethod.ParticleSwarmOptimization;
import org.jmagicbrain.trainmethod.TrainMethod;
import org.jmagicbrain.utils.Normalizers;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

public class WineAnalizer {
    public static void main(String[] args) {
        double NormalizationMax[] = {14.83, 5.8, 3.23, 30, 162, 3.88, 5.08, 0.66, 3.58, 13, 1.71, 4, 1680};
        double NormalizationMin[] = {11.03, 0.74, 1.36, 10.6, 70, 0.98, 0.34, 0.13, 0.41, 1.28, 0.48, 1.27, 278};

        int dslen = 178;
        Reader in = null;
        double[][] dataset = new double[dslen][13];
        double[][] expected = new double[dslen][3];

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
            for(int j = 1; j <= 13; j++){
                dataset[i][j - 1] = Normalizers.normalizeRange(Double.parseDouble(e.get(j)), NormalizationMin[j - 1], NormalizationMax[j - 1]);
                //dataset[i][j] = Double.parseDouble(e.get(j));
            }

            switch (Integer.parseInt(e.get(0))){
                case 1:
                    expected[i][0] = 1.0;
                    expected[i][1] = 0.0;
                    expected[i][2] = 0.0;
                    break;
                case 2:
                    expected[i][0] = 0.0;
                    expected[i][1] = 1.0;
                    expected[i][2] = 0.0;
                    break;
                case 3:
                    expected[i][0] = 0.0;
                    expected[i][1] = 0.0;
                    expected[i][2] = 1.0;
                    break;
            }
        }

        ActivationFunction activationFunction = new Sigmoid();
        ErrorFunction errorFunction = new MeanSquaredError();
        WeightInitializer weightInitializer = new DefaultInitializer();
        TrainMethod trainMethod = new ParticleSwarmOptimization(
<<<<<<< HEAD
                0.005,
                0.08,
                0.07,
                15,
                100.0,
                -100.0,
                100,
=======
                0.01,
                1.237,
                1.237,
                15,
                10.0,
                -10.0,
                1000,
>>>>>>> 3a89642e99464d4bf3221db59539fbb4279813ce
                0.1,
                dataset,
                expected,
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

        System.out.println("Starting train");
        nn.train();
        System.out.println("Ending train");

        nn.setInputLayer(dataset[dataset.length - 1]);
        nn.think();
        System.out.println(Arrays.toString(nn.getOutputLayer()));

    }
}
