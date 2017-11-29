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

public class AgaricusLepiota {
    public static void main(String[] args){
        Reader in = null;

        try {
            in = new FileReader("C:\\Desarrollo\\agaricus-lepiota.csv");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } finally {
            if(in == null) System.exit(1);
        }

        Iterable<CSVRecord> csvRecordIterable = null;
        try {
            csvRecordIterable = CSVFormat.EXCEL.parse(in);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if(csvRecordIterable == null) System.exit(1);
        }

        ArrayList<CSVRecord> recordArrayList = new ArrayList<>(8124);

        for (CSVRecord record : csvRecordIterable){
            recordArrayList.add(record);
        }

        double[][] expected = new double[recordArrayList.size()][2];
        double[][] ds;
        char[][] records = new char[recordArrayList.size()][recordArrayList.get(0).size() - 1];

        for(int i = 0; i < recordArrayList.size(); i++){
            if(recordArrayList.get(i).get(0).charAt(0) == 'e'){
                expected[i][1] = 1;
                expected[i][0] = 0;
            }else{
                expected[i][1] = 0;
                expected[i][0] = 1;
            }
            for(int j = 1; j < recordArrayList.get(i).size(); j++){
                records[i][j-1] = recordArrayList.get(i).get(j).charAt(0);
            }
        }

        ds = Normalizers.normalize(records);

        ActivationFunction activationFunction = new Sigmoid();
        ErrorFunction errorFunction = new MeanSquaredError();
        WeightInitializer weightInitializer = new DefaultInitializer();
        TrainMethod trainMethod = new ParticleSwarmOptimization(
                0.009,
                0.058,
                0.074,
                12,
                10.0,
                -10.0,
                700,
                0.1,
                ds,
                expected,
                errorFunction
        );

        NeuralNetwork nn = new NeuralNetwork.NeuralNetworkBuilder()
                .setActivationFunction(activationFunction)
                .setErrorFunction(errorFunction)
                .setTrainingMethod(trainMethod)
                .setWeightInitializer(weightInitializer)
                .addLayer(22)
                .addLayer(10)
                .addLayer(2)
                .build();

        System.out.println("Starting train");
        nn.train();
        System.out.println("Ending train");

        for (int i = 0; i < ds.length; i++ ) {
            nn.setInputLayer(ds[i]);
            nn.think();
            System.out.println(Arrays.toString(nn.getOutputLayer()) + "Expected: " + recordArrayList.get(i).get(0));
        }
    }
}
