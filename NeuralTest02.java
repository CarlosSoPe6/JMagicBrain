import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

public class NeuralTest02 {

    private double[][] layers;
    private double[][][] weightsMatrix;
    private double[] expectedOutput;
    private double[] errorVector;
    private double[][] sigmas;

    public final double MAX_ERROR = 0.01;
    private final double LEARNING_RATE = 0.2;

    public NeuralTest02(int... args){
        this.layers = new double[args.length][];
        this.weightsMatrix = new double[args.length - 1][][];

        this.expectedOutput = new double[args[args.length - 1]];
        this.errorVector = new double[args[args.length - 1]];
        this.sigmas = new double[args.length - 1][];

        for(int i = 0; i < args.length; i++){
            this.layers[i] = new double[args[i] + 1];
            layers[i][layers[i].length - 1] = 1;
            if(i > 0){
                this.sigmas[i - 1] = new double[args[i]];
                weightsMatrix[i-1] = new double[this.layers[i].length - 1][this.layers[i - 1].length];
            }
        }

        for (int i = 0; i < weightsMatrix.length; i++){
            for(int j = 0; j < weightsMatrix[i].length; j++){
                for(int k = 0; k < weightsMatrix[i][j].length; k++){
                    weightsMatrix[i][j][k] = Math.random();
                }
            }
        }
    }

    public void setInputValue(double value, int index){
        this.layers[0][index] = value;
    }

    public double[] getOutput(){
        return this.layers[this.layers.length -1];
    }

    public void setExpectedOutput(double... args){
        System.arraycopy(args, 0, this.expectedOutput, 0, this.expectedOutput.length);
    }

    public double normalize(double value, double min, double max){
        return (value - min) / (max - min);
    }

    public double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    public double dSigmoid(double x) {
        double sigmoid = sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }

    public void think(){
        double sum = 0;
        // Number of layers - 1
        for(int i = 0; i < weightsMatrix.length; i++){
            // The next layer
            for(int j = 0; j < weightsMatrix[i].length; j++){
                for(int k = 0; k < weightsMatrix[i][j].length; k++){
                    sum += layers[i][k] * weightsMatrix[i][j][k];
                }
                layers[i+1][j] = sigmoid(sum);
                sum = 0;
            }
        }
    }

    public void learn(){
        //System.out.printf("Total Err: %f, Vector %s\n", this.getTotalError(), Arrays.toString(this.errorVector));
        double sigma = 0;

        // Sigma for output, the last row
        for(int i = 0; i < sigmas[sigmas.length - 1].length; i++){
            sigmas[sigmas.length - 1][i] = (layers[sigmas.length][i] - expectedOutput[i]) * dSigmoid(layers[sigmas.length][i]);
        }

        // Others sigmas
        for(int i = sigmas.length - 2; i >= 0; i--){
            for(int j = 0; j < sigmas[i].length; j++){
                for(int k = 0; k < sigmas[i + 1].length; k++){
                    //System.out.printf("i: %d, j: %d, k: %d \n", i, j, k);
                    // TODO: Check this works properly
                    sigma += weightsMatrix[i][j][k] * sigmas[i + 1][k];
                }
                sigmas[i][j] = sigma * dSigmoid(layers[i + 1][j]);
                sigma = 0;
            }
        }

        // Update weights
        for(int i = 0; i < weightsMatrix.length; i++){
            for(int j = 0; j < weightsMatrix[i].length; j++){
                for(int k = 0; k < weightsMatrix[i][j].length; k++){
                    try {
                        weightsMatrix[i][j][k] = weightsMatrix[i][j][k] - LEARNING_RATE * layers[i][k] * sigmas[i][j];
                    }catch (Exception e){
                        System.out.printf("i: %d, j: %d, k: %d \n", i, j, k);
                        System.out.printf("x: %d, y: %d, z: %d \n", weightsMatrix.length,  weightsMatrix[i].length, weightsMatrix[i][j].length);
                        System.out.printf("0: %d, a: %d, b: %d \n", 0, layers[i].length, sigmas[i].length);
                        e.printStackTrace();
                        System.exit(2);
                    }
                }
            }
        }
    }

    public double getTotalError(){
        double error = 0;
        for(int i = 0; i < errorVector.length; i++){
            errorVector[i] = 0.5 * Math.pow(expectedOutput[i] - layers[layers.length - 1][i], 2);
        }
        for (double e : errorVector){
            error += e;
        }
        return  error;
    }

    public static void main(String[] args) throws IOException {
        NeuralTest02 nt2 = new NeuralTest02(13, 17, 3);
        double NormalizationMax[] = {14.83, 5.8, 3.23, 30, 162, 3.88, 5.08, 0.66, 3.58, 13, 1.71, 4, 1680};
        double NormalizationMin[] = {11.03, 0.74, 1.36, 10.6, 70, 0.98, 0.34, 0.13, 0.41, 1.28, 0.48, 1.27, 278};

        Reader in = null;

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
        Iterator<CSVRecord> iterator = records.iterator();;
        while (iterator.hasNext()) {
            record = iterator.next();
            recordArrayList.add(record);
        }
        int epochSize = 178;
        int hits = 0;
        int count = 0;
        double errt = 0;
        boolean endEvaluation = false;
        int epoch = 0;
        while(!endEvaluation && epoch < 10000) {
            for(int r = 0; r < recordArrayList.size(); r++) {
                record = recordArrayList.get(r);
                for (int i = 1; i < record.size(); i++) {
                    nt2.setInputValue(nt2.normalize(Double.parseDouble(record.get(i)), NormalizationMin[i - 1], NormalizationMax[i - 1]), i - 1);
                }
                nt2.think();
                // System.out.println(Arrays.toString(nt2.getOutput()));
                switch (Integer.parseInt(record.get(0))) {
                    case 1:
                        nt2.setExpectedOutput(1.0, 0, 0);
                        break;
                    case 2:
                        nt2.setExpectedOutput(0, 1.0, 0);
                        break;
                    case 3:
                        nt2.setExpectedOutput(0, 0, 1.0);
                        break;
                }

                count++;
                errt = nt2.getTotalError();
                if(nt2.MAX_ERROR > hits){
                    hits++;
                }else{
                    hits = 0;
                    nt2.learn();
                }

                if(hits == epochSize - 1){
                    endEvaluation = true;
                    break;
                }
            }

            if(errt <= 0.02){
                break;
            }
            System.out.printf("Total Error: %f. Epoc: %03d. Entry: %03d\n", errt, epoch, count);
            count = 0;
            epoch++;
        }

        System.out.println("------");
        System.out.printf("Total Error: %f. Epoc: %03d.\n", errt, epoch);
        System.out.println("------");

        for (CSVRecord aRecordArrayList : recordArrayList) {
            for (int i = 1; i < aRecordArrayList.size(); i++) {
                nt2.setInputValue(nt2.normalize(Double.parseDouble(aRecordArrayList.get(i)), NormalizationMin[i - 1], NormalizationMax[i - 1]), i - 1);
            }
            nt2.think();
            System.out.println(Arrays.toString(nt2.getOutput()) + "Expected: " + aRecordArrayList.get(0));
        }
    }
}
