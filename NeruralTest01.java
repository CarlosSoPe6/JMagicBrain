import java.util.Arrays;

/**
 * Prueba de algoritmo
 */
public class NeruralTest01 {

    public double[] inputVector = new double[13];
    public double[] outputVector = new double[3];
    private double[][] hiddenLayers = new double[1][17];
    private double[][][] weightMatrix = new double[2][][];
    private double[] expectedOutput = new double[3];
    private final double MAX_ERROR = 0.001;
    private final double LEARNING_RATE = 0.02;

    private final double[][] sigmas = new double[2][];

    public NeruralTest01(){
        weightMatrix[0] = new double[17][13];
        weightMatrix[1] = new double[3][17];

        for (int i = 0; i < weightMatrix.length; i++){
            for(int j = 0; j < weightMatrix[i].length; j++){
                for(int k = 0; k < weightMatrix[i][j].length; k++){
                    weightMatrix[i][j][k] = Math.random();
                }
            }
        }
    }

    public double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    public double dsigmoid(double x) {
        double sigmoid = sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }

    public void think(){
        double sum = 0;
        for(int j = 0; j < weightMatrix[0].length; j++){
            for(int k = 0; k < weightMatrix[0][j].length; k++){
                sum += inputVector[k] * weightMatrix[0][j][k];
            }
            hiddenLayers[0][j] = sigmoid(sum);
            sum = 0;
        }
        System.out.println(Arrays.toString(hiddenLayers[0]));
        for(int j = 0; j < weightMatrix[1].length; j++){
            for(int k = 0; k < weightMatrix[1][j].length; k++){
                sum += hiddenLayers[0][k] * weightMatrix[1][j][k];
            }
            outputVector[j] = sigmoid(sum);
            sum = 0;
        }
    }

    public void setInputValues(double... args){
        System.arraycopy(args, 0, inputVector, 0, inputVector.length);
    }

    public void learn(){
        double[] outputSigmas = new double[3];
        double totalError = 0;
        double error = 0;
        for(int i = 0; i < expectedOutput.length; i++){
            error = Math.pow(outputVector[i] - expectedOutput[i], 2);
            outputSigmas[i] = -error * dsigmoid(outputVector[i]);
            totalError += error;
        }
        totalError /= 2;
        System.out.println("Total error: " + totalError);

        sigmas[1] = outputSigmas;
        System.out.printf("Out[i] sigmas: %s \n", Arrays.toString(outputSigmas));

        double[] hiddenSigmas = new double[17];
        for (int i = 0; i < hiddenLayers[0].length; i++){
            for(int j = 0; j < sigmas[1].length; j++){
                hiddenSigmas[i] += sigmas[1][j] * hiddenLayers[0][i];
            }
            hiddenSigmas[i] *= dsigmoid(hiddenLayers[0][i]);
        }

        sigmas[0] = hiddenSigmas;

        System.out.printf("H[i] sigmas: %s \n", Arrays.toString(hiddenSigmas));

        for(int j = 0; j < weightMatrix[0].length; j++){
            for(int k = 0; k < weightMatrix[0][j].length; k++){
                weightMatrix[0][j][k] += sigmas[0][j] * weightMatrix[0][j][k] * inputVector[k];
            }
        }

        for(int j = 0; j < weightMatrix[1].length; j++){
            for(int k = 0; k < weightMatrix[1][j].length; k++){
                weightMatrix[1][j][k] += sigmas[1][j] * weightMatrix[1][j][k] * hiddenLayers[0][k];
            }
        }

    }

    public double normalize(double value, double min, double max){
        return (value - min) / (max - min);
    }

    public static void main(String[] args){
        NeruralTest01 nt2 = new NeruralTest01();
        nt2.setInputValues(
                nt2.normalize(14.23, 11.03, 14.83),
                nt2.normalize(1.71, 0.74, 5.8),
                nt2.normalize(2.43, 1.36, 3.23),
                nt2.normalize(15.6, 10.6, 30.0),
                nt2.normalize(127.0, 70.0, 162.0),
                nt2.normalize(2.8, 0.98, 3.88),
                nt2.normalize(3.06, 0.34, 5.08),
                nt2.normalize(0.28, 0.13, 0.66),
                nt2.normalize(2.29, 0.41, 3.58),
                nt2.normalize(5.64, 1.28, 13.0),
                nt2.normalize(1.04, 0.48, 1.71),
                nt2.normalize(3.92, 1.27, 4.0),
                nt2.normalize(1065.0, 278.0, 1680.0)
        );
        nt2.think();
        System.out.println("----------");
        System.out.println(Arrays.toString(nt2.outputVector));
        System.out.println("----------");
        System.out.println(Arrays.toString(nt2.outputVector));
    }
}
