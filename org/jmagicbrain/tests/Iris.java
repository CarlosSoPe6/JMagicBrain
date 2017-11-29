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

public class Iris {
    public static void main(String[] args){
        double[][] trainData = new double[24][];
        trainData[0] = new double[] { 6.3, 2.9, 5.6, 1.8};
        trainData[1] = new double[] { 6.9, 3.1, 4.9, 1.5};
        trainData[2] = new double[] { 4.6, 3.4, 1.4, 0.3};
        trainData[3] = new double[] { 7.2, 3.6, 6.1, 2.5};
        trainData[4] = new double[] { 4.7, 3.2, 1.3, 0.2};
        trainData[5] = new double[] { 4.9, 3, 1.4, 0.2, };
        trainData[6] = new double[] { 7.6, 3, 6.6, 2.1  };
        trainData[7] = new double[] { 4.9, 2.4, 3.3, 1  };
        trainData[8] = new double[] { 5.4, 3.9, 1.7, 0.4};
        trainData[9] = new double[] { 4.9, 3.1, 1.5, 0.1};
        trainData[10] = new double[] { 5, 3.6, 1.4, 0.2 };
        trainData[11] = new double[] { 6.4, 3.2, 4.5, 1.5};
        trainData[12] = new double[] { 4.4, 2.9, 1.4, 0.2};
        trainData[13] = new double[] { 5.8, 2.7, 5.1, 1.9};
        trainData[14] = new double[] { 6.3, 3.3, 6, 2.5, };
        trainData[15] = new double[] { 5.2, 2.7, 3.9, 1.4};
        trainData[16] = new double[] { 7, 3.2, 4.7, 1.4  };
        trainData[17] = new double[] { 6.5, 2.8, 4.6, 1.5};
        trainData[18] = new double[] { 4.9, 2.5, 4.5, 1.7};
        trainData[19] = new double[] { 5.7, 2.8, 4.5, 1.3};
        trainData[20] = new double[] { 5, 3.4, 1.5, 0.2};
        trainData[21] = new double[] { 6.5, 3, 5.8, 2.2};
        trainData[22] = new double[] { 5.5, 2.3, 4, 1.3};
        trainData[23] = new double[] { 6.7, 2.5, 5.8, 1.8};

        double[][] expected = new double[24][];
        expected[0] = new double[]{1, 0, 0  };
        expected[1] = new double[]{0, 1, 0  };
        expected[2] = new double[]{0, 0, 1  };
        expected[3] = new double[]{1, 0, 0  };
        expected[4] = new double[]{0, 0, 1  };
        expected[5] = new double[]{0, 0, 1  };
        expected[6] = new double[]{1, 0, 0  };
        expected[7] = new double[]{0, 1, 0  };
        expected[8] = new double[]{0, 0, 1  };
        expected[9] = new double[]{0, 0, 1  };
        expected[10] = new double[]{0, 0, 1 };
        expected[11] = new double[]{0, 1, 0 };
        expected[12] = new double[]{0, 0, 1 };
        expected[13] = new double[]{1, 0, 0 };
        expected[14] = new double[]{1, 0, 0 };
        expected[15] = new double[]{0, 1, 0 };
        expected[16] = new double[]{0, 1, 0 };
        expected[17] = new double[]{0, 1, 0 };
        expected[18] = new double[]{1, 0, 0 };
        expected[19] = new double[]{0, 1, 0 };
        expected[20] = new double[]{0, 0, 1 };
        expected[21] = new double[]{1, 0, 0 };
        expected[22] = new double[]{0, 1, 0 };
        expected[23] = new double[]{1, 0, 0 };

        ActivationFunction activationFunction = new Sigmoid();
        ErrorFunction errorFunction = new MeanSquaredError();
        WeightInitializer weightInitializer = new DefaultInitializer();
        TrainMethod trainMethod = new ParticleSwarmOptimization(
                0.01,
                2.4896,
                2.4896,
                12,
                10.0,
                -10.0,
                1111,
                0.060,
                trainData,
                expected,
                errorFunction
        );

        NeuralNetwork nn = new NeuralNetwork.NeuralNetworkBuilder()
                .setActivationFunction(activationFunction)
                .setErrorFunction(errorFunction)
                .setTrainingMethod(trainMethod)
                .setWeightInitializer(weightInitializer)
                .addLayer(4)
                .addLayer(6)
                .addLayer(3)
                .build();

        System.out.println("Starting train");
        nn.train();
        System.out.println("Ending train");
    }
}
