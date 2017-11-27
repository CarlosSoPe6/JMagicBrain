package org.jmagicbrain;

import org.jmagicbrain.errorfunction.ErrorFunction;

public class NeuralNetwork {

    private NeuralNetwork(){}

    public static class NeuralNetworkBuilder {
        private static ErrorFunction errorFunction;

        public static void setErrorFunction(ErrorFunction errorFunction) {
            NeuralNetworkBuilder.errorFunction = errorFunction;
        }

        public NeuralNetwork build() {
            return new NeuralNetwork();
        }
    }
}
