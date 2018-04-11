package org.jmagicbrain.functions;

public class Sigmoid implements ActivationFunction {
    @Override
    public double func(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }

    @Override
    public double deriv(double x) {
        double sigmoid = func(x);
        return sigmoid * (1 - sigmoid);
    }
}
