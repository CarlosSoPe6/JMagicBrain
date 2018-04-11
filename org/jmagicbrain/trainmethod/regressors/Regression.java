package org.jmagicbrain.trainmethod.regressors;

import org.jmagicbrain.trainmethod.Trainer;

public abstract class Regression implements Trainer{
    private double[][] dependentVariables;
    private double[][] independentVariables;

    public void setDependentVariables(double[][] dependentVariables) {
        this.dependentVariables = dependentVariables;
    }

    public void setIndependentVariables(double[][] independentVariables) {
        this.independentVariables = independentVariables;
    }

    public abstract double[][] predict(double[][] dataIn);
}
