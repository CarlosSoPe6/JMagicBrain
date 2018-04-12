package org.jmagicbrain.tests;

import org.jmagicbrain.trainmethod.regressors.LinearRegression;

import java.util.Arrays;

public class LinearRegressionTest {
    public static void main(String[] args){

        double[][] depVal = {{1, 2, 3, 4, 5}};
        double[][] indVal = {{1, 2, 1.30, 3.75, 5.25}};

        LinearRegression regression = new LinearRegression();
        regression.setDependentVariables(depVal);
        regression.setIndependentVariables(indVal);

        regression.train();

        double[][] dataToPredict = {{}};

        System.out.println(Arrays.toString(regression.predict(dataToPredict)[0]));
    }
}
