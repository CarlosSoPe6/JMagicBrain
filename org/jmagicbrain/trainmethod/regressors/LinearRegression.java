package org.jmagicbrain.trainmethod.regressors;

public class LinearRegression extends Regression {

    public LinearRegression(){
        super();
        this.regressionValues = new double[1][2];
    }

    @Override
    public double[][] predict(double[][] dataIn) {
        double prediction = (regressionValues[0][1] * dataIn[0][0]) + regressionValues[0][0];
        double[][] retValue = new double[1][1];
        retValue[0][0] = prediction;
        return retValue;
    }

    @Override
    public double train() {
        double averageX = average(independentVariables[independentVariables.length - 1]);
        double averageY = average(dependentVariables[dependentVariables.length - 1]);
        double deviationX = standardDeviation(independentVariables[independentVariables.length - 1], averageX);
        double deviationY = standardDeviation(dependentVariables[dependentVariables.length - 1], averageY);
        double r = correlation(independentVariables[independentVariables.length - 1], dependentVariables[dependentVariables.length - 1]);

        double b = r * (deviationY / deviationX);
        double a = averageY - (averageX * b);

        regressionValues[0][0] = b;
        regressionValues[0][1] = a;

        return 0;
    }

    public double correlation(double[] xData, double[] yData) {
        int n = xData.length;
        int j = 0;

        double r = 0;

        double x;
        double y;
        double xy;
        double sqrX;
        double sqrY;

        double sumXY = 0;
        double sumSqrX = 0;
        double sumSqrY = 0;

        for(int i = 0; i < n; i++){
            x = xData[i];
            y = yData[i];
            xy = x * y;
            sqrX = x * x;
            sqrY = y * y;

            sumXY += xy;
            sumSqrX += sqrX;
            sumSqrY += sqrY;
        }

        r = sumXY / Math.sqrt(sumSqrX * sumSqrY);

        return r;
    }

    private double average(double[] data){
        double avg = 0;
        for(double d : data){
            avg += d;
        }
        return avg / ((double) (data.length));
    }

    private double standardDeviation(double[] data, double average){
        double sd = 0;

        for (int i = 0; i < data.length; i++)
        {
            sd += Math.pow(data[i] - average, 2) / data.length;
        }

        return Math.sqrt(sd);
    }


}
